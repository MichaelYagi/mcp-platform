"""
Logging Module
Handles WebSocket log broadcasting and server log file tailing
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path


LOG_WEBSOCKET_CLIENTS = set()
MAIN_EVENT_LOOP = None


class WebSocketLogHandler(logging.Handler):
    """Custom log handler that broadcasts logs to WebSocket clients"""

    def emit(self, record):
        try:
            log_entry = {
                "timestamp": self.format_time(record),
                "level": record.levelname,
                "name": record.name,
                "message": record.getMessage()
            }

            # Broadcast to all connected log clients
            # Use run_coroutine_threadsafe to schedule the coroutine on the main event loop
            if MAIN_EVENT_LOOP is not None:
                asyncio.run_coroutine_threadsafe(
                    self.broadcast_log(log_entry),
                    MAIN_EVENT_LOOP
                )
        except Exception:
            self.handleError(record)

    def format_time(self, record):
        return datetime.fromtimestamp(record.created).isoformat()

    async def broadcast_log(self, log_entry):
        """Send log entry to all connected WebSocket clients"""
        if LOG_WEBSOCKET_CLIENTS:
            message = json.dumps({"type": "log", **log_entry})
            await asyncio.gather(
                *[ws.send(message) for ws in LOG_WEBSOCKET_CLIENTS],
                return_exceptions=True
            )


async def tail_log_file(filepath: Path, check_interval: float = 0.5):
    """
    Tail a log file and stream new lines to WebSocket clients.
    This allows us to capture server.py logs that are written to the shared log file.
    """
    if not filepath.exists():
        print(f"⏳ Waiting for log file to be created: {filepath}")
        # Wait for file to be created (max 30 seconds)
        for _ in range(30):
            await asyncio.sleep(1)
            if filepath.exists():
                break
        else:
            print(f"❌ Log file never created: {filepath}")
            return

    print(f"📋 Starting to tail log file: {filepath}")

    # Track last file position and size
    last_size = 0
    last_position = 0

    # Seek to end initially to only read new content
    try:
        last_size = filepath.stat().st_size
        last_position = last_size
        print(f"📋 Initial file size: {last_size} bytes, starting from end")
    except Exception as e:
        print(f"❌ Error getting initial file size: {e}")
        last_size = 0
        last_position = 0

    lines_read = 0

    while True:
        try:
            # Check if file size changed
            current_size = filepath.stat().st_size

            if current_size < last_size:
                # File was truncated/rotated, start from beginning
                print(f"📋 Log file was truncated, restarting from beginning")
                last_position = 0
                last_size = current_size

            if current_size > last_size:
                # New content available
                with open(filepath, 'r', encoding='utf-8', errors='replace') as file:
                    # Seek to last known position
                    file.seek(last_position)

                    # Read all new lines
                    new_lines = file.readlines()

                    for line in new_lines:
                        line = line.strip()
                        if not line:
                            continue

                        lines_read += 1

                        # Debug: print every 10th line to console
                        if lines_read % 10 == 0:
                            print(f"📋 Tailed {lines_read} lines from server log")

                        # Parse the log line — may be JSON (from JsonFormatter) or plain text
                        log_entry = {
                            "timestamp": datetime.now().isoformat(),
                            "level": "INFO",
                            "name": "SERVER",
                            "message": line
                        }

                        # Try to parse as JSON first (JsonFormatter output)
                        try:
                            parsed = json.loads(line)
                            if isinstance(parsed, dict):
                                # Map JsonFormatter fields → UI fields
                                # ts is Unix epoch float → convert to ISO string
                                ts = parsed.get("ts")
                                if ts:
                                    log_entry["timestamp"] = datetime.fromtimestamp(ts).isoformat()
                                log_entry["level"]   = parsed.get("level", "INFO")
                                log_entry["name"]    = parsed.get("logger", parsed.get("name", "SERVER"))
                                log_entry["message"] = parsed.get("msg", parsed.get("message", line))
                        except (json.JSONDecodeError, ValueError):
                            # Plain text log line — extract level and logger the old way
                            if "[DEBUG]" in line:
                                log_entry["level"] = "DEBUG"
                            elif "[INFO]" in line:
                                log_entry["level"] = "INFO"
                            elif "[WARNING]" in line or "[WARN]" in line:
                                log_entry["level"] = "WARNING"
                            elif "[ERROR]" in line:
                                log_entry["level"] = "ERROR"

                            if "] " in line:
                                try:
                                    parts = line.split("] ", 1)
                                    if len(parts) > 1:
                                        remaining = parts[1]
                                        if ": " in remaining:
                                            log_entry["name"] = remaining.split(": ", 1)[0]
                                except Exception:
                                    pass

                        # Broadcast to WebSocket clients
                        if LOG_WEBSOCKET_CLIENTS and MAIN_EVENT_LOOP:
                            message = json.dumps({"type": "log", **log_entry})
                            await asyncio.gather(
                                *[ws.send(message) for ws in LOG_WEBSOCKET_CLIENTS],
                                return_exceptions=True
                            )

                    # Update position
                    last_position = file.tell()
                    last_size = current_size

            # Wait before next check
            await asyncio.sleep(check_interval)

        except FileNotFoundError:
            print(f"❌ Log file disappeared: {filepath}")
            await asyncio.sleep(1)
        except Exception as e:
            print(f"❌ Error tailing log file: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(1)


async def log_websocket_handler(websocket):
    """Handle WebSocket connections for log streaming"""
    LOG_WEBSOCKET_CLIENTS.add(websocket)

    try:
        async for message in websocket:
            pass
    except Exception:
        pass
    finally:
        LOG_WEBSOCKET_CLIENTS.discard(websocket)


def setup_logging(client_log_file: Path, log_level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(client_log_file, encoding="utf-8"),
            logging.StreamHandler()
        ],
    )

    # Add WebSocket log handler to root logger
    ws_log_handler = WebSocketLogHandler()
    ws_log_handler.setLevel(logging.DEBUG)
    root_logger = logging.getLogger()
    root_logger.addHandler(ws_log_handler)

    # Set specific log levels for different components
    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("langchain").setLevel(logging.DEBUG)
    logging.getLogger("mcp").setLevel(logging.DEBUG)

    # Ensure propagation
    logging.getLogger("httpx").propagate = True
    logging.getLogger("langchain").propagate = True
    logging.getLogger("mcp").propagate = True


def set_event_loop(loop):
    """Set the main event loop for WebSocket log handler"""
    global MAIN_EVENT_LOOP
    MAIN_EVENT_LOOP = loop