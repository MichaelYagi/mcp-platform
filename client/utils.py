"""
Utilities Module
Helper functions for the MCP client
"""

import platform
import requests
import socket
import subprocess
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer
from pathlib import Path


def get_public_ip():
    """Get public IP address"""
    try:
        return requests.get("https://api.ipify.org").text
    except:
        return None


def get_venv_python(project_root: Path) -> str:
    """Return the correct Python executable path for the project's virtual environment."""
    venv = project_root / ".venv"

    if platform.system() == "Windows":
        candidates = [
            venv / "Scripts" / "python.exe",
            venv / "Scripts" / "python",
        ]
    else:
        candidates = [
            venv / "bin" / "python",
            project_root / ".venv-wsl" / "bin" / "python",
        ]

    for path in candidates:
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        f"No valid Python executable found. Checked: {', '.join(str(p) for p in candidates)}"
    )


def start_http_server(port=9000):
    """Serve index.html over HTTP on the network"""

    class QuietHTTPRequestHandler(SimpleHTTPRequestHandler):
        def log_message(self, format_log, *args):
            # Suppress noisy access logs - only errors will print
            pass

        def handle_one_request(self):
            try:
                super().handle_one_request()
            except (BrokenPipeError, ConnectionResetError):
                # Client disconnected mid-response - ignore silently
                pass

    def serve():
        with TCPServer(("0.0.0.0", port), QuietHTTPRequestHandler) as httpd:
            try:
                # Get actual network IP (not 127.0.1.1)
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()

                print(f"📄 HTTP server listening on 0.0.0.0:{port}")
                print(f"   Local: http://localhost:{port}/index.html")
                print(f"   Network: http://{local_ip}:{port}/index.html")
            except:
                print(f"📄 HTTP server running on 0.0.0.0:{port}")
            httpd.serve_forever()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()


def open_browser_file(path: Path):
    """Open a file in the default browser"""
    if "microsoft" in platform.uname().release.lower():
        windows_path = str(path).replace("/mnt/c", "C:").replace("/", "\\")
        subprocess.run(["cmd.exe", "/c", "start", windows_path], shell=False)
    else:
        webbrowser.open(f"file://{path}")


async def ensure_ollama_running(host: str = "http://127.0.0.1:11434"):
    """Check if Ollama server is running"""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            r = await client.get(f"{host}/api/tags")
            r.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"Ollama server is not running or unreachable at {host}. "
            f"Start it with 'ollama serve'. Original error: {e}"
        )