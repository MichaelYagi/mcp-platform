"""
Utilities Module
Helper functions for the MCP client
"""

import mimetypes
import platform
import requests
import socket
import subprocess
import threading
import urllib.parse
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
            pass

        def handle_one_request(self):
            try:
                super().handle_one_request()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            # /image?path=<absolute_path> — serve any local image file
            if self.path.startswith("/image"):
                parsed = urllib.parse.urlparse(self.path)
                params = urllib.parse.parse_qs(parsed.query)
                file_path = params.get("path", [""])[0]
                if not file_path:
                    self.send_error(400, "Missing path parameter")
                    return
                p = Path(file_path)
                if not p.exists() or not p.is_file():
                    self.send_error(404, f"File not found: {file_path}")
                    return
                # Only serve image files
                mime, _ = mimetypes.guess_type(str(p))
                if not mime or not mime.startswith("image/"):
                    mime = "image/jpeg"
                try:
                    data = p.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", mime)
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "max-age=3600")
                    self.end_headers()
                    self.wfile.write(data)
                except Exception as e:
                    self.send_error(500, str(e))
                return
            super().do_GET()

    # Resolve private IP before starting thread so caller can use it
    try:
        # Find the LAN IP (prefer 192.168.x.x over WSL2 172.x.x.x)
        private_ip = "127.0.0.1"
        for iface_ip in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = iface_ip[4][0]
            if ip.startswith("192.168."):
                private_ip = ip
                break
        # Fallback to UDP trick if no 192.168 found
        if private_ip == "127.0.0.1":
            _s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            _s.connect(("8.8.8.8", 80))
            private_ip = _s.getsockname()[0]
            _s.close()
    except Exception:
        private_ip = "127.0.0.1"

    def serve():
        with TCPServer(("0.0.0.0", port), QuietHTTPRequestHandler) as httpd:
            print(f"📄 HTTP server listening on 0.0.0.0:{port}")
            print(f"   Local: http://127.0.0.1:{port}/client/ui/index.html")
            print(f"   Network: http://{private_ip}:{port}/client/ui/index.html")
            httpd.serve_forever()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return private_ip


def open_browser_file(path: Path):
    """Open a file in the default browser"""
    if "microsoft" in platform.uname().release.lower():
        windows_path = str(path).replace("/mnt/c", "C:").replace("/", "\\")
        subprocess.run(["cmd.exe", "/c", "start", windows_path], shell=False)
    else:
        webbrowser.open(f"file://{path}")


def open_browser_url(url: str):
    """Open a URL in the default browser"""
    if "microsoft" in platform.uname().release.lower():
        subprocess.run(["cmd.exe", "/c", "start", url], shell=False)
    else:
        webbrowser.open(url)


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