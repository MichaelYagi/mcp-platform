"""
Shared logging initialiser for all MCP server subprocesses.
Call setup_server_logging() once at module level in each server.
"""
import logging
from pathlib import Path


def setup_server_logging(server_name: str, project_root: Path, json_formatter) -> logging.Logger:
    """Configure root logger with file + console handlers and return the named logger."""
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    formatter = (
        json_formatter()
        if json_formatter is not None
        else logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    file_handler = logging.FileHandler(log_dir / "mcp-server.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.getLogger("mcp").setLevel(logging.DEBUG)
    logging.getLogger(server_name).setLevel(logging.INFO)

    return logging.getLogger(server_name)
