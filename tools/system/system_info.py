import psutil
import platform
import shutil
import json
import os


def get_system_stats():
    """Gathers CPU, RAM, and Disk usage with WSL2 + Windows + Linux support."""

    # Detect environment
    system = platform.system()
    release = platform.release().lower()

    # CPU usage
    cpu_pct = psutil.cpu_percent(interval=1)

    # RAM usage
    mem = psutil.virtual_memory()

    # Determine disk path based on OS
    if system == "Windows":
        # Native Windows
        disk_path = "C:\\"

    elif system == "Linux" and "microsoft" in release:
        # WSL2 environment
        # Use Linux root filesystem for disk stats
        disk_path = "/"

    else:
        # Native Linux or macOS
        disk_path = "/"

    # Disk usage
    total, used, free = shutil.disk_usage(disk_path)

    stats = {
        "os": f"{system} (WSL2)" if system == "Linux" and "microsoft" in release else system,
        "cpu_usage_percent": cpu_pct,
        "memory_usage_percent": mem.percent,
        "disk_path_used": disk_path,
        "disk_free_gb": round(free / (2 ** 30), 2),
        "disk_total_gb": round(total / (2 ** 30), 2)
    }

    return json.dumps(stats, indent=2)