import psutil
import json

def list_processes(top_n=10):
    """Returns the top N processes by memory usage."""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_percent', 'cpu_percent']):
        try:
            processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Sort by memory usage
    processes.sort(key=lambda x: x['memory_percent'], reverse=True)
    return json.dumps(processes[:top_n], indent=2)


def kill_process(pid):
    """Terminates a process by its PID."""
    try:
        proc = psutil.Process(pid)
        name = proc.name()
        proc.terminate()
        return json.dumps({"status": "success", "message": f"Terminated {name} (PID: {pid})"})
    except psutil.NoSuchProcess:
        return json.dumps({"status": "error", "message": "Process ID not found."})
    except psutil.AccessDenied:
        return json.dumps({"status": "error", "message": "Permission denied."})