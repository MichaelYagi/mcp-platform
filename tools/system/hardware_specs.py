import platform
import subprocess
import logging
from typing import Dict, Any

logger = logging.getLogger("mcp_server")


def get_hardware_specs() -> Dict[str, Any]:
    """
    Get detailed hardware specifications across different platforms.

    Returns:
        Dictionary with CPU, GPU, RAM, and system information
    """
    system = platform.system()

    specs = {
        "os": f"{platform.system()} {platform.release()}",
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "cpu": {},
        "gpu": {},
        "ram": {}
    }

    try:
        # Get CPU info
        specs["cpu"] = _get_cpu_info(system)

        # Get GPU info
        specs["gpu"] = _get_gpu_info(system)

        # Get RAM info
        specs["ram"] = _get_ram_info(system)

    except Exception as e:
        logger.error(f"Error getting hardware specs: {e}")
        specs["error"] = str(e)

    return specs


def _get_cpu_info(system: str) -> Dict[str, Any]:
    """Get CPU information based on OS"""
    cpu_info = {
        "model": platform.processor() or "Unknown",
        "cores": {"physical": None, "logical": None}
    }

    try:
        if system == "Windows":
            # Use WMIC for detailed CPU info
            try:
                result = subprocess.check_output(
                    ["wmic", "cpu", "get", "Name,NumberOfCores,NumberOfLogicalProcessors"],
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                lines = [line.strip() for line in result.strip().split('\n') if line.strip()]
                if len(lines) > 1:
                    parts = lines[1].split()
                    if len(parts) >= 3:
                        cpu_info["model"] = ' '.join(parts[:-2])
                        cpu_info["cores"]["physical"] = int(parts[-2])
                        cpu_info["cores"]["logical"] = int(parts[-1])
            except:
                pass

        elif system == "Linux":
            # Parse /proc/cpuinfo
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    content = f.read()

                    # Get model name
                    for line in content.split('\n'):
                        if 'model name' in line:
                            cpu_info["model"] = line.split(':')[1].strip()
                            break

                    # Get core counts
                    physical_cores = len([line for line in content.split('\n') if 'core id' in line])
                    logical_cores = len([line for line in content.split('\n') if line.startswith('processor')])

                    if physical_cores > 0:
                        cpu_info["cores"]["physical"] = physical_cores
                    if logical_cores > 0:
                        cpu_info["cores"]["logical"] = logical_cores
            except:
                pass

        elif system == "Darwin":  # macOS
            try:
                # Get CPU brand
                result = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True)
                cpu_info["model"] = result.strip()

                # Get core counts
                physical = subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True)
                logical = subprocess.check_output(["sysctl", "-n", "hw.logicalcpu"], text=True)

                cpu_info["cores"]["physical"] = int(physical.strip())
                cpu_info["cores"]["logical"] = int(logical.strip())
            except:
                pass

    except Exception as e:
        logger.error(f"Error getting CPU info: {e}")

    return cpu_info


def _get_gpu_info(system: str) -> Dict[str, Any]:
    """Get GPU information based on OS"""
    gpu_info = {
        "devices": []
    }

    try:
        if system == "Windows":
            # Use WMIC for GPU info
            try:
                result = subprocess.check_output(
                    ["wmic", "path", "win32_VideoController", "get", "Name,AdapterRAM"],
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                lines = [line.strip() for line in result.strip().split('\n') if line.strip()]

                for i, line in enumerate(lines[1:]):  # Skip header
                    if line:
                        parts = line.rsplit(None, 1)
                        if len(parts) == 2:
                            name, vram_bytes = parts
                            try:
                                vram_gb = int(vram_bytes) / (1024 ** 3)
                                gpu_info["devices"].append({
                                    "name": name.strip(),
                                    "vram_gb": round(vram_gb, 2)
                                })
                            except:
                                gpu_info["devices"].append({"name": name.strip()})
            except:
                pass

        elif system == "Linux":
            # Try lspci for GPU info
            try:
                result = subprocess.check_output(
                    ["lspci"],
                    text=True,
                    stderr=subprocess.DEVNULL
                )

                for line in result.split('\n'):
                    if 'VGA' in line or '3D controller' in line:
                        # Extract GPU name after the colon
                        if ':' in line:
                            gpu_name = line.split(':', 2)[-1].strip()
                            gpu_info["devices"].append({"name": gpu_name})
            except:
                pass

            # Try nvidia-smi for NVIDIA GPUs
            try:
                result = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                    text=True,
                    stderr=subprocess.DEVNULL
                )

                gpu_info["devices"] = []  # Replace with nvidia-smi data
                for line in result.strip().split('\n'):
                    if line:
                        parts = line.split(',')
                        if len(parts) == 2:
                            name = parts[0].strip()
                            vram = parts[1].strip().replace(' MiB', '')
                            try:
                                vram_gb = int(vram) / 1024
                                gpu_info["devices"].append({
                                    "name": name,
                                    "vram_gb": round(vram_gb, 2)
                                })
                            except:
                                gpu_info["devices"].append({"name": name})
            except:
                pass

        elif system == "Darwin":  # macOS
            try:
                result = subprocess.check_output(
                    ["system_profiler", "SPDisplaysDataType"],
                    text=True
                )

                for line in result.split('\n'):
                    if 'Chipset Model:' in line or 'Graphics:' in line:
                        gpu_name = line.split(':', 1)[1].strip()
                        gpu_info["devices"].append({"name": gpu_name})
            except:
                pass

    except Exception as e:
        logger.error(f"Error getting GPU info: {e}")

    return gpu_info


def _get_ram_info(system: str) -> Dict[str, Any]:
    """Get RAM information based on OS"""
    ram_info = {
        "total_gb": None,
        "type": None,
        "speed_mhz": None
    }

    try:
        if system == "Windows":
            # Get total RAM
            try:
                result = subprocess.check_output(
                    ["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"],
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                lines = [line.strip() for line in result.strip().split('\n') if line.strip()]
                if len(lines) > 1:
                    total_bytes = int(lines[1])
                    ram_info["total_gb"] = round(total_bytes / (1024 ** 3), 2)
            except:
                pass

            # Get RAM type and speed
            try:
                result = subprocess.check_output(
                    ["wmic", "memorychip", "get", "MemoryType,Speed"],
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                lines = [line.strip() for line in result.strip().split('\n') if line.strip()]
                if len(lines) > 1:
                    parts = lines[1].split()
                    if len(parts) >= 2:
                        # Memory type codes: 20=DDR, 21=DDR2, 24=DDR3, 26=DDR4, 34=DDR5
                        type_map = {20: "DDR", 21: "DDR2", 24: "DDR3", 26: "DDR4", 34: "DDR5"}
                        try:
                            mem_type = int(parts[0])
                            ram_info["type"] = type_map.get(mem_type, f"Type {mem_type}")
                        except:
                            pass

                        try:
                            ram_info["speed_mhz"] = int(parts[1])
                        except:
                            pass
            except:
                pass

        elif system == "Linux":
            # Get total RAM from /proc/meminfo
            try:
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            total_kb = int(line.split()[1])
                            ram_info["total_gb"] = round(total_kb / (1024 ** 2), 2)
                            break
            except:
                pass

            # Try dmidecode for detailed info (requires sudo)
            try:
                result = subprocess.check_output(
                    ["sudo", "dmidecode", "-t", "memory"],
                    text=True,
                    stderr=subprocess.DEVNULL
                )

                for line in result.split('\n'):
                    if 'Type:' in line and 'DDR' in line:
                        ram_info["type"] = line.split(':', 1)[1].strip()
                    elif 'Speed:' in line and 'MHz' in line:
                        try:
                            speed = line.split(':', 1)[1].strip().replace(' MHz', '')
                            ram_info["speed_mhz"] = int(speed)
                        except:
                            pass
            except:
                pass

        elif system == "Darwin":  # macOS
            try:
                # Get total RAM
                result = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
                total_bytes = int(result.strip())
                ram_info["total_gb"] = round(total_bytes / (1024 ** 3), 2)

                # Get RAM type
                result = subprocess.check_output(
                    ["system_profiler", "SPMemoryDataType"],
                    text=True
                )

                for line in result.split('\n'):
                    if 'Type:' in line:
                        ram_info["type"] = line.split(':', 1)[1].strip()
                    elif 'Speed:' in line:
                        try:
                            speed = line.split(':', 1)[1].strip().replace(' MHz', '')
                            ram_info["speed_mhz"] = int(speed)
                        except:
                            pass
            except:
                pass

    except Exception as e:
        logger.error(f"Error getting RAM info: {e}")

    return ram_info