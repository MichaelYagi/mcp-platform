"""
Real-time System Monitor for MCP Client
Captures CPU and GPU stats like Windows Task Manager
"""

import psutil
import asyncio
import json
from typing import Dict, Any, Optional

# Try to import GPU libraries
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️  GPUtil not installed. GPU monitoring disabled. Install with: pip install gputil")

try:
    import pynvml
    NVML_AVAILABLE = True
    pynvml.nvmlInit()
except:
    NVML_AVAILABLE = False


class SystemMonitor:
    """Monitors system resources in real-time"""

    def __init__(self):
        self.cpu_count = psutil.cpu_count()
        self.cpu_freq_base = psutil.cpu_freq().max if psutil.cpu_freq() else 0

    def get_cpu_stats(self) -> Dict[str, Any]:
        """Get CPU usage and frequency"""
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=False)
        cpu_freq = psutil.cpu_freq()

        return {
            "usage_percent": round(cpu_percent, 1),
            "frequency_ghz": round(cpu_freq.current / 1000, 2) if cpu_freq else 0,
            "max_frequency_ghz": round(cpu_freq.max / 1000, 2) if cpu_freq else 0,
            "cores": self.cpu_count,
            "per_core": [round(x, 1) for x in psutil.cpu_percent(interval=0, percpu=True)]
        }

    def get_gpu_stats(self) -> Optional[Dict[str, Any]]:
        """Get GPU usage and temperature"""
        if not GPU_AVAILABLE:
            return None

        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return None

            gpu = gpus[0]  # Primary GPU

            stats = {
                "name": gpu.name,
                "usage_percent": round(gpu.load * 100, 1),
                "memory_used_mb": round(gpu.memoryUsed, 0),
                "memory_total_mb": round(gpu.memoryTotal, 0),
                "memory_percent": round((gpu.memoryUsed / gpu.memoryTotal) * 100, 1),
                "temperature_c": round(gpu.temperature, 1),
            }

            # Try to get more detailed stats with pynvml
            if NVML_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

                    # Get clock speeds
                    graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                    stats["clock_mhz"] = graphics_clock

                    # Get power usage
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert to watts
                    stats["power_watts"] = round(power_usage, 1)

                except:
                    pass

            return stats

        except Exception as e:
            print(f"Error getting GPU stats: {e}")
            return None

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get RAM usage"""
        mem = psutil.virtual_memory()

        return {
            "used_gb": round(mem.used / (1024**3), 2),
            "total_gb": round(mem.total / (1024**3), 2),
            "percent": round(mem.percent, 1),
            "available_gb": round(mem.available / (1024**3), 2)
        }

    def get_all_stats(self) -> Dict[str, Any]:
        """Get all system stats"""
        stats = {
            "cpu": self.get_cpu_stats(),
            "memory": self.get_memory_stats(),
            "timestamp": asyncio.get_event_loop().time()
        }

        gpu_stats = self.get_gpu_stats()
        if gpu_stats:
            stats["gpu"] = gpu_stats

        return stats


async def system_monitor_loop(websocket_clients, update_interval=1.0):
    """
    Continuously monitor system and broadcast stats to WebSocket clients

    Args:
        websocket_clients: Set of connected WebSocket clients
        update_interval: Seconds between updates (default: 1.0)
    """
    monitor = SystemMonitor()

    print(f"📊 System monitor started (update interval: {update_interval}s)")

    while True:
        try:
            # Get stats
            stats = monitor.get_all_stats()

            # Broadcast to all connected clients
            if websocket_clients:
                message = json.dumps({
                    "type": "system_stats",
                    **stats
                })

                await asyncio.gather(
                    *[ws.send(message) for ws in websocket_clients],
                    return_exceptions=True
                )

            # Wait for next update
            await asyncio.sleep(update_interval)

        except Exception as e:
            print(f"❌ Error in system monitor: {e}")
            await asyncio.sleep(update_interval)


# For testing
if __name__ == "__main__":
    monitor = SystemMonitor()

    print("\n" + "="*60)
    print("System Monitor Test")
    print("="*60)

    stats = monitor.get_all_stats()

    print("\n🖥️  CPU:")
    cpu = stats['cpu']
    print(f"  Usage: {cpu['usage_percent']}%")
    print(f"  Frequency: {cpu['frequency_ghz']} GHz")
    print(f"  Cores: {cpu['cores']}")

    print("\n💾 Memory:")
    mem = stats['memory']
    print(f"  Used: {mem['used_gb']} GB / {mem['total_gb']} GB ({mem['percent']}%)")

    if 'gpu' in stats:
        print("\n🎮 GPU:")
        gpu = stats['gpu']
        print(f"  Name: {gpu['name']}")
        print(f"  Usage: {gpu['usage_percent']}%")
        print(f"  Memory: {gpu['memory_used_mb']} MB / {gpu['memory_total_mb']} MB ({gpu['memory_percent']}%)")
        print(f"  Temperature: {gpu['temperature_c']}°C")
        if 'clock_mhz' in gpu:
            print(f"  Clock: {gpu['clock_mhz']} MHz")
        if 'power_watts' in gpu:
            print(f"  Power: {gpu['power_watts']} W")
    else:
        print("\n⚠️  GPU monitoring not available")

    print("\n" + "="*60)