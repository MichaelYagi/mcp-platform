"""
Agent Health Monitoring System
Tracks agent status, performance, and health metrics
"""

import time
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


class HealthStatus(Enum):
    """Agent health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthMetrics:
    """Health metrics for an agent"""
    agent_id: str
    status: HealthStatus
    last_heartbeat: float
    uptime: float

    # Performance metrics
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_response_time: float = 0.0
    current_load: float = 0.0

    # Resource metrics
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    queue_size: int = 0

    # Error tracking
    error_count: int = 0
    last_error: Optional[str] = None
    last_error_time: Optional[float] = None

    # Recent history
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    error_history: deque = field(default_factory=lambda: deque(maxlen=50))


@dataclass
class HealthAlert:
    """Health alert notification"""
    alert_id: str
    agent_id: str
    level: AlertLevel
    message: str
    timestamp: float
    metric: str
    value: Any
    threshold: Any


class HealthMonitor:
    """
    Monitors agent health and performance
    """

    def __init__(self, logger):
        self.logger = logger
        self.agent_metrics: Dict[str, HealthMetrics] = {}
        self.alerts: List[HealthAlert] = []

        # Thresholds for alerts
        self.thresholds = {
            "max_response_time": 10.0,  # seconds
            "max_error_rate": 0.2,  # 20%
            "max_queue_size": 50,
            "heartbeat_timeout": 30.0,  # seconds
            "max_memory_usage": 0.9,  # 90%
            "max_cpu_usage": 0.9  # 90%
        }

        # Monitoring state
        self.monitoring_active = False
        self.monitor_task = None

    def register_agent(self, agent_id: str):
        """Register an agent for monitoring"""
        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = HealthMetrics(
                agent_id=agent_id,
                status=HealthStatus.HEALTHY,
                last_heartbeat=time.time(),
                uptime=0.0
            )
            self.logger.info(f"💚 Monitoring agent: {agent_id}")

    def unregister_agent(self, agent_id: str):
        """Stop monitoring an agent"""
        if agent_id in self.agent_metrics:
            del self.agent_metrics[agent_id]
            self.logger.info(f"💔 Stopped monitoring: {agent_id}")

    def heartbeat(self, agent_id: str):
        """Record agent heartbeat"""
        if agent_id in self.agent_metrics:
            metrics = self.agent_metrics[agent_id]
            current_time = time.time()

            # Update uptime
            if metrics.last_heartbeat:
                metrics.uptime += current_time - metrics.last_heartbeat

            metrics.last_heartbeat = current_time

            # Update status
            if metrics.status == HealthStatus.OFFLINE:
                metrics.status = HealthStatus.HEALTHY
                self.logger.info(f"💚 Agent {agent_id} back online")

    def record_task_completion(self, agent_id: str, response_time: float, success: bool):
        """Record task completion metrics"""
        if agent_id not in self.agent_metrics:
            return

        metrics = self.agent_metrics[agent_id]

        if success:
            metrics.tasks_completed += 1
        else:
            metrics.tasks_failed += 1

        # Update response time
        metrics.response_times.append(response_time)
        if metrics.response_times:
            metrics.avg_response_time = sum(metrics.response_times) / len(metrics.response_times)

        # Check for performance degradation
        if response_time > self.thresholds["max_response_time"]:
            self._create_alert(
                agent_id,
                AlertLevel.WARNING,
                f"Slow response time: {response_time:.2f}s",
                "response_time",
                response_time,
                self.thresholds["max_response_time"]
            )

        # Check error rate
        total_tasks = metrics.tasks_completed + metrics.tasks_failed
        if total_tasks > 10:
            error_rate = metrics.tasks_failed / total_tasks
            if error_rate > self.thresholds["max_error_rate"]:
                self._create_alert(
                    agent_id,
                    AlertLevel.ERROR,
                    f"High error rate: {error_rate:.1%}",
                    "error_rate",
                    error_rate,
                    self.thresholds["max_error_rate"]
                )
                metrics.status = HealthStatus.DEGRADED

    def record_error(self, agent_id: str, error: str):
        """Record an error"""
        if agent_id not in self.agent_metrics:
            return

        metrics = self.agent_metrics[agent_id]
        metrics.error_count += 1
        metrics.last_error = error
        metrics.last_error_time = time.time()

        metrics.error_history.append({
            "error": error,
            "timestamp": time.time()
        })

        self.logger.warning(f"⚠️ Error in {agent_id}: {error}")

    def update_resource_usage(self, agent_id: str, memory: float = None,
                              cpu: float = None, queue_size: int = None):
        """Update resource usage metrics"""
        if agent_id not in self.agent_metrics:
            return

        metrics = self.agent_metrics[agent_id]

        if memory is not None:
            metrics.memory_usage = memory
            if memory > self.thresholds["max_memory_usage"]:
                self._create_alert(
                    agent_id,
                    AlertLevel.WARNING,
                    f"High memory usage: {memory:.1%}",
                    "memory_usage",
                    memory,
                    self.thresholds["max_memory_usage"]
                )

        if cpu is not None:
            metrics.cpu_usage = cpu
            if cpu > self.thresholds["max_cpu_usage"]:
                self._create_alert(
                    agent_id,
                    AlertLevel.WARNING,
                    f"High CPU usage: {cpu:.1%}",
                    "cpu_usage",
                    cpu,
                    self.thresholds["max_cpu_usage"]
                )

        if queue_size is not None:
            metrics.queue_size = queue_size
            if queue_size > self.thresholds["max_queue_size"]:
                self._create_alert(
                    agent_id,
                    AlertLevel.WARNING,
                    f"Large queue: {queue_size} messages",
                    "queue_size",
                    queue_size,
                    self.thresholds["max_queue_size"]
                )

            # Update current load (queue size as a proxy)
            metrics.current_load = min(queue_size / 10.0, 1.0)

    def _create_alert(self, agent_id: str, level: AlertLevel, message: str,
                      metric: str, value: Any, threshold: Any):
        """Create a health alert"""
        import uuid

        alert = HealthAlert(
            alert_id=f"alert_{uuid.uuid4().hex[:8]}",
            agent_id=agent_id,
            level=level,
            message=message,
            timestamp=time.time(),
            metric=metric,
            value=value,
            threshold=threshold
        )

        self.alerts.append(alert)

        # Log based on level
        log_msg = f"🚨 {agent_id}: {message}"
        if level == AlertLevel.CRITICAL:
            self.logger.critical(log_msg)
        elif level == AlertLevel.ERROR:
            self.logger.error(log_msg)
        elif level == AlertLevel.WARNING:
            self.logger.warning(log_msg)
        else:
            self.logger.info(log_msg)

    async def start_monitoring(self, check_interval: float = 5.0):
        """Start background health monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_task = asyncio.create_task(self._monitor_loop(check_interval))
        self.logger.info(f"💚 Health monitoring started (interval: {check_interval}s)")

    async def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("💔 Health monitoring stopped")

    async def _monitor_loop(self, check_interval: float):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                self._check_heartbeats()
                self._update_health_status()
                await asyncio.sleep(check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}")

    def _check_heartbeats(self):
        """Check for missed heartbeats"""
        current_time = time.time()
        timeout = self.thresholds["heartbeat_timeout"]

        for agent_id, metrics in self.agent_metrics.items():
            time_since_heartbeat = current_time - metrics.last_heartbeat

            if time_since_heartbeat > timeout:
                if metrics.status != HealthStatus.OFFLINE:
                    metrics.status = HealthStatus.OFFLINE
                    self._create_alert(
                        agent_id,
                        AlertLevel.CRITICAL,
                        f"Agent offline (no heartbeat for {time_since_heartbeat:.1f}s)",
                        "heartbeat",
                        time_since_heartbeat,
                        timeout
                    )

    def _update_health_status(self):
        """Update overall health status for each agent"""
        for agent_id, metrics in self.agent_metrics.items():
            if metrics.status == HealthStatus.OFFLINE:
                continue

            # Check multiple factors
            issues = []

            # Error rate
            total_tasks = metrics.tasks_completed + metrics.tasks_failed
            if total_tasks > 10:
                error_rate = metrics.tasks_failed / total_tasks
                if error_rate > self.thresholds["max_error_rate"]:
                    issues.append("high_error_rate")

            # Response time
            if metrics.avg_response_time > self.thresholds["max_response_time"]:
                issues.append("slow_response")

            # Queue size
            if metrics.queue_size > self.thresholds["max_queue_size"]:
                issues.append("overloaded")

            # Update status
            if len(issues) >= 2:
                metrics.status = HealthStatus.UNHEALTHY
            elif len(issues) == 1:
                metrics.status = HealthStatus.DEGRADED
            else:
                metrics.status = HealthStatus.HEALTHY

    def get_agent_health(self, agent_id: str) -> Optional[HealthMetrics]:
        """Get health metrics for an agent"""
        return self.agent_metrics.get(agent_id)

    def get_all_health(self) -> Dict[str, HealthMetrics]:
        """Get health metrics for all agents"""
        return self.agent_metrics.copy()

    def get_recent_alerts(self, limit: int = 50, level: AlertLevel = None) -> List[HealthAlert]:
        """Get recent alerts, optionally filtered by level"""
        alerts = self.alerts[-limit:] if limit else self.alerts

        if level:
            alerts = [a for a in alerts if a.level == level]

        return list(reversed(alerts))  # Most recent first

    def clear_alerts(self, older_than: float = None):
        """Clear old alerts"""
        if older_than:
            current_time = time.time()
            self.alerts = [
                a for a in self.alerts
                if current_time - a.timestamp < older_than
            ]
        else:
            self.alerts.clear()

    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary"""
        if not self.agent_metrics:
            return {"status": "no_agents", "agents": {}}

        summary = {
            "total_agents": len(self.agent_metrics),
            "healthy": 0,
            "degraded": 0,
            "unhealthy": 0,
            "offline": 0,
            "total_tasks": 0,
            "total_errors": 0,
            "avg_response_time": 0.0,
            "recent_alerts": len([a for a in self.alerts if time.time() - a.timestamp < 300])  # Last 5 min
        }

        response_times = []

        for metrics in self.agent_metrics.values():
            # Count by status
            if metrics.status == HealthStatus.HEALTHY:
                summary["healthy"] += 1
            elif metrics.status == HealthStatus.DEGRADED:
                summary["degraded"] += 1
            elif metrics.status == HealthStatus.UNHEALTHY:
                summary["unhealthy"] += 1
            elif metrics.status == HealthStatus.OFFLINE:
                summary["offline"] += 1

            # Aggregate metrics
            summary["total_tasks"] += metrics.tasks_completed + metrics.tasks_failed
            summary["total_errors"] += metrics.tasks_failed

            if metrics.response_times:
                response_times.extend(metrics.response_times)

        if response_times:
            summary["avg_response_time"] = sum(response_times) / len(response_times)

        # Overall status
        if summary["offline"] > 0 or summary["unhealthy"] > 0:
            summary["status"] = "unhealthy"
        elif summary["degraded"] > 0:
            summary["status"] = "degraded"
        else:
            summary["status"] = "healthy"

        return summary