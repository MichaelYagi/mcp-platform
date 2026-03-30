"""
Agent Performance Metrics System
Tracks detailed performance metrics and provides analytics
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics


@dataclass
class TaskMetrics:
    """Metrics for a single task"""
    task_id: str
    agent_id: str
    task_type: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    tools_used: List[str] = field(default_factory=list)
    llm_calls: int = 0
    tokens_used: int = 0
    error: Optional[str] = None


@dataclass
class AgentPerformanceProfile:
    """Performance profile for an agent"""
    agent_id: str

    # Task statistics
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0

    # Timing statistics
    total_duration: float = 0.0
    avg_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0

    # Recent performance
    recent_durations: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_successes: deque = field(default_factory=lambda: deque(maxlen=100))

    # Tool usage
    tool_usage: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # LLM statistics
    total_llm_calls: int = 0
    total_tokens: int = 0

    # Task type breakdown
    task_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Time-based metrics
    hourly_tasks: Dict[int, int] = field(default_factory=lambda: defaultdict(int))


class PerformanceMetrics:
    """
    Tracks and analyzes agent performance metrics
    """

    def __init__(self, logger):
        self.logger = logger

        # Per-agent profiles
        self.agent_profiles: Dict[str, AgentPerformanceProfile] = {}

        # Task history
        self.task_history: List[TaskMetrics] = []
        self.max_history_size = 10000

        # Comparative metrics
        self.comparative_stats = {}

        # Performance trends
        self.trend_data = {
            "timestamps": deque(maxlen=1000),
            "success_rates": deque(maxlen=1000),
            "avg_durations": deque(maxlen=1000)
        }

    def record_task(self, task_metrics: TaskMetrics):
        """Record task completion metrics"""
        agent_id = task_metrics.agent_id

        # Initialize profile if needed
        if agent_id not in self.agent_profiles:
            self.agent_profiles[agent_id] = AgentPerformanceProfile(agent_id=agent_id)

        profile = self.agent_profiles[agent_id]

        # Update task counts
        profile.total_tasks += 1
        if task_metrics.success:
            profile.successful_tasks += 1
        else:
            profile.failed_tasks += 1

        # Update timing statistics
        duration = task_metrics.duration
        profile.total_duration += duration
        profile.avg_duration = profile.total_duration / profile.total_tasks
        profile.min_duration = min(profile.min_duration, duration)
        profile.max_duration = max(profile.max_duration, duration)

        # Update recent performance
        profile.recent_durations.append(duration)
        profile.recent_successes.append(task_metrics.success)

        # Update tool usage
        for tool in task_metrics.tools_used:
            profile.tool_usage[tool] += 1

        # Update LLM statistics
        profile.total_llm_calls += task_metrics.llm_calls
        profile.total_tokens += task_metrics.tokens_used

        # Update task type breakdown
        profile.task_types[task_metrics.task_type] += 1

        # Update hourly metrics
        hour = int(task_metrics.start_time / 3600) % 24
        profile.hourly_tasks[hour] += 1

        # Add to task history
        self.task_history.append(task_metrics)
        if len(self.task_history) > self.max_history_size:
            self.task_history.pop(0)

        # Update trends
        self._update_trends()

        self.logger.debug(f"📊 Recorded metrics for {agent_id}: {duration:.2f}s, success={task_metrics.success}")

    def _update_trends(self):
        """Update performance trends"""
        current_time = time.time()
        self.trend_data["timestamps"].append(current_time)

        # Calculate recent success rate (last 10 tasks)
        recent_tasks = self.task_history[-10:]
        if recent_tasks:
            success_rate = sum(1 for t in recent_tasks if t.success) / len(recent_tasks)
            self.trend_data["success_rates"].append(success_rate)

            avg_duration = sum(t.duration for t in recent_tasks) / len(recent_tasks)
            self.trend_data["avg_durations"].append(avg_duration)

    def get_agent_performance(self, agent_id: str) -> Optional[AgentPerformanceProfile]:
        """Get performance profile for an agent"""
        return self.agent_profiles.get(agent_id)

    def get_all_performance(self) -> Dict[str, AgentPerformanceProfile]:
        """Get performance profiles for all agents"""
        return self.agent_profiles.copy()

    def get_comparative_stats(self) -> Dict[str, Any]:
        """Get comparative statistics across all agents"""
        if not self.agent_profiles:
            return {}

        stats = {
            "agents": {}
        }

        all_success_rates = []
        all_avg_durations = []

        for agent_id, profile in self.agent_profiles.items():
            success_rate = profile.successful_tasks / profile.total_tasks if profile.total_tasks > 0 else 0

            stats["agents"][agent_id] = {
                "success_rate": success_rate,
                "avg_duration": profile.avg_duration,
                "total_tasks": profile.total_tasks,
                "recent_performance": self._calculate_recent_performance(profile)
            }

            all_success_rates.append(success_rate)
            all_avg_durations.append(profile.avg_duration)

        # Overall statistics
        if all_success_rates:
            stats["overall"] = {
                "avg_success_rate": statistics.mean(all_success_rates),
                "avg_duration": statistics.mean(all_avg_durations),
                "best_performer": max(stats["agents"].items(), key=lambda x: x[1]["success_rate"])[0],
                "fastest_agent": min(stats["agents"].items(), key=lambda x: x[1]["avg_duration"])[0]
            }

        return stats

    def _calculate_recent_performance(self, profile: AgentPerformanceProfile) -> Dict[str, float]:
        """Calculate recent performance metrics"""
        if not profile.recent_durations:
            return {"trend": "insufficient_data"}

        recent_success_rate = sum(profile.recent_successes) / len(profile.recent_successes)
        recent_avg_duration = statistics.mean(profile.recent_durations)

        # Calculate trend
        if len(profile.recent_durations) > 10:
            first_half = list(profile.recent_durations)[:len(profile.recent_durations) // 2]
            second_half = list(profile.recent_durations)[len(profile.recent_durations) // 2:]

            trend_value = statistics.mean(second_half) - statistics.mean(first_half)

            if trend_value < -0.5:
                trend = "improving"
            elif trend_value > 0.5:
                trend = "degrading"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        return {
            "success_rate": recent_success_rate,
            "avg_duration": recent_avg_duration,
            "trend": trend
        }

    def get_task_type_analysis(self) -> Dict[str, Any]:
        """Analyze performance by task type"""
        task_type_stats = defaultdict(lambda: {
            "count": 0,
            "success_count": 0,
            "total_duration": 0.0,
            "agents": defaultdict(int)
        })

        for task in self.task_history:
            stats = task_type_stats[task.task_type]
            stats["count"] += 1
            if task.success:
                stats["success_count"] += 1
            stats["total_duration"] += task.duration
            stats["agents"][task.agent_id] += 1

        # Calculate averages
        result = {}
        for task_type, stats in task_type_stats.items():
            result[task_type] = {
                "count": stats["count"],
                "success_rate": stats["success_count"] / stats["count"],
                "avg_duration": stats["total_duration"] / stats["count"],
                "most_common_agent": max(stats["agents"].items(), key=lambda x: x[1])[0] if stats["agents"] else None
            }

        return result

    def get_tool_usage_analysis(self) -> Dict[str, Any]:
        """Analyze tool usage across agents"""
        overall_tool_usage = defaultdict(int)
        tool_success_rates = defaultdict(lambda: {"success": 0, "total": 0})

        for task in self.task_history:
            for tool in task.tools_used:
                overall_tool_usage[tool] += 1
                tool_success_rates[tool]["total"] += 1
                if task.success:
                    tool_success_rates[tool]["success"] += 1

        result = {}
        for tool, count in overall_tool_usage.items():
            success_data = tool_success_rates[tool]
            result[tool] = {
                "usage_count": count,
                "success_rate": success_data["success"] / success_data["total"]
            }

        return result

    def get_time_of_day_analysis(self) -> Dict[int, Dict[str, Any]]:
        """Analyze performance by time of day"""
        hourly_stats = defaultdict(lambda: {
            "task_count": 0,
            "success_count": 0,
            "total_duration": 0.0
        })

        for task in self.task_history:
            hour = int(task.start_time / 3600) % 24
            stats = hourly_stats[hour]
            stats["task_count"] += 1
            if task.success:
                stats["success_count"] += 1
            stats["total_duration"] += task.duration

        result = {}
        for hour, stats in hourly_stats.items():
            if stats["task_count"] > 0:
                result[hour] = {
                    "task_count": stats["task_count"],
                    "success_rate": stats["success_count"] / stats["task_count"],
                    "avg_duration": stats["total_duration"] / stats["task_count"]
                }

        return result

    def get_performance_trends(self) -> Dict[str, List]:
        """Get performance trends over time"""
        return {
            "timestamps": list(self.trend_data["timestamps"]),
            "success_rates": list(self.trend_data["success_rates"]),
            "avg_durations": list(self.trend_data["avg_durations"])
        }

    def get_bottleneck_analysis(self) -> Dict[str, Any]:
        """Identify performance bottlenecks"""
        bottlenecks = []

        for agent_id, profile in self.agent_profiles.items():
            issues = []

            # Check success rate
            success_rate = profile.successful_tasks / profile.total_tasks if profile.total_tasks > 0 else 0
            if success_rate < 0.8:
                issues.append(f"Low success rate: {success_rate:.1%}")

            # Check average duration
            if profile.avg_duration > 5.0:
                issues.append(f"Slow average response: {profile.avg_duration:.2f}s")

            # Check recent performance
            if profile.recent_durations:
                recent_avg = statistics.mean(profile.recent_durations)
                if recent_avg > profile.avg_duration * 1.5:
                    issues.append(
                        f"Recent performance degradation: {recent_avg:.2f}s vs {profile.avg_duration:.2f}s avg")

            if issues:
                bottlenecks.append({
                    "agent_id": agent_id,
                    "issues": issues,
                    "total_tasks": profile.total_tasks
                })

        return {"bottlenecks": bottlenecks}

    def get_summary_report(self) -> str:
        """Generate a human-readable summary report"""
        if not self.agent_profiles:
            return "No performance data available."

        lines = ["=" * 60, "AGENT PERFORMANCE SUMMARY", "=" * 60, ""]

        # Overall statistics
        total_tasks = sum(p.total_tasks for p in self.agent_profiles.values())
        total_success = sum(p.successful_tasks for p in self.agent_profiles.values())
        overall_success_rate = total_success / total_tasks if total_tasks > 0 else 0

        lines.append(f"Total Tasks: {total_tasks}")
        lines.append(f"Overall Success Rate: {overall_success_rate:.1%}")
        lines.append("")

        # Per-agent breakdown
        lines.append("Per-Agent Performance:")
        lines.append("-" * 60)

        for agent_id, profile in sorted(self.agent_profiles.items()):
            success_rate = profile.successful_tasks / profile.total_tasks if profile.total_tasks > 0 else 0
            lines.append(
                f"{agent_id:20} | Tasks: {profile.total_tasks:4} | Success: {success_rate:5.1%} | Avg: {profile.avg_duration:5.2f}s")

        lines.append("")

        # Bottlenecks
        bottlenecks = self.get_bottleneck_analysis()
        if bottlenecks["bottlenecks"]:
            lines.append("⚠️  Performance Bottlenecks:")
            lines.append("-" * 60)
            for bottleneck in bottlenecks["bottlenecks"]:
                lines.append(f"{bottleneck['agent_id']}:")
                for issue in bottleneck["issues"]:
                    lines.append(f"  - {issue}")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)