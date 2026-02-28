"""Budget manager for tracking usage and sending alerts."""

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class BudgetUsage:
    """Track budget usage."""
    monthly_usd: float = 0.0  # -1 means unlimited
    monthly_calls: int = 0    # 0 means unlimited
    current_cost: float = 0.0
    current_calls: int = 0
    alert_thresholds: list[float] = field(default_factory=lambda: [0.6, 0.8])
    alerted_thresholds: list[float] = field(default_factory=list)
    reset_date: datetime = field(default_factory=lambda: datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0))
    
    def is_unlimited(self) -> bool:
        """Check if budget is unlimited."""
        return self.monthly_usd < 0 and self.monthly_calls <= 0
    
    def get_cost_usage_ratio(self) -> float:
        """Get current cost usage ratio (0.0 - 1.0+)."""
        if self.monthly_usd < 0:
            return 0.0
        if self.monthly_usd <= 0:
            return 0.0
        return self.current_cost / self.monthly_usd
    
    def get_calls_usage_ratio(self) -> float:
        """Get current calls usage ratio (0.0 - 1.0+)."""
        if self.monthly_calls <= 0:
            return 0.0
        return self.current_calls / self.monthly_calls
    
    def should_reset(self) -> bool:
        """Check if usage should be reset (new month)."""
        now = datetime.now()
        return now >= self.reset_date + timedelta(days=32)  # Simple month check
    
    def reset_if_needed(self):
        """Reset usage if new month."""
        if self.should_reset():
            self.current_cost = 0.0
            self.current_calls = 0
            self.alerted_thresholds = []
            self.reset_date = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            logger.info("Budget usage reset for new month")


class BudgetManager:
    """Manage budget tracking and alerts."""
    
    def __init__(
        self,
        monthly_usd: float = 0.0,
        monthly_calls: int = 0,
        alert_thresholds: list[float] | None = None,
        on_alert: Callable[[str, float], None] | None = None,
    ):
        self.usage = BudgetUsage(
            monthly_usd=monthly_usd,
            monthly_calls=monthly_calls,
            alert_thresholds=alert_thresholds or [0.6, 0.8],
        )
        self.on_alert = on_alert  # Callback for sending alerts
    
    def record_call(self, cost: float = 0.0):
        """Record a API call with optional cost."""
        self.usage.reset_if_needed()
        self.usage.current_calls += 1
        self.usage.current_cost += cost
        
        # Check alerts for cost budget
        if self.usage.monthly_usd > 0:
            ratio = self.usage.get_cost_usage_ratio()
            self._check_alerts(ratio, "cost")
        
        # Check alerts for calls budget
        if self.usage.monthly_calls > 0:
            ratio = self.usage.get_calls_usage_ratio()
            self._check_alerts(ratio, "calls")
    
    def _check_alerts(self, ratio: float, budget_type: str):
        """Check if any alert thresholds are reached."""
        for threshold in self.usage.alert_thresholds:
            if ratio >= threshold and threshold not in self.usage.alerted_thresholds:
                self.usage.alerted_thresholds.append(threshold)
                percentage = int(threshold * 100)
                message = f"Budget alert: {budget_type} usage reached {percentage}% threshold"
                logger.warning(message)
                if self.on_alert:
                    self.on_alert(message, ratio)
    
    def get_status(self) -> dict[str, Any]:
        """Get current budget status."""
        self.usage.reset_if_needed()
        return {
            "monthly_usd": self.usage.monthly_usd,
            "monthly_calls": self.usage.monthly_calls,
            "current_cost": self.usage.current_cost,
            "current_calls": self.usage.current_calls,
            "cost_usage_ratio": self.usage.get_cost_usage_ratio(),
            "calls_usage_ratio": self.usage.get_calls_usage_ratio(),
            "alert_thresholds": self.usage.alert_thresholds,
            "alerted_thresholds": self.usage.alerted_thresholds,
            "is_unlimited": self.usage.is_unlimited(),
        }


# Global budget manager instance
_budget_manager: BudgetManager | None = None


def get_budget_manager() -> BudgetManager | None:
    """Get the global budget manager instance."""
    return _budget_manager


def init_budget_manager(
    monthly_usd: float = 0.0,
    monthly_calls: int = 0,
    alert_thresholds: list[float] | None = None,
    on_alert: Callable[[str, float], None] | None = None,
) -> BudgetManager:
    """Initialize the global budget manager."""
    global _budget_manager
    _budget_manager = BudgetManager(
        monthly_usd=monthly_usd,
        monthly_calls=monthly_calls,
        alert_thresholds=alert_thresholds,
        on_alert=on_alert,
    )
    return _budget_manager