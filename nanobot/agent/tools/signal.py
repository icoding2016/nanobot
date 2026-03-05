"""Signal tool for SubAgent to communicate with Orchestrator."""

import json
import time
from typing import Any

from nanobot.agent.tools.base import Tool


class SignalTool(Tool):
    """
    Tool for SubAgent to send signals to the Orchestrator.

    Signals allow SubAgents to proactively report status or request help,
    enabling more effective monitoring and intervention.
    """

    name = "signal"
    description = "Send a status signal to the orchestrator. Use this to report progress, warn about long operations, or request help when stuck."

    # Signal types with descriptions
    SIGNAL_TYPES = {
        "progress": "Report normal progress (resets idle timer)",
        "will_take_long": "Warn that current operation will take a long time (e.g., installing packages, running tests)",
        "stuck": "Report that you're stuck and need a hint",
        "need_help": "Request model switch/rescue due to inability to proceed",
    }

    def __init__(self, supervisor: Any = None, task_id: str | None = None):
        """
        Initialize the signal tool.

        Args:
            supervisor: The AgentSupervisor instance to receive signals.
                       If None, signals will be logged but not processed.
            task_id: The ID of the task using this tool.
        """
        self.supervisor = supervisor
        self.task_id = task_id

    @property
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "signal_type": {
                    "type": "string",
                    "enum": list(self.SIGNAL_TYPES.keys()),
                    "description": "The type of signal to send:\n" + "\n".join(f"- {k}: {v}" for k, v in self.SIGNAL_TYPES.items())
                },
                "message": {
                    "type": "string",
                    "description": "Optional message describing the situation"
                }
            },
            "required": ["signal_type"]
        }

    async def execute(self, signal_type: str, message: str = "", **kwargs: Any) -> str:
        """
        Execute the signal tool.

        Args:
            signal_type: The type of signal to send
            message: Optional message describing the situation

        Returns:
            Confirmation message
        """
        if signal_type not in self.SIGNAL_TYPES:
            return f"Error: Unknown signal type '{signal_type}'. Valid types: {list(self.SIGNAL_TYPES.keys())}"

        # Log the signal
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] Signal: {signal_type}"
        if message:
            log_entry += f" - {message}"

        # If supervisor is available, process the signal
        if self.supervisor and self.task_id:
            # For 'progress' signal, mark heartbeat as detected
            if signal_type == "progress":
                # The supervisor will detect heartbeat naturally through ActivityLog
                pass

            # For 'stuck' signal, could trigger immediate hint
            elif signal_type == "stuck":
                # Log that SubAgent reported being stuck
                from loguru import logger
                logger.warning(f"SubAgent [{self.task_id}] reported stuck: {message}")
                # Trigger L1 intervention immediately if supervisor supports it
                if hasattr(self.supervisor, "trigger_intervention"):
                     await self.supervisor.trigger_intervention(self.task_id, "stuck", message)

            # For 'need_help' signal, could trigger L2 intervention
            elif signal_type == "need_help":
                from loguru import logger
                logger.warning(f"SubAgent [{self.task_id}] requested help: {message}")
                if hasattr(self.supervisor, "trigger_intervention"):
                     await self.supervisor.trigger_intervention(self.task_id, "need_help", message)

        return f"Signal '{signal_type}' sent successfully. {message if message else ''}"
