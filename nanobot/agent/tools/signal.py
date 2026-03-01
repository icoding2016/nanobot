"""Signal tool for sending notifications to the orchestrator."""

from typing import Any
from nanobot.agent.tools.base import Tool


class SignalTool(Tool):
    """
    Tool for SubAgents to send signals/notifications to the Orchestrator.
    
    This allows SubAgents to report important events, request help,
    or notify about progress without ending the task.
    """
    
    name = "signal"
    description = "Send a signal notification to the orchestrator. Use this to report important events, request guidance, or notify about blockers."
    parameters = {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["progress", "blocker", "help", "checkpoint", "custom"],
                "description": "Type of signal to send"
            },
            "message": {
                "type": "string",
                "description": "The signal message content"
            },
            "severity": {
                "type": "string",
                "enum": ["info", "warning", "error"],
                "default": "info",
                "description": "Severity level of the signal"
            },
            "data": {
                "type": "object",
                "description": "Optional additional data to include with the signal"
            }
        },
        "required": ["type", "message"]
    }
    
    def __init__(self, bus=None, task_id: str | None = None):
        """
        Initialize the signal tool.
        
        Args:
            bus: The message bus for publishing signals.
            task_id: The task ID of the SubAgent using this tool.
        """
        self.bus = bus
        self.task_id = task_id
    
    def set_context(self, task_id: str) -> None:
        """Set the task ID context for this tool."""
        self.task_id = task_id
    
    async def execute(
        self,
        type: str,
        message: str,
        severity: str = "info",
        data: dict[str, Any] | None = None,
    ) -> str:
        """
        Execute the signal tool.
        
        Args:
            type: Type of signal (progress, blocker, help, checkpoint, custom).
            message: The signal message content.
            severity: Severity level (info, warning, error).
            data: Optional additional data.
        
        Returns:
            Confirmation message.
        """
        from datetime import datetime
        from loguru import logger
        
        signal_data = {
            "type": type,
            "message": message,
            "severity": severity,
            "task_id": self.task_id,
            "timestamp": datetime.now().isoformat(),
            "data": data or {}
        }
        
        # Log the signal
        log_level = {
            "info": logger.info,
            "warning": logger.warning,
            "error": logger.error
        }.get(severity, logger.info)
        
        log_level(f"Signal [{type}] from task {self.task_id}: {message}")
        
        # If bus is available, publish the signal
        if self.bus:
            try:
                from nanobot.bus.events import InboundMessage
                msg = InboundMessage(
                    channel="system",
                    sender_id=f"subagent:{self.task_id}",
                    chat_id="system:signals",
                    content=f"[Signal:{type}] {message}",
                    metadata=signal_data
                )
                await self.bus.publish_inbound(msg)
            except Exception as e:
                logger.error(f"Failed to publish signal: {e}")
        
        return f"Signal sent: [{severity}] {type} - {message}"