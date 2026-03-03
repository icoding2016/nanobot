"""Agent loop: the core processing engine."""

import asyncio
from contextlib import AsyncExitStack
from datetime import datetime
import json
import json_repair
from pathlib import Path
import time
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.context import ContextBuilder
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.memory import MemoryStore
from nanobot.agent.subagent import SubagentManager
from nanobot.session.manager import Session, SessionManager


# Default heartbeat timeouts by profile (in seconds)
DEFAULT_HEARTBEAT_TIMEOUTS = {
    "debugger": 600,      # 10 minutes
    "developer": 900,     # 15 minutes
    "designer": 1800,     # 30 minutes
    "researcher": 1800,   # 30 minutes
    "assistant": 900,     # 15 minutes
}
DEFAULT_HEARTBEAT_TIMEOUT = 900  # 15 minutes default


def _extract_command(content: str) -> tuple[str | None, str]:
    """
    Extract command prefix and rest from content.
    
    Supports both '/' (Telegram native) and '!' (cross-platform) prefixes.
    
    Returns:
        Tuple of (prefix, rest) where prefix is '/' or '!' or None.
    """
    content = content.strip()
    if content.startswith(('/', '!')):
        return content[0], content[1:]
    return None, content


class InterventionTracker:
    """
    Tracks intervention history for SubAgent tasks.
    
    Records when interventions happen and their results,
    to avoid duplicate interventions and enable escalation.
    """
    
    def __init__(self):
        self._history: dict[str, list[dict]] = {}
    
    def record(self, task_id: str, level: int, action: str) -> None:
        """Record an intervention event."""
        if task_id not in self._history:
            self._history[task_id] = []
        
        self._history[task_id].append({
            "time": time.time(),
            "level": level,
            "action": action,
            "result": None  # "success" / "failed" / None (pending)
        })
    
    def get_last_intervention(self, task_id: str) -> dict | None:
        """Get the most recent intervention for a task."""
        history = self._history.get(task_id, [])
        return history[-1] if history else None
    
    def should_escalate(self, task_id: str, timeout_seconds: float = 300) -> bool:
        """Check if intervention should escalate (L1 -> L2)."""
        last = self.get_last_intervention(task_id)
        if not last:
            return False
        
        # L1 intervention pending for timeout_seconds -> escalate
        if last["level"] == 1 and last["result"] is None:
            if time.time() - last["time"] > timeout_seconds:
                return True
        
        return False
    
    def mark_result(self, task_id: str, result: str) -> None:
        """Mark the result of the last intervention."""
        last = self.get_last_intervention(task_id)
        if last:
            last["result"] = result
    
    def clear(self, task_id: str) -> None:
        """Clear intervention history for a completed task."""
        self._history.pop(task_id, None)


class AgentSupervisor:
    """
    Monitors SubAgent health and performs interventions.
    
    Runs a background monitoring loop that checks SubAgent heartbeats
    and triggers interventions when tasks appear stuck.
    """
    
    def __init__(
        self,
        subagent_manager: SubagentManager,
        bus: MessageBus,
        heartbeat_timeouts: dict[str, int] | None = None,
        check_interval: int = 60,
    ):
        self.subagents = subagent_manager
        self.bus = bus
        self.heartbeat_timeouts = heartbeat_timeouts or DEFAULT_HEARTBEAT_TIMEOUTS
        self.check_interval = check_interval
        self.interventions = InterventionTracker()
        self._running = False
        self._task: asyncio.Task | None = None
        
        # Messages to inject into SubAgent loops (task_id -> message)
        self._pending_injections: dict[str, str] = {}
    
    def start(self) -> None:
        """Start the monitoring loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("AgentSupervisor monitoring loop started")
    
    def stop(self) -> None:
        """Stop the monitoring loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        logger.info("AgentSupervisor monitoring loop stopped")
    
    def get_timeout_for_profile(self, profile: str | None) -> int:
        """Get heartbeat timeout for a profile."""
        if profile and profile in self.heartbeat_timeouts:
            return self.heartbeat_timeouts[profile]
        return DEFAULT_HEARTBEAT_TIMEOUT
    
    async def _monitor_loop(self) -> None:
        """Main monitoring loop - checks all running tasks periodically."""
        while self._running:
            try:
                await asyncio.sleep(self.check_interval)
                await self._check_all_tasks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"AgentSupervisor error: {e}")
    
    async def _check_all_tasks(self) -> None:
        """Check status of all running SubAgent tasks."""
        statuses = self.subagents.get_all_task_statuses()
        
        for status in statuses:
            task_id = status.get("id")
            profile = status.get("profile")
            idle_seconds = status.get("idle_seconds", 0)
            
            timeout = self.get_timeout_for_profile(profile)
            
            if idle_seconds > timeout:
                await self._handle_stuck_task(task_id, status)
    
    async def _handle_stuck_task(self, task_id: str, status: dict) -> None:
        """Handle a potentially stuck task."""
        idle_seconds = status.get("idle_seconds", 0)
        profile = status.get("profile", "unknown")
        label = status.get("label", "task")
        
        # Check if we should escalate
        if self.interventions.should_escalate(task_id):
            # L1 failed, escalate to L2
            logger.warning(f"Task [{task_id}] L1 intervention failed, escalating to L2")
            await self._intervene_l2(task_id, status)
        else:
            # First intervention: L1 hint injection
            last = self.interventions.get_last_intervention(task_id)
            if not last:
                logger.info(f"Task [{task_id}] stuck (idle {idle_seconds:.0f}s), injecting L1 hint")
                await self._intervene_l1(task_id, status)
    
    async def _intervene_l1(self, task_id: str, status: dict) -> None:
        """
        L1 intervention: Inject a hint message into the SubAgent.
        
        This is a lightweight intervention that just adds a message
        to prompt the SubAgent to continue or report status.
        """
        idle_seconds = int(status.get("idle_seconds", 0))
        label = status.get("label", "task")
        
        hint = f"""[System Check] I noticed you've been idle for {idle_seconds // 60} minutes.

If you're stuck, please:
1. Update your board with current status
2. Try a different approach
3. Or describe what's blocking you

Please continue or report your status."""
        
        # Queue the injection
        self._pending_injections[task_id] = hint
        self.interventions.record(task_id, 1, "hint_injection")
        
        logger.info(f"L1 intervention: hint queued for task [{task_id}]")
    
    async def _intervene_l2(self, task_id: str, status: dict) -> None:
        """
        L2 intervention: Model switch with handoff.
        
        This is a heavier intervention that requires the current model
        to generate a handoff document and switches to a different model.
        
        Note: This requires integration with the SubAgent loop to:
        1. Pause execution
        2. Generate handoff
        3. Switch model
        4. Continue with new context
        """
        # For now, log and mark as requiring manual intervention
        # Full implementation would need deeper integration with _run_subagent
        logger.warning(f"L2 intervention: model switch requested for task [{task_id}]")
        self.interventions.record(task_id, 2, "model_switch")
        
        # Broadcast a system message about the stuck task
        from nanobot.bus.events import InboundMessage
        msg = InboundMessage(
            channel="system",
            sender_id="supervisor",
            chat_id="system:alerts",
            content=f"[Supervisor Alert] Task '{status.get('label', task_id)}' requires attention. "
                    f"Idle for {int(status.get('idle_seconds', 0) // 60)} minutes. "
                    f"Profile: {status.get('profile', 'unknown')}"
        )
        await self.bus.publish_inbound(msg)
    
    def get_pending_injection(self, task_id: str) -> str | None:
        """Get and clear a pending injection for a task."""
        return self._pending_injections.pop(task_id, None)
    
    def heartbeat_detected(self, task_id: str) -> None:
        """Called when a heartbeat is detected, marking intervention as successful."""
        last = self.interventions.get_last_intervention(task_id)
        if last and last["result"] is None:
            self.interventions.mark_result(task_id, "success")
            logger.info(f"Task [{task_id}] recovered after L{last['level']} intervention")


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        models: list[str] | str | None = None,
        max_iterations: int = 20,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        memory_window: int = 50,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        cron_service: "CronService | None" = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        agent_profiles: dict[str, Any] | None = None,
        orchestrator_rules: list[str] | None = None,
        shortcuts: dict[str, Any] | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig
        from nanobot.cron.service import CronService
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.models = models
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self.shortcuts = shortcuts or {}
        self.agent_profiles = agent_profiles or {}

        self.context = ContextBuilder(workspace, self.agent_profiles, orchestrator_rules)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            models=models,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
            agent_profiles=self.agent_profiles,
        )
        
        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        
        # Budget monitoring
        self._budget_alerts_sent: set[float] = set()  # Track sent alerts to avoid duplicates
        
        # Agent Supervisor for SubAgent monitoring
        self.supervisor = AgentSupervisor(
            subagent_manager=self.subagents,
            bus=self.bus,
        )
        
        self._register_default_tools()
    
    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools (restrict to workspace if configured)
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        self.tools.register(ReadFileTool(allowed_dir=allowed_dir))
        self.tools.register(WriteFileTool(allowed_dir=allowed_dir))
        self.tools.register(EditFileTool(allowed_dir=allowed_dir))
        self.tools.register(ListDirTool(allowed_dir=allowed_dir))
        
        # Shell tool
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ))
        
        # Web tools
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())
        
        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)
        
        # Spawn tool (for subagents)
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool)
        
        # Cron tool (for scheduling)
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))
    
    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or not self._mcp_servers:
            return
        self._mcp_connected = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        self._mcp_stack = AsyncExitStack()
        await self._mcp_stack.__aenter__()
        await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)

    def _set_tool_context(self, channel: str, chat_id: str) -> None:
        """Update context for all tools that need routing info."""
        if hasattr(self.provider, "set_context"):
            try:
                self.provider.set_context(channel, f"{channel}:{chat_id}")
            except Exception:
                pass
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.set_context(channel, chat_id)

        if spawn_tool := self.tools.get("spawn"):
            if isinstance(spawn_tool, SpawnTool):
                spawn_tool.set_context(channel, chat_id)

        if cron_tool := self.tools.get("cron"):
            if isinstance(cron_tool, CronTool):
                cron_tool.set_context(channel, chat_id)

    async def _run_agent_loop(self, initial_messages: list[dict]) -> tuple[str | None, list[str]]:
        """
        Run the agent iteration loop.

        Args:
            initial_messages: Starting messages for the LLM conversation.

        Returns:
            Tuple of (final_content, list_of_tools_used).
        """
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []

        while iteration < self.max_iterations:
            iteration += 1

            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                models=self.models,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # Check budget thresholds after LLM call
            await self._check_budget_thresholds()

            if response.has_tool_calls:
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )

                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
                messages.append({"role": "user", "content": "Reflect on the results and decide next steps."})
            else:
                final_content = response.content
                break

        return final_content, tools_used

    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        await self._connect_mcp()
        
        # Start the SubAgent supervisor
        self.supervisor.start()
        
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )
                try:
                    response = await self._process_message(msg)
                    if response:
                        await self.bus.publish_outbound(response)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}"
                    ))
            except asyncio.TimeoutError:
                continue
    
    async def close_mcp(self) -> None:
        """Close MCP connections."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        self.supervisor.stop()
        logger.info("Agent loop stopping")
    
    async def _process_message(self, msg: InboundMessage, session_key: str | None = None) -> OutboundMessage | None:
        """
        Process a single inbound message.
        
        Args:
            msg: The inbound message to process.
            session_key: Override session key (used by process_direct).
        
        Returns:
            The response message, or None if no response needed.
        """
        # System messages route back via chat_id ("channel:chat_id")
        if msg.channel == "system":
            return await self._process_system_message(msg)
        
        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}: {preview}")
        
        base_key = session_key or msg.session_key
        base_session = self.sessions.get_or_create(base_key)
        
        # Handle commands (support both / and ! prefixes)
        raw_content = msg.content.strip()
        prefix, rest = _extract_command(raw_content)
        cmd = rest.lower() if prefix else ""
        
        if prefix and cmd == "new":
            current_topic = base_session.metadata.get("active_topic")
            if current_topic:
                from datetime import datetime
                new_topic = datetime.now().strftime("topic-%Y%m%d-%H%M%S")
                base_session.metadata["active_topic"] = new_topic
                self.sessions.save(base_session)
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content=f"New topic started: {new_topic}")
            messages_to_archive = base_session.messages.copy()
            base_session.clear()
            base_session.metadata.pop("active_topic", None)
            self.sessions.save(base_session)
            self.sessions.invalidate(base_session.key)

            async def _consolidate_and_cleanup():
                temp_session = Session(key=base_session.key)
                temp_session.messages = messages_to_archive
                await self._consolidate_memory(temp_session, archive_all=True)

            asyncio.create_task(_consolidate_and_cleanup())
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started. Memory consolidation in progress.")
        
        if prefix and cmd == "help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🐈 nanobot commands:\n/new — Start a new conversation\n/topic <name> — Switch to a named topic\n/status — Show status\n/help — Show available commands\n\nRouting:\n@<agent> <task> — Send task to a specific agent\n/<cmd> <task> — Execute a configured shortcut command\n\nNote: Both / and ! prefixes work (e.g., /status or !status)")

        if prefix and cmd == "status":
            orchestrator_models = self._get_orchestrator_models()
            profiles = []
            for name, profile in self.agent_profiles.items():
                models = profile.models if getattr(profile, "models", None) else []
                profiles.append(f"{name}: {', '.join(models) if models else '(no model)'}")

            running = self.subagents.list_running()
            running_lines = []
            for idx, item in enumerate(running, start=1):
                label = item.get("label", "task")
                profile = item.get("profile") or "generic"
                board = item.get("board_rel") or ""
                suffix = f" [{board}]" if board else ""
                running_lines.append(f"{idx}. {profile} - {label}{suffix}")
            running_text = "\n".join(running_lines) if running_lines else "(none)"

            topics = self._list_topics(base_key)
            topic_lines = [f"{i}. {t['name']}" for i, t in enumerate(topics, start=1)]
            topic_text = "\n".join(topic_lines) if topic_lines else "(none)"
            active_topic = base_session.metadata.get("active_topic") or "(none)"

            boards = self._list_boards()
            board_lines = [f"{i}. {b['rel']}" for i, b in enumerate(boards, start=1)]
            board_text = "\n".join(board_lines) if board_lines else "(none)"

            cost_total, cost_models, cost_budget = self._read_cost_summary()
            cron_text = ""
            if self.cron_service:
                cron_status = self.cron_service.status()
                next_ms = cron_status.get("next_wake_at_ms")
                if next_ms:
                    from datetime import datetime
                    next_text = datetime.fromtimestamp(next_ms / 1000).isoformat()
                else:
                    next_text = "(none)"
                cron_text = f"jobs={cron_status.get('jobs', 0)}, next={next_text}"

            content_lines = [
                "Status:",
                f"- Orchestrator models: {', '.join(orchestrator_models)}",
                f"- Workspace: {self.workspace}",
                "- Agent profiles:",
            ]
            content_lines.extend(
                [f"  {line}" for line in profiles] if profiles else ["  (none)"]
            )
            content_lines.append("- Running tasks:")
            content_lines.extend(
                [f"  {line}" for line in running_text.splitlines()] if running_text else ["  (none)"]
            )
            if cron_text:
                content_lines.append(f"- Cron: {cron_text}")
            content_lines.append(f"- Active topic: {active_topic}")
            content_lines.append("- Topics:")
            content_lines.extend(
                [f"  {line}" for line in topic_text.splitlines()] if topic_text else ["  (none)"]
            )
            content_lines.append("- Boards:")
            content_lines.extend(
                [f"  {line}" for line in board_text.splitlines()] if board_text else ["  (none)"]
            )
            content_lines.append(f"- Cost total: {cost_total}")
            if cost_budget:
                content_lines.append(f"- Budget: {cost_budget}")
            content_lines.append("- Cost by model:")
            content_lines.extend(
                [f"  {line}" for line in cost_models] if cost_models else ["  (none)"]
            )
            content = "\n".join(content_lines)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)

        if prefix and cmd.startswith("topic"):
            # Extract topic name from the command (handle both /topic and !topic)
            parts = rest.split(maxsplit=1)
            topics = self._list_topics(base_key)
            if len(parts) < 2 or not parts[1].strip():
                if not topics:
                    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                          content="No topics found.")
                lines = []
                for idx, item in enumerate(topics, start=1):
                    lines.append(f"{idx}. {item['name']}")
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content="Topics:\n" + "\n".join(lines))
            selector = parts[1].strip()
            chosen = None
            if selector.isdigit():
                index = int(selector)
                if 1 <= index <= len(topics):
                    chosen = topics[index - 1]["name"]
            else:
                for item in topics:
                    if item["name"] == selector:
                        chosen = item["name"]
                        break
            if not chosen:
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content="Topic not found.")
            base_session.metadata["active_topic"] = chosen
            board_rel = self._ensure_topic_board(chosen)
            boards_map = base_session.metadata.get("topic_boards", {})
            if isinstance(boards_map, dict):
                boards_map[chosen] = board_rel
            base_session.metadata["topic_boards"] = boards_map
            self._record_coord_event(base_key, "switch_topic", chosen, "manual")
            self.sessions.save(base_session)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content=f"Topic switched to: {chosen}")

        active_topic = base_session.metadata.get("active_topic")
        auto_topic_notice = None
        if not active_topic:
            detected_topic, reason, task_path = self._detect_topic(raw_content)
            if detected_topic:
                base_session.metadata["active_topic"] = detected_topic
                board_rel = self._ensure_topic_board(detected_topic)
                boards_map = base_session.metadata.get("topic_boards", {})
                if isinstance(boards_map, dict):
                    boards_map[detected_topic] = board_rel
                base_session.metadata["topic_boards"] = boards_map
                if task_path:
                    self._bind_task_topic(task_path, detected_topic, board_rel)
                self._record_coord_event(base_key, "auto_topic", detected_topic, reason)
                self.sessions.save(base_session)
                active_topic = detected_topic
                auto_topic_notice = f"主题已切换到: {detected_topic}"
        if active_topic:
            key = f"{base_key}:{active_topic}"
            session = self.sessions.get_or_create(key)
        else:
            session = base_session

        content = raw_content
        if active_topic:
            boards_map = base_session.metadata.get("topic_boards", {})
            board_rel = None
            if isinstance(boards_map, dict):
                board_rel = boards_map.get(active_topic)
            if not board_rel:
                board_rel = self._ensure_topic_board(active_topic)
                if isinstance(boards_map, dict):
                    boards_map[active_topic] = board_rel
                base_session.metadata["topic_boards"] = boards_map
                self.sessions.save(base_session)
            content = self._attach_topic_board_reference(content, board_rel)
        if content.startswith("@"):
            token = content.split(maxsplit=1)[0]
            profile = token[1:]
            if profile and profile in self.agent_profiles:
                task = content[len(token):].strip() or content
                self._set_tool_context(msg.channel, msg.chat_id)
                result = await self.tools.execute("spawn", {
                    "task": task,
                    "profile": profile,
                    "label": f"{profile} task",
                })
                session.add_message("user", msg.content)
                session.add_message("assistant", result, tools_used=["spawn"])
                self.sessions.save(session)
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=result)

        if content.startswith("/"):
            token = content.split(maxsplit=1)[0]
            rule = self.shortcuts.get(token)
            if rule:
                action = rule.action if hasattr(rule, "action") else rule.get("action")
                profile = rule.profile if hasattr(rule, "profile") else rule.get("profile")
                label = rule.label if hasattr(rule, "label") else rule.get("label")
                task = content[len(token):].strip() or content
                if action == "spawn" and profile:
                    self._set_tool_context(msg.channel, msg.chat_id)
                    result = await self.tools.execute("spawn", {
                        "task": task,
                        "profile": profile,
                        "label": label or f"{profile} task",
                    })
                    session.add_message("user", msg.content)
                    session.add_message("assistant", result, tools_used=["spawn"])
                    self.sessions.save(session)
                    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=result)
        
        if len(session.messages) > self.memory_window:
            asyncio.create_task(self._consolidate_memory(session))

        expanded_content = self._expand_references(content)
        self._set_tool_context(msg.channel, msg.chat_id)
        initial_messages = self.context.build_messages(
            history=session.get_history(max_messages=self.memory_window),
            current_message=expanded_content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )
        final_content, tools_used = await self._run_agent_loop(initial_messages)

        if final_content is None:
            final_content = "I've completed processing but have no response to give."
        if auto_topic_notice:
            final_content = f"{auto_topic_notice}\n\n{final_content}"
        
        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info(f"Response to {msg.channel}:{msg.sender_id}: {preview}")
        
        session.add_message("user", msg.content)
        session.add_message("assistant", final_content,
                            tools_used=tools_used if tools_used else None)
        self.sessions.save(session)
        
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=msg.metadata or {},  # Pass through for channel-specific needs (e.g. Slack thread_ts)
        )
    
    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).
        
        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")
        
        # Parse origin from chat_id (format: "channel:chat_id")
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            # Fallback
            origin_channel = "cli"
            origin_chat_id = msg.chat_id
        
        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)
        self._set_tool_context(origin_channel, origin_chat_id)
        initial_messages = self.context.build_messages(
            history=session.get_history(max_messages=self.memory_window),
            current_message=msg.content,
            channel=origin_channel,
            chat_id=origin_chat_id,
        )
        final_content, _ = await self._run_agent_loop(initial_messages)

        if final_content is None:
            final_content = "Background task completed."
        
        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        
        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=final_content
        )
    
    async def _consolidate_memory(self, session, archive_all: bool = False) -> None:
        """Consolidate old messages into MEMORY.md + HISTORY.md.

        Args:
            archive_all: If True, clear all messages and reset session (for /new command).
                       If False, only write to files without modifying session.
        """
        memory = MemoryStore(self.workspace)

        if archive_all:
            old_messages = session.messages
            keep_count = 0
            logger.info(f"Memory consolidation (archive_all): {len(session.messages)} total messages archived")
        else:
            keep_count = self.memory_window // 2
            if len(session.messages) <= keep_count:
                logger.debug(f"Session {session.key}: No consolidation needed (messages={len(session.messages)}, keep={keep_count})")
                return

            messages_to_process = len(session.messages) - session.last_consolidated
            if messages_to_process <= 0:
                logger.debug(f"Session {session.key}: No new messages to consolidate (last_consolidated={session.last_consolidated}, total={len(session.messages)})")
                return

            old_messages = session.messages[session.last_consolidated:-keep_count]
            if not old_messages:
                return
            logger.info(f"Memory consolidation started: {len(session.messages)} total, {len(old_messages)} new to consolidate, {keep_count} keep")

        lines = []
        for m in old_messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}")
        conversation = "\n".join(lines)
        word_count = len(conversation.split())
        max_words = max(1000, min(4000, int(word_count * 0.25))) if word_count else 1000
        current_memory = memory.read_long_term()

        prompt = f"""You are a memory consolidation agent. Process this conversation and return a JSON object with exactly two keys:

1. "history_entry": A paragraph (2-5 sentences) summarizing the key events/decisions/topics. Start with a timestamp like [YYYY-MM-DD HH:MM]. Include enough detail to be useful when found by grep search later.

2. "memory_update": The updated long-term memory content. Add any new facts: user location, preferences, personal info, habits, project context, technical decisions, tools/services used. If nothing new, return the existing content unchanged. Keep this under {max_words} words.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{conversation}

Respond with ONLY valid JSON, no markdown fences."""

        try:
            response = await self.provider.chat(
                messages=[
                    {"role": "system", "content": "You are a memory consolidation agent. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                models=self.models,
            )
            
            # Check budget thresholds after LLM call
            await self._check_budget_thresholds()
            text = (response.content or "").strip()
            if not text:
                logger.warning("Memory consolidation: LLM returned empty response, skipping")
                return
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json_repair.loads(text)
            if not isinstance(result, dict):
                logger.warning(f"Memory consolidation: unexpected response type, skipping. Response: {text[:200]}")
                return

            if entry := result.get("history_entry"):
                memory.append_history(entry)
            if update := result.get("memory_update"):
                if update != current_memory:
                    memory.write_long_term(update)

            if archive_all:
                session.last_consolidated = 0
            else:
                session.last_consolidated = len(session.messages) - keep_count
            logger.info(f"Memory consolidation done: {len(session.messages)} messages, last_consolidated={session.last_consolidated}")
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")

    def _expand_references(self, content: str) -> str:
        tokens = []
        for part in content.split():
            if part.startswith("#file:") or part.startswith("#board:"):
                tokens.append(part)
        if not tokens:
            return content

        references = []
        for token in tokens:
            kind, target = token[1:].split(":", 1)
            target = target.strip()
            if not target:
                continue
            if kind == "file":
                rel = Path(target)
                path = (self.workspace / rel).resolve()
            else:
                if target.isdigit():
                    boards = self._list_boards()
                    index = int(target)
                    if 1 <= index <= len(boards):
                        rel = Path(boards[index - 1]["rel"])
                        path = (self.workspace / rel).resolve()
                    else:
                        continue
                else:
                    rel = Path(target)
                    if rel.suffix:
                        path = (self.workspace / "boards" / rel).resolve()
                    else:
                        path = (self.workspace / "boards" / rel / "board.md").resolve()

            if not str(path).startswith(str(self.workspace.resolve())):
                continue
            if not path.exists() or not path.is_file():
                continue
            try:
                data = path.read_text(encoding="utf-8")
            except Exception:
                continue
            references.append(f"[Reference: {token}]\n{data}")

        if not references:
            return content
        return content + "\n\n" + "\n\n".join(references)

    def _coord_log_path(self) -> Path:
        return self.workspace / "task" / "coord.md"

    def _record_coord_event(self, base_key: str, event: str, topic: str | None, reason: str | None) -> None:
        path = self._coord_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"## {ts}"]
        lines.append(f"- Session: {base_key}")
        lines.append(f"- Event: {event}")
        if topic:
            lines.append(f"- Topic: {topic}")
        if reason:
            lines.append(f"- Reason: {reason}")
        lines.append("")
        path.write_text(path.read_text(encoding="utf-8") + "\n".join(lines) + "\n", encoding="utf-8") if path.exists() else path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _ensure_topic_board(self, topic: str) -> str:
        safe_topic = "".join(c if c.isalnum() or c in "-_" else "_" for c in topic).strip("_")[:50]
        board_dir = self.workspace / "boards" / "orchestrator" / safe_topic
        board_dir.mkdir(parents=True, exist_ok=True)
        board_path = board_dir / "board.md"
        if not board_path.exists():
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            board_path.write_text(
                f"# Topic Board: {topic}\n\nCreated: {now}\n\n## Summary\n\n## Notes\n",
                encoding="utf-8",
            )
        return f"boards/orchestrator/{safe_topic}/board.md"

    def _attach_topic_board_reference(self, content: str, board_rel: str) -> str:
        token = f"#board:{Path(board_rel).parent.as_posix()}"
        if token in content:
            return content
        board_path = (self.workspace / board_rel).resolve()
        if not board_path.exists():
            return content
        try:
            if board_path.stat().st_size > 12000:
                return content
        except OSError:
            return content
        return f"{content}\n\n{token}"

    def _strip_task_date(self, stem: str) -> str:
        parts = stem.split("-")
        if len(parts) >= 2:
            last = parts[-1]
            if last.isdigit() and len(last) in {8}:
                return "-".join(parts[:-1])
            if len(last) == 10 and last[4] == "-" and last[7] == "-":
                return "-".join(parts[:-1])
        return stem

    def _task_candidates(self) -> list[dict[str, Any]]:
        task_dir = self.workspace / "task"
        if not task_dir.exists():
            return []
        items = []
        for path in task_dir.glob("task-*.md"):
            stem = path.stem
            title = stem[5:] if stem.startswith("task-") else stem
            title = self._strip_task_date(title)
            items.append({"title": title, "path": path})
        return items

    def _match_topic_from_tasks(self, content: str) -> tuple[str | None, Path | None]:
        text = content.lower()
        best_title = None
        best_path = None
        best_score = 0
        for item in self._task_candidates():
            title = item["title"]
            tokens = [t for t in title.replace("_", " ").replace("-", " ").split() if t]
            score = sum(1 for t in tokens if t.lower() in text)
            if score > best_score:
                best_score = score
                best_title = title
                best_path = item["path"]
        return (best_title, best_path) if best_score > 0 else (None, None)

    def _detect_topic(self, content: str) -> tuple[str | None, str | None, Path | None]:
        lowered = content.lower()
        keys = [
            "topic:", "topic：",
            "task:", "task：",
            "project:", "project：",
            "mission:", "mission：",
            "theme:", "theme：",
            "subject:", "subject：",
            "主题:", "主题：",
            "话题:", "话题：",
            "项目:", "项目：",
            "任务:", "任务：",
        ]
        for key in keys:
            idx = lowered.find(key)
            if idx >= 0:
                raw = content[idx + len(key):].strip()
                candidate = raw.split()[0] if raw else ""
                if candidate:
                    return candidate, f"explicit:{key}", None
        title, path = self._match_topic_from_tasks(content)
        if title:
            return title, "task_match", path
        if "```" in content or len(content) >= 80:
            snippet = content.strip().splitlines()[0][:50]
            candidate = "".join(c if c.isalnum() or c in "-_" else "_" for c in snippet).strip("_")
            if candidate:
                return candidate[:50], "auto_long", None
        return None, None, None

    def _bind_task_topic(self, task_path: Path, topic: str, board_rel: str) -> None:
        if not task_path.exists():
            return
        try:
            text = task_path.read_text(encoding="utf-8")
        except Exception:
            return
        if "Topic:" in text:
            return
        header = f"# Task: {task_path.stem}\nTopic: {topic}\nBoard: {board_rel}\n\n"
        task_path.write_text(header + text, encoding="utf-8")

    def _list_topics(self, base_key: str) -> list[dict[str, Any]]:
        topics = []
        prefix = f"{base_key}:"
        for item in self.sessions.list_sessions():
            key = item.get("key", "")
            if key.startswith(prefix):
                topic_name = key[len(prefix):]
                topics.append({
                    "name": topic_name,
                    "updated_at": item.get("updated_at"),
                    "created_at": item.get("created_at"),
                })
        topics.sort(key=lambda x: x.get("updated_at") or x.get("created_at") or "", reverse=True)
        return topics

    def _list_boards(self) -> list[dict[str, Any]]:
        boards = []
        boards_root = self.workspace / "boards"
        if not boards_root.exists():
            return boards
        for path in boards_root.rglob("board.md"):
            try:
                rel = path.relative_to(self.workspace).as_posix()
            except ValueError:
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0
            boards.append({"rel": rel, "mtime": mtime})
        boards.sort(key=lambda x: x["mtime"], reverse=True)
        return boards

    def _get_orchestrator_models(self) -> list[str]:
        if isinstance(self.models, list) and self.models:
            return self.models
        if isinstance(self.models, str) and self.models:
            return [self.models]
        try:
            default_model = self.provider.get_default_model()
            return [default_model] if default_model else ["(unknown)"]
        except Exception:
            return ["(unknown)"]

    def _read_cost_summary(self) -> tuple[str, list[str], str]:
        log_path = self.workspace / "logs" / "costs.json"
        if not log_path.exists():
            return "no cost log", [], ""
        try:
            data = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            return "invalid cost log", [], ""

        if not isinstance(data, dict):
            return "invalid cost log", [], ""

        current = data.get("current") if isinstance(data.get("current"), dict) else {}
        totals = current.get("totals") if isinstance(current.get("totals"), dict) else {}
        models = current.get("models") if isinstance(current.get("models"), dict) else {}
        budget = current.get("budget") if isinstance(current.get("budget"), dict) else {}

        total_tokens = int(totals.get("total_tokens", 0))
        total_cost = float(totals.get("total_cost", 0.0))
        if total_tokens == 0 and total_cost == 0 and not models:
            return "no cost data", [], ""

        model_lines = []
        for name, stats in sorted(models.items(), key=lambda x: float(x[1].get("total_cost", 0.0)), reverse=True):
            in_tokens = int(stats.get("prompt_tokens", 0))
            out_tokens = int(stats.get("completion_tokens", 0))
            total_tokens_item = int(stats.get("total_tokens", 0))
            in_cost = float(stats.get("prompt_cost", 0.0))
            out_cost = float(stats.get("completion_cost", 0.0))
            total_cost_item = float(stats.get("total_cost", 0.0))
            price_in = stats.get("prompt_cost_per_token")
            price_out = stats.get("completion_cost_per_token")
            price_in_text = f"{price_in:.8f}" if isinstance(price_in, (int, float)) else "n/a"
            price_out_text = f"{price_out:.8f}" if isinstance(price_out, (int, float)) else "n/a"
            model_lines.append(
                f"{name}: in_tokens={in_tokens}, out_tokens={out_tokens}, "
                f"in_cost={in_cost:.6f}, out_cost={out_cost:.6f}, "
                f"total_tokens={total_tokens_item}, total_cost={total_cost_item:.6f}, "
                f"price_in={price_in_text}, price_out={price_out_text}"
            )

        budget_limit = budget.get("monthly_usd") if isinstance(budget.get("monthly_usd"), (int, float)) else None
        if isinstance(budget_limit, (int, float)) and budget_limit > 0:
            percent = (total_cost / float(budget_limit) * 100) if budget_limit else 0.0
            status = " exceeded" if total_cost > float(budget_limit) else ""
            budget_text = f"{total_cost:.6f}/{budget_limit:.6f} USD ({percent:.1f}%){status}"
        else:
            budget_text = ""

        return f"tokens={total_tokens}, cost={total_cost:.6f}", model_lines, budget_text

    async def _check_budget_thresholds(self) -> None:
        """Check budget usage and send alerts when thresholds are crossed."""
        from nanobot.config.schema import Config
        
        # Load config to get budget settings
        try:
            config = Config()
            budget_config = config.budget
            monthly_budget = budget_config.monthly_usd
            alert_thresholds = budget_config.alert_thresholds
            
            if not monthly_budget or monthly_budget <= 0:
                return
                
            if not alert_thresholds:
                return
        except Exception:
            return
        
        # Get current cost data
        cost_total, cost_models, cost_budget = self._read_cost_summary()
        
        # Extract total cost from cost_total string (format: "tokens=X, cost=Y.ZZZZZZ")
        total_cost = 0.0
        if cost_total and "cost=" in cost_total:
            try:
                total_cost = float(cost_total.split("cost=")[1])
            except (IndexError, ValueError):
                return
        
        if total_cost <= 0:
            return
            
        # Calculate usage percentage
        usage_percentage = total_cost / monthly_budget
        
        # Check each threshold
        for threshold in alert_thresholds:
            if threshold <= 0 or threshold > 1:
                continue
                
            # Check if this threshold has been crossed and not yet alerted
            if usage_percentage >= threshold and threshold not in self._budget_alerts_sent:
                # Mark this threshold as alerted
                self._budget_alerts_sent.add(threshold)
                
                # Send alert to all enabled channels
                await self._broadcast_budget_alert(threshold, usage_percentage, total_cost, monthly_budget)
    
    async def _broadcast_budget_alert(self, threshold: float, usage: float, current_cost: float, budget: float) -> None:
        """Broadcast budget alert to all enabled channels."""
        from nanobot.config.schema import Config
        
        try:
            config = Config()
            
            # Build alert message
            percentage = usage * 100
            threshold_percentage = threshold * 100
            
            alert_message = (
                f"🚨 Budget Alert: {percentage:.1f}% of monthly budget used\n"
                f"Current cost: ${current_cost:.2f} / ${budget:.2f}\n"
                f"Threshold crossed: {threshold_percentage:.0f}%"
            )
            
            # Get list of enabled channels
            enabled_channels = []
            
            # Check each channel type
            if config.channels.telegram.enabled:
                enabled_channels.append("telegram")
            if config.channels.whatsapp.enabled:
                enabled_channels.append("whatsapp")
            if config.channels.discord.enabled:
                enabled_channels.append("discord")
            if config.channels.feishu.enabled:
                enabled_channels.append("feishu")
            if config.channels.dingtalk.enabled:
                enabled_channels.append("dingtalk")
            if config.channels.email.enabled:
                enabled_channels.append("email")
            if config.channels.slack.enabled:
                enabled_channels.append("slack")
            if config.channels.mochat.enabled:
                enabled_channels.append("mochat")
            if config.channels.qq.enabled:
                enabled_channels.append("qq")
            
            # Send alert to each enabled channel
            for channel in enabled_channels:
                try:
                    # For broadcast, we'll use a special chat_id or broadcast mechanism
                    # Different channels may have different broadcast approaches
                    if channel == "telegram":
                        # Telegram: send to allowed users
                        for user_id in config.channels.telegram.allow_from:
                            msg = OutboundMessage(
                                channel="telegram",
                                chat_id=user_id,
                                content=alert_message
                            )
                            await self.bus.publish_outbound(msg)
                    
                    elif channel == "whatsapp":
                        # WhatsApp: send to allowed phones
                        for phone in config.channels.whatsapp.allow_from:
                            msg = OutboundMessage(
                                channel="whatsapp",
                                chat_id=phone,
                                content=alert_message
                            )
                            await self.bus.publish_outbound(msg)
                    
                    elif channel == "discord":
                        # Discord: this would need special handling for DMs
                        # For now, we'll skip as it requires more complex setup
                        pass
                    
                    elif channel == "email":
                        # Email: send to allowed addresses
                        for email in config.channels.email.allow_from:
                            msg = OutboundMessage(
                                channel="email",
                                chat_id=email,
                                content=alert_message
                            )
                            await self.bus.publish_outbound(msg)
                    
                    elif channel in ["feishu", "dingtalk", "slack", "mochat", "qq"]:
                        # These channels would need their specific broadcast mechanisms
                        # For now, we'll use a generic approach if available
                        # In practice, each channel handler would need to implement broadcast
                        pass
                        
                except Exception as e:
                    # Log error but continue trying other channels
                    logger.error(f"Failed to send budget alert to {channel}: {e}")
                    
        except Exception as e:
            logger.error(f"Error broadcasting budget alert: {e}")

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
    ) -> str:
        """
        Process a message directly (for CLI or cron usage).
        
        Args:
            content: The message content.
            session_key: Session identifier (overrides channel:chat_id for session lookup).
            channel: Source channel (for tool context routing).
            chat_id: Source chat ID (for tool context routing).
        
        Returns:
            The agent's response.
        """
        await self._connect_mcp()
        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content
        )
        
        response = await self._process_message(msg, session_key=session_key)
        return response.content if response else ""
