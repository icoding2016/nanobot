"""Subagent manager for background task execution."""

import asyncio
import json
import uuid
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
import json_repair

from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
from nanobot.agent.memory import MemoryStore


class ActivityLog:
    """
    Tracks SubAgent activity for heartbeat monitoring.
    
    Records heartbeat events (LLM responses, tool calls, board updates)
    to allow Orchestrator to monitor SubAgent health.
    """
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.last_heartbeat: float = time.time()
        self.iteration: int = 0
        self.events: list[dict[str, Any]] = []
        self._max_events = 50
    
    def record_heartbeat(self, event_type: str, detail: str = "") -> None:
        """
        Record a heartbeat event.
        
        Args:
            event_type: "llm_response", "tool_call", "board_update", "error"
            detail: Brief description (truncated to 100 chars)
        """
        now = time.time()
        self.last_heartbeat = now
        self.events.append({
            "time": now,
            "type": event_type,
            "detail": detail[:100] if detail else ""
        })
        # Keep only recent events
        if len(self.events) > self._max_events:
            self.events = self.events[-self._max_events:]
    
    def get_idle_seconds(self) -> float:
        """Get seconds since last heartbeat."""
        return time.time() - self.last_heartbeat
    
    def get_recent_events(self, count: int = 5) -> list[dict[str, Any]]:
        """Get most recent events."""
        return self.events[-count:] if self.events else []
    
    def to_dict(self) -> dict[str, Any]:
        """Export status as dict."""
        return {
            "task_id": self.task_id,
            "last_heartbeat": self.last_heartbeat,
            "idle_seconds": self.get_idle_seconds(),
            "iteration": self.iteration,
            "event_count": len(self.events),
            "recent_events": self.get_recent_events()
        }


class SubagentManager:
    """
    Manages background subagent execution.
    
    Subagents are lightweight agent instances that run in the background
    to handle specific tasks. They share the same LLM provider but have
    isolated context and a focused system prompt.
    """
    
    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        bus: MessageBus,
        models: list[str] | str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        restrict_to_workspace: bool = False,
        agent_profiles: dict[str, Any] | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig
        self.provider = provider
        self.workspace = workspace
        self.bus = bus
        self.models = models
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.restrict_to_workspace = restrict_to_workspace
        self.agent_profiles = agent_profiles or {}
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
        self._running_meta: dict[str, dict[str, Any]] = {}
        self._activity_logs: dict[str, ActivityLog] = {}
    
    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        profile: str | None = None,
    ) -> str:
        """
        Spawn a subagent to execute a task in the background.
        
        Args:
            task: The task description for the subagent.
            label: Optional human-readable label for the task.
            origin_channel: The channel to announce results to.
            origin_chat_id: The chat ID to announce results to.
            profile: Optional agent profile name to use.
        
        Returns:
            Status message indicating the subagent was started.
        """
        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")
        
        origin = {
            "channel": origin_channel,
            "chat_id": origin_chat_id,
        }
        
        # Determine models from profile
        models_to_use = self.models
        if profile and profile in self.agent_profiles:
            profile_config = self.agent_profiles[profile]
            if profile_config.models:
                models_to_use = profile_config.models
                logger.info(f"Using profile '{profile}' models: {models_to_use}")
        
        # Create background task
        bg_task = asyncio.create_task(
            self._run_subagent(task_id, task, display_label, origin, models_to_use, profile)
        )
        self._running_tasks[task_id] = bg_task
        self._running_meta[task_id] = {
            "id": task_id,
            "label": display_label,
            "profile": profile,
            "started_at": datetime.now().isoformat(),
            "board_rel": None,
        }
        
        # Create activity log for heartbeat monitoring
        self._activity_logs[task_id] = ActivityLog(task_id)
        
        # Cleanup when done
        def _cleanup(_: asyncio.Task[None]) -> None:
            self._running_tasks.pop(task_id, None)
            self._running_meta.pop(task_id, None)
            self._activity_logs.pop(task_id, None)
        bg_task.add_done_callback(_cleanup)
        
        logger.info(f"Spawned subagent [{task_id}]: {display_label}")
        return f"Subagent [{display_label}] started (id: {task_id}). I'll notify you when it completes."
    
    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, str],
        models: list[str] | str | None,
        profile_name: str | None,
    ) -> None:
        """Execute the subagent task and announce the result."""
        logger.info(f"Subagent [{task_id}] starting task: {label}")
        
        # Setup Blackboard — boards/<agent>/<task_id>-<label>/board.md
        safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:30]
        agent_dir_name = profile_name or "generic"
        task_dir = self.workspace / "boards" / agent_dir_name / f"{task_id}-{safe_label}"
        task_dir.mkdir(parents=True, exist_ok=True)
        board_path = task_dir / "board.md"
        board_rel = f"boards/{agent_dir_name}/{task_id}-{safe_label}/board.md"
        if task_id in self._running_meta:
            self._running_meta[task_id]["board_rel"] = board_rel
        
        initial_board_content = f"""# Task Board: {label}
ID: {task_id}
Profile: {profile_name or 'generic'}
Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Task Description
{task}

## Status
In Progress

## Notes & Findings
(Subagent will write here)
"""
        board_path.write_text(initial_board_content, encoding="utf-8")
        
        try:
            # Build subagent tools (no message tool, no spawn tool)
            tools = ToolRegistry()
            allowed_dir = self.workspace if self.restrict_to_workspace else None
            tools.register(ReadFileTool(allowed_dir=allowed_dir))
            tools.register(WriteFileTool(allowed_dir=allowed_dir))
            tools.register(EditFileTool(allowed_dir=allowed_dir))
            tools.register(ListDirTool(allowed_dir=allowed_dir))
            tools.register(ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.restrict_to_workspace,
            ))
            tools.register(WebSearchTool(api_key=self.brave_api_key))
            tools.register(WebFetchTool())
            
            # Build messages with subagent-specific prompt
            system_prompt = self._build_subagent_prompt(task, board_rel)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please start working on the task defined in {board_rel}."},
            ]
            
            # Run agent loop (limited iterations)
            max_iterations = 15
            iteration = 0
            final_result: str | None = None
            
            # Get activity log for heartbeat recording
            activity_log = self._activity_logs.get(task_id)
            
            while iteration < max_iterations:
                iteration += 1
                
                # Record iteration in activity log
                if activity_log:
                    activity_log.iteration = iteration
                
                response = await self.provider.chat(
                    messages=messages,
                    tools=tools.get_definitions(),
                    models=models,  # Use profile-specific models
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                
                # Record heartbeat: LLM response received
                if activity_log:
                    activity_log.record_heartbeat("llm_response", f"iteration {iteration}")
                
                if response.has_tool_calls:
                    # Add assistant message with tool calls
                    tool_call_dicts = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in response.tool_calls
                    ]
                    messages.append({
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": tool_call_dicts,
                    })
                    
                    # Execute tools
                    for tool_call in response.tool_calls:
                        args_str = json.dumps(tool_call.arguments)
                        logger.debug(f"Subagent [{task_id}] executing: {tool_call.name} with arguments: {args_str}")
                        
                        # Record heartbeat: tool call
                        if activity_log:
                            activity_log.record_heartbeat("tool_call", f"{tool_call.name}")
                        
                        result = await tools.execute(tool_call.name, tool_call.arguments)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.name,
                            "content": result,
                        })
                        
                        # Record board update if edit_file/write_file was called
                        if activity_log and tool_call.name in ("edit_file", "write_file"):
                            activity_log.record_heartbeat("board_update", f"{tool_call.name}")
                else:
                    final_result = response.content
                    break
            
            if final_result is None:
                final_result = "Task completed but no final response was generated."
            
            # Extract Insights before archiving
            await self._extract_insights(task, final_result, board_path)
            
            # Archive Board
            archive_dir = self.workspace / "archive" / agent_dir_name
            archive_dir.mkdir(parents=True, exist_ok=True)
            if task_dir.exists():
                shutil.move(str(task_dir), str(archive_dir / f"{task_id}-{safe_label}"))
            
            logger.info(f"Subagent [{task_id}] completed successfully")
            await self._announce_result(task_id, label, task, final_result, origin, "ok")
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(f"Subagent [{task_id}] failed: {e}")
            await self._announce_result(task_id, label, task, error_msg, origin, "error")
    
    async def _extract_insights(self, task: str, result: str, board_path: Path) -> None:
        """Extract insights from the completed task board into Memory/Knowledge."""
        if not board_path.exists():
            return
            
        board_content = board_path.read_text(encoding="utf-8")
        memory = MemoryStore(self.workspace)
        
        prompt = f"""You are a Knowledge Extraction Agent. Analyze the completed task board below.
Extract valuable information into two categories:

1. "memory_update": User preferences, personal facts. Leave empty if none.
2. "knowledge_items": A list of reusable technical knowledge snippets. Each item should be an object with:
   - "category": a short lowercase category name (e.g., "python", "docker", "git", "react").
   - "name": a short unique name for this knowledge snippet (e.g., "async_subprocess_pattern", "fix_numpy_import").
   - "content": the actual knowledge/rule/fix in markdown.
   Leave as empty list if none.

Return ONLY valid JSON with keys "memory_update" (string) and "knowledge_items" (list).

## Task Board Content
{board_content}

## Task Result
{result}
"""
        try:
            response = await self.provider.chat(
                messages=[
                    {"role": "system", "content": "You are a Knowledge Extraction Agent. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                models=self.models,  # Use default models for extraction
            )
            
            text = (response.content or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            
            data = json_repair.loads(text)
            
            if mem_update := data.get("memory_update"):
                memory.write_long_term(mem_update)
            
            # Write each knowledge item to its own categorized file
            knowledge_items = data.get("knowledge_items", [])
            if isinstance(knowledge_items, list):
                knowledge_base_dir = self.workspace / "memory" / "knowledge"
                knowledge_base_dir.mkdir(parents=True, exist_ok=True)
                
                for item in knowledge_items:
                    if not isinstance(item, dict):
                        continue
                    category = item.get("category", "general").strip().lower()
                    name = item.get("name", "insight").strip().lower()
                    content = item.get("content", "").strip()
                    if not content:
                        continue
                    
                    # Sanitize filename components
                    safe_cat = "".join(c if c.isalnum() or c == "_" else "_" for c in category)
                    safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
                    knowledge_file = knowledge_base_dir / f"{safe_cat}_{safe_name}.md"
                    
                    timestamp = datetime.now().strftime("%Y-%m-%d")
                    if knowledge_file.exists():
                        # Append update
                        existing = knowledge_file.read_text(encoding="utf-8")
                        knowledge_file.write_text(
                            existing + f"\n\n---\n*Updated: {timestamp}*\n{content}",
                            encoding="utf-8"
                        )
                    else:
                        knowledge_file.write_text(
                            f"# {category.title()}: {name.replace('_', ' ').title()}\n*Created: {timestamp}*\n\n{content}",
                            encoding="utf-8"
                        )
                    logger.debug(f"Knowledge saved: {knowledge_file.name}")
                    
        except Exception as e:
            logger.error(f"Failed to extract insights: {e}")

    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        origin: dict[str, str],
        status: str,
    ) -> None:
        """Announce the subagent result to the main agent via the message bus."""
        status_text = "completed successfully" if status == "ok" else "failed"
        
        announce_content = f"""[Subagent '{label}' {status_text}]

Task: {task}

Result:
{result}

Summarize this naturally for the user. Keep it brief (1-2 sentences). Do not mention technical details like "subagent" or task IDs."""
        
        # Inject as system message to trigger main agent
        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=announce_content,
        )
        
        await self.bus.publish_inbound(msg)
        logger.debug(f"Subagent [{task_id}] announced result to {origin['channel']}:{origin['chat_id']}")
    
    def _build_subagent_prompt(self, task: str, board_rel: str) -> str:
        """Build a focused system prompt for the subagent."""
        from datetime import datetime
        import time as _time
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = _time.strftime("%Z") or "UTC"

        return f"""# Subagent

## Current Time
{now} ({tz})

You are a subagent spawned by the main agent to complete a specific task.

## Context & Blackboard
You have been assigned a specific Project Board file: `{board_rel}`.
1. READ this file immediately using `read_file` to understand your task and history.
2. UPDATE this file frequently with your findings, plan, and progress using `edit_file`.
3. This board is your primary memory — write your progress, sub-tasks, and findings here.

## Rules
1. Stay focused — complete only the assigned task, nothing else
2. Your final response will be reported back to the main agent
3. Do not initiate conversations or take on side tasks
4. Be concise but informative in your findings

## What You Can Do
- Read and write files in the workspace
- Execute shell commands
- Search the web and fetch web pages
- Complete the task thoroughly

## What You Cannot Do
- Send messages directly to users (no message tool available)
- Spawn other subagents
- Access the main agent's conversation history

## Workspace
Your workspace is at: {self.workspace}
Skills are available at: {self.workspace}/skills/ (read SKILL.md files as needed)

When you have completed the task, provide a clear summary of your findings or actions."""
    
    def get_running_count(self) -> int:
        """Return the number of currently running subagents."""
        return len(self._running_tasks)

    def list_running(self) -> list[dict[str, Any]]:
        items = list(self._running_meta.values())
        items.sort(key=lambda x: x.get("started_at", ""), reverse=True)
        return items
    
    def get_task_status(self, task_id: str) -> dict[str, Any] | None:
        """
        Get detailed status of a running task for monitoring.
        
        Args:
            task_id: The task ID to query
            
        Returns:
            Task status dict with heartbeat info, or None if task not found
        """
        meta = self._running_meta.get(task_id)
        if not meta:
            return None
        
        activity_log = self._activity_logs.get(task_id)
        
        status = {
            **meta,
            "idle_seconds": activity_log.get_idle_seconds() if activity_log else 0,
            "iteration": activity_log.iteration if activity_log else 0,
            "event_count": len(activity_log.events) if activity_log else 0,
            "recent_events": activity_log.get_recent_events() if activity_log else [],
        }
        
        return status
    
    def get_all_task_statuses(self) -> list[dict[str, Any]]:
        """
        Get status of all running tasks.
        
        Returns:
            List of task status dicts
        """
        statuses = []
        for task_id in self._running_tasks:
            status = self.get_task_status(task_id)
            if status:
                statuses.append(status)
        # Sort by started_at, newest first
        statuses.sort(key=lambda x: x.get("started_at", ""), reverse=True)
        return statuses