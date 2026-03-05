import sys
from unittest.mock import MagicMock, AsyncMock

# Mock dependencies
sys.modules['json_repair'] = MagicMock()
sys.modules['loguru'] = MagicMock()
sys.modules['loguru'].logger = MagicMock()
sys.modules['litellm'] = MagicMock()
sys.modules['tenacity'] = MagicMock()
sys.modules['rich'] = MagicMock()
sys.modules['rich.console'] = MagicMock()
sys.modules['rich.prompt'] = MagicMock()
sys.modules['httpx'] = MagicMock()
sys.modules['requests'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['openai'] = MagicMock()
sys.modules['oauth_cli_kit'] = MagicMock()

import unittest
from unittest.mock import patch
import time

# Import modules under test
from nanobot.agent.subagent import ActivityLog, SubagentManager
from nanobot.agent.loop import InterventionTracker, AgentSupervisor

class TestActivityLog(unittest.TestCase):
    def test_initial_state(self):
        log = ActivityLog("test_task")
        self.assertEqual(log.task_id, "test_task")
        # Should be very close to 0 since it initializes last_heartbeat
        self.assertTrue(log.get_idle_seconds() < 1.0)

    def test_update_activity(self):
        log = ActivityLog("test_task")
        time.sleep(0.1)
        log.record_heartbeat("llm_response")
        self.assertTrue(log.get_idle_seconds() < 0.1)
        self.assertEqual(len(log.events), 1)
        self.assertEqual(log.events[0]["type"], "llm_response")

    def test_to_dict(self):
        log = ActivityLog("test_task")
        status = log.to_dict()
        self.assertIn("idle_seconds", status)
        self.assertIn("task_id", status)

class TestInterventionTracker(unittest.TestCase):
    def test_record_and_get(self):
        tracker = InterventionTracker()
        tracker.record("task1", 1, "hint")
        
        last = tracker.get_last_intervention("task1")
        self.assertIsNotNone(last)
        self.assertEqual(last["level"], 1)
        self.assertEqual(last["action"], "hint")
        
    def test_should_escalate(self):
        tracker = InterventionTracker()
        tracker.record("task1", 1, "hint")
        
        # Not enough time passed
        self.assertFalse(tracker.should_escalate("task1", timeout_seconds=10))
        
        # Manually adjust time to simulate timeout
        tracker._history["task1"][0]["time"] -= 20
        self.assertTrue(tracker.should_escalate("task1", timeout_seconds=10))
        
    def test_mark_result(self):
        tracker = InterventionTracker()
        tracker.record("task1", 1, "hint")
        tracker.mark_result("task1", "success")
        
        last = tracker.get_last_intervention("task1")
        self.assertEqual(last["result"], "success")

class TestAgentSupervisor(unittest.TestCase):
    def setUp(self):
        self.mock_subagent_manager = MagicMock()
        self.mock_bus = MagicMock()
        self.mock_bus.publish_inbound = AsyncMock()
        self.supervisor = AgentSupervisor(self.mock_subagent_manager, self.mock_bus)
        # Mock the internal tracker
        self.supervisor.interventions = MagicMock()

    def test_check_health_healthy(self):
        # Mock a healthy task
        self.mock_subagent_manager.get_all_task_statuses.return_value = [{
            "id": "task1",
            "idle_seconds": 10,
            "profile": "researcher"
        }]
        
        # Run check
        # We need to run the async method synchronously
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.supervisor._check_all_tasks())
        loop.close()
        
        # Should not intervene
        self.mock_subagent_manager.post_event.assert_not_called()

    def test_check_health_l1_intervention(self):
        # Mock a stalled task (L1 threshold is usually around 120s for default)
        self.mock_subagent_manager.get_all_task_statuses.return_value = [{
            "id": "task_stalled",
            "idle_seconds": 2000, # > default timeout
            "profile": "debugger" 
        }]
        
        # Mock tracker: should_escalate = False, get_last_intervention = None (so L1 triggers)
        self.supervisor.interventions.should_escalate.return_value = False
        self.supervisor.interventions.get_last_intervention.return_value = None
        
        # Run check
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.supervisor._check_all_tasks())
        loop.close()
        
        # Should trigger L1 intervention (inject hint)
        self.mock_subagent_manager.post_event.assert_called()
        args = self.mock_subagent_manager.post_event.call_args
        self.assertEqual(args[0][0], "task_stalled")
        self.assertEqual(args[0][1]["type"], "inject_hint")

    def test_check_health_l2_intervention(self):
        # Mock a very stalled task
        self.mock_subagent_manager.get_all_task_statuses.return_value = [{
            "id": "task_dead",
            "idle_seconds": 2000,
            "profile": "general"
        }]
        
        # Mock tracker: should_escalate = True
        self.supervisor.interventions.should_escalate.return_value = True
        
        # Run check
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.supervisor._check_all_tasks())
        loop.close()
        
        # Should trigger L2 intervention (rescue/switch)
        self.mock_subagent_manager.post_event.assert_called()
        args = self.mock_subagent_manager.post_event.call_args
        self.assertEqual(args[0][0], "task_dead")
        self.assertEqual(args[0][1]["type"], "rescue_switch")
        
        # Should also publish system alert
        self.mock_bus.publish_inbound.assert_called()

if __name__ == '__main__':
    unittest.main()
