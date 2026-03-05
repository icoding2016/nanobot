import sys
import unittest
from unittest.mock import MagicMock

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

from nanobot.agent.loop import _extract_command

class TestAgentLoopCommands(unittest.TestCase):
    def test_extract_command(self):
        # Test default ! prefix
        prefix, cmd = _extract_command("!help")
        self.assertEqual(prefix, "!")
        self.assertEqual(cmd, "help")
        
        # Test Telegram / prefix
        prefix, cmd = _extract_command("/help")
        self.assertEqual(prefix, "/")
        self.assertEqual(cmd, "help")
        
        # Test case insensitivity (handled in process_message usually, but here just extract)
        prefix, cmd = _extract_command("!New")
        self.assertEqual(prefix, "!")
        self.assertEqual(cmd, "New")
        
        # Test no prefix
        prefix, cmd = _extract_command("hello world")
        self.assertIsNone(prefix)
        self.assertEqual(cmd, "hello world")
        
        # Test empty
        prefix, cmd = _extract_command("")
        self.assertIsNone(prefix)
        self.assertEqual(cmd, "")

if __name__ == '__main__':
    unittest.main()
