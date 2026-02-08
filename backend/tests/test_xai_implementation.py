
import unittest
import sys
import os
import asyncio

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from xai_stream import inject_thought, get_thoughts, clear_thoughts, get_thought_count
from agents.explainable_agent import create_xai_wrapper

class MockAgent:
    def __init__(self, name="MockAgent"):
        self.name = name
        
    def run(self, *args, **kwargs):
        return {"status": "success", "summary": "Mock task complete"}

class AsyncMockAgent:
    def __init__(self, name="AsyncMockAgent"):
        self.name = name
        
    async def run(self, *args, **kwargs):
        await asyncio.sleep(0.01)
        return {"status": "success", "summary": "Async mock task complete"}

class TestXAIStream(unittest.TestCase):
    def setUp(self):
        clear_thoughts()
        
    def test_basic_injection(self):
        inject_thought("TestAgent", "Hello World")
        thoughts = get_thoughts(clear=False)
        self.assertEqual(len(thoughts), 1)
        self.assertEqual(thoughts[0]["agent"], "TestAgent")
        self.assertEqual(thoughts[0]["text"], "Hello World")
        
    def test_buffer_limit(self):
        # Inject 60 thoughts
        for i in range(60):
            inject_thought("TestAgent", f"Thought {i}")
            
        thoughts = get_thoughts(clear=False)
        self.assertEqual(len(thoughts), 50) # Max limit 50
        self.assertEqual(thoughts[-1]["text"], "Thought 59")
        
    def test_destructive_read(self):
        inject_thought("TestAgent", "Thought 1")
        thoughts1 = get_thoughts(clear=True)
        self.assertEqual(len(thoughts1), 1)
        
        thoughts2 = get_thoughts(clear=False)
        self.assertEqual(len(thoughts2), 0)
        
    def test_explainable_wrapper_sync(self):
        original_agent = MockAgent("SyncAgent")
        wrapped = create_xai_wrapper(original_agent)
        
        asyncio.run(wrapped.run(intent="Testing Sync Wrapper"))
        
        thoughts = get_thoughts(clear=False)
        # Should have Start + Finish
        self.assertEqual(len(thoughts), 2)
        self.assertIn("Starting: Testing Sync Wrapper", thoughts[0]["text"])
        self.assertIn("Finished", thoughts[1]["text"])
        
    def test_explainable_wrapper_async(self):
        original_agent = AsyncMockAgent("AsyncAgent")
        wrapped = create_xai_wrapper(original_agent)
        
        asyncio.run(wrapped.run(intent="Testing Async Wrapper"))
        
        thoughts = get_thoughts(clear=False)
        self.assertEqual(len(thoughts), 2)
        self.assertIn("Starting: Testing Async Wrapper", thoughts[0]["text"])
        self.assertIn("Finished", thoughts[1]["text"])

if __name__ == '__main__':
    unittest.main()
