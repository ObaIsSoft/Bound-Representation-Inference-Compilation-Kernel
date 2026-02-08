from typing import Any, Dict, Optional
import time
import asyncio
import logging
from xai_stream import inject_thought

logger = logging.getLogger(__name__)

class ExplainableAgent:
    """
    Wrapper for agents to automatically inject thoughts into the XAI stream.
    
    This wrapper intercepts the `run` method and:
    1. Injects a "Starting..." thought.
    2. Executes the underlying agent.
    3. Injects a "Finished..." thought with a summary.
    4. Handles errors gracefully.
    """
    
    def __init__(self, agent: Any, name: Optional[str] = None):
        self.agent = agent
        self.name = name or getattr(agent, "name", agent.__class__.__name__)
        
    async def run(self, *args, **kwargs) -> Any:
        """
        Execute the wrapped agent with observability.
        """
        start_time = time.time()
        
        # Extract intent if available (heuristic)
        intent = kwargs.get("intent") or kwargs.get("user_intent") or "Executing task"
        if isinstance(args, tuple) and len(args) > 1:
             # Heuristic for some agents where intent is 2nd arg
            if isinstance(args[1], str):
                intent = args[1]
                
        inject_thought(self.name, f"Starting: {intent}")
        
        try:
            # Check if agent.run is async
            if asyncio.iscoroutinefunction(self.agent.run):
                result = await self.agent.run(*args, **kwargs)
            else:
                result = self.agent.run(*args, **kwargs)
                # If it returned a coroutine (some valid patterns do this), await it
                if asyncio.iscoroutine(result):
                    result = await result
            
            duration = time.time() - start_time
            
            # Try to summarize result
            summary = "Task completed successfully."
            if isinstance(result, dict):
                if "status" in result:
                     summary = f"Completed with status: {result['status']}"
                if "summary" in result:
                    summary = result["summary"]
            
            inject_thought(self.name, f"Finished in {duration:.2f}s. {summary}")
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Agent {self.name} failed: {e}")
            inject_thought(self.name, f"Failed after {duration:.2f}s: {str(e)}")
            raise e
            
    def __getattr__(self, name):
        """Delegate other attribute access to the wrapped agent."""
        return getattr(self.agent, name)

def create_xai_wrapper(agent: Any, **kwargs) -> ExplainableAgent:
    """
    Factory function to create an ExplainableAgent wrapper.
    Helper to maintain compatibility with existing code that expects a factory.
    """
    return ExplainableAgent(agent, **kwargs)
