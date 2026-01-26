
import sys
import os
import inspect

# Add backend to path
# Add backend to path so we can resolve 'schema', 'agents', etc.
sys.path.append(os.getcwd()) # For 'backend.physics...'
sys.path.append(os.path.join(os.getcwd(), "backend")) # For 'schema', 'agents'

from orchestrator import get_agent_registry, build_graph
from langgraph.graph import StateGraph

def check_structure():
    print("--- 1. AGENT REGISTRY AUDIT ---")
    try:
        registry = get_agent_registry()
        print(f"✅ Registry Loaded. Total Agents: {len(registry)}")
    except Exception as e:
        print(f"❌ Registry Failed: {e}")
        return

    protocol_passes = 0
    failures = []

    for name, agent in registry.items():
        # Protocol Check: Must have 'run' method
        if hasattr(agent, "run") and callable(agent.run):
            protocol_passes += 1
            # print(f"  [+] {name}: OK ({type(agent).__name__})")
        else:
            failures.append(name)
            print(f"  [-] {name}: MISSING 'run' method")

    print(f"\nProtocol Compliance: {protocol_passes}/{len(registry)}")
    if failures:
        print(f"FAILED AGENTS: {failures}")
    else:
        print("✅ ALL AGENTS IMPLEMENT RUN PROTOCOL")

    print("\n--- 2. GRAPH CONNECTIVITY AUDIT ---")
    try:
        graph = build_graph()
        # Access internal nodes - LangGraph generic access
        # This depends on LangGraph version, but we can try to extract nodes
        # If compiled, access 'nodes' might be hidden. 
        # We can just check if compile() succeeded.
        print("✅ Graph Compiled Successfully")
        
        # We can manually inspect orchestrator.py for edges (static analysis) 
        # but compilation proves struct integrity.
    except Exception as e:
        print(f"❌ Graph Compilation Failed: {e}")

if __name__ == "__main__":
    check_structure()
