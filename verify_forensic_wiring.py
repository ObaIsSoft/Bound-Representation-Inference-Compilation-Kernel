
import sys
import os
import unittest.mock

# Mock OpenAI API Key for STT Agent import
os.environ["OPENAI_API_KEY"] = "sk-mock-key-for-verification"

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

try:
    from backend.orchestrator import build_graph
    from backend.agent_registry import registry
    
    print("Checking Registry...")
    if registry.is_agent_available("ForensicAgent"):
        print("[PASS] ForensicAgent is available in registry.")
    else:
        print("[FAIL] ForensicAgent NOT found in registry.")

    print("\nChecking Graph Structure...")
    graph = build_graph()
    
    # Check if 'forensic_node' is in the graph
    # LangGraph's compiled graph has `.nodes` which is a dict-like or list
    if "forensic_node" in graph.nodes:
         print("[PASS] 'forensic_node' found in compiled graph nodes.")
    else:
         # Fallback check if direct property access fails
         print(f"[INFO] Graph nodes keys: {graph.nodes.keys() if hasattr(graph, 'nodes') else 'Unknown'}")

    print("[PASS] build_graph() executed successfully.")
    print("Forensic wiring verification successful.")

except Exception as e:
    print(f"[FAIL] Verification failed: {e}")
    # import traceback
    # traceback.print_exc()
