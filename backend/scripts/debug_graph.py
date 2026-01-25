import sys
import os

# Add parent dir (backend) AND grandparent (project root) to path
current = os.path.dirname(os.path.abspath(__file__))
backend_path = os.path.dirname(current)
project_root = os.path.dirname(backend_path)
sys.path.append(backend_path)
sys.path.append(project_root)

try:
    from orchestrator import build_graph
    print("Import successful. Attempting to compile graph...")
    graph = build_graph()
    print("✅ Graph compiled successfully.")
except Exception as e:
    print(f"❌ Graph Build Failed: {e}")
    import traceback
    traceback.print_exc()
