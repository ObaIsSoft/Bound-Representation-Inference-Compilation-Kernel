import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from agents.manifold_agent import ManifoldAgent
from vmk_kernel import ToolProfile

def test_manifold_reconstruction():
    print("Initializing ManifoldAgent...")
    agent = ManifoldAgent()
    
    # 1. Define Stock and History
    stock_dims = [10.0, 10.0, 10.0]
    
    # Simple history: Cut a sphere from center
    history = [
        {
            "tool_id": "ball_mill_1", 
            "radius": 2.0,
            "path": [[0.0, 0.0, 10.0], [0.0, 0.0, -10.0]], # Vertical cut through center
            "type": "CUT"
        }
    ]
    
    print(f"Running repair_topology on {len(history)} instructions...")
    
    result = agent.repair_topology(history, stock_dims)
    
    print("--- Result ---")
    print(f"Status: {result.get('status')}")
    if result.get('error'):
        print(f"Error: {result.get('error')}")
    
    print("Logs:")
    for l in result.get("logs", []):
        print(f"  {l}")
        
    # Assertions
    if result.get("status") == "repaired":
        print(f"Is Watertight? {result.get('is_watertight')}")
        print(f"Vertices: {result.get('new_vertex_count')}")
        print(f"Faces: {result.get('new_face_count')}")
        
        if result.get("is_watertight") and result.get("new_face_count") > 0:
            print("SUCCESS: Mesh reconstructed and watertight.")
            sys.exit(0)
        else:
            print("FAILURE: Mesh not watertight or empty.")
            sys.exit(1)
    else:
        print("FAILURE: Operation failed.")
        sys.exit(1)

if __name__ == "__main__":
    test_manifold_reconstruction()
