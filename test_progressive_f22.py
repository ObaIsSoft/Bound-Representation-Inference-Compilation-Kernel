
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from backend.agents.openscad_agent import OpenSCADAgent

def test_f22_progressive():
    agent = OpenSCADAgent()
    
    # Read the F-22 code
    with open("f22_raptor.scad", "r") as f:
        code = f.read()
        
    print("Starting Progressive Compilation...")
    
    # Compile
    generator = agent.compile_assembly_progressive(code)
    
    parts = []
    errors = []
    
    for event in generator:
        if event["event"] == "part":
            v_count = len(event.get('vertices', []))
            p_idx = event.get('part_index', '?')
            print(f"✓ Part [{p_idx}] Compiled: {event['part_name']} ({v_count} verts)")
            parts.append(event)
        elif event["event"] == "part_error":
            print(f"✗ Part Failed: {event['part_id']} - {event['error']}")
            errors.append(event)
        elif event["event"] == "error":
            print(f"CRITICAL ERROR: {event['error']}")
            errors.append(event)
            
    print(f"\nSummary: {len(parts)} parts compiled, {len(errors)} errors.")
    
    if len(errors) == 0 and len(parts) > 0:
        print("SUCCESS: Progressive compilation working with globals.")
        sys.exit(0)
    else:
        print("FAILURE: Errors detected.")
        sys.exit(1)

if __name__ == "__main__":
    test_f22_progressive()
