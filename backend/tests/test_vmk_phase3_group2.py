
import sys
import os
import math
print("DEBUG: Starting Phase 3 Group 2 Script")
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from agents.structural_agent import StructuralAgent
from agents.chemistry_agent import ChemistryAgent
from agents.surrogate_agent import SurrogateAgent

def test_phase3_group2_integration():
    print("--- VMK PHASE 3 (GROUP 2) ANALYSIS TEST ---")
    
    # 1. Structural Agent (Stress Risers)
    print("\n[1] Testing StructuralAgent.detect_stress_risers...")
    struct = StructuralAgent()
    
    # Create geometry with a sharp internal corner (Small Tool)
    # Tool R=0.5mm < Critical 1.0mm
    history = [
        {"tool_id": "endmill_small", "radius": 0.5, "path": [[0,0,0], [10,0,0]]},
        {"tool_id": "endmill_large", "radius": 2.0, "path": [[0,10,0], [10,10,0]]}
    ]
    
    risers = struct.detect_stress_risers(history, critical_radius=1.0)
    for r in risers:
        print(f"  - {r}")
        
    if len(risers) == 1 and "endmill_small" in risers[0]:
        print("Stress Riser Detection: SUCCESS")
    else:
        print(f"Stress Riser Detection: FAILURE (Found {len(risers)})")
        
        
    # 2. Chemistry Agent (Reactive Surface)
    print("\n[2] Testing ChemistryAgent.calculate_reactive_surface...")
    chem = ChemistryAgent()
    
    # Sphere R=10mm inside 30x30x30 Box.
    # Area = 4 * PI * 10^2 = 4 * 3.14159 * 100 = 1256.6 mm2
    
    dims = [30, 30, 30]
    # To create a Sphere R=10 in VMK (Subtractive):
    # VMK starts as Solid Block. 
    # If we want a Sphere Surface, we can't easily make a SOLID sphere by cutting.
    # UNLESS we treat the "Void" as the object?
    # Our ChemistryAgent logic sums Area where |SDF| < eps.
    # Whether it's a hole or a hill, the surface area is the same.
    # So we cut a Sphere R=10 from the block. The surface area of the hole is equal to area of sphere.
    
    sphere_op = [{"tool_id": "ball", "radius": 10.0, "path": [[0,0,0], [0,0,0]]}]
    
    area = chem.calculate_reactive_surface(sphere_op, dims)
    print(f"Calculated Area: {area:.2f} mm^2")
    
    expected = 4 * math.pi * 100
    print(f"Expected Area: {expected:.2f} mm^2")
    
    # MC error can be 10-15% with 5000 samples.
    error = abs(area - expected) / expected
    print(f"Error: {error * 100:.1f}%")
    
    if error < 0.25: # Generous 25% tolerance for simplified MC
        print("Surface Area Logic: SUCCESS (Within MC Tolerance)")
    else:
        print("Surface Area Logic: FAILURE (Too inaccurate)")


    # 3. Surrogate Agent (Ground Truth Validation)
    print("\n[3] Testing SurrogateAgent.validate_prediction...")
    surrogate = SurrogateAgent()
    
    # Scenario: Point inside Rock.
    # Terrain: [0,0,0] is Rock (SDF < 0). Wait, depends on terrain logic.
    # In PhysicsAgent test: Terrain = Obstacles (Cuts).
    # Stock = Rock. Cuts = Air.
    # So if NO cuts, everything is Rock.
    
    state = {
        "position": [0,0,0],
        "context": {
            "terrain": {
                "dims": [100,100,100],
                "obstacles": [] # Solid Block of Rock
            }
        }
    }
    
    # Surrogate hallucinates "Safe"
    prediction = {
        "predicted_safety_score": 0.95
    }
    
    validation = surrogate.validate_prediction(prediction, state)
    print(f"Validation Result: {validation}")
    
    if validation["drift_alert"] == True and validation["ground_truth"] == "COLLISION":
        print("Ground Truth Verification: SUCCESS")
    else:
        print("Ground Truth Verification: FAILURE")

if __name__ == "__main__":
    test_phase3_group2_integration()
