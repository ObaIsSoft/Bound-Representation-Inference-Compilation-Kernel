import sys
import os
print("DEBUG: Starting Script")
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

print("DEBUG: Importing MaterialAgent")
from agents.material_agent import MaterialAgent
print("DEBUG: Importing EnvironmentAgent")
from agents.environment_agent import EnvironmentAgent
print("DEBUG: Imports Done")

def test_phase2_group2_integration():
    print("--- VMK PHASE 2 (GROUP 2) INTEGRATION TEST ---")
    
    # 1. Material Agent (Exact Mass)
    print("\n[1] Testing MaterialAgent.calculate_exact_mass_sdf...")
    mat_agent = MaterialAgent()
    
    # Scenario: Block 10x10x10mm (Volume 1000mm^3 = 1e-6 m^3)
    # Drill Hole: Radius 2mm, Length 10mm. Volume = pi * r^2 * h = pi*4*10 = ~125.6mm^3
    # Net Volume = 1000 - 125.6 = 874.4mm^3 = 8.744e-7 m^3
    
    # Stock Dims need to be slightly larger than cut? Or cut equal to stock.
    dims = [10, 10, 10]
    
    toolpaths = [
        # Tool t1 radius 2mm
        {"tool_id": "drill", "path": [[0,0,-6], [0,0,6]]} # Through hole. Stock center 0,0,0. Z from -5 to 5.
    ]
    
    # Using Generic Aluminum (Density 2700)
    # Expected Mass = 8.744e-7 * 2700 = 0.00236 kg = 2.36g
    
    result = mat_agent.calculate_exact_mass_sdf("Aluminum", dims, toolpaths, precision=5000)
    
    print(f"Calculated Mass: {result['mass_kg']:.6f} kg")
    print(f"Volume: {result['volume_m3']:.9f} m^3")
    print(f"Samples: {result['samples']}")
    
    # Verification: Check if close to expected (Monte Carlo has variance)
    expected_vol = 8.744e-7
    error_pct = abs(result['volume_m3'] - expected_vol) / expected_vol * 100
    print(f"Volume Error: {error_pct:.2f}% (Expect < 5% with N=5000)")
    
    
    # 2. Environment Agent (Terrain SDF)
    print("\n[2] Testing EnvironmentAgent.evaluate_terrain_sdf...")
    env_agent = EnvironmentAgent()
    
    # Terrain: Cave System.
    # Stock 100x100x100.
    # Obstacle: Tunnel Radius 10 at Z=0.
    # SDF < 0 = In Rock. SDF > 0 = In Air (Tunnel).
    
    terrain = {
        "dims": [100, 100, 100],
        "obstacles": [
            {"tool_id": "tunnel", "radius": 10.0, "path": [[-50,0,0], [50,0,0]]}
        ]
    }
    
    # Point A: In Tunnel (Air). [0,0,0]. SDF should be > 0. 
    # Logic note: VMK = Stock - Cut. 
    # Box SDF(0,0,0) is Neg (Inside Box).
    # Cut SDF(0,0,0) is Neg (Inside Capsule).
    # max(Box, -Cut) = max(Neg, Pos) = Pos.
    # Pos = Air.
    
    pt_air = [0,0,0]
    res_air = env_agent.evaluate_terrain_sdf(pt_air, terrain)
    print(f"Point {pt_air} (Air): SDF={res_air['sdf']:.2f}, Underground={res_air['is_underground']}")
    
    # Point B: In Rock. [0, 20, 0].
    # Box SDF(0,20,0) is Neg.
    # Cut SDF(0,20,0) is Pos (Outside Capsule dist 10).
    # -Cut is Neg (-10).
    # max(Neg, Neg) = Neg.
    # Neg = Rock.
    
    pt_rock = [0,20,0]
    res_rock = env_agent.evaluate_terrain_sdf(pt_rock, terrain)
    print(f"Point {pt_rock} (Rock): SDF={res_rock['sdf']:.2f}, Underground={res_rock['is_underground']}")

    # 3. Conversational Agent (Ground Truth)
    print("\n[3] Testing ConversationalAgent.query_vmk...")
    from agents.conversational_agent import ConversationalAgent
    conv_agent = ConversationalAgent()
    
    # Query: is point [0,20,0] inside material? (Using same terrain data)
    vmk_params = {
        "dims": [100, 100, 100],
        "history": terrain["obstacles"],
        "point": [0, 20, 0]
    }
    
    # Note: ConversationalAgent needs tools to be auto-registered inside query_vmk too?
    # Checked impl: It executes history. VMKInstruction executes.
    # But VMKInstruction needs tool_id in kernel.tools.
    # My ConversationalAgent implementation just does `for op in history: kernel.execute(op)`.
    # It does NOT auto-register tools.
    # It will crash with KeyError if tool not in history?
    # Wait, the history comes from params. 
    # If the history items have tool_id, and kernel doesn't have them --> Crash.
    # I need to fix ConversationalAgent auto-reg logic FIRST.
    
    # Executing now that fix is applied:
    result_conv = conv_agent.query_vmk("distance", vmk_params)
    print(f"Conversation Query (Point [0,20,0]): {result_conv}")
    
    # Expect: SDF ~ -10 (Inside Rock). inside_material = True.
    if result_conv.get("inside_material"):
        print("Conversation Agent successfully identified material state.")
    else:
        print("Conversation Agent failed identification.")

if __name__ == "__main__":
    test_phase2_group2_integration()

