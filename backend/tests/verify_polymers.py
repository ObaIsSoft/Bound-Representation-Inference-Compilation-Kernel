
import sys
import os
import logging
from pprint import pprint

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

# Mock imports if needed, or rely on real path
try:
    from agents.materials_oracle.adapters.polymers_adapter import PolymersAdapter
except ImportError:
    print("Failed to import PolymersAdapter")
    sys.exit(1)

def test_polymer_design():
    print("🚀 Starting Polymer Design Verification")
    
    adapter = PolymersAdapter()
    
    # 1. Synthesize Kevlar (PPD-T Fiber)
    print("\n--- Synthesis: Aramid Fiber (Kevlar) ---")
    kevlar_params = {
        "type": "SYNTHESIS",
        "monomer": "PPD-T",
        "chain_length": 5000,
        "chain_alignment": 0.95, # Very high alignment (spun fiber)
        "cross_link_density": 0.2
    }
    kevlar = adapter.run_simulation(kevlar_params)
    pprint(kevlar)
    
    props = kevlar["properties"]
    if props["classification"] == "HIGH_PERFORMANCE_FIBER":
        print("✅ Correctly identified as High Performance Fiber")
    else:
        print("❌ Classification Failed")

    # 2. Synthesize Polycarbonate (Transparent Armor)
    print("\n--- Synthesis: Polycarbonate ---")
    pc_params = {
        "type": "SYNTHESIS",
        "monomer": "BISPHENOL-A",
        "chain_length": 2000,
        "chain_alignment": 0.1, # Amorphous (Transparent)
        "cross_link_density": 0.1
    }
    pc = adapter.run_simulation(pc_params)
    pprint(pc)
    
    # 3. Ballistic Test: 50 Cal vs Kevlar
    print("\n--- Ballistics: .50 BMG vs 25mm Kevlar Plate ---")
    ballistic_params = {
        "type": "BALLISTIC",
        "projectile": "50_BMG",
        "thickness_mm": 25.0, # Thick plate
        "material_properties": props
    }
    result = adapter.run_simulation(ballistic_params)
    pprint(result)
    
    if result["impact_energy_j"] > 10000:
        print("✅ Simulation successfully engaged High-Energy Physics (15kJ+)")
        
    # 4. Space Qualification: Polyethylene (Spectra)
    print("\n--- Space Qual: Polyethylene (Spectra) ---")
    pe_params = {
        "type": "SPACE",
        "monomer_name": "ETHYLENE",
        "material_properties": {"density_g_cc": 0.97}
    }
    space_res = adapter.run_simulation(pe_params)
    pprint(space_res)
    
    if space_res["radiation_shielding_score"] > 90:
        print("✅ Polyethylene correctly identified as excellent radiation shield")

if __name__ == "__main__":
    test_polymer_design()
