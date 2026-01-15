import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.mass_properties_agent import MassPropertiesAgent
from agents.dfm_agent import DfmAgent
from agents.gnc_agent import GncAgent
from isa import Unit

def verify_critical_agents():
    print("--- Verifying Critical Agents (Real Physics) ---")
    
    # 1. Mass Properties
    mass_agent = MassPropertiesAgent()
    mass_res = mass_agent.run({
        "volume_cm3": 1000.0,
        "material_density": 2.7, # Al
        "bounding_box": [10.0, 10.0, 10.0]
    })
    mass_val = mass_res["mass"]["magnitude"]
    print(f"MassProps: Vol=1000cc, Den=2.7 -> Mass={mass_val}kg")
    assert abs(mass_val - 2.7) < 0.01
    print("PASS: Mass Calculation")
    
    # 2. DfM Agent
    dfm_agent = DfmAgent()
    # Case A: Good wall
    dfm_res_good = dfm_agent.run({"min_wall_thickness_mm": 2.0, "method": "FDM"})
    assert dfm_res_good["manufacturable"]
    # Case B: Bad wall
    dfm_res_bad = dfm_agent.run({"min_wall_thickness_mm": 0.5, "method": "FDM"})
    assert not dfm_res_bad["manufacturable"]
    print("PASS: DfM Checks (Wall Thickness)")
    
    # 3. GNC Agent
    gnc_agent = GncAgent()
    # Case A: Good T/W (Hover capable)
    gnc_res_good = gnc_agent.run({"mass_kg": 1.0, "thrust_n": 15.0, "environment": "EARTH"})
    print(f"GNC: Mass=1kg, Thrust=15N -> T/W={gnc_res_good['tw_ratio']:.2f}")
    assert gnc_res_good["flight_ready"]
    assert gnc_res_good["tw_ratio"] > 1.5
    
    # Case B: Heavy (Crash)
    gnc_res_bad = gnc_agent.run({"mass_kg": 2.0, "thrust_n": 15.0, "environment": "EARTH"})
    assert not gnc_res_bad["flight_ready"]
    print("PASS: GNC Stability Checks")
    
    print("\nAll Critical Agents Verified Successfully. 🚀")

if __name__ == "__main__":
    verify_critical_agents()
