
import sys
import os
import math
print("DEBUG: Starting Thermodynamics Verification Script")
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from agents.physics_oracle.physics_oracle import PhysicsOracle

def test_thermo_integration():
    print("--- THERMODYNAMICS & ENERGY ORACLE TEST ---")
    
    oracle = PhysicsOracle()
    
    # [1] HEAT ENGINE: Nuclear to Electric
    print("\n[1] Testing Heat Engine (Rankine Cycle)...")
    # 1GW Thermal Input (from reactor), 1000K Source, 300K Sink.
    eng_res = oracle.solve(
        query="Convert Nuclear Heat",
        domain="THERMODYNAMICS",
        params={
            "type": "ENGINE",
            "cycle": "RANKINE",
            "T_hot_k": 1000.0,
            "T_cold_k": 300.0,
            "input_thermal_power_mw": 1000.0
        }
    )
    print(f"Engine Result:\n{eng_res}")
    
    p_elec = eng_res.get("output_electric_power_mw", 0.0)
    p_waste = eng_res.get("waste_heat_mw", 0.0)
    print(f"Electric Output: {p_elec:.2f} MW")
    print(f"Waste Heat: {p_waste:.2f} MW")
    
    if eng_res["status"] == "solved" and p_elec > 0 and p_waste > 0:
         print("Thermo Solver (Engine): SUCCESS")
    else:
         print("Thermo Solver (Engine): FAILURE")

    # [2] RADIATOR: Cooling System
    print("\n[2] Testing Radiator Sizing (Space Cooling)...")
    # Need to verify filtering the waste heat. Let's say we need to dump 650MW at 800K.
    # Sigma = 5.67e-8. Flux = 0.9 * Sigma * 800^4 ~ 20kW/m2
    # Area ~ 650e6 / 20e3 ~ 32500 m2.
    rad_res = oracle.solve(
        query="Size Radiators",
        domain="THERMODYNAMICS",
        params={
             "type": "RADIATOR",
             "waste_heat_mw": 650.0,
             "radiator_temp_k": 800.0,
             "emissivity": 0.9
        }
    )
    print(f"Radiator Result:\n{rad_res}")
    area = rad_res.get("required_area_m2", 0.0)
    print(f"Required Area: {area:,.0f} m^2")
    
    if rad_res["status"] == "solved" and area > 1000:
         print("Thermo Solver (Radiator): SUCCESS")
    else:
         print("Thermo Solver (Radiator): FAILURE")

    # [3] SOLAR: Mars Power
    print("\n[3] Testing Solar Power (Mars)...")
    solar_res = oracle.solve(
        query="Solar at Mars",
        domain="THERMODYNAMICS",
        params={
            "type": "SOLAR",
            "distance_au": 1.52, # Mars
            "panel_area_m2": 100.0,
            "efficiency": 0.25 
        }
    )
    print(f"Solar Result:\n{solar_res}")
    flux = solar_res.get("solar_flux_w_m2", 0.0)
    power = solar_res.get("output_power_kw", 0.0)
    
    # Earth Flux ~1361. Mars Flux ~ 1361 / 1.52^2 ~ 589 W/m2
    print(f"Mars Solar Flux: {flux:.1f} W/m^2 (Theory ~590)")
    print(f"Array Power: {power:.1f} kW")
    
    if solar_res["status"] == "solved" and abs(flux - 589) < 20:
         print("Thermo Solver (Solar): SUCCESS")
    else:
         print("Thermo Solver (Solar): FAILURE")

if __name__ == "__main__":
    test_thermo_integration()
