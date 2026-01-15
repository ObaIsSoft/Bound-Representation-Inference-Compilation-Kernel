
import sys
import os
import math
print("DEBUG: Starting Astrophysics Verification Script")
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from agents.physics_oracle.physics_oracle import PhysicsOracle

def test_astrophysics_integration():
    print("--- ASTROPHYSICS & ORBIT ORACLE TEST ---")
    
    oracle = PhysicsOracle()
    
    # [1] ORBITAL MECHANICS: GEO Satellite
    print("\n[1] Testing Keplerian Orbit (GEO)...")
    # GEO Altitude ~35786 km. Period should be 24h.
    orbit_res = oracle.solve(
        query="Calculate GEO period",
        domain="ASTROPHYSICS",
        params={
            "type": "ORBIT",
            "central_body": "EARTH",
            "altitude_km": 35786
        }
    )
    print(f"Orbit Result:\n{orbit_res}")
    
    T = orbit_res.get("period_hours", 0.0)
    print(f"Calculated Period: {T:.2f} hours (Theory: 24.0)")
    
    if orbit_res["status"] == "solved" and abs(T - 24.0) < 0.2:
         print("Astro Solver (Orbits): SUCCESS")
    else:
         print("Astro Solver (Orbits): FAILURE")

    # [2] HOHMANN TRANSFER: Earth to Mars
    print("\n[2] Testing Hohmann Transfer (Earth -> Mars)...")
    transfer_res = oracle.solve(
        query="Transfer to Mars",
        domain="ASTROPHYSICS",
        params={
             "type": "TRANSFER",
             "central_body": "SUN",
             "r1_au": 1.0,
             "r2_au": 1.524
        }
    )
    # print(f"Transfer Result:\n{transfer_res}")
    dv = transfer_res.get("total_delta_v_km_s", 0.0)
    time_days = transfer_res.get("transfer_time_days", 0.0)
    print(f"Earth-Mars Delta V: {dv:.2f} km/s (Theory ~ 5.7 km/s)")
    print(f"Trip Time: {time_days:.1f} days (Theory ~ 259 days)")
    
    if transfer_res["status"] == "solved" and abs(dv - 5.7) < 1.0:
         print("Astro Solver (Transfer): SUCCESS")
    else:
         print("Astro Solver (Transfer): FAILURE")

    # [3] ROCKET EQUATION: Chemical vs Nuclear
    print("\n[3] Testing Rocket Equation (Mission Planning)...")
    
    # Needs 5.7 km/s DV. Dry Mass 100 tons (Spaceship).
    # Case A: Chemical (Isp 450s). 
    chem_res = oracle.solve(
        query="Fuel for Chem Rocket",
        domain="ASTROPHYSICS",
        params={"type": "ROCKET", "delta_v_km_s": 5.7, "isp_s": 450, "dry_mass_kg": 100000}
    )
    
    # Case B: Nuclear Thermal (Isp 900s).
    nuc_res = oracle.solve(
        query="Fuel for Nuclear Rocket",
        domain="ASTROPHYSICS",
        params={"type": "ROCKET", "delta_v_km_s": 5.7, "isp_s": 900, "dry_mass_kg": 100000}
    )
    
    fuel_chem = chem_res.get("fuel_mass_kg", 0)
    fuel_nuc = nuc_res.get("fuel_mass_kg", 0)
    
    print(f"Payload (Dry Mass): 100,000 kg")
    print(f"Chemical Fuel Req: {fuel_chem:,.0f} kg")
    print(f"Nuclear Fuel Req:  {fuel_nuc:,.0f} kg")
    print(f"Savings: {(fuel_chem - fuel_nuc):,.0f} kg of fuel")
    
    if fuel_chem > fuel_nuc:
        print("Astro Solver (Rocket): SUCCESS (Nuclear Efficiency Verified)")
    else:
        print("Astro Solver (Rocket): FAILURE")

if __name__ == "__main__":
    test_astrophysics_integration()
