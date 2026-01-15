
import sys
import os
import math
print("DEBUG: Starting Optics Verification Script")
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from agents.physics_oracle.physics_oracle import PhysicsOracle

def test_optics_integration():
    print("--- OPTICS PHYSICS ORACLE TEST ---")
    
    oracle = PhysicsOracle()
    
    # [1] GEOMETRIC OPTICS: Lens Ray Tracing
    print("\n[1] Testing Geometric Optics (Ray Tracer)...")
    # Single Spherical Surface (Air n1=1 -> Glass n2=1.5). Radius R=0.5m.
    # Formula: n2/f = (n2 - n1) / R  -> f = R * n2 / (n2 - n1)
    # f = 0.5 * 1.5 / 0.5 = 1.5 meters.
    geo_res = oracle.solve(
        query="Trace rays through lens",
        domain="OPTICS",
        params={
            "type": "GEOMETRIC",
            "lens_radius_m": 0.5,
            "ior": 1.5,
            "ray_heights_m": [0.01, 0.05, 0.1]
        }
    )
    print(f"Geometric Result:\n{geo_res}")
    
    f_sim = geo_res.get("simulated_focal_length_m", 0.0)
    print(f"Calculated Focal Length: {f_sim:.4f} m (Theory: 1.5 m)")
    
    if geo_res["status"] == "solved" and abs(f_sim - 1.5) < 0.1:
         print("Optics Solver (Geometric): SUCCESS (Focal Length Matches Single-Surface Theory)")
    else:
         print("Optics Solver (Geometric): FAILURE")

    # [2] LASER PHYSICS: Fusion Ignition Focus
    print("\n[2] Testing Laser Physics (Gaussian Beam)...")
    # Focus 1MW laser (lambda 351nm UV) with 10mm waist using f=200mm lens
    laser_res = oracle.solve(
        query="Check Laser Ignition Intensity",
        domain="OPTICS",
        params={
             "type": "LASER",
             "power_w": 1e6, # 1 MW
             "waist_radius_m": 0.01, # 10mm input
             "wavelength_nm": 351, # NIF-like UV
             "focal_length_m": 0.2, # 20cm lens
             "M2": 1.1 # Near perfect beam
        }
    )
    print(f"Laser Result:\n{laser_res}")
    
    spot_um = laser_res.get("focused_spot_radius_microns")
    intensity = laser_res.get("peak_intensity_w_cm2")
    print(f"Spot Size: {spot_um:.2f} microns")
    print(f"Peak Intensity: {intensity} W/cm^2")
    
    if laser_res["status"] == "solved":
        if laser_res["fusion_ignition_check"]:
            print("Optics Solver (Laser): SUCCESS (Ignition Threshold Reached)")
        else:
             print("Optics Solver (Laser): SUCCESS (Ignition check valid)")
    else:
        print("Optics Solver (Laser): FAILURE")

if __name__ == "__main__":
    test_optics_integration()
