
import sys
import os
import json
import logging

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from agents.physics_oracle.adapters.optics_adapter import OpticsAdapter

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_wave_optics():
    print("🚀 Starting Wave Optics Verification")
    adapter = OpticsAdapter()
    
    # 1. Double Slit (Young's)
    print("\n--- Testing Double Slit ---")
    d_slit_params = {
        "type": "WAVE",
        "setup": "DOUBLE_SLIT",
        "wavelength_nm": 633, # HeNe Red
        "slit_separation_m": 0.0001, # 0.1mm
        "screen_distance_m": 1.0     # 1m
    }
    res = adapter.run_simulation(d_slit_params)
    print(json.dumps(res, indent=2))
    
    # Check fringe spacing: y = (633e-9 * 1) / 1e-4 = 6.33e-3 m = 6.33 mm
    expected = 6.33
    actual = res.get("fringe_spacing_mm", 0)
    if abs(actual - expected) < 0.1:
        print("✅ Double Slit Logic Correct")
    else:
        print(f"❌ Double Slit Failed. Expected ~{expected}, got {actual}")

    # 2. Diffraction Grating
    print("\n--- Testing Diffraction Grating ---")
    grating_params = {
        "type": "WAVE",
        "setup": "GRATING",
        "wavelength_nm": 532, # Green
        "lines_per_mm": 1000  # High density
    }
    res = adapter.run_simulation(grating_params)
    print(json.dumps(res, indent=2))
    
    # d = 1e-6 m (1000 lines/mm). lambda = 5.32e-7.
    # sin(theta) = 1 * 5.32e-7 / 1e-6 = 0.532
    # theta = arcsin(0.532) = ~32.14 deg
    expected_deg = 32.14
    if res["spectral_lines"]:
        actual_deg = res["spectral_lines"][0]["angle_deg"]
        if abs(actual_deg - expected_deg) < 1.0:
            print("✅ Grating Logic Correct")
        else:
             print(f"❌ Grating Failed. Expected ~{expected_deg}, got {actual_deg}")
    
    # 3. Michelson Interferometer
    print("\n--- Testing Interferometer ---")
    interferometer_params = {
        "type": "WAVE",
        "setup": "INTERFEROMETER",
        "wavelength_nm": 633,
        "mirror_move_m": 6.33e-7 # Move 1 wavelength
    }
    # 2d = m lambda -> m = 2d / lambda = 2 * 6.33e-7 / 6.33e-7 = 2 full fringes
    res = adapter.run_simulation(interferometer_params)
    print(json.dumps(res, indent=2))
    
    expected_fringes = 2.0
    actual_fringes = res.get("fringe_shift_count", 0)
    if abs(actual_fringes - expected_fringes) < 0.01:
        print("✅ Interferometer Logic Correct")
    else:
        print(f"❌ Interferometer Failed. Expected {expected_fringes}, got {actual_fringes}")

if __name__ == "__main__":
    test_wave_optics()
