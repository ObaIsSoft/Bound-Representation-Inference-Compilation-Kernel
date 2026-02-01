
import logging
import sys
from backend.physics.domains.materials import MaterialsDomain
from backend.materials.materials_api import UnifiedMaterialsAPI

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def stress_test_materials():
    logger.info("Starting Materials Domain Stress Test...")
    
    domain = MaterialsDomain(providers={})
    
    test_materials = [
        # Common Metals
        {"name": "Aluminum", "expected_density_range": (2600, 2800)},
        {"name": "Titanium", "expected_density_range": (4400, 4600)},
        {"name": "Copper", "expected_density_range": (8800, 9000)},
        {"name": "Gold", "expected_density_range": (19000, 19500)},
        {"name": "Silver", "expected_density_range": (10400, 10600)},
        
        # Semiconductors / Others
        {"name": "Silicon", "expected_density_range": (2300, 2400)},
        
        # Compounds (Formulas)
        # Ranges relaxed to accommodate DFT data spread and polymorphs
        {"name": "Fe2O3", "expected_density_range": (4000, 5500)}, 
        {"name": "SiO2", "expected_density_range": (2000, 2800)},
        
        # Chemicals / Polymers (Thermo Library supported)
        {"name": "Polyethylene", "expected_density_range": (900, 1000)},
        {"name": "Water", "expected_density_range": (990, 1000)},
        {"name": "Ethanol", "expected_density_range": (780, 800)},
    ]
    
    failures = []
    
    for mat in test_materials:
        name = mat["name"]
        logger.info(f"--- Testing {name} ---")
        
        try:
            # 1. Test Density
            density = domain.get_property(name, "density")
            logger.info(f"  Density: {density:.2f} kg/m³")
            
            min_d, max_d = mat["expected_density_range"]
            if not (min_d <= density <= max_d):
                msg = f"Density out of range for {name}: {density} (Expected {min_d}-{max_d})"
                logger.error(f"❌ {msg}")
                failures.append(msg)
            else:
                logger.info("  ✅ Density OK")
                
            # 2. Test Young's Modulus (Elasticity)
            # Many might fail here if MP data is missing
            try:
                youngs = domain.get_property(name, "youngs_modulus")
                if youngs > 0:
                    logger.info(f"  Young's Modulus: {youngs/1e9:.2f} GPa")
                else:
                    logger.warning(f"  ⚠️ Young's Modulus zero or missing")
            except Exception as e:
                logger.warning(f"  ⚠️ Young's Modulus retrieval failed: {e}")

            # 3. Test Advanced Properties (Verbose Output)
            # Try to fetch these for ALL materials to satisfy user request
            for prop in ["specific_heat", "thermal_conductivity", "energy_above_hull"]:
                try:
                    val = domain.get_property(name, prop)
                    if val is not None:
                        unit = "J/kg/K" if prop == "specific_heat" else "W/m/K" if prop == "thermal_conductivity" else "eV/atom"
                        logger.info(f"  {prop}: {val:.4f} {unit}")
                except Exception:
                    pass # Optional property

        except Exception as e:
            msg = f"Exception for {name}: {e}"
            logger.error(f"❌ {msg}")
            failures.append(msg)
            
    logger.info("\n--- Stress Test Summary ---")
    if failures:
        logger.error(f"{len(failures)} Failures detected in Basic Test:")
        for f in failures:
            logger.error(f"  - {f}")
    else:
        logger.info("✅ Basic stress test passed.")
        
    return len(failures) == 0


def test_thermal_dependency():
    """Test temperature dependent properties."""
    print("\n--- Testing Temperature Dependence (Water) ---")
    api = UnifiedMaterialsAPI()
    
    # Water density should decrease with temperature
    rho_293 = api.get_property("Water", "density", temperature=293)
    rho_350 = api.get_property("Water", "density", temperature=350)
    
    print(f"Density @ 293K: {rho_293:.2f} kg/m3")
    print(f"Density @ 350K: {rho_350:.2f} kg/m3")
    
    if rho_350 < rho_293:
        print("✅ Density decreases with temperature correctly.")
    else:
        print(f"❌ Density failed to show thermal expansion: {rho_350} >= {rho_293}")

    # Specific Heat of Water
    try:
        cp = api.get_property("Water", "specific_heat", temperature=298)
        print(f"Specific Heat (Cp) @ 298K: {cp:.2f} J/kg/K")
    except Exception as e:
        print(f"❌ Failed Cp: {e}")

def test_stability_ehull():
    """Test energy above hull (stability)."""
    print("\n--- Testing Stability (Fe2O3) ---")
    api = UnifiedMaterialsAPI()
    
    try:
        e_hull = api.get_property("Fe2O3", "energy_above_hull")
        print(f"Energy Above Hull: {e_hull} eV/atom")
    except ValueError as e:
        print(f"⚠️ Could not retrieve E_hull: {e}")

def test_polymer_advanced():
    """Test advanced properties for Polymers (Polyethylene)."""
    print("\n--- Testing Advanced Polymer Properties (Polyethylene) ---")
    api = UnifiedMaterialsAPI()
    material = "Polyethylene"
    
    # 1. Temperature Dependence of Density
    try:
        r_293 = api.get_property(material, "density", temperature=293)
        print(f"Density @ 293K: {r_293:.2f} kg/m3")
        
        # 2. Specific Heat
        cp = api.get_property(material, "specific_heat", temperature=298)
        print(f"Specific Heat (Cp): {cp:.2f} J/kg/K")
        
        # 3. Thermal Conductivity
        k = api.get_property(material, "thermal_conductivity", temperature=298)
        print(f"Thermal Conductivity (k): {k:.2f} W/m/K")
        
    except Exception as e:
        print(f"❌ Polymer Check Failed: {e}")

if __name__ == "__main__":
    # Run ALL tests unconditionally
    stress_test_materials()
    test_thermal_dependency()
    test_stability_ehull()
    test_polymer_advanced()
    print("\n✅ Verification Complete.")
