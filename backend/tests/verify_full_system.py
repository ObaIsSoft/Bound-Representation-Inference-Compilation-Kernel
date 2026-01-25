
import asyncio
import json
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from physics.kernel import get_physics_kernel
from agents.geometry_agent import GeometryAgent
from agents.material_agent import MaterialAgent
from agents.thermal_agent import ThermalAgent

async def verify_system():
    print("=" * 60)
    print("PHYSICS SYSTEM END-TO-END VERIFICATION")
    print("=" * 60)
    
    # 1. Initialize Physics Kernel
    print("\n1. Initializing Physics Kernel...")
    
    # Bypass LLM discovery to prevent hangs in test env
    class DummyLLM:
        def retrieve_equation(self, *args, **kwargs): return None
        
    kernel = get_physics_kernel(llm_provider=DummyLLM())
    print("   ✓ Kernel initialized")
    print(f"   ✓ Providers: {list(kernel.providers.keys())}")
    
    # 2. Simulate Design Flow (Geometry)
    print("\n2. Simulating Design (Geometry)...")
    geo_agent = GeometryAgent()
    # Mock geometry for validation
    geometry = {
        "type": "block",
        "dimensions": [0.1, 0.05, 0.2], # m
        "volume": 0.001, # m^3
        "cross_section_area": 0.005 # m^2
    }
    print(f"   ✓ Geometry defined: {geometry['dimensions']}")
    
    # 3. Apply Material (Physics Properties)
    print("\n3. Applying Material (Physics Properties)...")
    material_agent = MaterialAgent()
    material_name = "Aluminum 6061"
    
    # Manually fetch properties via kernel for verification
    materials_domain = kernel.domains['materials']
    density = materials_domain.get_property(material_name, "density")
    yield_strength = materials_domain.get_property(material_name, "yield_strength")
    
    print(f"   ✓ Material: {material_name}")
    print(f"   ✓ Density: {density} kg/m^3")
    print(f"   ✓ Yield Strength: {yield_strength/1e6} MPa")
    
    # 4. Validate Physics Constraints
    print("\n4. Validating Physics Constraints...")
    validation = kernel.validate_geometry(geometry, material_name)
    
    print(f"   ✓ Self-weight: {validation['self_weight']:.4f} N")
    print(f"   ✓ Stress: {validation['stress']/1e3:.4f} kPa")
    print(f"   ✓ Factor of Safety: {validation['fos']:.2f}")
    
    if validation['feasible']:
        print("   ✅ Design is FEASIBLE")
    else:
        print(f"   ❌ Design is INFEASIBLE: {validation['reason']}")
        
    # 5. Thermal Analysis
    print("\n5. Running Thermal Analysis...")
    thermal_agent = ThermalAgent()
    thermo = kernel.domains['thermodynamics']
    
    # Simple heat transfer scenario
    t_ambient = 293.15 # 20C
    t_object = 373.15  # 100C
    area = 0.5 # m^2
    h = 10.0 # W/m^2K (convection coeff)
    
    qa = thermo.calculate_heat_transfer(t_ambient, t_object, area, h)
    
    print(f"   ✓ Heat Transfer Rate: {qa:.2f} W")
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(verify_system())
