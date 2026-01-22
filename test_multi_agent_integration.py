#!/usr/bin/env python3
"""
Multi-Agent Integration Test: Randomized Drone Design Problem
Tests all evolved agents working together on a complex design task.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

import numpy as np
import random
from typing import Dict, Any

def generate_random_drone_design():
    """Generate randomized drone design parameters"""
    return {
        "name": f"Drone_{random.randint(1000, 9999)}",
        "geometry": [
            {
                "type": "cylinder",
                "params": {
                    "radius": random.uniform(0.5, 2.0),
                    "length": random.uniform(2.0, 8.0),
                    "material": random.choice(["aluminum", "carbon_fiber", "titanium"])
                }
            }
        ],
        "environment": random.choice(["AERIAL", "MARINE", "SPACE"]),
        "mission": {
            "duration_hours": random.uniform(1.0, 24.0),
            "payload_kg": random.uniform(1.0, 50.0),
            "max_velocity_ms": random.uniform(10.0, 100.0)
        }
    }

def test_multi_agent_workflow():
    """Test all agents on a randomized drone design"""
    print("="*70)
    print("MULTI-AGENT INTEGRATION TEST: Randomized Drone Design")
    print("="*70)
    
    # Generate random design
    design = generate_random_drone_design()
    print(f"\n📋 Design: {design['name']}")
    print(f"   Environment: {design['environment']}")
    print(f"   Geometry: {design['geometry'][0]['type']} (r={design['geometry'][0]['params']['radius']:.2f}m, l={design['geometry'][0]['params']['length']:.2f}m)")
    print(f"   Material: {design['geometry'][0]['params']['material']}")
    print(f"   Mission: {design['mission']['duration_hours']:.1f}h, {design['mission']['payload_kg']:.1f}kg payload")
    
    results = {}
    
    # 1. MaterialAgent - Get material properties
    print("\n" + "="*70)
    print("1️⃣  MaterialAgent: Analyzing material properties...")
    print("="*70)
    try:
        from agents.material_agent import MaterialAgent
        material_agent = MaterialAgent()
        material_result = material_agent.run(
            material_name=design['geometry'][0]['params']['material'],
            temperature=300.0
        )
        results['material'] = material_result
        print(f"✓ Material: {material_result.get('name', 'Unknown')}")
        props = material_result.get('properties', {})
        print(f"  Density: {props.get('density', 0):.0f} kg/m³")
        print(f"  Young's Modulus: {props.get('youngs_modulus', 0)/1e9:.0f} GPa")
    except Exception as e:
        print(f"✗ MaterialAgent failed: {e}")
        results['material'] = {"error": str(e)}
    
    # 2. ChemistryAgent - Check environmental compatibility
    print("\n" + "="*70)
    print("2️⃣  ChemistryAgent: Checking environmental compatibility...")
    print("="*70)
    try:
        from agents.chemistry_agent import ChemistryAgent
        chem_agent = ChemistryAgent()
        chem_result = chem_agent.run(
            materials=[design['geometry'][0]['params']['material']],
            environment_type=design['environment']
        )
        results['chemistry'] = chem_result
        print(f"✓ Compatibility: {chem_result.get('status', 'unknown')}")
        if chem_result.get('issues'):
            for issue in chem_result['issues'][:3]:
                print(f"  ⚠️  {issue}")
    except Exception as e:
        print(f"✗ ChemistryAgent failed: {e}")
        results['chemistry'] = {"error": str(e)}
    
    # 3. FluidAgent - Aerodynamic analysis
    print("\n" + "="*70)
    print("3️⃣  FluidAgent: Computing aerodynamic forces...")
    print("="*70)
    try:
        from agents.fluid_agent import FluidAgent
        fluid_agent = FluidAgent()
        fluid_result = fluid_agent.run(
            geometry_tree=design['geometry'],
            context={
                "velocity": design['mission']['max_velocity_ms'],
                "density": 1.225 if design['environment'] == 'AERIAL' else 1000.0
            }
        )
        results['fluid'] = fluid_result
        print(f"✓ Solver: {fluid_result.get('solver', 'unknown')}")
        print(f"  Drag Coefficient: {fluid_result.get('cd', 0):.3f}")
        print(f"  Drag Force: {fluid_result.get('drag_n', 0):.1f} N")
        if 'critique' in fluid_result:
            print(f"  Critic Confidence: {fluid_result['critique'].get('confidence', 0):.2f}")
    except Exception as e:
        print(f"✗ FluidAgent failed: {e}")
        results['fluid'] = {"error": str(e)}
    
    # 4. StructuralAgent - Stress analysis
    print("\n" + "="*70)
    print("4️⃣  StructuralAgent: Analyzing structural integrity...")
    print("="*70)
    try:
        from agents.structural_agent import StructuralAgent
        struct_agent = StructuralAgent()
        
        # Calculate load from drag + payload
        total_load = results.get('fluid', {}).get('drag_n', 100.0) + design['mission']['payload_kg'] * 9.81
        
        struct_result = struct_agent.run({
            "geometry_tree": design['geometry'],
            "load_n": total_load,
            "material": design['geometry'][0]['params']['material']
        })
        results['structural'] = struct_result
        print(f"✓ Status: {struct_result.get('status', 'unknown')}")
        print(f"  Max Stress: {struct_result.get('max_stress_mpa', 0):.1f} MPa")
        print(f"  Safety Factor: {struct_result.get('safety_factor', 0):.2f}")
    except Exception as e:
        print(f"✗ StructuralAgent failed: {e}")
        results['structural'] = {"error": str(e)}
    
    # 5. ThermalAgent - Heat analysis
    print("\n" + "="*70)
    print("5️⃣  ThermalAgent: Computing thermal profile...")
    print("="*70)
    try:
        from agents.thermal_agent import ThermalAgent
        thermal_agent = ThermalAgent()
        
        # Heat flux from aerodynamic heating
        heat_flux = (design['mission']['max_velocity_ms'] ** 2) * 0.01  # Simplified
        
        thermal_result = thermal_agent.run({
            "geometry_tree": design['geometry'],
            "ambient_temp": 25.0,
            "heat_flux": heat_flux
        })
        results['thermal'] = thermal_result
        print(f"✓ Status: {thermal_result.get('status', 'unknown')}")
        print(f"  Max Temperature: {thermal_result.get('max_temp_c', 0):.1f} °C")
    except Exception as e:
        print(f"✗ ThermalAgent failed: {e}")
        results['thermal'] = {"error": str(e)}
    
    # 6. DesignExplorationAgent - Optimize parameters
    print("\n" + "="*70)
    print("6️⃣  DesignExplorationAgent: Exploring design space...")
    print("="*70)
    try:
        from agents.design_exploration_agent import DesignExplorationAgent
        explore_agent = DesignExplorationAgent()
        explore_result = explore_agent.run({
            "parameters": {
                "radius": (0.5, 2.0, 0.1),
                "length": (2.0, 8.0, 0.5)
            },
            "objectives": ["minimize_mass", "maximize_strength"],
            "num_samples": 5
        })
        results['exploration'] = explore_result
        print(f"✓ Candidates: {len(explore_result.get('candidates', []))}")
        best = explore_result.get('best_candidate', {})
        print(f"  Best Score: {best.get('score', 0):.3f}")
        print(f"  Best Mass: {best.get('mass_kg', 0):.1f} kg")
    except Exception as e:
        print(f"✗ DesignExplorationAgent failed: {e}")
        results['exploration'] = {"error": str(e)}
    
    # 7. TemplateDesignAgent - Apply template
    print("\n" + "="*70)
    print("7️⃣  TemplateDesignAgent: Applying design template...")
    print("="*70)
    try:
        from agents.template_design_agent import TemplateDesignAgent
        template_agent = TemplateDesignAgent()
        template_result = template_agent.run({
            "template_id": "quadcopter_frame_x",
            "parameters": {"scale": 1.0, "rotation": 0.0}
        })
        results['template'] = template_result
        print(f"✓ Template: {template_result.get('template', {}).get('type', 'unknown')}")
        if 'quality_scores' in template_result:
            scores = template_result['quality_scores']
            print(f"  Manufacturability: {scores.get('manufacturability', 0):.2f}")
            print(f"  Performance: {scores.get('performance', 0):.2f}")
    except Exception as e:
        print(f"✗ TemplateDesignAgent failed: {e}")
        results['template'] = {"error": str(e)}
    
    # Summary
    print("\n" + "="*70)
    print("INTEGRATION TEST SUMMARY")
    print("="*70)
    
    total_agents = 7
    successful = sum(1 for r in results.values() if 'error' not in r)
    
    print(f"\n✓ Agents Tested: {total_agents}")
    print(f"✓ Successful: {successful}/{total_agents} ({successful/total_agents*100:.0f}%)")
    print(f"✗ Failed: {total_agents - successful}/{total_agents}")
    
    if successful == total_agents:
        print("\n🎉 ALL AGENTS WORKING TOGETHER SUCCESSFULLY!")
    else:
        print("\n⚠️  Some agents need fixes:")
        for agent, result in results.items():
            if 'error' in result:
                print(f"  - {agent}: {result['error']}")
    
    return results

if __name__ == "__main__":
    test_multi_agent_workflow()
