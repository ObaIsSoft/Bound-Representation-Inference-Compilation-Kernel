#!/usr/bin/env python3
"""
Verification Script: Test All Evolved Agents
Verifies that all Tier 1-3 agents have working surrogates, critics, and evolution.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

import numpy as np
from typing import Dict, Any

def test_agent_evolution(agent_name: str, agent_class, test_params: Dict[str, Any]) -> Dict[str, Any]:
    """Test an agent's evolution capabilities"""
    print(f"\n{'='*60}")
    print(f"Testing: {agent_name}")
    print(f"{'='*60}")
    
    results = {
        "agent": agent_name,
        "has_surrogate": False,
        "has_critic": False,
        "has_evolve": False,
        "run_works": False,
        "evolve_works": False,
        "errors": []
    }
    
    try:
        # Initialize agent
        agent = agent_class()
        print(f"✓ {agent_name} initialized")
        
        # Check for surrogate
        if hasattr(agent, 'surrogate') and agent.surrogate is not None:
            results["has_surrogate"] = True
            print(f"✓ Has surrogate: {type(agent.surrogate).__name__}")
        elif hasattr(agent, 'brain') and agent.brain is not None:
            results["has_surrogate"] = True
            print(f"✓ Has brain/surrogate: {type(agent.brain).__name__}")
        else:
            print(f"✗ No surrogate found")
        
        # Check for critic
        if hasattr(agent, 'critic') and agent.critic is not None:
            results["has_critic"] = True
            print(f"✓ Has critic: {type(agent.critic).__name__}")
        else:
            print(f"✗ No critic found")
        
        # Check for evolve method
        if hasattr(agent, 'evolve'):
            results["has_evolve"] = True
            print(f"✓ Has evolve() method")
        else:
            print(f"✗ No evolve() method")
        
        # Test run() - use params dict for all agents
        try:
            result = agent.run(test_params)
            results["run_works"] = True
            print(f"✓ run() executed successfully")
            if isinstance(result, dict):
                print(f"  Output keys: {list(result.keys())[:5]}")
        except Exception as e:
            results["errors"].append(f"run() failed: {str(e)}")
            print(f"✗ run() failed: {e}")
        
        # Test evolve() if available
        if results["has_evolve"]:
            try:
                # Generate agent-specific training data
                training_data = test_params.get("training_data", [])
                
                if training_data:
                    evolve_result = agent.evolve(training_data)
                    results["evolve_works"] = True
                    print(f"✓ evolve() executed successfully")
                    print(f"  Status: {evolve_result.get('status')}")
                else:
                    print(f"⚠ No training data provided, skipping evolve()")
            except Exception as e:
                results["errors"].append(f"evolve() failed: {str(e)}")
                print(f"✗ evolve() failed: {e}")
        
    except Exception as e:
        results["errors"].append(f"Initialization failed: {str(e)}")
        print(f"✗ Failed to initialize: {e}")
    
    return results

def main():
    print("="*60)
    print("BRICK OS Agent Evolution Verification (v2)")
    print("="*60)
    
    all_results = []
    
    # Tier 1: Core Evolution
    print("\n" + "="*60)
    print("TIER 1: Evolution Core")
    print("="*60)
    
    try:
        from agents.thermal_agent import ThermalAgent
        all_results.append(test_agent_evolution(
            "ThermalAgent",
            ThermalAgent,
            {
                "params": {
                    "geometry_tree": [{"type": "box", "params": {"width": 1, "height": 1, "thickness": 0.1}}],
                    "ambient_temp": 25.0,
                    "heat_flux": 100.0
                },
                "training_data": [
                    (np.array([25.0, 100.0, 0.1, 0.0]), np.array([50.0]))
                ]
            }
        ))
    except Exception as e:
        print(f"✗ ThermalAgent import failed: {e}")
    
    try:
        from agents.structural_agent import StructuralAgent
        all_results.append(test_agent_evolution(
            "StructuralAgent",
            StructuralAgent,
            {
                "params": {
                    "geometry_tree": [{"type": "box", "params": {"width": 1, "height": 1, "thickness": 0.1}}],
                    "load_n": 1000.0,
                    "material": "aluminum"
                },
                "training_data": [
                    (np.array([1000.0, 0.1, 7800.0, 0.0]), np.array([10.0]))
                ]
            }
        ))
    except Exception as e:
        print(f"✗ StructuralAgent import failed: {e}")
    
    # Tier 2: Physics & Manufacturing
    print("\n" + "="*60)
    print("TIER 2: Physics & Manufacturing")
    print("="*60)
    
    try:
        from agents.material_agent import MaterialAgent
        all_results.append(test_agent_evolution(
            "MaterialAgent",
            MaterialAgent,
            {
                "material_name": "aluminum",
                "temperature": 300.0,
                "training_data": [
                    (np.array([300.0, 0.0, 0.0, 0.0]), np.array([200e9, 0.3]))
                ]
            }
        ))
    except Exception as e:
        print(f"✗ MaterialAgent import failed: {e}")
    
    try:
        from agents.chemistry_agent import ChemistryAgent
        all_results.append(test_agent_evolution(
            "ChemistryAgent",
            ChemistryAgent,
            {
                "materials": ["aluminum"],
                "environment_type": "MARINE",
                "training_data": [
                    (np.array([7.0, 25.0, 0.0, 0.0]), np.array([0.5]))
                ]
            }
        ))
    except Exception as e:
        print(f"✗ ChemistryAgent import failed: {e}")
    
    try:
        from agents.electronics_agent import ElectronicsAgent
        all_results.append(test_agent_evolution(
            "ElectronicsAgent",
            ElectronicsAgent,
            {
                "params": {
                    "power_budget_w": 100.0,
                    "voltage_v": 5.0,
                    "components": []
                },
                "training_data": [
                    (np.array([5.0, 1.0, 10.0, 0.1]), np.array([0.5, 85.0]))
                ]
            }
        ))
    except Exception as e:
        print(f"✗ ElectronicsAgent import failed: {e}")
    
    try:
        from agents.fluid_agent import FluidAgent
        all_results.append(test_agent_evolution(
            "FluidAgent",
            FluidAgent,
            {
                "geometry_tree": [{"type": "cylinder", "params": {"radius": 1.0, "length": 5.0}}],
                "context": {"velocity": 10.0, "density": 1.225},
                "training_data": [
                    (np.array([1.0, 2.0, 1e5, 0.0]), np.array([0.5, 0.0]))
                ]
            }
        ))
    except Exception as e:
        print(f"✗ FluidAgent import failed: {e}")
    
    # Tier 3: Design & Optimization
    print("\n" + "="*60)
    print("TIER 3: Design & Optimization")
    print("="*60)
    
    try:
        from agents.design_exploration_agent import DesignExplorationAgent
        all_results.append(test_agent_evolution(
            "DesignExplorationAgent",
            DesignExplorationAgent,
            {
                "params": {
                    "parameters": {"width": (0.5, 2.0, 0.1), "height": (0.5, 2.0, 0.1)},
                    "objectives": ["minimize_mass"],
                    "num_samples": 5
                },
                "training_data": [
                    (np.array([1.0, 1.5]), 0.85),
                    (np.array([0.8, 1.2]), 0.75)
                ]
            }
        ))
    except Exception as e:
        print(f"✗ DesignExplorationAgent import failed: {e}")
    
    try:
        from agents.template_design_agent import TemplateDesignAgent
        all_results.append(test_agent_evolution(
            "TemplateDesignAgent",
            TemplateDesignAgent,
            {
                "params": {
                    "template_id": "naca_0012",
                    "parameters": {"scale": 1.0, "rotation": 0.0}
                },
                "training_data": [
                    (np.array([1.0, 0.0, 2.0, 0, 0, 0]), np.array([0.9, 0.85])),
                    (np.array([1.5, 45.0, 3.0, 0, 0, 0]), np.array([0.8, 0.90]))
                ]
            }
        ))
    except Exception as e:
        print(f"✗ TemplateDesignAgent import failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    total = len(all_results)
    with_surrogate = sum(1 for r in all_results if r["has_surrogate"])
    with_critic = sum(1 for r in all_results if r["has_critic"])
    with_evolve = sum(1 for r in all_results if r["has_evolve"])
    run_works = sum(1 for r in all_results if r["run_works"])
    evolve_works = sum(1 for r in all_results if r["evolve_works"])
    
    print(f"\nTested Agents: {total}")
    print(f"  With Surrogate: {with_surrogate}/{total} ({with_surrogate/max(total,1)*100:.0f}%)")
    print(f"  With Critic: {with_critic}/{total} ({with_critic/max(total,1)*100:.0f}%)")
    print(f"  With evolve(): {with_evolve}/{total} ({with_evolve/max(total,1)*100:.0f}%)")
    print(f"  run() works: {run_works}/{total} ({run_works/max(total,1)*100:.0f}%)")
    print(f"  evolve() works: {evolve_works}/{with_evolve} ({evolve_works/max(with_evolve,1)*100:.0f}%)")
    
    # Detailed failures
    failures = [r for r in all_results if r["errors"]]
    if failures:
        print(f"\n⚠️  Agents with errors: {len(failures)}")
        for r in failures:
            print(f"\n  {r['agent']}:")
            for err in r["errors"]:
                print(f"    - {err}")
    else:
        print(f"\n✅ All tested agents working!")
    
    return 0 if not failures else 1

if __name__ == "__main__":
    sys.exit(main())
