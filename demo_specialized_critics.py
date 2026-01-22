"""
Comprehensive Demonstration of Specialized Critics for BRICK OS

This demo shows all critics working together:
- ChemistryCritic
- ElectronicsCritic  
- MaterialCritic
- ComponentCritic
- OracleCritic

Simulates a complete design cycle with agent monitoring.
"""

import sys
sys.path.append('backend')

import numpy as np
from agents.critics.ChemistryCritic import ChemistryCritic
from agents.critics.ElectronicsCritic import ElectronicsCritic
from agents.critics.MaterialCritic import MaterialCritic
from agents.critics.ComponentCritic import ComponentCritic
from agents.critics.OracleCritic import OracleCritic

def print_section(title):
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)

def print_report(critic_name, report):
    print(f"\n📊 {critic_name} Report:")
    print(f"  Observations: {report.get('timestamp', 0)}")
    print(f"  Confidence: {report.get('confidence', 0):.0%}")
    
    # Key metrics (varies by critic)
    if "safety_accuracy" in report:
        print(f"  Safety Accuracy: {report['safety_accuracy']:.0%}")
    if "avg_power_margin_w" in report:
        print(f"  Avg Power Margin: {report['avg_power_margin_w']:.1f}W")
    if "db_coverage" in report:
        print(f"  Database Coverage: {report['db_coverage']:.0%}")
    if "catalog_coverage" in report:
        print(f"  Catalog Coverage: {report['catalog_coverage']:.0%}")
    if "conservation_violations" in report:
        print(f"  Conservation Violations: {report['conservation_violations']}")
    
    # Failure modes
    if report.get("failure_modes"):
        print(f"\n  ⚠️  Failure Modes:")
        for mode in report["failure_modes"][:3]:  # Show first 3
            print(f"    • {mode}")
    
    # Recommendations
    if report.get("recommendations"):
        print(f"\n  💡 Recommendations:")
        for rec in report["recommendations"][:3]:  # Show first 3
            print(f"    • {rec}")

# ============================================================================
# PHASE 1: INITIALIZE CRITICS
# ============================================================================

print_section("PHASE 1: INITIALIZING SPECIALIZED CRITICS")

chemistry_critic = ChemistryCritic(window_size=50)
electronics_critic = ElectronicsCritic(window_size=50)
material_critic = MaterialCritic(window_size=50)
component_critic = ComponentCritic(window_size=50)
oracle_critic = OracleCritic(window_size=50)

print("\n✓ Initialized 5 specialized critics")
print("  • ChemistryCritic")
print("  • ElectronicsCritic")
print("  • MaterialCritic")
print("  • ComponentCritic")
print("  • OracleCritic (monitors all oracle systems)")

# ============================================================================
# PHASE 2: SIMULATE DESIGN CYCLE WITH AGENT OBSERVATIONS
# ============================================================================

print_section("PHASE 2: SIMULATING 30 DESIGN ITERATIONS")

np.random.seed(42)

for iteration in range(30):
    # Simulate design parameters
    temperature = np.random.uniform(20, 180)  # Operating temperature
    voltage = np.random.uniform(3.3, 48)
    is_marine = np.random.random() < 0.3  # 30% marine environment
    
    # ========================================================================
    # 1. CHEMISTRY AGENT
    # ========================================================================
    materials = ["Aluminum 6061", "Steel"] if np.random.random() < 0.7 else ["Aluminum 6061", "Magnesium"]
    env_type = "MARINE" if is_marine else "STANDARD"
    
    # Simulate chemistry agent output
    has_mg_in_marine = "Magnesium" in materials and is_marine
    chemistry_safe = not has_mg_in_marine
    issues = ["HAZARD: Magnesium dissolves rapidly in salt water"] if has_mg_in_marine else []
    
    chemistry_output = {
        "chemical_safe": chemistry_safe,
        "issues": issues,
        "report": [f"Checked {len(materials)} materials for {env_type}"]
    }
    
    chemistry_critic.observe(
        input_state={"materials": materials, "environment_type": env_type},
        chemistry_output=chemistry_output,
        safety_outcome=chemistry_safe  # In real system, this comes from field data
    )
    
    # ========================================================================
    # 2. ELECTRONICS AGENT
    # ========================================================================
    num_components = np.random.randint(3, 8)
    supply_w = np.random.uniform(50, 500)
    demand_w = supply_w * np.random.uniform(0.6, 1.2)  # Sometimes deficit
    
    power_deficit = demand_w > supply_w
    margin_w = supply_w - demand_w
    
    electronics_output = {
        "status": "critical" if power_deficit else "success",
        "scale": "MESO",
        "power_analysis": {
            "supply_w": supply_w,
            "demand_w": demand_w,
            "margin_w": margin_w,
            "status": "critical" if power_deficit else "success"
        },
        "validation_issues": ["POWER_DEFICIT"] if power_deficit else []
    }
    
    electronics_critic.observe(
        input_state={"components": [{"category": "battery"}] * num_components},
        electronics_output=electronics_output
    )
    
    # ========================================================================
    # 3. MATERIAL AGENT
    # ========================================================================
    material_name = np.random.choice(materials)
    
    # Simulate material lookup (50% found in DB, 50% fallback)
    found_in_db = np.random.random() < 0.5
    is_melted = temperature > 582  # Aluminum melting point
    
    strength_factor = max(0.3, 1.0 - (temperature - 150) * 0.003) if temperature > 150 else 1.0
    
    material_output = {
        "name": material_name if found_in_db else f"Generic Aluminum (Fallback for {material_name})",
        "properties": {
            "density": 2700,
            "yield_strength": 276e6 * strength_factor,
            "melting_point": 582,
            "is_melted": is_melted,
            "strength_factor": strength_factor
        }
    }
    
    material_critic.observe(
        input_state={"material_name": material_name, "temperature": temperature},
        material_output=material_output
    )
    
    # ========================================================================
    # 4. COMPONENT AGENT
    # ========================================================================
    requirements = {"min_power_w": demand_w * 0.5, "max_cost": 100}
    
    # Simulate component selection (70% success)
    selection_success = np.random.random() < 0.7
    num_selected = np.random.randint(1, 4) if selection_success else 0
    
    selected_components = []
    for i in range(num_selected):
        selected_components.append({
            "id": f"comp_{iteration}_{i}",
            "name": f"Component {i}",
            "category": "power_supply",
            "mass_g": np.random.uniform(10, 200),
            "cost_usd": np.random.uniform(10, 150),
            "specs": {"power_w": demand_w * np.random.uniform(0.6, 2.5)}  # Sometimes over-spec
        })
    
    component_output = {
        "selection": selected_components,
        "count": num_selected,
        "logs": [f"Selected {num_selected} components"]
    }
    
    component_critic.observe(
        requirements=requirements,
        selection_output=component_output,
        user_accepted=np.random.random() < 0.8  # 80% user acceptance
    )
    
    # ========================================================================
    # 5. ORACLE CALLS (Most critical!)
    # ========================================================================
    
    # PhysicsOracle: Circuit analysis
    if voltage > 0 and demand_w > 0:
        current = demand_w / voltage
        resistance = voltage / current
        
        # Simulate oracle output (with occasional errors)
        has_oracle_error = np.random.random() < 0.05  # 5% error rate
        
        if has_oracle_error:
            # Introduce Kirchhoff violation
            oracle_output = {
                "status": "success",
                "current_in": current,
                "current_out": current * 1.15,  # Violation!
                "resistance": resistance
            }
        else:
            oracle_output = {
                "status": "success",
                "current_in": current,
                "current_out": current,
                "resistance": resistance
            }
        
        oracle_critic.observe(
            oracle_type="physics",
            domain="CIRCUIT",
            input_params={"voltage": voltage, "power": demand_w},
            oracle_output=oracle_output
        )
    
    # ChemistryOracle: Corrosion rate
    if not is_melted:
        corrosion_rate = np.random.uniform(0.01, 0.1)  # mm/year
        
        oracle_output = {
            "status": "success",
            "corrosion_rate_mm_year": corrosion_rate,
            "lifetime_years": 5.0 / corrosion_rate  # 5mm thickness
        }
        
        oracle_critic.observe(
            oracle_type="chemistry",
            domain="CORROSION",
            input_params={"material": material_name, "environment": env_type},
            oracle_output=oracle_output
        )
    
    # MaterialsOracle: Thermal expansion
    thermal_expansion = 23e-6 * (temperature - 20)  # Aluminum CTE
    
    oracle_output = {
        "status": "success",
        "thermal_strain": thermal_expansion,
        "delta_length_mm": thermal_expansion * 100  # 100mm part
    }
    
    oracle_critic.observe(
        oracle_type="materials",
        domain="THERMAL_EXPANSION",
        input_params={"material": material_name, "temperature": temperature},
        oracle_output=oracle_output
    )

print(f"\n✓ Completed {iteration + 1} design iterations")
print("  All critics observed agent behavior in real-time")

# ============================================================================
# PHASE 3: CRITIC ANALYSIS
# ============================================================================

print_section("PHASE 3: CRITIC REPORTS")

# Generate reports
chem_report = chemistry_critic.analyze()
elec_report = electronics_critic.analyze()
mat_report = material_critic.analyze()
comp_report = component_critic.analyze()
oracle_report = oracle_critic.analyze()

print_report("ChemistryCritic", chem_report)
print_report("ElectronicsCritic", elec_report)
print_report("MaterialCritic", mat_report)
print_report("ComponentCritic", comp_report)
print_report("OracleCritic", oracle_report)

# ============================================================================
# PHASE 4: EVOLUTION DECISIONS
# ============================================================================

print_section("PHASE 4: EVOLUTION DECISIONS")

critics = {
    "ChemistryCritic": chemistry_critic,
    "ElectronicsCritic": electronics_critic,
    "MaterialCritic": material_critic,
    "ComponentCritic": component_critic,
    "OracleCritic": oracle_critic
}

evolution_queue = []

for critic_name, critic in critics.items():
    should_evolve, reason, strategy = critic.should_evolve()
    
    if should_evolve:
        evolution_queue.append({
            "critic": critic_name,
            "reason": reason,
            "strategy": strategy
        })
        print(f"\n⚠️  {critic_name}")
        print(f"   Should Evolve: YES")
        print(f"   Reason: {reason}")
        print(f"   Strategy: {strategy}")
    else:
        print(f"\n✓ {critic_name}")
        print(f"   Should Evolve: NO")
        print(f"   Reason: {reason}")

# ============================================================================
# PHASE 5: EXPORT REPORTS
# ============================================================================

print_section("PHASE 5: EXPORTING DETAILED REPORTS")

import os
os.makedirs("/tmp/critic_reports", exist_ok=True)

chemistry_critic.export_report("/tmp/critic_reports/chemistry_report.json")
electronics_critic.export_report("/tmp/critic_reports/electronics_report.json")
material_critic.export_report("/tmp/critic_reports/material_report.json")
component_critic.export_report("/tmp/critic_reports/component_report.json")
oracle_critic.export_report("/tmp/critic_reports/oracle_report.json")

print("\n✓ Exported 5 detailed reports:")
print("  • /tmp/critic_reports/chemistry_report.json")
print("  • /tmp/critic_reports/electronics_report.json")
print("  • /tmp/critic_reports/material_report.json")
print("  • /tmp/critic_reports/component_report.json")
print("  • /tmp/critic_reports/oracle_report.json")

# ============================================================================
# PHASE 6: SUMMARY
# ============================================================================

print_section("SUMMARY: MULTI-AGENT SELF-EVOLUTION SYSTEM")

print("\n🎯 Key Achievements:")
print(f"  • Monitored {iteration + 1} complete design cycles")
print(f"  • Tracked 5 agents + 3 oracles simultaneously")
print(f"  • Detected {len(evolution_queue)} agents requiring evolution")
print(f"  • Generated {5} detailed analysis reports")

print("\n📊 Critical Insights:")
if oracle_critic.conservation_violations:
    print(f"  ⚠️  {len(oracle_critic.conservation_violations)} conservation law violations (CRITICAL)")
if material_critic.melting_failures > 0:
    print(f"  🔥 {material_critic.melting_failures} designs exceeded melting point")
if chemistry_critic.false_negatives > 0:
    print(f"  🚨 {chemistry_critic.false_negatives} chemistry false negatives (safety issue)")

print("\n🔄 Evolution Queue:")
if evolution_queue:
    for item in evolution_queue:
        print(f"  • {item['critic']}: {item['strategy']}")
else:
    print("  ✅ No evolution needed - all agents performing nominally")

print("\n💡 System Highlights:")
print("  1. CHEMISTRY CRITIC monitors corrosion, compatibility, safety predictions")
print("  2. ELECTRONICS CRITIC tracks power balance, shorts, DRC violations")
print("  3. MATERIAL CRITIC watches degradation, mass accuracy, DB coverage")
print("  4. COMPONENT CRITIC observes selection quality, installations, user preferences")
print("  5. ORACLE CRITIC validates ALL fundamental calculations (most critical!)")

print("\n🚀 Next Steps:")
print("  → Review detailed JSON reports in /tmp/critic_reports/")
print("  → Address evolution queue items")
print("  → Integrate critics into orchestrator for real-time monitoring")

print("\n" + "=" * 80)
