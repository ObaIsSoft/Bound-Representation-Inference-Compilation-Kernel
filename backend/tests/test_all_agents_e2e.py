"""
End-to-end agent test suite — covers every pipeline agent with a realistic payload.
Run from backend/: python3 -m pytest tests/test_all_agents_e2e.py -v
or directly:       python3 tests/test_all_agents_e2e.py
"""
import sys
import os
import asyncio
import traceback

_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_project_root = os.path.dirname(_backend_dir)
sys.path.insert(0, _backend_dir)
sys.path.insert(0, _project_root)

# Load .env so Supabase / API credentials are available before any import
try:
    from dotenv import load_dotenv
    for _env in [os.path.join(_backend_dir, ".env"), os.path.join(_project_root, ".env")]:
        if os.path.exists(_env):
            load_dotenv(_env, override=False)
            break
except ImportError:
    pass

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"
results = []


def record(name, status, detail=""):
    tag = f"[{status}]"
    print(f"  {tag:<8} {name}" + (f" — {detail}" if detail else ""))
    results.append((name, status, detail))


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ─────────────────────────────────────────────────────────────────────────────
# BATCH 1 — Thermal, Structural, Fluid, Material, Cost
# ─────────────────────────────────────────────────────────────────────────────

def test_thermal_agent():
    from agents.thermal_agent import ThermalAgent
    agent = ThermalAgent()
    result = run(agent.run({
        "power_watts": 150,
        "ambient_temp": 25,
        "material": "aluminum",
        "fluid": "Air",
        "geometry": {"length": 0.2, "width": 0.1, "height": 0.05},
        "flow_velocity_mps": 5.0,
        "analysis_type": "steady_state",
    }))
    assert "max_temperature_c" in result, f"Missing max_temperature_c: {result}"
    assert result.get("status") in ("success", "warning_exceeded"), f"Bad status: {result.get('status')}"
    record("ThermalAgent.run()", PASS, f"T_max={result.get('max_temperature_c', '?'):.1f}°C, solver={result.get('solver_used','?')}")


def test_thermal_agent_space():
    from agents.thermal_agent import ThermalAgent
    agent = ThermalAgent()
    result = run(agent.run({
        "power_watts": 80,
        "ambient_temp": 3,
        "environment_type": "SPACE",
        "emissivity": 0.85,
        "material": "titanium",
        "geometry": {"length": 0.3, "width": 0.3, "height": 0.02},
    }))
    assert "max_temperature_c" in result
    record("ThermalAgent.run() [space/radiation]", PASS, f"T_max={result.get('max_temperature_c', '?'):.1f}°C")


def test_structural_agent():
    from agents.structural_agent import StructuralAgent
    agent = StructuralAgent()
    result = run(agent.run({
        "design_parameters": {"length_m": 0.5, "width_m": 0.05, "height_m": 0.05},
        "material_properties": {"E": 200e9, "yield_strength": 250e6, "density": 7850},
        "loads": {"force": 5000, "direction": "vertical"},
    }))
    assert "max_stress_mpa" in result, f"Missing max_stress_mpa: {result}"
    assert "safety_factor" in result
    record("StructuralAgent.run()", PASS, f"σ_max={result.get('max_stress_mpa','?'):.1f} MPa, SF={result.get('safety_factor','?'):.2f}")


def test_fluid_agent():
    from agents.fluid_agent import FluidAgent
    agent = FluidAgent()
    geometry = [{"type": "box", "dimensions": {"length": 1.0, "width": 0.3, "height": 0.2}}]
    context = {"velocity": 30.0, "density": 1.225, "temperature": 288.15}
    result = agent.run(geometry, context)
    assert result is not None, "FluidAgent returned None"
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    record("FluidAgent.run()", PASS, f"keys={list(result.keys())[:4]}")


def test_material_agent():
    from agents.material_agent import MaterialAgent
    agent = MaterialAgent()
    result = run(agent.run("aluminum_6061", temperature=25.0))
    assert result is not None
    assert isinstance(result, dict)
    # Should have some material property
    has_prop = any(k in result for k in ("density", "thermal_conductivity", "yield_strength", "properties", "data"))
    assert has_prop, f"No material property keys found: {list(result.keys())}"
    record("MaterialAgent.run()", PASS, f"keys={list(result.keys())[:4]}")


def test_cost_agent():
    from agents.cost_agent import CostAgent
    agent = CostAgent()
    result = run(agent.quick_estimate({
        "mass_kg": 2.5,
        "material_name": "aluminum_6061",
        "process_type": "cnc_milling",
        "quantity": 10,
        "geometry": {"length": 0.2, "width": 0.1, "height": 0.05},
    }))
    assert isinstance(result, dict), f"Expected dict: {result}"
    has_cost = any(k in result for k in ("total_cost", "breakdown", "estimated_cost_usd", "cost_usd", "unit_cost", "estimated_cost", "total_estimate"))
    assert has_cost, f"No cost key found: {list(result.keys())}"
    record("CostAgent.quick_estimate()", PASS, f"keys={list(result.keys())[:4]}")


# ─────────────────────────────────────────────────────────────────────────────
# BATCH 2 — Electronics, DFM, Tolerance, Codegen, Physics
# ─────────────────────────────────────────────────────────────────────────────

def test_electronics_agent():
    from agents.electronics_agent import ElectronicsAgent
    agent = ElectronicsAgent()
    result = run(agent.run({
        "components": [
            {"type": "microcontroller", "name": "STM32F4", "voltage": 3.3, "current_ma": 100},
            {"type": "sensor", "name": "IMU", "voltage": 3.3, "current_ma": 5},
            {"type": "motor_driver", "name": "DRV8833", "voltage": 5.0, "current_ma": 1500},
        ],
        "power_supply": {"voltage": 12.0, "capacity_mah": 3000},
        "environment_type": "GROUND",
    }))
    assert isinstance(result, dict)
    record("ElectronicsAgent.run()", PASS, f"keys={list(result.keys())[:4]}")


def test_dfm_agent():
    from agents.dfm_agent import DfmAgent
    agent = DfmAgent()
    result = agent.run({
        "design_parameters": {"length": 100, "width": 50, "height": 30},
        "material": "aluminum_6061",
        "process_type": "cnc_milling",
    })
    assert isinstance(result, dict)
    has_key = any(k in result for k in ("overall_manufacturability_score", "score", "manufacturability_score", "issues", "manufacturable", "dfm_score"))
    assert has_key, f"No manufacturability key: {list(result.keys())}"
    record("DfmAgent.run()", PASS, f"keys={list(result.keys())[:4]}")


def test_tolerance_agent():
    from agents.tolerance_agent import ToleranceAgent
    agent = ToleranceAgent()
    result = agent.run({
        "design_parameters": {"length_mm": 100.0, "width_mm": 50.0},
        "tolerances": [
            {"name": "shaft_diameter", "nominal": 25.0, "plus": 0.0, "minus": -0.021, "unit": "mm"},
            {"name": "bore_diameter", "nominal": 25.0, "plus": 0.021, "minus": 0.0, "unit": "mm"},
        ],
    })
    assert isinstance(result, dict)
    has_key = any(k in result for k in ("rss_tolerance", "worst_case", "monte_carlo", "total_tolerance", "stack_result"))
    assert has_key, f"No tolerance key: {list(result.keys())}"
    record("ToleranceAgent.run()", PASS, f"keys={list(result.keys())[:4]}")


def test_codegen_agent_cpp():
    from agents.codegen_agent import CodegenAgent
    agent = CodegenAgent()
    result = agent.run({
        "language": "C++",
        "platform": "ESP32",
        "components": [
            {"type": "sensor", "name": "temperature_sensor", "interface": "I2C", "address": "0x48"},
            {"type": "actuator", "name": "servo", "interface": "PWM", "pin": 9},
        ],
        "project_name": "thermal_controller",
    })
    assert isinstance(result, dict)
    has_key = any(k in result for k in ("files", "main_file", "source_files", "project_files", "code"))
    assert has_key, f"No code key: {list(result.keys())}"
    record("CodegenAgent.run() [C++]", PASS, f"keys={list(result.keys())[:4]}")


def test_codegen_agent_micropython():
    from agents.codegen_agent import CodegenAgent
    agent = CodegenAgent()
    result = agent.run({
        "language": "MicroPython",
        "platform": "ESP32",
        "components": [
            {"type": "sensor", "name": "dht22", "interface": "GPIO", "pin": 4},
            {"type": "display", "name": "oled_128x64", "interface": "I2C"},
        ],
        "project_name": "env_monitor",
    })
    assert isinstance(result, dict)
    has_key = any(k in result for k in ("files", "main_file", "source_files", "project_files", "code"))
    assert has_key, f"No code key: {list(result.keys())}"
    record("CodegenAgent.run() [MicroPython]", PASS, f"keys={list(result.keys())[:4]}")


def test_physics_agent():
    from agents.physics_agent import PhysicsAgent
    agent = PhysicsAgent()
    result = agent.run({
        "environment": {"type": "GROUND", "gravity": 9.81, "temperature": 288.15, "pressure": 101325},
        "geometry_tree": [{"type": "box", "dimensions": {"length": 0.5, "width": 0.3, "height": 0.2}, "material": "aluminum"}],
        "design_parameters": {"mass_kg": 5.0, "velocity_mps": 0.0},
    })
    assert isinstance(result, dict)
    record("PhysicsAgent.run()", PASS, f"keys={list(result.keys())[:5]}")


# ─────────────────────────────────────────────────────────────────────────────
# BATCH 3 — Chemistry, Manufacturing, MassProps, GNC, Control
# ─────────────────────────────────────────────────────────────────────────────

def test_chemistry_agent():
    from agents.chemistry_agent import ChemistryAgent
    agent = ChemistryAgent()
    result = agent.run({
        "environment": {"type": "MARINE"},
        "material": "steel_1018",
        "design_parameters": {"material": "steel_1018"},
    })
    assert isinstance(result, dict)
    record("ChemistryAgent.run()", PASS, f"keys={list(result.keys())[:4]}")


def test_manufacturing_agent():
    from agents.manufacturing_agent import ManufacturingAgent
    agent = ManufacturingAgent()
    geometry_tree = [{"type": "box", "dimensions": {"length": 0.2, "width": 0.1, "height": 0.05}, "material": "aluminum_6061"}]
    result = run(agent.run(geometry_tree, "aluminum_6061", process_type="cnc_milling", region="global"))
    assert isinstance(result, dict)
    has_key = any(k in result for k in ("components", "bom_analysis", "manufacturing_plan", "process", "operations"))
    assert has_key, f"No manufacturing key: {list(result.keys())}"
    record("ManufacturingAgent.run()", PASS, f"keys={list(result.keys())[:4]}")


def test_mass_properties_agent():
    from agents.mass_properties_agent import MassPropertiesAgent
    agent = MassPropertiesAgent()
    # Pass volume_cm3 + material_density directly — bypasses Supabase, uses inline calc
    result = agent.run({
        "volume_cm3": 500.0,
        "material_density": 2.7,
        "bounding_box": [10.0, 8.0, 6.0],
    })
    assert isinstance(result, dict)
    assert "mass" in result
    mass_val = result["mass"]["magnitude"] if isinstance(result["mass"], dict) else result["mass"]
    assert abs(mass_val - 1.35) < 0.1, f"Expected ~1.35 kg, got {mass_val}"
    record("MassPropertiesAgent.run()", PASS, f"mass={mass_val:.3f} kg")


def test_gnc_agent():
    from agents.gnc_agent import GncAgent
    agent = GncAgent()
    # Hover-capable drone
    r1 = agent.run({"mass_kg": 1.0, "thrust_n": 20.0, "environment": "EARTH"})
    assert r1.get("flight_ready") or r1.get("tw_ratio", 0) > 1.5, f"Should be flight-ready: {r1}"
    # Over-weight
    r2 = agent.run({"mass_kg": 5.0, "thrust_n": 10.0, "environment": "EARTH"})
    assert not r2.get("flight_ready", True), f"Should NOT be flight-ready: {r2}"
    record("GncAgent.run()", PASS, f"T/W hover={r1.get('tw_ratio','?'):.2f}, heavy={r2.get('tw_ratio','?'):.2f}")


def test_control_agent():
    from agents.control_agent import ControlAgent
    agent = ControlAgent()
    result = agent.run({
        "system_type": "pid",
        "plant": {"mass": 1.0, "damping": 0.5, "stiffness": 10.0},
        "setpoint": 1.0,
        "Kp": 10.0,
        "Ki": 1.0,
        "Kd": 0.5,
    })
    assert isinstance(result, dict)
    record("ControlAgent.run()", PASS, f"keys={list(result.keys())[:4]}")


# ─────────────────────────────────────────────────────────────────────────────
# BATCH 4 — Optimization, Forensic, Safety, Compliance, Sustainability, Validator, Lattice
# ─────────────────────────────────────────────────────────────────────────────

def test_optimization_agent():
    from agents.optimization_agent import OptimizationAgent
    agent = OptimizationAgent()
    result = agent.run({
        "isa_state": {
            "constraints": {
                "max_mass_kg":    5.0,
                "max_stress_mpa": 200.0,
                "min_safety_factor": 1.5,
            }
        },
        "config": {"population_size": 5, "generations": 3, "enable_red_team": False},
        "objective": {"type": "VOLUME"},
    })
    assert isinstance(result, dict)
    record("OptimizationAgent.run()", PASS, f"keys={list(result.keys())[:4]}")


def test_forensic_agent():
    from agents.forensic_agent import ForensicAgent
    agent = ForensicAgent()
    result = agent.run({
        "failure_description": "Component fractured at weld joint under cyclic loading",
        "validation_flags": {"physics_safe": False, "geometry_physics_compatible": True},
        "max_stress_mpa": 320,
        "material": "steel_1018",
        "load_cycles": 100000,
    })
    assert isinstance(result, dict)
    record("ForensicAgent.run()", PASS, f"keys={list(result.keys())[:4]}")


def test_safety_agent():
    from agents.safety_agent import SafetyAgent
    agent = SafetyAgent()
    result = run(agent.run({
        "design_parameters": {"max_pressure_psi": 150, "operating_temp_c": 80},
        "material": "aluminum_6061",
        "environment_type": "GROUND",
        "components": [{"type": "pressure_vessel", "max_rating_psi": 200}],
    }))
    assert isinstance(result, dict)
    record("SafetyAgent.run()", PASS, f"keys={list(result.keys())[:4]}")


def test_compliance_agent():
    from agents.compliance_agent import ComplianceAgent
    agent = ComplianceAgent()
    result = agent.run({
        "design_parameters": {"voltage_v": 12.0, "current_a": 2.0},
        "environment_type": "GROUND",
        "material": "aluminum_6061",
        "standards": ["CE", "RoHS"],
    })
    assert isinstance(result, dict)
    record("ComplianceAgent.run()", PASS, f"keys={list(result.keys())[:4]}")


def test_sustainability_agent():
    from agents.sustainability_agent import SustainabilityAgent
    agent = SustainabilityAgent()
    result = run(agent.run({
        "materials": [{"material_id": "aluminum_6061", "mass_kg": 2.5}],
        "manufacturing_process": "cnc_milling",
        "energy_source": "grid_mix",
        "lifetime_years": 10,
        "end_of_life": "recycle",
        "use_phase_energy_kwh_per_year": 50,
    }))
    assert isinstance(result, dict)
    record("SustainabilityAgent.run()", PASS, f"keys={list(result.keys())[:4]}")


def test_validator_agent():
    from agents.validator_agent import ValidatorAgent
    agent = ValidatorAgent()
    result = agent.run({
        "geometry_tree": [{"type": "box", "dimensions": {"length": 0.2, "width": 0.1, "height": 0.05}}],
        "material": "aluminum_6061",
        "max_stress_mpa": 150,
        "safety_factor": 2.1,
        "max_temperature_c": 75,
        "design_parameters": {"budget_usd": 500},
        "estimated_cost_usd": 120,
    })
    assert isinstance(result, dict)
    record("ValidatorAgent.run()", PASS, f"keys={list(result.keys())[:4]}")


def test_lattice_synthesis_agent():
    from agents.lattice_synthesis_agent import LatticeSynthesisAgent
    agent = LatticeSynthesisAgent()
    result = run(agent.run({
        "operation": "synthesize",
        "formula": "Si",
        "crystal_system": "cubic",
    }))
    assert isinstance(result, dict)
    record("LatticeSynthesisAgent.run()", PASS, f"keys={list(result.keys())[:4]}")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

TESTS = [
    # Batch 1
    test_thermal_agent,
    test_thermal_agent_space,
    test_structural_agent,
    test_fluid_agent,
    test_material_agent,
    test_cost_agent,
    # Batch 2
    test_electronics_agent,
    test_dfm_agent,
    test_tolerance_agent,
    test_codegen_agent_cpp,
    test_codegen_agent_micropython,
    test_physics_agent,
    # Batch 3
    test_chemistry_agent,
    test_manufacturing_agent,
    test_mass_properties_agent,
    test_gnc_agent,
    test_control_agent,
    # Batch 4
    test_optimization_agent,
    test_forensic_agent,
    test_safety_agent,
    test_compliance_agent,
    test_sustainability_agent,
    test_validator_agent,
    test_lattice_synthesis_agent,
]


def main():
    print("\n" + "="*65)
    print("  BRICK OS — End-to-End Agent Test Suite")
    print("="*65)

    batches = [
        ("Batch 1 — Thermal / Structural / Fluid / Material / Cost", TESTS[0:6]),
        ("Batch 2 — Electronics / DFM / Tolerance / Codegen / Physics", TESTS[6:12]),
        ("Batch 3 — Chemistry / Manufacturing / MassProps / GNC / Control", TESTS[12:17]),
        ("Batch 4 — Optimization / Forensic / Safety / Compliance / Sustainability / Validator / Lattice", TESTS[17:]),
    ]

    for batch_name, batch_tests in batches:
        print(f"\n{batch_name}")
        print("-" * 65)
        for test_fn in batch_tests:
            try:
                test_fn()
            except Exception as e:
                record(test_fn.__name__.replace("test_", "").replace("_", " ").title(),
                       FAIL, str(e).split("\n")[0][:80])
                traceback.print_exc()

    passed = sum(1 for _, s, _ in results if s == PASS)
    failed = sum(1 for _, s, _ in results if s == FAIL)
    skipped = sum(1 for _, s, _ in results if s == SKIP)
    total = len(results)

    print("\n" + "="*65)
    print(f"  Results: {passed}/{total} passed  |  {failed} failed  |  {skipped} skipped")
    print("="*65)

    if failed:
        print("\nFailed tests:")
        for name, status, detail in results:
            if status == FAIL:
                print(f"  ✗ {name}: {detail}")
        sys.exit(1)
    else:
        print("\n  All agents operational.")
        sys.exit(0)


if __name__ == "__main__":
    main()
