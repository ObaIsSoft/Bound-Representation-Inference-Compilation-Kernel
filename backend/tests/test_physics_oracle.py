
import sys
import os
import math
print("DEBUG: Starting Physics Oracle Verification Script")
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from agents.physics_oracle.physics_oracle import PhysicsOracle

def test_physics_oracle_integration():
    print("--- PHYSICS ORACLE: THEORY OF EVERYTHING TEST ---")
    
    oracle = PhysicsOracle()
    
    # [1] FLUID DYNAMICS (Navier-Stokes)
    print("\n[1] Testing Fluid Domain (LBM Solver)...")
    # Scenario: 10cm Cylinder in 34m/s flow (Mach 0.1)
    fluid_res = oracle.solve(
        query="Calculate drag on cylinder",
        domain="FLUID",
        params={"velocity": 0.1} # Normalized lattice units
    )
    print(f"Fluid Result: {fluid_res}")
    
    if fluid_res["status"] == "solved" and fluid_res["drag_coefficient"] > 0:
        print("Fluid Solver: SUCCESS (Computed Drag)")
    else:
        print("Fluid Solver: FAILURE")

    # [2] CIRCUITS (MNA Solver)
    print("\n[2] Testing Circuit Domain (MNA Solver)...")
    # Scenario: Voltage Divider (10V -> 100R -> Node1 -> 100R -> GND)
    # Expected V_node_1 = 5.0V
    netlist = {
        "components": [
            {"type": "V", "id": "V1", "nodes": [1, 0], "value": 10.0},
            {"type": "R", "id": "R1", "nodes": [1, 2], "value": 100.0},
            {"type": "R", "id": "R2", "nodes": [2, 0], "value": 100.0}
        ]
    }
    circuit_res = oracle.solve(
        query="Solve Voltage Divider",
        domain="CIRCUIT",
        params=netlist
    )
    print(f"Circuit Result: {circuit_res}")
    
    v_mid = circuit_res.get("voltages", {}).get("V_node_2", 0.0)
    if circuit_res["status"] == "solved" and abs(v_mid - 5.0) < 0.1:
        print(f"Circuit Solver: SUCCESS (Node 2 = {v_mid}V)")
    else:
        print("Circuit Solver: FAILURE")

    # [3] EXOTIC / QUANTUM (First Principles)
    print("\n[3] Testing Exotic Domain (SymPy Derivation)...")
    # Scenario: Quantum Harmonic Oscillator Energy (n=1)
    # E = (1 + 0.5) * hbar * w = 1.5 hbar w
    exotic_res = oracle.solve(
        query="Calculate Harmonic Oscillator Energy",
        domain="EXOTIC",
        params={"equation_type": "harmonic_oscillator_energy", "n": 1, "omega": 1e15}
    )
    print(f"Exotic Result: {exotic_res}")
    
    if exotic_res["status"] == "derived" and "numeric_result_Joules" in exotic_res:
        print("Exotic Solver: SUCCESS (Derived Physics)")
    else:
        print("Exotic Solver: FAILURE")

if __name__ == "__main__":
    test_physics_oracle_integration()
