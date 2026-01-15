
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from agents.tolerance_agent import ToleranceAgent
from agents.mitigation_agent import MitigationAgent
from agents.manufacturing_agent import ManufacturingAgent
from vmk_kernel import SymbolicMachiningKernel, ToolProfile, VMKInstruction

def test_zero_tolerance_integration():
    print("--- VMK ZERO-TOLERANCE INTEGRATION TEST ---")
    
    # 1. Tolerance Agent (Clearance Check)
    print("\n[1] Testing ToleranceAgent.verify_fit_precision...")
    tol_agent = ToleranceAgent()
    
    # Scenario: 10mm Hole, 9.9mm Shaft (Clearance = 0.05mm radial)
    # We simulate the hole cut, then check points at r=4.95
    fit_spec = {
        "type": "clearance",
        "hole_radius": 5.0, # 10mm dia
        "hole_path": [[0,0,10], [0,0,0]], # Drill down Z
        "shaft_surface_points": [
            [4.95, 0, 5],  # Should be Clear (SDF > 0)
            [5.05, 0, 5],  # Should Interfere (SDF < 0) - FAILURE CASE SIMULATION
        ]
    }
    
    result = tol_agent.verify_fit_precision(fit_spec)
    print(f"Result verified: {result['verified']}")
    if not result['verified']:
        print(f"Deviations caught: {result['deviations']}")

    # 2. Mitigation Agent (Critical Dimension 1nm check)
    print("\n[2] Testing MitigationAgent.verify_critical_dimensions...")
    mit_agent = MitigationAgent()
    
    # Scenario: Verify Surface is exact
    plan = {
        "toolpaths": [
            {"tool_id": "t1", "path": [[-10,0,5], [10,0,5]]} # Cut channel
        ],
        "check_points": [
             # Center of channel (r=0.1 cut). Surface at y = +/- 0.1? No tool is 0.1 radius.
             # Path y=0. Surface at distance 0.1 from center
             {"coord": [0, 0.1, 5], "expected_surface": True},
             {"coord": [0, 0.100002, 5], "expected_surface": True} # 2nm error -> FAIL
        ]
    }
    
    # Need to register tool inside agent? No, agent registers default. 
    # But agent creates NEW kernel.
    # MitigationAgent implementation assumes standard tool or we need to pass it?
    # Our impl adds "verify_tool" r=0.1.
    
    result = mit_agent.verify_critical_dimensions(plan)
    print(f"Result verified: {result['verified']}")
    print(f"Max Deviation (nm): {result['max_deviation_nm']:.2f}")
    
    # 3. Manufacturing Agent (Gouging)
    print("\n[3] Testing ManufacturingAgent.verify_toolpath_accuracy...")
    mfg_agent = ManufacturingAgent()
    
    toolpaths = [
        # Safe move
        {"tool_id": "t1", "path": [[0,0,10], [0,0,0]]},
        # Gouge: Start deep inside material immediately
        {"tool_id": "t1", "path": [[5,5,2.5], [6,5,2.5]]} # 2.5 is deep in 5mm stock
    ]
    
    result = mfg_agent.verify_toolpath_accuracy(toolpaths)
    print(f"Result verified: {result['verified']}")
    if result['collisions']:
        print(f"Collisions: {len(result['collisions'])}")
        print(result['collisions'][0])

if __name__ == "__main__":
    test_zero_tolerance_integration()
