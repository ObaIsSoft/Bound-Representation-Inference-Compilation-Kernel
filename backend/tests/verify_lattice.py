
import sys
import os
import numpy as np
import logging

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from agents.lattice_synthesis_agent import LatticeSynthesisAgent
from vmk_kernel import SymbolicMachiningKernel, ToolProfile, VMKInstruction

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_lattice_logic():
    print("🚀 Starting Lattice Verification")
    
    agent = LatticeSynthesisAgent()
    
    # 1. Test Agent Logic (Gyroid)
    print("\n--- Testing Gyroid Math ---")
    
    # At p=(0,0,0): sin(0)cos(0) + ... = 0 + 0 + 0 = 0.
    # Dist = abs(0) - t = -t (Inside wall). Correct.
    p_zero = np.array([0.0, 0.0, 0.0]) 
    sdf_0 = agent.generate_unit_cell_sdf(p_zero, type="GYROID", thickness=0.5)
    print(f"SDF at (0,0,0): {sdf_0} (Expected -0.5)")
    
    # At pi/2 periodicity check? 
    # generate_unit_cell_sdf expects 2PI wrapped or raw? 
    # "p is expected to be in Unit Cell Space (0..2pi)"
    # So if we pass 2PI, it should equal 0.
    p_2pi = np.array([2*np.pi, 2*np.pi, 2*np.pi])
    sdf_2pi = agent.generate_unit_cell_sdf(p_2pi, type="GYROID", thickness=0.5)
    print(f"SDF at (2pi,2pi,2pi): {sdf_2pi} (Expected -0.5)")
    
    if abs(sdf_0 - -0.5) < 1e-5 and abs(sdf_2pi - -0.5) < 1e-5:
        print("✅ Gyroid Periodicity Correct")
    else:
        print("❌ Gyroid Math Check Failed")

    # 2. Test VMK Integration
    print("\n--- Testing VMK Lattice Instruction ---")
    kernel = SymbolicMachiningKernel(stock_dims=[200.0, 200.0, 200.0]) # Increased to avoid boundary clipping at p=50
    
    # Scale: 200um box. Period = 50um. 
    # Should see repetitions.
    
    instr = VMKInstruction(
        tool_id="lattice_gen",
        type="LATTICE",
        lattice_type="GYROID",
        period=50.0,
        thickness=5.0
    )
    kernel.execute_gcode(instr)
    
    # Query Points
    # 0,0,0 -> Inside Wall (SDF < 0)
    # 25, 25, 25 -> Half period?
    # 2pi * 25/50 = pi.
    # sin(pi) = 0. 
    # SDF should be similar (-thickness).
    
    q1 = np.array([0.0, 0.0, 0.0])
    res1 = kernel.get_sdf(q1)
    
    q2 = np.array([50.0, 0.0, 0.0]) # One full period
    res2 = kernel.get_sdf(q2)
    
    print(f"VMK Surface at 0: {res1}")
    print(f"VMK Surface at 50 (Period): {res2}")
    
    # Intersect Logic check:
    # d_stock at 0 is approx -50.
    # d_lattice at 0 is -5.0.
    # result = max(-50, -5) = -5. Correct.
    
    if abs(res1 - -5.0) < 0.1 and abs(res2 - res1) < 0.01:
        print("✅ VMK Lattice Integration Verified")
    else:
        print("❌ VMK Lattice Check Failed")

if __name__ == "__main__":
    test_lattice_logic()
