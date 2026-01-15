
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from agents.electronics_agent import ElectronicsAgent
from agents.physics_agent import PhysicsAgent

def test_phase2_integration():
    print("--- VMK PHASE 2 INTEGRATION TEST ---")
    
    # 1. Electronics Agent (Nano-DRC)
    print("\n[1] Testing ElectronicsAgent.verify_drc...")
    elec_agent = ElectronicsAgent()
    
    # Trace Layout: 
    # T1: Line from (0,0) to (10,0). Width 0.1
    # T2: Line from (0, 0.14) to (10, 0.14). Width 0.1
    # Center-to-Center Y distance = 0.14
    # Radius = 0.05.
    # Gap = 0.14 - 0.05 - 0.05 = 0.04mm.
    # Min Clearance = 0.05mm.
    # Expect FAILURE (0.04 < 0.05).
    
    layout = {
        "traces": [
            {"id": "t1", "path": [[0,0,0], [10,0,0]], "width": 0.1},
            {"id": "t2", "path": [[0,0.14,0], [10,0.14,0]], "width": 0.1}
        ],
        "min_clearance_mm": 0.05
    }
    
    drc_result = elec_agent.verify_drc(layout)
    print(f"DRC Verified: {drc_result['verified']}")
    if not drc_result['verified']:
        print(f"Violations: {drc_result['violations'][0]}")
    
    # 2. Physics Agent (SDF Collision)
    print("\n[2] Testing PhysicsAgent.check_collision_sdf...")
    phys_agent = PhysicsAgent()
    
    # Environment: 
    # Stock 100x100x100.
    # Tunnel (Cut) radius 10 along X axis.
    # Inside tunnel (Air) is safe. In wall is collision.
    
    env_map = {
        "stock_dims": [100, 100, 100],
        "obstacles": [
            {"tool_id": "tunnel_bore", "path": [[-50,0,0], [50,0,0]]} # Radius will be used from ToolProfile or assumed? 
            # The agent executes instructions. We need to pass tool def? 
            # PhysicsAgent logic: "for op in obstacles... kernel.execute(op)".
            # But the OP needs a tool_id. And the kernel needs the tool registered.
            # PhysicsAgent implementation didn't auto-register tools! It assumes they are known.
            # Wait, SymbolicMachiningKernel has no tools by default.
            # I need to FIX PhysicsAgent to register default tool if missing, similar to MitigationAgent.
        ]
    }
    
    # Wait, PhysicsAgent implementation:
    # Kernel init. Loop ops. Execute. 
    # If op has tool_id 'tunnel_bore', and 'tunnel_bore' is not in kernel.tools => Crash.
    
    # Let's try to pass the tool definition in the map? Agent doesn't look for it yet.
    # I'll just run it to confirm the crash, then fix it.
    
    # Update: Fix applied in Agent. Now safe to call.
    position = [0, 0, 0] # Deep inside the tunnel (Tunnel is radius 10 at 0,0,0)
    # Wait, tunnel path [-50,0,0] to [50,0,0]. Radius 10.
    # Point 0,0,0 is on the centerline.
    # SDF should be negative?
    # Logic: Stock = Box. Tunnel = Subtraction.
    # max(Box, -Tunnel).
    # Box SDF at 0,0,0 is negative (inside).
    # Tunnel SDF at 0,0,0 is -10 (inside capsule).
    # -Tunnel is +10.
    # max(neg, +10) = +10.
    # Result: +10.
    # PhysicsAgent Logic: Collision if dist < radius?
    # If dist > 0 (Air), we are safe?
    # Wait, PhysicsAgent says:
    # "If dist < vehicle_radius, collision."
    # If dist is +10 (Air), and vehicle radius is 2.
    # 10 < 2 is False. No collision. Correct.
    
    # Let's test collision Case: Inside Wall.
    # Point [0, 15, 0]. Tunnel radius 10.
    # Tunnel SDF = 5 (Outside tunnel). -SDF = -5.
    # Box SDF = neg.
    # max(neg, -5) = -5.
    # Dist = -5.
    # -5 < 2 (radius). True. Collision.
    
    result = phys_agent.check_collision_sdf([0, 15, 0], 2.0, env_map)
    print(f"Collision Check (Wall): {result['collision']}")
    print(f"Distance: {result['distance_to_obstacle']}")
    
    result_safe = phys_agent.check_collision_sdf([0, 0, 0], 2.0, env_map)
    print(f"Collision Check (Air): {result_safe['collision']}")
    print(f"Distance: {result_safe['distance_to_obstacle']}")


if __name__ == "__main__":
    test_phase2_integration()
