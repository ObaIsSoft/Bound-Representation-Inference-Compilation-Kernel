import numpy as np
from agents.optimization_agent import OptimizationAgent, ObjectiveFunction

def verify_smart_snap():
    print("--- Verifying Smart Snap (OptimizationAgent) ---")
    
    agent = OptimizationAgent()
    objective = ObjectiveFunction(id="test", target="MINIMIZE", metric="DRAG") # DRAG mode aligns to flow
    
    # 1. Create a jagged, wobbly line
    # Intended line: y=0, z=0 (along x-axis)
    # Wobbly input: deviating in y/z
    points = []
    for x in range(10):
        y = np.random.uniform(-0.5, 0.5) if 1 < x < 8 else 0.0 # Pin ends, wobble middle
        z = np.random.uniform(-0.2, 0.2) if 1 < x < 8 else 0.0
        points.append([float(x), y, z])
        
    print(f"Original Points (Sample): {points[2:5]}")
    
    # 2. Run Optimization
    optimized = agent.optimize_sketch_curve(points, objective)
    
    # 3. Verify Smoothing
    # Calculate total jitter (variance from straight line)
    def calc_variance(pts):
        ys = [p[1] for p in pts]
        return np.std(ys)
        
    orig_var = calc_variance(points)
    opt_var = calc_variance(optimized)
    
    print(f"Original Y-Variance: {orig_var:.4f}")
    print(f"Optimized Y-Variance: {opt_var:.4f}")
    
    assert opt_var < orig_var, "Optimization did not reduce jitter!"
    
    print("[SUCCESS] Smart Snap significantly smoothed the curve.")

if __name__ == "__main__":
    verify_smart_snap()
