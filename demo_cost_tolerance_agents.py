#!/usr/bin/env python3
"""
Demo: CostAgent and ToleranceAgent Production Implementation

This demonstrates the production-ready cost estimation and tolerance
analysis capabilities of BRICK OS.
"""

import asyncio
import sys
sys.path.insert(0, '/Users/obafemi/Documents/dev/brick')

from backend.agents.cost_agent_production import (
    ProductionCostAgent,
    ManufacturingProcess,
    quick_cost_estimate
)
from backend.agents.tolerance_agent_production import (
    ProductionToleranceAgent,
    ToleranceSpec,
    DistributionType,
    quick_rss_analysis,
    analyze_feature_position
)


async def demo_cost_agent():
    """Demonstrate CostAgent capabilities."""
    print("=" * 60)
    print("COST AGENT DEMO")
    print("=" * 60)
    
    agent = ProductionCostAgent(use_ml=False)
    
    # Example 1: Aluminum CNC part
    print("\n1. Aluminum 6061 CNC Milled Part")
    print("-" * 40)
    
    geom = agent.calculate_geometry_complexity(
        surface_area_mm2=15000,
        volume_mm3=150000,
        bounding_box_volume_mm3=200000,
        n_features=8,
        n_holes=4,
        tightest_tolerance_mm=0.05
    )
    
    estimate = await agent.estimate_cost(
        geometry=geom,
        material_key="aluminum_6061",
        process=ManufacturingProcess.CNC_MILLING,
        quantity=50,
        region="us"
    )
    
    print(f"Quantity: 50 parts")
    print(f"Material: Aluminum 6061")
    print(f"Process: CNC Milling (US rates)")
    print(f"\nTotal Cost: ${estimate.total_cost:,.2f}")
    print(f"Cost per part: ${estimate.total_cost/50:.2f}")
    print(f"\nCost Breakdown:")
    for component, cost in estimate.breakdown.to_dict().items():
        if cost > 0:
            pct = cost / estimate.total_cost * 100
            print(f"  {component:12s}: ${cost:8,.2f} ({pct:4.1f}%)")
    
    print(f"\nConfidence Interval (95%): [${estimate.confidence_interval[0]:.2f}, ${estimate.confidence_interval[1]:.2f}]")
    print(f"Method: {estimate.method}")
    if estimate.warnings:
        print(f"\nWarnings:")
        for w in estimate.warnings:
            print(f"  ⚠️  {w}")
    
    # Example 2: Quick estimate
    print("\n\n2. Quick Cost Estimate")
    print("-" * 40)
    
    result = await quick_cost_estimate(
        volume_cm3=25.0,
        material="steel_4140",
        process="cnc_milling",
        quantity=100
    )
    
    print(f"Volume: 25 cm³, Material: Steel 4140, Qty: 100")
    print(f"Total Cost: ${result['total_cost_usd']:,.2f}")
    print(f"Cost per part: ${result['cost_per_part_usd']:.2f}")
    
    # Example 3: Quantity scaling
    print("\n\n3. Quantity Scaling Analysis")
    print("-" * 40)
    
    quantities = [1, 10, 50, 100, 500, 1000]
    print(f"{'Qty':>6s} {'Total':>12s} {'Per Part':>12s}")
    print("-" * 32)
    
    for qty in quantities:
        est = await agent.estimate_cost(
            geometry=geom,
            material_key="aluminum_6061",
            process=ManufacturingProcess.CNC_MILLING,
            quantity=qty
        )
        print(f"{qty:>6d} ${est.total_cost:>10,.0f} ${est.total_cost/qty:>10.2f}")


def demo_tolerance_agent():
    """Demonstrate ToleranceAgent capabilities."""
    print("\n\n" + "=" * 60)
    print("TOLERANCE AGENT DEMO")
    print("=" * 60)
    
    agent = ProductionToleranceAgent(default_mc_iterations=10000)
    
    # Example 1: Simple RSS stack
    print("\n1. Simple RSS Tolerance Stack")
    print("-" * 40)
    
    tolerances = [
        ToleranceSpec("hole1", 50.0, 0.1),
        ToleranceSpec("hole2", 30.0, 0.15),
        ToleranceSpec("thickness", 10.0, 0.05)
    ]
    
    result = agent.analyze_stack(
        tolerances,
        stack_description="Hole-to-hole distance",
        design_target=(90.0, 0.4)
    )
    
    print(f"Stack: 50 ± 0.1 + 30 ± 0.15 + 10 ± 0.05")
    print(f"\nRSS Analysis:")
    print(f"  Nominal: {result.rss.nominal_stack:.3f} mm")
    print(f"  RSS Tolerance: ±{result.rss.rss_tolerance:.3f} mm")
    print(f"  Upper Limit: {result.rss.upper_limit:.3f} mm")
    print(f"  Lower Limit: {result.rss.lower_limit:.3f} mm")
    print(f"  Cpk: {result.rss.cpk:.2f}" if result.rss.cpk else "  Cpk: N/A")
    
    print(f"\nMonte Carlo (10,000 iterations):")
    print(f"  Mean: {result.monte_carlo.mean:.3f} mm")
    print(f"  Std Dev: {result.monte_carlo.std_dev:.4f} mm")
    print(f"  99% Range: [{result.monte_carlo.percentiles['1%']:.3f}, {result.monte_carlo.percentiles['99%']:.3f}] mm")
    print(f"  % Outside Spec: {result.monte_carlo.percent_outside_limits:.4f}%")
    
    print(f"\nWorst Case:")
    print(f"  Upper: {result.worst_case.upper_limit:.3f} mm")
    print(f"  Lower: {result.worst_case.lower_limit:.3f} mm")
    print(f"  Range: ±{result.worst_case.tolerance_range:.3f} mm")
    
    print(f"\nPasses Specification: {'✅ Yes' if result.passes_specification else '❌ No'}")
    
    # Example 2: Sensitivity analysis
    print("\n\n2. Tolerance Sensitivity Analysis")
    print("-" * 40)
    
    sens = agent.sensitivity_analysis(
        tolerances,
        design_target=(90.0, 0.3),
        variation_percent=20.0
    )
    
    print(f"{'Tolerance':<15s} {'Current':>10s} {'Impact':>10s} {'Rank':>6s}")
    print("-" * 45)
    for s in sens["tolerance_sensitivities"]:
        print(f"{s['tolerance_name']:<15s} {s['baseline_yield']:>9.1f}% {s['yield_improvement']:+>9.1f}% {s['sensitivity_rank']:>6d}")
    
    print(f"\nMost Critical: {sens['most_critical']}")
    
    # Example 3: GD&T True Position
    print("\n\n3. GD&T True Position Analysis (ASME Y14.5)")
    print("-" * 40)
    
    position = analyze_feature_position(
        x_deviation=0.08,
        y_deviation=0.05,
        position_tolerance=0.25,  # Diameter
        mmc_bonus=0.05
    )
    
    print(f"Deviation: X={position['x_deviation']:.3f}, Y={position['y_deviation']:.3f}")
    print(f"Actual Position: {position['actual_position_deviation']:.4f} mm")
    print(f"Tolerance Zone: {position['position_tolerance_radius']:.4f} mm radius")
    print(f"Bonus Tolerance: {position['bonus_tolerance']:.3f} mm")
    print(f"\nWithin Tolerance: {'✅ Yes' if position['within_tolerance'] else '❌ No'}")
    print(f"Utilization: {position['utilization_percent']:.1f}%")
    print(f"Remaining: {position['remaining_tolerance']:.4f} mm")
    
    # Example 4: Quick RSS
    print("\n\n4. Quick RSS Analysis")
    print("-" * 40)
    
    quick = quick_rss_analysis(
        [("A", 20.0, 0.1), ("B", 15.0, 0.08), ("C", 10.0, 0.05)],
        target_mm=(45.0, 0.3)
    )
    
    print(f"Stack: 20 ± 0.1 + 15 ± 0.08 + 10 ± 0.05 = 45 ± 0.3 (target)")
    print(f"\nRSS: {quick['nominal_stack']:.3f} ± {quick['rss_tolerance']:.3f} mm")
    print(f"Worst Case: ±{(quick['worst_case_upper'] - quick['worst_case_lower'])/2:.3f} mm")
    print(f"Predicted Yield: {100 - quick['percent_outside_spec']:.2f}%")
    print(f"Passes: {'✅ Yes' if quick['passes_spec'] else '❌ No'}")
    
    print("\n\nTop Contributors:")
    for c in quick['contributions'][:3]:
        print(f"  {c['tolerance']}: {c['contribution_percent']:.1f}%")


async def main():
    """Run all demos."""
    await demo_cost_agent()
    demo_tolerance_agent()
    
    print("\n\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
