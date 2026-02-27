#!/usr/bin/env python3
"""
Demo: CostAgent and ToleranceAgent Production Implementation

Demonstrates production-ready cost estimation and tolerance analysis.
"""

import asyncio
import sys
sys.path.insert(0, '/Users/obafemi/Documents/dev/brick')

from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timedelta

# ToleranceAgent (works without external services)
from backend.agents.tolerance_agent_production import (
    ProductionToleranceAgent,
    ToleranceSpec,
    DistributionType,
    quick_rss_analysis,
    analyze_feature_position
)

# CostAgent (requires mocked services for demo)
from backend.agents.cost_agent_production import ProductionCostAgent, ManufacturingProcess


def demo_tolerance_agent():
    """Demonstrate ToleranceAgent - fully functional without external services."""
    print("=" * 60)
    print("TOLERANCE AGENT DEMO")
    print("=" * 60)
    
    agent = ProductionToleranceAgent(default_mc_iterations=10000)
    
    # Example 1: Simple RSS stack
    print("\n1. RSS Tolerance Stack Analysis")
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
    print(f"  Cpk: {result.rss.cpk:.2f}")
    
    print(f"\nMonte Carlo (10,000 iterations):")
    print(f"  Mean: {result.monte_carlo.mean:.3f} mm")
    print(f"  Std Dev: {result.monte_carlo.std_dev:.4f} mm")
    print(f"  99% Range: [{result.monte_carlo.percentiles['1%']:.3f}, {result.monte_carlo.percentiles['99%']:.3f}] mm")
    print(f"  % Outside Spec: {result.monte_carlo.percent_outside_limits:.4f}%")
    
    print(f"\nWorst Case:")
    print(f"  Upper: {result.worst_case.upper_limit:.3f} mm")
    print(f"  Lower: {result.worst_case.lower_limit:.3f} mm")
    
    print(f"\nPasses Specification: {'✅ Yes' if result.passes_specification else '❌ No'}")
    
    # Example 2: GD&T True Position
    print("\n\n2. GD&T True Position Analysis (ASME Y14.5)")
    print("-" * 40)
    
    position = analyze_feature_position(
        x_deviation=0.08,
        y_deviation=0.05,
        position_tolerance=0.25,
        mmc_bonus=0.05
    )
    
    print(f"Deviation: X={position['x_deviation']:.3f}, Y={position['y_deviation']:.3f}")
    print(f"Actual Position: {position['actual_position_deviation']:.4f} mm")
    print(f"Tolerance Zone: {position['position_tolerance_radius']:.4f} mm radius")
    print(f"Bonus Tolerance: {position['bonus_tolerance']:.3f} mm")
    print(f"\nWithin Tolerance: {'✅ Yes' if position['within_tolerance'] else '❌ No'}")
    print(f"Utilization: {position['utilization_percent']:.1f}%")
    print(f"Remaining: {position['remaining_tolerance']:.4f} mm")


async def demo_cost_agent_mocked():
    """Demonstrate CostAgent with mocked services (real data unavailable in demo)."""
    print("\n\n" + "=" * 60)
    print("COST AGENT DEMO (with mocked services)")
    print("=" * 60)
    
    agent = ProductionCostAgent(use_ml=False)
    agent._initialized = True
    
    # Mock pricing service
    from backend.services.pricing_service import PricePoint
    mock_price = PricePoint(
        price=3.50,
        currency="USD",
        unit="kg",
        source="metals-api",
        timestamp=datetime.now(),
        expires_at=datetime.now() + timedelta(hours=24)
    )
    agent.pricing_service = AsyncMock()
    agent.pricing_service.get_material_price = AsyncMock(return_value=mock_price)
    
    # Mock supabase
    agent.supabase = AsyncMock()
    agent.supabase.get_material = AsyncMock(return_value={
        "density_kg_m3": 2700,
        "cost_per_kg_usd": 3.50
    })
    agent.supabase.get_manufacturing_rates = AsyncMock(return_value={
        "machine_hourly_rate_usd": 85.0,
        "setup_cost_usd": 150.0,
        "setup_time_hr": 2.0,
        "data_source": "supplier_quote"
    })
    
    # Example 1: Aluminum CNC part
    print("\n1. Aluminum 6061 CNC Milled Part")
    print("-" * 40)
    
    estimate = await agent.estimate_cost_abc(
        volume_mm3=150000,  # 150 cm³
        material_key="aluminum_6061",
        process=ManufacturingProcess.CNC_MILLING,
        quantity=50,
        n_features=8,
        n_holes=4,
        tightest_tolerance_mm=0.05,
        region="us"
    )
    
    print(f"Quantity: 50 parts")
    print(f"Volume: 150 cm³")
    print(f"Material: Aluminum 6061 ($3.50/kg from {estimate.data_sources.get('material_price', 'unknown')})")
    print(f"Process: CNC Milling (US rates from {estimate.data_sources.get('manufacturing_rate', 'unknown')})")
    print(f"\nTotal Cost: ${estimate.total_cost:,.2f}")
    print(f"Cost per part: ${estimate.total_cost/50:.2f}")
    print(f"Confidence: {estimate.confidence*100:.0f}%")
    print(f"\nCost Breakdown:")
    for component, cost in estimate.breakdown.to_dict().items():
        if cost > 0:
            pct = cost / estimate.total_cost * 100
            print(f"  {component:12s}: ${cost:8,.2f} ({pct:4.1f}%)")
    
    # Example 2: Quantity scaling
    print("\n\n2. Quantity Scaling Analysis")
    print("-" * 40)
    
    quantities = [1, 10, 50, 100, 500]
    print(f"{'Qty':>6s} {'Total':>12s} {'Per Part':>12s}")
    print("-" * 32)
    
    for qty in quantities:
        est = await agent.estimate_cost_abc(
            volume_mm3=150000,
            material_key="aluminum_6061",
            process=ManufacturingProcess.CNC_MILLING,
            quantity=qty,
            n_features=8,
            n_holes=4
        )
        print(f"{qty:>6d} ${est.total_cost:>10,.0f} ${est.total_cost/qty:>10.2f}")
    
    print("\n\n3. Fail-Fast Example (missing data)")
    print("-" * 40)
    
    # Show fail-fast behavior
    agent.pricing_service.get_material_price = AsyncMock(return_value=None)
    agent.supabase.get_material = AsyncMock(return_value=None)
    
    try:
        await agent.get_material_price("unobtainium_999")
    except ValueError as e:
        print(f"Expected error for unknown material:")
        print(f"  {str(e).split(chr(10))[0]}")


async def main():
    """Run all demos."""
    demo_tolerance_agent()
    await demo_cost_agent_mocked()
    
    print("\n\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nKey Features Demonstrated:")
    print("  ✅ RSS tolerance stack-up (ISO/ASME standard)")
    print("  ✅ Monte Carlo simulation (10,000 iterations)")
    print("  ✅ GD&T true position per ASME Y14.5")
    print("  ✅ Activity-Based Costing with external data")
    print("  ✅ Fail-fast error handling (no hardcoded fallbacks)")
    print("  ✅ Data provenance tracking")


if __name__ == "__main__":
    asyncio.run(main())
