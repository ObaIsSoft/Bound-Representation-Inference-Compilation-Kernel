#!/usr/bin/env python3
"""
Test the cost estimation API endpoints
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))

async def test_pricing_check():
    """Test the pricing check endpoint"""
    print("=" * 60)
    print("Testing /api/pricing/check")
    print("=" * 60)
    
    try:
        from backend.main import check_pricing_status
        result = await check_pricing_status()
        print(f"\nResult: {result}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

async def test_cost_estimate():
    """Test the cost estimate endpoint"""
    print("\n" + "=" * 60)
    print("Testing /api/cost/estimate")
    print("=" * 60)
    
    try:
        from backend.main import estimate_cost, CostEstimateRequest
        
        req = CostEstimateRequest(
            mass_kg=5.0,
            material_name="Aluminum 6061-T6",
            complexity="moderate",
            currency="USD"
        )
        
        result = await estimate_cost(req)
        print(f"\nResult: {result}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

async def main():
    await test_pricing_check()
    await test_cost_estimate()

if __name__ == "__main__":
    asyncio.run(main())
