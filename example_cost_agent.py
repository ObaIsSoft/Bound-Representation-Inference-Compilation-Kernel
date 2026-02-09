#!/usr/bin/env python3
"""
Example: CostAgent handles missing prices gracefully

This shows how the migrated CostAgent works without API keys.
When prices are unavailable, it returns helpful error messages.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backend.agents.cost_agent import CostAgent


async def test_cost_agent():
    """Test CostAgent with missing prices"""
    
    print("=" * 60)
    print("Testing Migrated CostAgent")
    print("=" * 60)
    print()
    
    agent = CostAgent()
    
    # Test 1: Material with no price configured
    print("Test 1: Requesting price for Aluminum (no API configured)")
    print("-" * 60)
    
    result = await agent.quick_estimate({
        "mass_kg": 5.0,
        "material_name": "Aluminum 6061-T6"
    }, currency="USD")
    
    print(f"Result: {result}")
    print()
    
    # Test 2: Show what user should do
    print("Test 2: Expected behavior when prices unavailable")
    print("-" * 60)
    print("""
When prices are unavailable, the CostAgent returns:

{
    "error": "No price available for Aluminum 6061-T6",
    "solution": "Configure LME_API_KEY or set price manually..."
}

To fix this, you have 3 options:

1. FREE - Sign up for Metals-API (200 calls/month):
   export METALS_API_KEY=your_key
   
2. FREE - Use Yahoo Finance (no key, install yfinance):
   pip install yfinance
   
3. FREE - Set prices manually in your code:
   await pricing_service.set_material_price(
       material="Aluminum 6061-T6",
       price=3.50,
       currency="USD",
       source="supplier_quote"
   )
""")
    
    print("=" * 60)
    print("Summary: CostAgent now fails gracefully instead of using guesses")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_cost_agent())
