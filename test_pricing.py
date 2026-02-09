#!/usr/bin/env python3
"""
Quick test of pricing service with yfinance (completely free, no API key)
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.services.pricing_service import pricing_service


async def test_yahoo_finance_pricing():
    """Test Yahoo Finance pricing (free, no API key)"""
    
    print("=" * 60)
    print("Testing Pricing Service with Yahoo Finance (FREE)")
    print("=" * 60)
    print()
    
    # Initialize the service
    await pricing_service.initialize()
    
    # Test materials that Yahoo Finance supports
    test_materials = [
        "Aluminum 6061-T6",  # Maps to ALI=F futures
        "Copper",             # Maps to HG=F futures
        "Gold",               # Maps to GC=F futures
        "Silver",             # Maps to SI=F futures
    ]
    
    for material in test_materials:
        print(f"\nFetching price for: {material}")
        print("-" * 40)
        
        try:
            price = await pricing_service.get_material_price(material, "USD")
            
            if price:
                print(f"  ✅ Price: ${price.price:.2f} {price.currency}/{price.unit}")
                print(f"  📊 Source: {price.source}")
                print(f"  🕐 Timestamp: {price.timestamp}")
            else:
                print(f"  ❌ No price available (returned None)")
                print(f"  💡 Tip: Check if material name matches Yahoo Finance ticker mapping")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print()
    print("=" * 60)
    print("Test Complete!")
    print("=" * 60)
    print()
    print("Note: Yahoo Finance uses futures/ETF prices as proxies.")
    print("Actual spot prices may differ slightly.")


if __name__ == "__main__":
    asyncio.run(test_yahoo_finance_pricing())
