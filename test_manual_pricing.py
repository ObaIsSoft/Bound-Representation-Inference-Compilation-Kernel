#!/usr/bin/env python3
"""
Test manual pricing (no APIs needed)
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backend.services import pricing_service


async def test_manual_pricing():
    """Test setting and retrieving manual prices"""
    
    print("=" * 60)
    print("Testing Manual Pricing (No APIs Required)")
    print("=" * 60)
    print()
    
    # Set prices manually (from your supplier quotes)
    materials_to_set = [
        ("Aluminum 6061-T6", 3.50, "USD", "supplier_quote"),
        ("Steel A36", 0.80, "USD", "supplier_quote"),
        ("Stainless Steel 304", 4.00, "USD", "supplier_quote"),
    ]
    
    print("Setting manual prices...")
    for material, price, currency, source in materials_to_set:
        success = await pricing_service.set_material_price(
            material=material,
            price=price,
            currency=currency,
            source=source
        )
        if success:
            print(f"  ✅ Set {material}: ${price}/{currency} ({source})")
        else:
            print(f"  ❌ Failed to set {material}")
    
    print()
    print("Retrieving prices...")
    print("-" * 40)
    
    # Now retrieve them
    for material, _, _, _ in materials_to_set:
        price = await pricing_service.get_material_price(material, "USD")
        if price:
            print(f"  ✅ {material}: ${price.price:.2f} {price.currency}/{price.unit}")
            print(f"     Source: {price.source}")
        else:
            print(f"  ❌ {material}: Not found")
    
    print()
    print("=" * 60)
    print("Done! Manual pricing works without any API keys.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_manual_pricing())
