"""
Pricing Service - Real-time Pricing from External APIs

Fetches live pricing for materials and components.
Implements caching to respect rate limits.
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

# Try to import httpx for API calls
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from .supabase_service import supabase

logger = logging.getLogger(__name__)


@dataclass
class PricePoint:
    """Price data with metadata"""
    price: float
    currency: str
    unit: str  # "kg", "each", "m", etc.
    source: str  # "lme", "digikey", "cache", etc.
    timestamp: datetime
    expires_at: datetime


class PricingService:
    """
    Real-time pricing service with caching.
    
    Supported sources:
    - LME (London Metal Exchange) - metals
    - Fastmarkets - industrial materials
    - DigiKey - electronic components
    - Mouser - electronic components
    """
    
    def __init__(self):
        self.http_client: Optional[Any] = None
        self._initialized = False
        
        # API keys
        self.lme_api_key = os.getenv("LME_API_KEY")
        self.digikey_client_id = os.getenv("DIGIKEY_CLIENT_ID")
        self.digikey_secret = os.getenv("DIGIKEY_SECRET")
        self.mouser_api_key = os.getenv("MOUSER_API_KEY")
        self.climatiq_api_key = os.getenv("CLIMATIQ_API_KEY")
        
    async def initialize(self):
        """Initialize HTTP client"""
        if self._initialized:
            return
            
        if HAS_HTTPX:
            self.http_client = httpx.AsyncClient(timeout=30.0)
        
        self._initialized = True
    
    async def get_material_price(
        self,
        material: str,
        currency: str = "USD",
        use_cache: bool = True
    ) -> Optional[PricePoint]:
        """
        Get material price from cache or external API.
        
        Args:
            material: Material name (e.g., "Aluminum 6061")
            currency: Currency code
            use_cache: Use cached price if available
            
        Returns:
            PricePoint or None if unavailable
        """
        await self.initialize()
        
        # 1. Try Supabase cache first
        if use_cache:
            try:
                material_data = await supabase.get_material(material)
                price_column = f"cost_per_kg_{currency.lower()}"
                price = material_data.get(price_column)
                
                if price and material_data.get("updated_at"):
                    updated = datetime.fromisoformat(material_data["updated_at"].replace('Z', '+00:00'))
                    age_hours = (datetime.now() - updated).total_seconds() / 3600
                    
                    # If price is fresh (< 24 hours), use it
                    if age_hours < 24:
                        return PricePoint(
                            price=float(price),
                            currency=currency,
                            unit="kg",
                            source=material_data.get("data_source", "database"),
                            timestamp=updated,
                            expires_at=updated + timedelta(hours=24)
                        )
            except ValueError:
                pass  # Material not in database
        
        # 2. Try external APIs
        if "aluminum" in material.lower() or "copper" in material.lower():
            price = await self._fetch_lme_price(material, currency)
            if price:
                return price
        
        # 3. Return None - no hardcoded fallback!
        logger.warning(f"No price available for {material}")
        return None
    
    async def _fetch_lme_price(
        self,
        material: str,
        currency: str
    ) -> Optional[PricePoint]:
        """
        Fetch price from London Metal Exchange.
        
        Note: LME requires commercial API access.
        This is a placeholder implementation.
        """
        if not self.lme_api_key or not HAS_HTTPX:
            return None
        
        try:
            # Map material names to LME symbols
            lme_symbols = {
                "aluminum": "AL",
                "aluminium": "AL",
                "copper": "CU",
                "zinc": "ZN",
                "nickel": "NI",
                "lead": "PB",
                "tin": "SN",
            }
            
            material_lower = material.lower()
            symbol = None
            for key, sym in lme_symbols.items():
                if key in material_lower:
                    symbol = sym
                    break
            
            if not symbol:
                return None
            
            # LME API call (placeholder - real implementation needs proper endpoint)
            # url = f"https://api.lme.com/v1/price/{symbol}"
            # headers = {"Authorization": f"Bearer {self.lme_api_key}"}
            # response = await self.http_client.get(url, headers=headers)
            
            # For now, return None - real implementation when API available
            logger.debug(f"LME API not configured, skipping {material}")
            return None
            
        except Exception as e:
            logger.error(f"LME API error: {e}")
            return None
    
    async def get_component_price(
        self,
        mpn: str,
        quantity: int = 1,
        currency: str = "USD"
    ) -> Optional[PricePoint]:
        """
        Get component price from DigiKey or Mouser.
        
        Args:
            mpn: Manufacturer part number
            quantity: Quantity to price
            currency: Currency code
            
        Returns:
            PricePoint or None
        """
        await self.initialize()
        
        # Try database first
        try:
            component = await supabase.get_component(mpn)
            pricing = component.get("pricing", {})
            
            if pricing:
                # Find price for quantity tier
                price = self._get_price_for_quantity(pricing, quantity, currency)
                if price:
                    return PricePoint(
                        price=price,
                        currency=currency,
                        unit="each",
                        source="component_catalog",
                        timestamp=datetime.now(),
                        expires_at=datetime.now() + timedelta(hours=24)
                    )
        except ValueError:
            pass
        
        # Try DigiKey API
        if self.digikey_client_id:
            price = await self._fetch_digikey_price(mpn, quantity, currency)
            if price:
                return price
        
        return None
    
    def _get_price_for_quantity(
        self,
        pricing: Dict,
        quantity: int,
        currency: str
    ) -> Optional[float]:
        """Extract price for quantity tier"""
        tiers = pricing.get("tiers", [])
        currency_prices = pricing.get(currency.lower(), {})
        
        if not tiers and currency_prices:
            # Flat pricing
            return float(currency_prices.get("unit", 0))
        
        # Find appropriate tier
        applicable_tier = None
        for tier in sorted(tiers, key=lambda x: x.get("min_qty", 0)):
            if tier.get("min_qty", 0) <= quantity:
                applicable_tier = tier
            else:
                break
        
        if applicable_tier:
            return float(applicable_tier.get("price", 0))
        
        return None
    
    async def _fetch_digikey_price(
        self,
        mpn: str,
        quantity: int,
        currency: str
    ) -> Optional[PricePoint]:
        """Fetch price from DigiKey API"""
        if not HAS_HTTPX:
            return None
        
        try:
            # DigiKey API requires OAuth2 - simplified placeholder
            # Real implementation needs token refresh
            logger.debug(f"DigiKey API not fully implemented, skipping {mpn}")
            return None
            
        except Exception as e:
            logger.error(f"DigiKey API error: {e}")
            return None
    
    async def get_carbon_footprint(
        self,
        material: str,
        mass_kg: float
    ) -> Optional[Dict[str, Any]]:
        """
        Get carbon footprint from Climatiq API.
        
        Args:
            material: Material name
            mass_kg: Mass in kg
            
        Returns:
            Carbon footprint data
        """
        # Try database first
        try:
            material_data = await supabase.get_material(material)
            factor = material_data.get("carbon_footprint_kg_co2_per_kg")
            
            if factor:
                return {
                    "material": material,
                    "mass_kg": mass_kg,
                    "factor_kg_co2_per_kg": factor,
                    "total_kg_co2": factor * mass_kg,
                    "source": material_data.get("data_source", "database"),
                    "unit": "kg CO2e"
                }
        except ValueError:
            pass
        
        # Try Climatiq API
        if self.climatiq_api_key and HAS_HTTPX:
            return await self._fetch_climatiq_footprint(material, mass_kg)
        
        return None
    
    async def _fetch_climatiq_footprint(
        self,
        material: str,
        mass_kg: float
    ) -> Optional[Dict[str, Any]]:
        """Fetch carbon footprint from Climatiq"""
        try:
            # Climatiq API endpoint
            url = "https://api.climatiq.io/data/v1/estimate"
            headers = {
                "Authorization": f"Bearer {self.climatiq_api_key}",
                "Content-Type": "application/json"
            }
            
            # Map material to Climatiq activity ID
            activity_map = {
                "aluminum": "material-type_aluminium",
                "steel": "material-type_steel",
                "plastic": "material-type_plastic",
            }
            
            material_lower = material.lower()
            activity_id = None
            for key, act_id in activity_map.items():
                if key in material_lower:
                    activity_id = act_id
                    break
            
            if not activity_id:
                return None
            
            payload = {
                "activity_id": activity_id,
                "parameters": {
                    "mass": mass_kg,
                    "mass_unit": "kg"
                }
            }
            
            # response = await self.http_client.post(url, json=payload, headers=headers)
            # data = response.json()
            
            # Placeholder - implement when API key available
            logger.debug(f"Climatiq API not fully configured")
            return None
            
        except Exception as e:
            logger.error(f"Climatiq API error: {e}")
            return None
    
    async def batch_get_material_prices(
        self,
        materials: List[str],
        currency: str = "USD"
    ) -> Dict[str, Optional[PricePoint]]:
        """
        Get prices for multiple materials concurrently.
        
        Args:
            materials: List of material names
            currency: Currency code
            
        Returns:
            Dictionary of material -> PricePoint
        """
        tasks = [
            self.get_material_price(m, currency)
            for m in materials
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            material: result if not isinstance(result, Exception) else None
            for material, result in zip(materials, results)
        }


# Global instance
pricing_service = PricingService()
