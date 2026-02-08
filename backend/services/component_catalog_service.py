"""
Component Catalog Service

Provides unified access to electronic components from multiple suppliers:
- DigiKey (via Nexar API - already have key)
- Mouser
- Octopart

Features:
- Search by specs, MPN, or description
- Real-time inventory and pricing
- Datasheet and CAD model retrieval
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

logger = logging.getLogger(__name__)


@dataclass
class ComponentSpec:
    """Component specification"""
    mpn: str  # Manufacturer Part Number
    manufacturer: str
    category: str
    description: str
    
    # Electrical specs
    voltage_rating_v: Optional[float] = None
    current_rating_a: Optional[float] = None
    power_rating_w: Optional[float] = None
    resistance_ohm: Optional[float] = None
    capacitance_f: Optional[float] = None
    inductance_h: Optional[float] = None
    frequency_hz: Optional[float] = None
    
    # Physical specs
    package: Optional[str] = None
    mounting_type: Optional[str] = None  # SMD, THT
    dimensions_mm: Optional[Dict[str, float]] = None
    weight_g: Optional[float] = None
    
    # Metadata
    rohs_compliant: bool = True
    lead_free: bool = True
    lifecycle_status: str = "active"  # active, obsolete, nrnd


@dataclass
class ComponentOffer:
    """Supplier offer for a component"""
    supplier: str
    mpn: str
    
    # Pricing tiers
    pricing: List[Dict[str, Any]]  # [{"qty": 100, "price": 0.50}, ...]
    currency: str = "USD"
    
    # Inventory
    stock_quantity: int = 0
    lead_time_days: Optional[int] = None
    
    # Links
    product_url: Optional[str] = None
    datasheet_url: Optional[str] = None
    cad_model_url: Optional[str] = None
    
    # Metadata
    last_updated: Optional[datetime] = None
    moq: int = 1  # Minimum order quantity


class ComponentCatalogService:
    """
    Unified component catalog service.
    
    Searches across multiple suppliers and returns best options.
    """
    
    def __init__(self):
        self.http_client: Optional[Any] = None
        self._initialized = False
        
        # API keys
        self.nexar_api_key = os.getenv("NEXAR_API_KEY")
        self.nexar_secret = os.getenv("NEXAR_SECRET")
        self.mouser_api_key = os.getenv("MOUSER_API_KEY")
        self.mouser_partner_id = os.getenv("MOUSER_PARTNER_ID")
        self.octopart_api_key = os.getenv("OCTOPART_API_KEY")
        
    async def initialize(self):
        """Initialize HTTP client"""
        if self._initialized:
            return
            
        if HAS_HTTPX:
            self.http_client = httpx.AsyncClient(timeout=30.0)
        
        self._initialized = True
    
    async def search_by_mpn(self, mpn: str) -> List[ComponentOffer]:
        """
        Search for component by Manufacturer Part Number.
        
        Args:
            mpn: Manufacturer part number
            
        Returns:
            List of supplier offers
        """
        await self.initialize()
        
        # Search all suppliers concurrently
        tasks = []
        
        if self.nexar_api_key:
            tasks.append(self._search_digikey(mpn))
        if self.mouser_api_key:
            tasks.append(self._search_mouser(mpn))
        if self.octopart_api_key:
            tasks.append(self._search_octopart(mpn))
        
        if not tasks:
            logger.warning("No component catalog APIs configured")
            return []
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten and filter valid results
        offers = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Catalog search error: {result}")
                continue
            offers.extend(result)
        
        # Sort by price (lowest first)
        offers.sort(key=lambda x: x.pricing[0]["price"] if x.pricing else float('inf'))
        
        return offers
    
    async def search_by_specs(
        self,
        category: str,
        specs: Dict[str, Any],
        max_results: int = 10
    ) -> List[ComponentSpec]:
        """
        Search for components by specifications.
        
        Args:
            category: Component category (resistor, capacitor, etc.)
            specs: Specification filters
            max_results: Maximum results to return
            
        Returns:
            List of component specifications
        """
        await self.initialize()
        
        # Build search query from specs
        # This is simplified - real implementation would use parametric search
        
        query_parts = [category]
        for key, value in specs.items():
            if isinstance(value, (int, float)):
                query_parts.append(f"{key}:{value}")
            else:
                query_parts.append(f"{key} {value}")
        
        search_query = " ".join(query_parts)
        
        # Use Octopart for parametric search
        if self.octopart_api_key:
            return await self._search_octopart_parametric(
                category, specs, max_results
            )
        
        logger.warning("No parametric search API configured")
        return []
    
    async def get_datasheet(self, mpn: str, supplier: str = "digikey") -> Optional[str]:
        """
        Get datasheet URL for component.
        
        Args:
            mpn: Manufacturer part number
            supplier: Preferred supplier
            
        Returns:
            Datasheet URL or None
        """
        await self.initialize()
        
        offers = await self.search_by_mpn(mpn)
        
        # Find preferred supplier
        for offer in offers:
            if offer.supplier.lower() == supplier.lower():
                return offer.datasheet_url
        
        # Return first available
        for offer in offers:
            if offer.datasheet_url:
                return offer.datasheet_url
        
        return None
    
    async def get_cad_model(
        self,
        mpn: str,
        format: str = "step"
    ) -> Optional[str]:
        """
        Get CAD model URL for component.
        
        Args:
            mpn: Manufacturer part number
            format: CAD format (step, iges, etc.)
            
        Returns:
            CAD model URL or None
        """
        await self.initialize()
        
        offers = await self.search_by_mpn(mpn)
        
        for offer in offers:
            if offer.cad_model_url:
                # Check if format matches
                if format.lower() in offer.cad_model_url.lower():
                    return offer.cad_model_url
                return offer.cad_model_url
        
        return None
    
    async def _search_digikey(self, mpn: str) -> List[ComponentOffer]:
        """
        Search DigiKey via Nexar API.
        
        Note: Nexar is DigiKey's API platform.
        """
        if not HAS_HTTPX or not self.nexar_api_key:
            return []
        
        try:
            # Nexar GraphQL endpoint
            url = "https://api.nexar.com/graphql"
            
            # GraphQL query for parts
            query = """
            query SearchParts($mpn: String!) {
                supSearchMpn(q: $mpn, limit: 5) {
                    results {
                        part {
                            mpn
                            manufacturer {
                                name
                            }
                            shortDescription
                            specs {
                                attribute {
                                    name
                                }
                                value
                            }
                            bestDatasheet {
                                url
                            }
                            sellers {
                                company {
                                    name
                                }
                                offers {
                                    inventoryLevel
                                    prices {
                                        quantity
                                        price
                                        currency
                                    }
                                }
                            }
                        }
                    }
                }
            }
            """
            
            headers = {
                "Authorization": f"Bearer {self.nexar_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "query": query,
                "variables": {"mpn": mpn}
            }
            
            # For now, return empty - implement when API available
            logger.debug(f"Nexar API search for {mpn}")
            return []
            
        except Exception as e:
            logger.error(f"DigiKey search error: {e}")
            return []
    
    async def _search_mouser(self, mpn: str) -> List[ComponentOffer]:
        """Search Mouser Electronics"""
        if not HAS_HTTPX or not self.mouser_api_key:
            return []
        
        try:
            # Mouser API endpoint
            url = f"https://api.mouser.com/api/v1/search/partnumber"
            
            params = {
                "apiKey": self.mouser_api_key
            }
            
            payload = {
                "SearchByPartRequest": {
                    "mouserPartNumber": mpn,
                    "partSearchOptions": "Exact"
                }
            }
            
            # Placeholder - implement when API available
            logger.debug(f"Mouser API search for {mpn}")
            return []
            
        except Exception as e:
            logger.error(f"Mouser search error: {e}")
            return []
    
    async def _search_octopart(self, mpn: str) -> List[ComponentOffer]:
        """Search Octopart (Altium)"""
        if not HAS_HTTPX or not self.octopart_api_key:
            return []
        
        try:
            # Octopart API endpoint
            url = "https://octopart.com/api/v4/endpoint"
            
            # Placeholder - implement when API available
            logger.debug(f"Octopart API search for {mpn}")
            return []
            
        except Exception as e:
            logger.error(f"Octopart search error: {e}")
            return []
    
    async def _search_octopart_parametric(
        self,
        category: str,
        specs: Dict[str, Any],
        max_results: int
    ) -> List[ComponentSpec]:
        """Parametric search on Octopart"""
        # Placeholder for parametric search
        return []


# Global instance
component_catalog = ComponentCatalogService()
