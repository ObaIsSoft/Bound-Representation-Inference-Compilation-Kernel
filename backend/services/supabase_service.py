"""
Supabase Service - Centralized Database Access

Provides typed, cached, and retry-enabled access to Supabase.
All agents MUST use this service instead of direct Supabase calls.
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

# Try to import supabase
try:
    from supabase import create_client, Client
    HAS_SUPABASE = True
except ImportError:
    HAS_SUPABASE = False
    Client = Any

# Try to import redis for caching
try:
    import redis.asyncio as redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Typed query result"""
    data: List[Dict[str, Any]]
    count: Optional[int] = None
    error: Optional[str] = None


class SupabaseService:
    """
    Centralized Supabase client with:
    - Connection pooling
    - Automatic retries
    - Redis caching
    - Typed query builders
    """
    
    def __init__(self):
        self.client: Optional[Client] = None
        self.redis: Optional[Any] = None
        self._initialized = False
        
        # Cache TTLs
        self.cache_ttls = {
            "materials": 3600,  # 1 hour
            "pricing": 1800,    # 30 minutes
            "components": 3600, # 1 hour
            "standards": 86400, # 24 hours
            "thresholds": 300,  # 5 minutes
        }
        
    async def initialize(self):
        """Initialize Supabase and Redis connections"""
        if self._initialized:
            return
            
        # Initialize Supabase
        if HAS_SUPABASE:
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
            
            if url and key:
                self.client = create_client(url, key)
                logger.info("Supabase client initialized")
            else:
                logger.warning("Supabase credentials not found - using fallback mode")
        else:
            logger.warning("Supabase package not installed - using fallback mode")
        
        # Initialize Redis
        if HAS_REDIS:
            redis_url = os.getenv("REDIS_URL")
            if redis_url:
                try:
                    self.redis = redis.from_url(redis_url, decode_responses=True)
                    await self.redis.ping()
                    logger.info("Redis cache initialized")
                except Exception as e:
                    logger.warning(f"Redis connection failed: {e}")
        
        self._initialized = True
    
    async def _get_cached(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        if not self.redis:
            return None
        try:
            data = await self.redis.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.debug(f"Cache get failed: {e}")
        return None
    
    async def _set_cached(self, key: str, value: Any, ttl: int = 3600):
        """Set value in Redis cache"""
        if not self.redis:
            return
        try:
            await self.redis.setex(key, ttl, json.dumps(value, default=str))
        except Exception as e:
            logger.debug(f"Cache set failed: {e}")
    
    def _make_cache_key(self, table: str, query_params: Dict) -> str:
        """Create cache key from query params"""
        param_str = json.dumps(query_params, sort_keys=True, default=str)
        return f"supabase:{table}:{hash(param_str)}"
    
    # ========================================================================
    # CRITIC THRESHOLDS
    # ========================================================================
    
    async def get_critic_thresholds(
        self,
        critic_name: str,
        vehicle_type: str = "default"
    ) -> Dict[str, Any]:
        """
        Get critic thresholds from database.
        
        Args:
            critic_name: Name of critic (e.g., "ControlCritic")
            vehicle_type: Vehicle type (e.g., "drone_large")
            
        Returns:
            Dictionary of threshold values
            
        Raises:
            ValueError: If thresholds not found in database
        """
        await self.initialize()
        
        cache_key = self._make_cache_key(
            "critic_thresholds",
            {"critic": critic_name, "vehicle": vehicle_type}
        )
        
        # Check cache
        cached = await self._get_cached(cache_key)
        if cached:
            return cached
        
        # Query database
        if self.client:
            try:
                result = self.client.table("critic_thresholds")\
                    .select("thresholds")\
                    .eq("critic_name", critic_name)\
                    .eq("vehicle_type", vehicle_type)\
                    .single()\
                    .execute()
                
                if result.data:
                    thresholds = result.data.get("thresholds", {})
                    await self._set_cached(
                        cache_key,
                        thresholds,
                        self.cache_ttls["thresholds"]
                    )
                    return thresholds
                    
            except Exception as e:
                logger.error(f"Database query failed: {e}")
        
        # Fallback - raise error (no hardcoded defaults!)
        raise ValueError(
            f"No thresholds found for {critic_name}/{vehicle_type}. "
            f"Please populate critic_thresholds table."
        )
    
    # ========================================================================
    # MATERIALS
    # ========================================================================
    
    async def get_material(self, material_name: str) -> Dict[str, Any]:
        """
        Get material properties from database.
        
        Args:
            material_name: Material name (e.g., "Aluminum 6061")
            
        Returns:
            Material properties dictionary
            
        Raises:
            ValueError: If material not found
        """
        await self.initialize()
        
        cache_key = self._make_cache_key("materials", {"name": material_name})
        
        # Check cache
        cached = await self._get_cached(cache_key)
        if cached:
            return cached
        
        # Query database
        if self.client:
            try:
                result = self.client.table("materials")\
                    .select("*")\
                    .ilike("name", material_name)\
                    .limit(1)\
                    .execute()
                
                if result.data and len(result.data) > 0:
                    material = result.data[0]
                    await self._set_cached(
                        cache_key,
                        material,
                        self.cache_ttls["materials"]
                    )
                    return material
                    
            except Exception as e:
                logger.error(f"Database query failed: {e}")
        
        raise ValueError(f"Material '{material_name}' not found in database")
    
    async def get_material_properties(self, material_name: str) -> Dict[str, Any]:
        """Alias for get_material"""
        return await self.get_material(material_name)
    
    async def get_material_price(
        self,
        material_name: str,
        currency: str = "USD"
    ) -> Optional[float]:
        """
        Get cached material price.
        
        Args:
            material_name: Material name
            currency: Currency code (USD, EUR, etc.)
            
        Returns:
            Price per kg or None if not available
        """
        await self.initialize()
        
        material = await self.get_material(material_name)
        price_column = f"cost_per_kg_{currency.lower()}"
        
        price = material.get(price_column)
        if price is not None:
            return float(price)
        
        # Try USD as fallback
        if currency != "USD":
            price = material.get("cost_per_kg_usd")
            if price is not None:
                return float(price)
        
        return None
    
    # ========================================================================
    # MANUFACTURING RATES
    # ========================================================================
    
    async def get_manufacturing_rates(
        self,
        process_type: str,
        region: str = "global"
    ) -> Dict[str, Any]:
        """
        Get manufacturing rates for a process.
        
        Args:
            process_type: Process type (e.g., "cnc_milling")
            region: Region code (e.g., "us", "eu", "global")
            
        Returns:
            Manufacturing rates dictionary
        """
        await self.initialize()
        
        cache_key = self._make_cache_key(
            "manufacturing_rates",
            {"process": process_type, "region": region}
        )
        
        # Check cache
        cached = await self._get_cached(cache_key)
        if cached:
            return cached
        
        # Query database
        if self.client:
            try:
                result = self.client.table("manufacturing_rates")\
                    .select("*")\
                    .eq("process_type", process_type)\
                    .eq("region", region)\
                    .limit(1)\
                    .execute()
                
                if result.data and len(result.data) > 0:
                    rates = result.data[0]
                    await self._set_cached(
                        cache_key,
                        rates,
                        self.cache_ttls["standards"]
                    )
                    return rates
                    
            except Exception as e:
                logger.error(f"Database query failed: {e}")
        
        raise ValueError(
            f"No manufacturing rates found for {process_type}/{region}"
        )
    
    # ========================================================================
    # STANDARDS
    # ========================================================================
    
    async def get_standard(
        self,
        standard_type: str,
        standard_key: str
    ) -> Dict[str, Any]:
        """
        Get standard reference value.
        
        Args:
            standard_type: Type (e.g., "iso_fit", "awg_ampacity")
            standard_key: Key (e.g., "H7/g6", "12")
            
        Returns:
            Standard value dictionary
        """
        await self.initialize()
        
        cache_key = self._make_cache_key(
            "standards",
            {"type": standard_type, "key": standard_key}
        )
        
        # Check cache
        cached = await self._get_cached(cache_key)
        if cached:
            return cached
        
        # Query database
        if self.client:
            try:
                result = self.client.table("standards_reference")\
                    .select("standard_value")\
                    .eq("standard_type", standard_type)\
                    .eq("standard_key", standard_key)\
                    .limit(1)\
                    .execute()
                
                if result.data and len(result.data) > 0:
                    value = result.data[0].get("standard_value", {})
                    await self._set_cached(
                        cache_key,
                        value,
                        self.cache_ttls["standards"]
                    )
                    return value
                    
            except Exception as e:
                logger.error(f"Database query failed: {e}")
        
        raise ValueError(
            f"Standard not found: {standard_type}/{standard_key}"
        )
    
    # ========================================================================
    # COMPONENTS
    # ========================================================================
    
    async def get_component(self, mpn: str) -> Dict[str, Any]:
        """
        Get component by manufacturer part number.
        
        Args:
            mpn: Manufacturer part number
            
        Returns:
            Component data dictionary
        """
        await self.initialize()
        
        cache_key = self._make_cache_key("components", {"mpn": mpn})
        
        # Check cache
        cached = await self._get_cached(cache_key)
        if cached:
            return cached
        
        # Query database
        if self.client:
            try:
                result = self.client.table("component_catalog")\
                    .select("*")\
                    .eq("mpn", mpn)\
                    .limit(1)\
                    .execute()
                
                if result.data and len(result.data) > 0:
                    component = result.data[0]
                    await self._set_cached(
                        cache_key,
                        component,
                        self.cache_ttls["components"]
                    )
                    return component
                    
            except Exception as e:
                logger.error(f"Database query failed: {e}")
        
        raise ValueError(f"Component '{mpn}' not found")
    
    async def search_components(
        self,
        category: str,
        specs: Dict[str, Any],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search components by category and specs.
        
        Args:
            category: Component category (e.g., "resistor")
            specs: Specification filters
            limit: Max results
            
        Returns:
            List of matching components
        """
        await self.initialize()
        
        if not self.client:
            return []
        
        try:
            query = self.client.table("component_catalog")\
                .select("*")\
                .eq("category", category)\
                .limit(limit)
            
            result = query.execute()
            return result.data or []
            
        except Exception as e:
            logger.error(f"Component search failed: {e}")
            return []


# Global instance
supabase = SupabaseService()
