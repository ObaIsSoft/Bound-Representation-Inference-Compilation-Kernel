"""
Supabase Service - Centralized Database Access

Provides typed, cached, and retry-enabled access to Supabase.
All agents MUST use this service instead of direct Supabase calls.

PRODUCTION: Supabase ONLY - no local fallbacks
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
    
    PRODUCTION: Requires Supabase credentials. No fallbacks.
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
                logger.error("Supabase credentials not found in environment")
                raise RuntimeError(
                    "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set"
                )
        else:
            raise RuntimeError(
                "Supabase package not installed. Run: pip install supabase"
            )
        
        # Initialize Redis (optional)
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
    
    def _ensure_initialized(self):
        """Ensure service is initialized"""
        if not self._initialized:
            raise RuntimeError("SupabaseService not initialized. Call initialize() first.")
        if not self.client:
            raise RuntimeError("Supabase client not available")
    
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
    
    async def _invalidate_cache(self, key: str):
        """Invalidate a cached value"""
        if not self.redis:
            return
        try:
            await self.redis.delete(key)
        except Exception as e:
            logger.debug(f"Cache invalidate failed: {e}")
    
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
            RuntimeError: If Supabase not initialized
        """
        await self.initialize()
        self._ensure_initialized()
        
        cache_key = self._make_cache_key(
            "critic_thresholds",
            {"critic": critic_name, "vehicle": vehicle_type}
        )
        
        # Check cache
        cached = await self._get_cached(cache_key)
        if cached:
            return cached
        
        # Query database
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
        
        raise ValueError(
            f"No thresholds found for {critic_name}/{vehicle_type}. "
            f"Please populate critic_thresholds table."
        )
    
    # ========================================================================
    # MATERIALS
    # ========================================================================
    
    async def get_material(self, material_name: str) -> Dict[str, Any]:
        """
        Get material properties from Supabase.
        
        Args:
            material_name: Material name (e.g., "Aluminum 6061")
            
        Returns:
            Material properties dictionary
            
        Raises:
            ValueError: If material not found
            RuntimeError: If Supabase not initialized
        """
        await self.initialize()
        self._ensure_initialized()
        
        cache_key = self._make_cache_key("materials", {"name": material_name})
        
        # Check cache
        cached = await self._get_cached(cache_key)
        if cached:
            return cached
        
        # Query Supabase
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
        Get material price from Supabase.
        
        Args:
            material_name: Material name
            currency: Currency code (USD, EUR, etc.)
            
        Returns:
            Price per kg or None if not available
        """
        material = await self.get_material(material_name)
        price_column = f"cost_per_kg_{currency.lower()}"
        
        price = material.get(price_column)
        if price is not None:
            return float(price)
        
        # Try USD as fallback currency
        if currency != "USD":
            price = material.get("cost_per_kg_usd")
            if price is not None:
                return float(price)
        
        return None
    
    async def update_material_price(
        self,
        material_name: str,
        price: float,
        currency: str = "USD",
        source: str = "manual"
    ) -> bool:
        """
        Update or insert material price in Supabase.
        
        Args:
            material_name: Material name
            price: Price per kg
            currency: Currency code
            source: Price source
            
        Returns:
            True if successful
        """
        await self.initialize()
        self._ensure_initialized()
        
        # Check if material exists
        existing = self.client.table("materials")\
            .select("id")\
            .ilike("name", material_name)\
            .execute()
        
        price_column = f"cost_per_kg_{currency.lower()}"
        
        if existing.data:
            # Update existing
            material_id = existing.data[0]["id"]
            self.client.table("materials")\
                .update({
                    price_column: price,
                    "property_data_source": source
                })\
                .eq("id", material_id)\
                .execute()
        else:
            # Insert new
            self.client.table("materials")\
                .insert({
                    "name": material_name,
                    price_column: price,
                    "property_data_source": source
                })\
                .execute()
        
        # Invalidate cache
        cache_key = self._make_cache_key("materials", {"name": material_name})
        await self._invalidate_cache(cache_key)
        
        return True
    
    # ========================================================================
    # MANUFACTURING RATES
    # ========================================================================
    
    async def get_manufacturing_rates(
        self,
        process_type: str,
        region: str = "global"
    ) -> Dict[str, Any]:
        """
        Get manufacturing rates from Supabase.
        
        Args:
            process_type: Process type (e.g., "cnc_milling")
            region: Region code (e.g., "us", "eu", "global")
            
        Returns:
            Manufacturing rates dictionary
        """
        await self.initialize()
        self._ensure_initialized()
        
        cache_key = self._make_cache_key(
            "manufacturing_rates",
            {"process": process_type, "region": region}
        )
        
        # Check cache
        cached = await self._get_cached(cache_key)
        if cached:
            return cached
        
        # Query Supabase
        result = self.client.table("manufacturing_rates")\
            .select("*")\
            .eq("process", process_type)\
            .eq("region", region)\
            .single()\
            .execute()
        
        if result.data:
            await self._set_cached(
                cache_key,
                result.data,
                self.cache_ttls["pricing"]
            )
            return result.data
        
        raise ValueError(
            f"No manufacturing rates found for {process_type}/{region}"
        )
    
    # ========================================================================
    # COMPONENTS
    # ========================================================================
    
    async def get_component(
        self,
        component_name: str
    ) -> Dict[str, Any]:
        """
        Get component specifications from Supabase.
        
        Args:
            component_name: Component name or ID
            
        Returns:
            Component specifications
        """
        await self.initialize()
        self._ensure_initialized()
        
        cache_key = self._make_cache_key("components", {"name": component_name})
        
        # Check cache
        cached = await self._get_cached(cache_key)
        if cached:
            return cached
        
        # Query Supabase
        result = self.client.table("components")\
            .select("*")\
            .or_(f"name.ilike.%{component_name}%,id.ilike.%{component_name}%")\
            .limit(1)\
            .execute()
        
        if result.data and len(result.data) > 0:
            await self._set_cached(
                cache_key,
                result.data[0],
                self.cache_ttls["components"]
            )
            return result.data[0]
        
        raise ValueError(f"Component '{component_name}' not found")
    
    async def get_components_by_category(
        self,
        category: str
    ) -> List[Dict[str, Any]]:
        """
        Get all components in a category.
        
        Args:
            category: Component category
            
        Returns:
            List of component specifications
        """
        await self.initialize()
        self._ensure_initialized()
        
        result = self.client.table("components")\
            .select("*")\
            .eq("category", category)\
            .execute()
        
        return result.data or []


# Global service instance
supabase_service = SupabaseService()

# Backward compatibility alias
supabase = supabase_service


# Convenience functions for synchronous usage
def get_supabase_client() -> Optional[Client]:
    """Get the Supabase client (if initialized)"""
    return supabase_service.client


async def get_material(material_name: str) -> Dict[str, Any]:
    """Convenience function to get material"""
    return await supabase_service.get_material(material_name)


async def get_critic_thresholds(critic_name: str, vehicle_type: str = "default") -> Dict[str, Any]:
    """Convenience function to get critic thresholds"""
    return await supabase_service.get_critic_thresholds(critic_name, vehicle_type)


async def get_manufacturing_rates(process_type: str, region: str = "global") -> Dict[str, Any]:
    """Convenience function to get manufacturing rates"""
    return await supabase_service.get_manufacturing_rates(process_type, region)
