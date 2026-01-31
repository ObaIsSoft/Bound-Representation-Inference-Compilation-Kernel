"""
Cache Strategy Abstraction.

This module defines the interface for caching heavy computation results.
Strategies:
1. LRUCacheProvider (Default): In-memory Python caching.
   - Good for: Development, Single-instance.
   - Pros: Fastest (RAM access), Zero network overhead.
   - Cons: Cleared on restart, Local to process.

2. RedisCacheProvider (Future): Distributed caching.
   - Good for: Scaled production.
   - Pros: Persistent across restarts, Shared across workers.
"""

from typing import Any, Protocol, Optional, Union
import logging
from functools import lru_cache
import json
import time

logger = logging.getLogger(__name__)

class CacheProvider(Protocol):
    """Protocol defining the Cache Interface."""
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache."""
        ...
        
    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        """Set value in cache with Time-To-Live (seconds)."""
        ...
    
    def clear(self) -> None:
        """Clear the cache."""
        ...

# --- 1. In-Memory Implementation (Default) ---

class LRUCacheProvider:
    """
    In-memory cache using a dictionary with TTL support.
    Note: Python's lru_cache decorator is function-based. 
    This is a key-value store wrapper for manual caching.
    """
    def __init__(self, capacity: int = 1000):
        self.cache: dict = {}
        self.capacity = capacity
        # We store (value, expire_time)
        logger.info(f"LRUCacheProvider initialized (Capacity: {capacity})")

    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
            
        value, expire_time = self.cache[key]
        
        # Check Expiry
        if time.time() > expire_time:
            del self.cache[key]
            return None
            
        return value

    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        # Simple capacity check (random eviction if full - strictly not LRU but sufficient for MVP)
        if len(self.cache) >= self.capacity:
            # Just clear 10% to make space
            keys = list(self.cache.keys())[:int(self.capacity * 0.1)]
            for k in keys: del self.cache[k]
        
        expire_time = time.time() + ttl
        self.cache[key] = (value, expire_time)

    def clear(self) -> None:
        self.cache.clear()


# --- 2. Scaling Implementation (Future) ---

# #scale: Implement RedisCacheProvider here
# class RedisCacheProvider:
#     """
#     Production-grade cache using Redis.
#     """
#     def __init__(self, redis_url: str = "redis://localhost:6379/0"):
#         # import redis
#         # self.r = redis.from_url(redis_url)
#         pass
#
#     def get(self, key: str) -> Optional[Any]:
#         # val = self.r.get(key)
#         # if val: return json.loads(val)
#         return None
#
#     def set(self, key: str, value: Any, ttl: int = 300) -> None:
#         # self.r.setex(key, ttl, json.dumps(value))
#         pass
#
#     def clear(self) -> None:
#         # self.r.flushdb()
#         pass


# --- Factory ---

_cache_instance = None

def get_cache(provider: str = "memory") -> CacheProvider:
    """Singleton Factory for CacheProvider."""
    global _cache_instance
    if _cache_instance is None:
        if provider == "memory":
            _cache_instance = LRUCacheProvider()
        elif provider == "redis":
            # #scale: Switch to RedisCacheProvider
            # _cache_instance = RedisCacheProvider()
            logger.warning("Redis cache requested but not implemented. Falling back to Memory.")
            _cache_instance = LRUCacheProvider()
            
    return _cache_instance
