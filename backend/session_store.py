"""
Session storage implementations for BRICK OS.
Supports both in-memory (development) and Redis (production) backends.
"""

import json
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class SessionStore(ABC):
    """Abstract base class for session storage."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value with TTL."""
        pass
    
    @abstractmethod
    async def delete(self, key: str):
        """Delete key."""
        pass
    
    @abstractmethod
    async def get_discovery_state(self, session_id: str) -> Optional[Dict]:
        """Retrieve discovery state for a session."""
        pass
    
    @abstractmethod
    async def set_discovery_state(self, session_id: str, state: Dict, ttl: int = 3600):
        """Store discovery state with TTL."""
        pass
    
    @abstractmethod
    async def delete_discovery_state(self, session_id: str):
        """Delete discovery state."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Verify storage connectivity."""
        pass


class InMemorySessionStore(SessionStore):
    """
    In-memory session storage for development.
    NOT for production - data is lost on restart.
    """
    
    def __init__(self):
        self._store: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        async with self._lock:
            await self._cleanup_expired()
            return self._store.get(key)
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value with TTL."""
        async with self._lock:
            import time
            self._store[key] = value
            self._timestamps[key] = time.time() + ttl
    
    async def delete(self, key: str):
        """Delete key."""
        async with self._lock:
            self._store.pop(key, None)
            self._timestamps.pop(key, None)
    
    async def get_discovery_state(self, session_id: str) -> Optional[Dict]:
        async with self._lock:
            await self._cleanup_expired()
            return self._store.get(session_id)
    
    async def set_discovery_state(self, session_id: str, state: Dict, ttl: int = 3600):
        async with self._lock:
            import time
            self._store[session_id] = state
            self._timestamps[session_id] = time.time() + ttl
    
    async def delete_discovery_state(self, session_id: str):
        async with self._lock:
            self._store.pop(session_id, None)
            self._timestamps.pop(session_id, None)
    
    async def health_check(self) -> bool:
        return True
    
    async def _cleanup_expired(self):
        """Remove expired sessions."""
        import time
        now = time.time()
        expired = [
            sid for sid, exp in self._timestamps.items() 
            if now > exp
        ]
        for sid in expired:
            self._store.pop(sid, None)
            self._timestamps.pop(sid, None)


class RedisSessionStore(SessionStore):
    """
    Production-grade session storage with Redis.
    Supports TTL, persistence, and distributed access.
    """
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or self._get_redis_url()
        self._redis: Optional[Any] = None
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key from Redis."""
        try:
            redis = await self._get_redis()
            data = await redis.get(key)
            if data is None:
                return None
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return data
        except Exception as e:
            logger.error(f"Failed to get key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value with TTL in Redis."""
        try:
            redis = await self._get_redis()
            await redis.setex(key, ttl, json.dumps(value, default=str))
        except Exception as e:
            logger.error(f"Failed to set key {key}: {e}")
            raise
    
    async def delete(self, key: str):
        """Delete key from Redis."""
        try:
            redis = await self._get_redis()
            await redis.delete(key)
        except Exception as e:
            logger.error(f"Failed to delete key {key}: {e}")
    
    def _get_redis_url(self) -> str:
        """Get Redis URL from environment or default."""
        import os
        return os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    async def _get_redis(self):
        """Lazy initialization of Redis connection."""
        if self._redis is None:
            try:
                import aioredis
                self._redis = await aioredis.from_url(
                    self.redis_url,
                    encoding='utf-8',
                    decode_responses=True
                )
                logger.info(f"✅ Connected to Redis at {self.redis_url}")
            except ImportError:
                logger.error("❌ aioredis not installed. Run: pip install aioredis>=2.0")
                raise RuntimeError("aioredis required for Redis session store")
            except Exception as e:
                logger.error(f"❌ Failed to connect to Redis: {e}")
                raise RuntimeError(f"Redis connection failed: {e}") from e
        return self._redis
    
    async def get_discovery_state(self, session_id: str) -> Optional[Dict]:
        """Retrieve discovery state from Redis."""
        try:
            redis = await self._get_redis()
            key = f"brick:session:{session_id}"
            
            data = await redis.get(key)
            if data is None:
                return None
            
            # Extend TTL on access (session still active)
            await redis.expire(key, 3600)
            
            return json.loads(data)
            
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None
    
    async def set_discovery_state(self, session_id: str, state: Dict, ttl: int = 3600):
        """Persist discovery state to Redis with TTL."""
        try:
            redis = await self._get_redis()
            key = f"brick:session:{session_id}"
            
            # Add metadata
            state_with_meta = {
                **state,
                "_meta": {
                    "updated_at": datetime.utcnow().isoformat(),
                    "ttl": ttl
                }
            }
            
            await redis.setex(
                key,
                ttl,
                json.dumps(state_with_meta, default=str)
            )
            
            logger.debug(f"💾 Saved session {session_id} (TTL: {ttl}s)")
            
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
            raise RuntimeError(f"Session persistence failed: {e}") from e
    
    async def delete_discovery_state(self, session_id: str):
        """Delete discovery state from Redis."""
        try:
            redis = await self._get_redis()
            await redis.delete(f"brick:session:{session_id}")
            logger.debug(f"🗑️  Deleted session {session_id}")
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
    
    async def health_check(self) -> bool:
        """Verify Redis connectivity."""
        try:
            redis = await self._get_redis()
            await redis.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None


def create_session_store() -> SessionStore:
    """
    Factory function to create appropriate session store.
    Uses Redis if REDIS_URL is set, otherwise falls back to in-memory.
    """
    import os
    
    redis_url = os.getenv("REDIS_URL")
    
    if redis_url:
        logger.info(f"🔧 Using Redis session store: {redis_url}")
        return RedisSessionStore(redis_url)
    else:
        logger.warning("⚠️  Using InMemorySessionStore - sessions will NOT persist across restarts!")
        logger.warning("   Set REDIS_URL environment variable for production persistence.")
        return InMemorySessionStore()
