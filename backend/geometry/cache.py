from collections import OrderedDict
from typing import Optional, Dict, Any, Tuple

class GeometryCache:
    """LRU cache for compiled geometry"""
    
    def __init__(self, max_size_mb: int = 500):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache: OrderedDict[str, Tuple[bytes, float]] = OrderedDict()
        self.current_size = 0
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[bytes]:
        """Retrieve from cache"""
        if key in self.cache:
            self.hits += 1
            # Move to end (LRU)
            self.cache.move_to_end(key)
            return self.cache[key][0]
        self.misses += 1
        return None
    
    def put(self, key: str, data: bytes):
        """Store in cache with LRU eviction"""
        data_size = len(data)
        
        # Evict old entries if needed
        while self.current_size + data_size > self.max_size_bytes and self.cache:
            # OrderedDict.popitem(last=False) pops the first (oldest) item
            old_key, (old_data, old_size) = self.cache.popitem(last=False)
            self.current_size -= old_size
        
        self.cache[key] = (data, data_size)
        self.current_size += data_size
    
    def clear(self):
        """Clear entire cache"""
        self.cache.clear()
        self.current_size = 0
    
    def stats(self) -> Dict[str, Any]:
        """Cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        return {
            "size_mb": self.current_size / (1024 * 1024),
            "entries": len(self.cache),
            "hit_rate": hit_rate,
            "hits": self.hits,
            "misses": self.misses
        }
