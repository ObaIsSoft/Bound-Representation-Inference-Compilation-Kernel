from typing import Dict, Any, Set

class ProgressiveCompiler:
    """Only recompile changed components"""
    
    def __init__(self):
        self.component_cache: Dict[str, Any] = {}
        self.dirty_flags: Set[str] = set()
    
    def mark_dirty(self, component_id: str):
        """Mark component for recompilation"""
        self.dirty_flags.add(component_id)
    
    def is_dirty(self, component_id: str) -> bool:
        """Check if component needs recompilation"""
        return component_id in self.dirty_flags
    
    def get_cached(self, component_id: str):
        """Get cached compiled component"""
        return self.component_cache.get(component_id)
    
    def set_cached(self, component_id: str, compiled_data):
        """Cache compiled component"""
        self.component_cache[component_id] = compiled_data
        self.dirty_flags.discard(component_id)
    
    def clear_dirty(self):
        """Clear all dirty flags"""
        self.dirty_flags.clear()
