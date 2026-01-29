import logging
import asyncio
import time
import hashlib
import json
from typing import Dict, Any, List, Optional
from dataclasses import asdict

from .base_engine import GeometryRequest, GeometryResult
from .manifold_engine import ManifoldEngine
from .cadquery_engine import CadQueryEngine
from .enums import CompileMode, GeometryFormat
from .node import GeometryNode
from .cache import GeometryCache
from .progressive import ProgressiveCompiler

logger = logging.getLogger(__name__)

class HybridGeometryEngine:
    """
    Smart Router for Geometry compilation.
    - Preview/GLB -> Manifold (Hot Path)
    - Export/STEP -> CadQuery (Cold Path)
    
    Features:
    - LRU Caching
    - Progressive Compilation (Dirty Flags)
    - Safe Process Isolation (OCCT Worker)
    """
    
    def __init__(self, cache_size_mb: int = 500):
        self.hot_engine = ManifoldEngine()
        self.cold_engine = CadQueryEngine()
        
        # Caching & Optimization
        self.cache = GeometryCache(max_size_mb=cache_size_mb)
        self.progressive = ProgressiveCompiler()
        
        # Stats
        self.compile_count = 0
        self.total_compile_time = 0.0
        
        logger.info("Hybrid Geometry Engine Initialized: [Hot: Manifold] [Cold: CadQuery]")

    async def compile(self, tree: List[Dict], format: str = "glb", request_id: str = "req_0", fidelity: str = "medium") -> GeometryResult:
        """
        Main entry point for async compilation.
        """
        start_time = time.time()
        
        # Normailze Enums
        try: 
            fmt_enum = GeometryFormat(format.lower()) 
        except ValueError:
             # Fallback/Safety
             fmt_enum = GeometryFormat.GLTF if format in ["glb", "gltf"] else GeometryFormat.STEP
             
        try:
             mode_enum = CompileMode(fidelity.lower())
        except (ValueError, AttributeError):
             logger.debug(f"Invalid fidelity '{fidelity}', using STANDARD")
             mode_enum = CompileMode.STANDARD

        # Convert to GeometryNodes (for cache key generation)
        nodes = [self._dict_to_node(n) for n in tree]
        
        # Check Cache
        cache_key = self._generate_cache_key(nodes, mode_enum, fmt_enum)
        cached_data = self.cache.get(cache_key)
        
        if cached_data:
            elapsed = (time.time() - start_time) * 1000
            return GeometryResult(
                success=True,
                payload=cached_data,
                metadata={"engine": "cache", "compile_time_ms": elapsed, "cache_hit": True}
            )

        # Create Standard Request (Shared by engines)
        # Note: Engines currently expect raw dicts in tree, but we might want to pass nodes eventually.
        # For now, pass original tree dicts to maintain compatibility with existing base_engine.
        request = GeometryRequest(
            request_id=request_id,
            tree=tree, 
            output_format=format,
            fidelity=fidelity
        )
        
        result = None
        
        # 1. Routing Logic
        # HOT PATH (Preview / Standard + GLTF/STL)
        if mode_enum == CompileMode.PREVIEW or (mode_enum == CompileMode.STANDARD and fmt_enum in [GeometryFormat.GLTF, GeometryFormat.STL]):
            # Run in thread (Manifold)
            result = await asyncio.to_thread(self.hot_engine.build, request)
            if result.success:
                result.metadata["engine"] = "manifold"
                
        # COLD PATH (Export / Complex STEP)
        elif mode_enum == CompileMode.EXPORT or fmt_enum in [GeometryFormat.STEP, GeometryFormat.IGES]:
            # Run in thread (wrapper around subprocess)
            result = await asyncio.to_thread(self.cold_engine.build, request)
            if result.success:
                result.metadata["engine"] = "cadquery_worker"
                # If file path returned, read bytes for cache? 
                # Or just cache file path? For uniformity, our Result usually carries payload bytes or file path.
                # If payload is None but file_path exists, we might load it to cache it?
                # For large STEP files, caching bytes might kill memory. 
                # Let's NOT cache large STEP exports in memory for now, only GLBs.
                pass
        else:
             result = GeometryResult(success=False, error=f"Unsupported mode/format combo: {fidelity}/{format}")

        # Post-Processing
        elapsed = (time.time() - start_time) * 1000
        
        if result and result.success:
            result.metadata["compile_time_ms"] = elapsed
            result.metadata["cache_hit"] = False
            
            # Cache if payload exists (Hot Path)
            if result.payload:
                self.cache.put(cache_key, result.payload)
                
            self.compile_count += 1
            self.total_compile_time += elapsed
            
        return result

    def _dict_to_node(self, d: Dict) -> GeometryNode:
        return GeometryNode(
            id=d.get("id", "unknown"),
            type=d.get("type", "box"),
            params=d.get("params", {}),
            transform=None, # TODO: Parse transform list to numpy
            operation=d.get("operation", "UNION")
        )

    def _generate_cache_key(self, nodes: List[GeometryNode], mode: CompileMode, fmt: GeometryFormat) -> str:
        node_keys = [n.to_cache_key() for n in nodes]
        combined = f"{'-'.join(node_keys)}-{mode.value}-{fmt.value}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def stats(self) -> Dict[str, Any]:
        avg = self.total_compile_time / self.compile_count if self.compile_count > 0 else 0
        s = self.cache.stats()
        s["avg_compile_time_ms"] = avg
        s["total_compiles"] = self.compile_count
        return s

# --- Integration Helpers ---

_engine_instance: Optional[HybridGeometryEngine] = None

def get_engine() -> HybridGeometryEngine:
    """Singleton accessor for global engine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = HybridGeometryEngine()
    return _engine_instance

async def compile_geometry_task(tree: List[Dict], format: str = "glb", mode: str = "standard") -> Dict[str, Any]:
    """
    Public API helper.
    """
    engine = get_engine()
    res = await engine.compile(tree, format=format, fidelity=mode)
    
    # Return serializable dict
    import base64
    return {
        "success": res.success,
        "error": res.error,
        "payload_base64": base64.b64encode(res.payload).decode('utf-8') if res.payload else None,
        "file_path": res.file_path,
        "metadata": res.metadata
    }

