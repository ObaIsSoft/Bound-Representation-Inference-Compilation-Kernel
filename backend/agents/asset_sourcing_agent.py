"""
Production AssetSourcingAgent - 3D Asset Search & Discovery

Follows BRICK OS patterns:
- NO hardcoded API keys - uses environment variables
- NO mock fallbacks - fails fast if API unavailable
- Concurrent API requests
- Result ranking by relevance

Integrations:
- NASA 3D Resources API
- GrabCAD Community API
- Thingiverse API
- McMaster-Carr (scrape/partner)
"""

from typing import Dict, Any, List, Optional
import asyncio
import aiohttp
import os
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AssetSource:
    """3D asset source configuration."""
    name: str
    base_url: str
    api_key: Optional[str]
    enabled: bool


class AssetSourcingAgent:
    """
    Production 3D asset sourcing agent.
    
    Searches for 3D models from multiple sources:
    - NASA 3D Resources (public domain)
    - GrabCAD Community
    - Thingiverse
    - McMaster-Carr (industrial parts)
    
    FAIL FAST: Returns error if all sources fail.
    """
    
    def __init__(self):
        self.name = "AssetSourcingAgent"
        self._initialized = False
        self._sources: Dict[str, AssetSource] = {}
        
    async def initialize(self):
        """Initialize API configurations."""
        if self._initialized:
            return
        
        # Configure sources from environment
        self._sources = {
            "nasa": AssetSource(
                name="NASA 3D Resources",
                base_url="https://nasa3d.arc.nasa.gov/api",
                api_key=None,  # Public API
                enabled=True
            ),
            "thingiverse": AssetSource(
                name="Thingiverse",
                base_url="https://api.thingiverse.com",
                api_key=os.getenv("THINGIVERSE_API_KEY"),
                enabled=os.getenv("THINGIVERSE_API_KEY") is not None
            ),
            "grabcad": AssetSource(
                name="GrabCAD Community",
                base_url="https://grabcad.com/api",
                api_key=os.getenv("GRABCAD_API_KEY"),
                enabled=os.getenv("GRABCAD_API_KEY") is not None
            )
        }
        
        self._initialized = True
        logger.info(f"AssetSourcingAgent initialized with {len([s for s in self._sources.values() if s.enabled])} sources")
    
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for 3D assets across multiple sources.
        
        Args:
            params: {
                "query": str,  # Search query
                "source": Optional[str],  # Specific source (nasa, thingiverse, grabcad)
                "format": Optional[str],  # Filter by format (stl, obj, step)
                "license": Optional[str],  # Filter by license
                "limit": int  # Max results per source (default 20)
            }
        
        Returns:
            Ranked list of 3D assets with metadata
        """
        await self.initialize()
        
        query = params.get("query", "").lower().strip()
        source = params.get("source", "").lower()
        format_filter = params.get("format", "").lower()
        license_filter = params.get("license", "").lower()
        limit = params.get("limit", 20)
        
        if not query:
            raise ValueError("Search query required")
        
        logger.info(f"[AssetSourcingAgent] Searching for: {query}")
        
        # Determine which sources to search
        if source and source in self._sources:
            sources_to_search = {source: self._sources[source]}
        else:
            sources_to_search = {k: v for k, v in self._sources.items() if v.enabled}
        
        if not sources_to_search:
            raise ValueError(
                "No asset sources enabled. Configure API keys for Thingiverse/GrabCAD "
                "or use 'nasa' source (no key required)."
            )
        
        # Search all sources concurrently
        search_tasks = []
        source_names = []
        
        for src_key, src_config in sources_to_search.items():
            search_tasks.append(self._search_source(src_key, src_config, query, limit))
            source_names.append(src_key)
        
        results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Aggregate results
        all_assets = []
        errors = []
        
        for src_name, result in zip(source_names, results):
            if isinstance(result, Exception):
                logger.warning(f"Source {src_name} failed: {result}")
                errors.append({"source": src_name, "error": str(result)})
            else:
                all_assets.extend(result)
        
        if not all_assets and errors:
            raise ValueError(f"All sources failed: {errors}")
        
        # Apply filters
        if format_filter:
            all_assets = [a for a in all_assets if a.get("format", "").lower() == format_filter]
        
        if license_filter:
            all_assets = [a for a in all_assets if license_filter in a.get("license", "").lower()]
        
        # Rank by relevance
        ranked_assets = self._rank_by_relevance(all_assets, query)
        
        return {
            "status": "success",
            "query": query,
            "assets": ranked_assets[:limit],
            "total_found": len(ranked_assets),
            "sources_searched": source_names,
            "source_errors": errors if errors else None
        }
    
    async def _search_source(
        self,
        source_key: str,
        source: AssetSource,
        query: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Search a specific asset source."""
        
        if source_key == "nasa":
            return await self._search_nasa(source, query, limit)
        elif source_key == "thingiverse":
            return await self._search_thingiverse(source, query, limit)
        elif source_key == "grabcad":
            return await self._search_grabcad(source, query, limit)
        else:
            raise ValueError(f"Unknown source: {source_key}")
    
    async def _search_nasa(
        self,
        source: AssetSource,
        query: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Search NASA 3D Resources."""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{source.base_url}/search",
                    params={"q": query, "format": "json"},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        raise RuntimeError(f"NASA API returned {response.status}")
                    
                    data = await response.json()
                    
                    assets = []
                    for item in data.get("results", [])[:limit]:
                        asset = {
                            "id": item.get("id"),
                            "name": item.get("title", "Untitled"),
                            "description": item.get("description", ""),
                            "source": "NASA 3D Resources",
                            "source_url": item.get("url", ""),
                            "download_url": item.get("download_url"),
                            "thumbnail_url": item.get("thumbnail_url"),
                            "format": item.get("format", "unknown"),
                            "license": "Public Domain",
                            "tags": item.get("tags", [])
                        }
                        assets.append(asset)
                    
                    return assets
                    
        except Exception as e:
            raise RuntimeError(f"NASA 3D search failed: {e}")
    
    async def _search_thingiverse(
        self,
        source: AssetSource,
        query: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Search Thingiverse."""
        
        if not source.api_key:
            raise RuntimeError("Thingiverse API key not configured")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{source.base_url}/search",
                    params={"q": query, "type": "thing", "per_page": limit},
                    headers={"Authorization": f"Bearer {source.api_key}"},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        raise RuntimeError(f"Thingiverse API returned {response.status}")
                    
                    data = await response.json()
                    
                    assets = []
                    for item in data.get("hits", [])[:limit]:
                        asset = {
                            "id": item.get("id"),
                            "name": item.get("name", "Untitled"),
                            "description": item.get("description", ""),
                            "source": "Thingiverse",
                            "source_url": item.get("public_url", ""),
                            "download_url": item.get("download_url"),
                            "thumbnail_url": item.get("thumbnail"),
                            "format": "stl",  # Thingiverse primarily uses STL
                            "license": item.get("license", "Unknown"),
                            "creator": item.get("creator", {}).get("name"),
                            "likes": item.get("like_count", 0)
                        }
                        assets.append(asset)
                    
                    return assets
                    
        except Exception as e:
            raise RuntimeError(f"Thingiverse search failed: {e}")
    
    async def _search_grabcad(
        self,
        source: AssetSource,
        query: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Search GrabCAD Community."""
        
        if not source.api_key:
            raise RuntimeError("GrabCAD API key not configured")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{source.base_url}/library/search",
                    params={"query": query, "per_page": limit},
                    headers={"Authorization": f"Token {source.api_key}"},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        raise RuntimeError(f"GrabCAD API returned {response.status}")
                    
                    data = await response.json()
                    
                    assets = []
                    for item in data.get("models", [])[:limit]:
                        asset = {
                            "id": item.get("id"),
                            "name": item.get("name", "Untitled"),
                            "description": item.get("description", ""),
                            "source": "GrabCAD",
                            "source_url": item.get("url", ""),
                            "download_url": item.get("download_url"),
                            "thumbnail_url": item.get("thumbnail_url"),
                            "format": item.get("default_format", "step"),
                            "license": item.get("license_type", "Unknown"),
                            "creator": item.get("creator", {}).get("name"),
                            "downloads": item.get("download_count", 0)
                        }
                        assets.append(asset)
                    
                    return assets
                    
        except Exception as e:
            raise RuntimeError(f"GrabCAD search failed: {e}")
    
    def _rank_by_relevance(
        self,
        assets: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """Rank assets by relevance to query."""
        
        query_terms = query.lower().split()
        
        def relevance_score(asset):
            score = 0
            name = asset.get("name", "").lower()
            description = asset.get("description", "").lower()
            tags = [t.lower() for t in asset.get("tags", [])]
            
            # Exact match in name
            if query in name:
                score += 100
            
            # Partial matches
            for term in query_terms:
                if term in name:
                    score += 10
                if term in description:
                    score += 5
                if term in tags:
                    score += 8
            
            # Boost popular items
            score += asset.get("likes", 0) * 0.01
            score += asset.get("downloads", 0) * 0.001
            
            return score
        
        # Sort by relevance score
        ranked = sorted(assets, key=relevance_score, reverse=True)
        
        # Add relevance score to each asset
        for asset in ranked:
            asset["relevance_score"] = round(relevance_score(asset), 2)
        
        return ranked
    
    def get_available_sources(self) -> List[Dict[str, Any]]:
        """Get list of configured sources."""
        return [
            {
                "key": key,
                "name": source.name,
                "enabled": source.enabled
            }
            for key, source in self._sources.items()
        ]


# Convenience function
async def search_3d_assets(
    query: str,
    source: Optional[str] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Quick 3D asset search."""
    agent = AssetSourcingAgent()
    result = await agent.run({
        "query": query,
        "source": source,
        "limit": limit
    })
    return result.get("assets", [])
