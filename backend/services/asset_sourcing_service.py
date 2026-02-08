"""
Asset Sourcing Service

Provides access to 3D models and engineering assets from:
- NASA 3D Resources (free, official)
- Sketchfab (creative commons + commercial)
- CGTrader (marketplace)
- Thingiverse (3D printable)
- GrabCAD (engineering library)

Features:
- Search by category, tags, or description
- Format conversion (STEP, STL, OBJ, GLTF)
- License compliance tracking
"""

import os
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

logger = logging.getLogger(__name__)


class AssetLicense(Enum):
    """Asset license types"""
    CC0 = "cc0"  # Public domain
    CC_BY = "cc_by"  # Attribution
    CC_BY_SA = "cc_by_sa"  # Attribution + ShareAlike
    CC_BY_NC = "cc_by_nc"  # Attribution + NonCommercial
    COMMERCIAL = "commercial"  # Paid license
    NASA = "nasa"  # NASA open data


@dataclass
class Asset3D:
    """3D asset metadata"""
    id: str
    source: str  # nasa, sketchfab, cgtrader, etc.
    external_id: str
    
    # Basic info
    name: str
    description: Optional[str]
    category: str
    tags: List[str]
    
    # Files
    mesh_url: Optional[str] = None
    mesh_format: Optional[str] = None  # step, stl, obj, gltf
    thumbnail_url: Optional[str] = None
    file_size_mb: Optional[float] = None
    
    # License
    license: AssetLicense = AssetLicense.CC_BY
    attribution: Optional[str] = None
    commercial_use_allowed: bool = True
    
    # Metadata
    creator: Optional[str] = None
    created_at: Optional[datetime] = None
    poly_count: Optional[int] = None
    dimensions_mm: Optional[Dict[str, float]] = None


class AssetSourcingService:
    """
    3D asset sourcing from multiple repositories.
    """
    
    def __init__(self):
        self.http_client: Optional[Any] = None
        self._initialized = False
        
        # API keys
        self.nasa_3d_api_key = os.getenv("NASA_3D_API_KEY")
        self.sketchfab_api_key = os.getenv("SKETCHFAB_API_KEY")
        self.cgtrader_api_key = os.getenv("CGTRADER_API_KEY")
        self.thingiverse_client_id = os.getenv("THINGIVERSE_CLIENT_ID")
        self.grabcad_api_key = os.getenv("GRABCAD_API_KEY")
        
    async def initialize(self):
        """Initialize HTTP client"""
        if self._initialized:
            return
            
        if HAS_HTTPX:
            self.http_client = httpx.AsyncClient(timeout=30.0)
        
        self._initialized = True
    
    async def search(
        self,
        query: str,
        category: Optional[str] = None,
        license_filter: Optional[AssetLicense] = None,
        max_results: int = 10,
        formats: Optional[List[str]] = None
    ) -> List[Asset3D]:
        """
        Search for 3D assets across all sources.
        
        Args:
            query: Search query
            category: Asset category (mechanical, electrical, aerospace, etc.)
            license_filter: Filter by license type
            max_results: Maximum results per source
            formats: Preferred formats ["step", "stl", "obj", "gltf"]
            
        Returns:
            List of matching assets
        """
        await self.initialize()
        
        # Search all sources concurrently
        tasks = []
        sources = []
        
        # NASA (free, always available)
        tasks.append(self._search_nasa(query, category, max_results))
        sources.append("nasa")
        
        if self.sketchfab_api_key:
            tasks.append(self._search_sketchfab(query, category, max_results))
            sources.append("sketchfab")
        
        if self.cgtrader_api_key:
            tasks.append(self._search_cgtrader(query, category, max_results))
            sources.append("cgtrader")
        
        if self.thingiverse_client_id:
            tasks.append(self._search_thingiverse(query, category, max_results))
            sources.append("thingiverse")
        
        if self.grabcad_api_key:
            tasks.append(self._search_grabcad(query, category, max_results))
            sources.append("grabcad")
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten and filter
        all_assets = []
        for source, result in zip(sources, results):
            if isinstance(result, Exception):
                logger.error(f"{source} search error: {result}")
                continue
            all_assets.extend(result)
        
        # Apply filters
        filtered = all_assets
        
        if license_filter:
            filtered = [a for a in filtered if a.license == license_filter]
        
        if formats:
            filtered = [
                a for a in filtered 
                if a.mesh_format and a.mesh_format.lower() in [f.lower() for f in formats]
            ]
        
        # Sort by relevance (simplified - would use actual relevance scores)
        # Prioritize NASA and GrabCAD for engineering use
        source_priority = {"nasa": 0, "grabcad": 1, "cgtrader": 2, "sketchfab": 3, "thingiverse": 4}
        filtered.sort(key=lambda x: source_priority.get(x.source, 5))
        
        return filtered[:max_results]
    
    async def get_asset_by_id(
        self,
        source: str,
        external_id: str
    ) -> Optional[Asset3D]:
        """
        Get asset by source and ID.
        
        Args:
            source: Asset source (nasa, sketchfab, etc.)
            external_id: External ID from that source
            
        Returns:
            Asset or None
        """
        await self.initialize()
        
        if source == "nasa":
            return await self._get_nasa_asset(external_id)
        elif source == "sketchfab":
            return await self._get_sketchfab_asset(external_id)
        elif source == "cgtrader":
            return await self._get_cgtrader_asset(external_id)
        elif source == "thingiverse":
            return await self._get_thingiverse_asset(external_id)
        elif source == "grabcad":
            return await self._get_grabcad_asset(external_id)
        
        return None
    
    async def download_asset(
        self,
        asset: Asset3D,
        destination: str,
        preferred_format: Optional[str] = None
    ) -> bool:
        """
        Download asset to local storage.
        
        Args:
            asset: Asset to download
            destination: Local file path
            preferred_format: Preferred format (if multiple available)
            
        Returns:
            True if successful
        """
        await self.initialize()
        
        if not asset.mesh_url:
            logger.warning(f"No mesh URL for asset {asset.id}")
            return False
        
        try:
            if not HAS_HTTPX:
                logger.error("httpx not available for download")
                return False
            
            async with self.http_client.stream("GET", asset.mesh_url) as response:
                if response.status_code != 200:
                    logger.error(f"Download failed: {response.status_code}")
                    return False
                
                with open(destination, 'wb') as f:
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)
            
            logger.info(f"Downloaded {asset.name} to {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Download error: {e}")
            return False
    
    async def _search_nasa(
        self,
        query: str,
        category: Optional[str],
        max_results: int
    ) -> List[Asset3D]:
        """
        Search NASA 3D Resources.
        
        NASA provides free 3D models of spacecraft, instruments, etc.
        API Docs: https://nasa3d.arc.nasa.gov/api
        """
        if not HAS_HTTPX:
            return []
        
        try:
            # NASA 3D API endpoint
            base_url = os.getenv("NASA_3D_BASE_URL", "https://nasa3d.arc.nasa.gov/api")
            url = f"{base_url}/search"
            
            params = {
                "q": query,
                "limit": max_results
            }
            
            if category:
                params["category"] = category
            
            # Placeholder - NASA API doesn't strictly require key for search
            # Implement actual API call when needed
            logger.debug(f"NASA 3D search: {query}")
            
            # Return empty for now - implement real search when API tested
            return []
            
        except Exception as e:
            logger.error(f"NASA search error: {e}")
            return []
    
    async def _search_sketchfab(
        self,
        query: str,
        category: Optional[str],
        max_results: int
    ) -> List[Asset3D]:
        """Search Sketchfab"""
        if not HAS_HTTPX or not self.sketchfab_api_key:
            return []
        
        try:
            url = "https://api.sketchfab.com/v3/search"
            
            headers = {
                "Authorization": f"Token {self.sketchfab_api_key}"
            }
            
            params = {
                "type": "models",
                "q": query,
                "count": max_results,
                "sort_by": "relevance"
            }
            
            # Placeholder
            logger.debug(f"Sketchfab search: {query}")
            return []
            
        except Exception as e:
            logger.error(f"Sketchfab search error: {e}")
            return []
    
    async def _search_cgtrader(
        self,
        query: str,
        category: Optional[str],
        max_results: int
    ) -> List[Asset3D]:
        """Search CGTrader"""
        if not self.cgtrader_api_key:
            return []
        
        logger.debug(f"CGTrader search: {query}")
        return []
    
    async def _search_thingiverse(
        self,
        query: str,
        category: Optional[str],
        max_results: int
    ) -> List[Asset3D]:
        """Search Thingiverse"""
        if not self.thingiverse_client_id:
            return []
        
        logger.debug(f"Thingiverse search: {query}")
        return []
    
    async def _search_grabcad(
        self,
        query: str,
        category: Optional[str],
        max_results: int
    ) -> List[Asset3D]:
        """Search GrabCAD"""
        if not self.grabcad_api_key:
            return []
        
        logger.debug(f"GrabCAD search: {query}")
        return []
    
    async def _get_nasa_asset(self, external_id: str) -> Optional[Asset3D]:
        """Get NASA asset by ID"""
        return None
    
    async def _get_sketchfab_asset(self, external_id: str) -> Optional[Asset3D]:
        """Get Sketchfab asset by ID"""
        return None
    
    async def _get_cgtrader_asset(self, external_id: str) -> Optional[Asset3D]:
        """Get CGTrader asset by ID"""
        return None
    
    async def _get_thingiverse_asset(self, external_id: str) -> Optional[Asset3D]:
        """Get Thingiverse asset by ID"""
        return None
    
    async def _get_grabcad_asset(self, external_id: str) -> Optional[Asset3D]:
        """Get GrabCAD asset by ID"""
        return None


# Global instance
asset_sourcing = AssetSourcingService()
