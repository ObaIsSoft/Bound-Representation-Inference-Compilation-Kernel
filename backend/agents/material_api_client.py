"""
Material API Client - Dynamic material data from external sources

Architecture:
- Multiple API sources with priority order
- SQLite caching with TTL
- Automatic fallback chain
- Graceful degradation

Priority: MatWeb → NIST → Materials Project → Hardcoded
"""

import os
import json
import sqlite3
import hashlib
import asyncio
import aiohttp
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class APICacheEntry:
    """Cache entry for API responses"""
    query_hash: str
    source: str
    data: str  # JSON string
    timestamp: datetime
    ttl_hours: int = 168  # 1 week default
    
    def is_valid(self) -> bool:
        """Check if cache entry is still valid"""
        age = datetime.now() - self.timestamp
        return age < timedelta(hours=self.ttl_hours)


class MaterialAPICache:
    """SQLite-based cache for material API responses"""
    
    def __init__(self, cache_path: str = "./material_cache.db"):
        self.cache_path = Path(cache_path)
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite cache database"""
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS material_cache (
                    query_hash TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    data TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    ttl_hours INTEGER DEFAULT 168
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_source ON material_cache(source)
            """)
            conn.commit()
    
    def get(self, query_hash: str) -> Optional[APICacheEntry]:
        """Get cached entry if valid"""
        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.execute(
                "SELECT source, data, timestamp, ttl_hours FROM material_cache WHERE query_hash = ?",
                (query_hash,)
            )
            row = cursor.fetchone()
            
            if row:
                entry = APICacheEntry(
                    query_hash=query_hash,
                    source=row[0],
                    data=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    ttl_hours=row[3]
                )
                if entry.is_valid():
                    return entry
                else:
                    # Delete expired entry
                    conn.execute("DELETE FROM material_cache WHERE query_hash = ?", (query_hash,))
                    conn.commit()
        return None
    
    def set(self, query_hash: str, source: str, data: Dict, ttl_hours: int = 168):
        """Cache API response"""
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO material_cache 
                   (query_hash, source, data, timestamp, ttl_hours)
                   VALUES (?, ?, ?, ?, ?)""",
                (query_hash, source, json.dumps(data), datetime.now().isoformat(), ttl_hours)
            )
            conn.commit()
    
    def clear_expired(self):
        """Remove expired cache entries"""
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                "DELETE FROM material_cache WHERE datetime(timestamp, '+' || ttl_hours || ' hours') < datetime('now')"
            )
            conn.commit()


class MatWebClient:
    """
    MatWeb Material Property Database Client
    
    API: http://www.matweb.com/reference/apigateway.aspx
    Requires API key
    """
    
    BASE_URL = "http://www.matweb.com/api"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("MATWEB_API_KEY")
        self.cache = MaterialAPICache()
    
    async def search_material(self, designation: str) -> Optional[Dict[str, Any]]:
        """Search for material by designation"""
        if not self.api_key:
            logger.debug("MatWeb API key not configured")
            return None
        
        # Check cache
        query_hash = hashlib.md5(f"matweb:search:{designation}".encode()).hexdigest()
        cached = self.cache.get(query_hash)
        if cached:
            logger.info(f"MatWeb cache hit for {designation}")
            return json.loads(cached.data)
        
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "apikey": self.api_key,
                    "query": designation,
                    "action": "search"
                }
                async with session.get(self.BASE_URL, params=params, timeout=30) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.cache.set(query_hash, "matweb", data)
                        return data
                    elif resp.status == 429:
                        logger.warning("MatWeb rate limit exceeded")
                    else:
                        logger.warning(f"MatWeb API error: {resp.status}")
        except Exception as e:
            logger.error(f"MatWeb request failed: {e}")
        
        return None
    
    async def get_material_properties(self, matweb_id: str) -> Optional[Dict[str, Any]]:
        """Get full property set for material"""
        if not self.api_key:
            return None
        
        query_hash = hashlib.md5(f"matweb:props:{matweb_id}".encode()).hexdigest()
        cached = self.cache.get(query_hash)
        if cached:
            return json.loads(cached.data)
        
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "apikey": self.api_key,
                    "matid": matweb_id,
                    "action": "getproperties"
                }
                async with session.get(self.BASE_URL, params=params, timeout=30) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.cache.set(query_hash, "matweb", data)
                        return data
        except Exception as e:
            logger.error(f"MatWeb properties request failed: {e}")
        
        return None


class NISTClient:
    """
    NIST Material Database Client
    
    APIs:
    - NIST Structural Ceramics: https://www.nist.gov/programs-projects/structural-ceramics-database
    - NIST WebBook: https://webbook.nist.gov/chemistry/
    """
    
    CERAMICS_URL = "https://www.nist.gov/sites/default/files/documents/srd/ceramics.json"
    WEBBOOK_URL = "https://webbook.nist.gov/cgi/cbook.cgi"
    
    def __init__(self):
        self.cache = MaterialAPICache()
    
    async def get_ceramic_data(self, material_name: str) -> Optional[Dict[str, Any]]:
        """Get ceramic material data from NIST Structural Ceramics Database"""
        query_hash = hashlib.md5(f"nist:ceramic:{material_name}".encode()).hexdigest()
        cached = self.cache.get(query_hash)
        if cached:
            return json.loads(cached.data)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.CERAMICS_URL, timeout=30) as resp:
                    if resp.status == 200:
                        all_data = await resp.json()
                        # Search for material
                        for entry in all_data.get("materials", []):
                            if material_name.lower() in entry.get("name", "").lower():
                                self.cache.set(query_hash, "nist_ceramics", entry)
                                return entry
        except Exception as e:
            logger.error(f"NIST ceramics request failed: {e}")
        
        return None
    
    async def get_thermochemical_data(self, cas_number: str) -> Optional[Dict[str, Any]]:
        """
        Get thermochemical data from NIST WebBook
        
        Note: WebBook has thermochemical data (Cp, S, ΔHf), NOT mechanical properties
        """
        query_hash = hashlib.md5(f"nist:webbook:{cas_number}".encode()).hexdigest()
        cached = self.cache.get(query_hash)
        if cached:
            return json.loads(cached.data)
        
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "ID": cas_number,
                    "Units": "SI",
                    "Mask": "2",  # Thermodynamic data
                    "Type": "JSON"
                }
                async with session.get(self.WEBBOOK_URL, params=params, timeout=30) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.cache.set(query_hash, "nist_webbook", data)
                        return data
        except Exception as e:
            logger.error(f"NIST WebBook request failed: {e}")
        
        return None


class MaterialsProjectClient:
    """
    Materials Project DFT Database Client
    
    API: https://api.materialsproject.org
    Requires API key from https://materialsproject.org/api
    """
    
    BASE_URL = "https://api.materialsproject.org/materials/core"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("MATERIALS_PROJECT_API_KEY")
        self.cache = MaterialAPICache()
    
    async def search_by_formula(self, formula: str) -> Optional[Dict[str, Any]]:
        """Search for material by chemical formula"""
        if not self.api_key:
            logger.debug("Materials Project API key not configured")
            return None
        
        query_hash = hashlib.md5(f"mp:formula:{formula}".encode()).hexdigest()
        cached = self.cache.get(query_hash)
        if cached:
            return json.loads(cached.data)
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"X-API-KEY": self.api_key}
                params = {"formula": formula, "fields": "material_id,formula_pretty,band_gap"}
                async with session.get(
                    self.BASE_URL, 
                    headers=headers, 
                    params=params, 
                    timeout=30
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.cache.set(query_hash, "materials_project", data)
                        return data
                    elif resp.status == 429:
                        logger.warning("Materials Project rate limit exceeded")
        except Exception as e:
            logger.error(f"Materials Project request failed: {e}")
        
        return None
    
    async def get_elasticity(self, material_id: str) -> Optional[Dict[str, Any]]:
        """Get elastic constants from DFT calculations"""
        if not self.api_key:
            return None
        
        query_hash = hashlib.md5(f"mp:elastic:{material_id}".encode()).hexdigest()
        cached = self.cache.get(query_hash)
        if cached:
            return json.loads(cached.data)
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"X-API-KEY": self.api_key}
                url = f"https://api.materialsproject.org/materials/elasticity/{material_id}"
                async with session.get(url, headers=headers, timeout=30) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.cache.set(query_hash, "materials_project", data)
                        return data
        except Exception as e:
            logger.error(f"Materials Project elasticity request failed: {e}")
        
        return None


class MaterialAPIClient:
    """
    Unified material API client with fallback chain
    
    Priority:
    1. MatWeb (most comprehensive for engineering alloys)
    2. NIST Ceramics (for ceramics/composites)
    3. Materials Project (for DFT-calculated properties)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.matweb = MatWebClient(self.config.get("matweb_api_key"))
        self.nist = NISTClient()
        self.mp = MaterialsProjectClient(self.config.get("materials_project_api_key"))
        self.cache = MaterialAPICache()
    
    async def fetch_material(
        self,
        designation: str,
        category: str = "metal"
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch material data from best available source
        
        Args:
            designation: Material designation (UNS, AISI, etc.)
            category: "metal", "ceramic", "polymer", "composite"
        
        Returns:
            Material data with source attribution
        """
        result = None
        source = None
        
        # 1. Try MatWeb (best for metals)
        if category in ["metal", "alloy"]:
            result = await self.matweb.search_material(designation)
            if result:
                source = "matweb"
                logger.info(f"Found {designation} in MatWeb")
        
        # 2. Try NIST Ceramics (for ceramics)
        if not result and category in ["ceramic", "composite"]:
            result = await self.nist.get_ceramic_data(designation)
            if result:
                source = "nist_ceramics"
                logger.info(f"Found {designation} in NIST Ceramics")
        
        # 3. Try Materials Project (DFT data)
        if not result:
            # Convert designation to approximate formula
            formula = self._estimate_formula(designation)
            if formula:
                result = await self.mp.search_by_formula(formula)
                if result:
                    source = "materials_project"
                    logger.info(f"Found {designation} in Materials Project")
        
        if result:
            return {
                "data": result,
                "source": source,
                "query": designation,
                "timestamp": datetime.now().isoformat()
            }
        
        return None
    
    def _estimate_formula(self, designation: str) -> Optional[str]:
        """Rough estimation of chemical formula from designation"""
        # Very rough mapping for common materials
        mapping = {
            "6061": "AlMgSi",
            "7075": "AlZnMgCu",
            "4140": "FeCrMo",
            "304": "FeCrNi",
            "316": "FeCrNiMo",
            "ti6al4v": "TiAlV",
            "ti-6al-4v": "TiAlV",
        }
        
        designation_lower = designation.lower().replace(" ", "").replace("-", "")
        for key, formula in mapping.items():
            if key in designation_lower:
                return formula
        
        return None
    
    async def batch_fetch(
        self,
        designations: List[str],
        category: str = "metal"
    ) -> Dict[str, Optional[Dict]]:
        """Fetch multiple materials concurrently"""
        tasks = [self.fetch_material(d, category) for d in designations]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            d: r if not isinstance(r, Exception) else None
            for d, r in zip(designations, results)
        }


# Convenience functions
def get_material_api_client(config: Optional[Dict] = None) -> MaterialAPIClient:
    """Get configured material API client"""
    return MaterialAPIClient(config)


async def test_material_apis():
    """Test material API clients"""
    client = MaterialAPIClient()
    
    # Test materials
    test_materials = ["aluminum 6061", "steel 4140", "Ti-6Al-4V"]
    
    for mat in test_materials:
        print(f"\nTesting: {mat}")
        result = await client.fetch_material(mat)
        if result:
            print(f"  Source: {result['source']}")
            print(f"  Keys: {list(result['data'].keys())[:5]}")
        else:
            print("  Not found in any API")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_material_apis())
