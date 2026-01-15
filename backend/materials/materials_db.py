"""
Materials Database Helper Class
Provides easy access to comprehensive materials database.
Combines Supabase (Cloud) and External APIs.
Strictly NO Local SQLite Fallback.
"""
import json
import logging
from typing import Dict, List, Optional, Any
from .materials_api import UnifiedMaterialsAPI

try:
    from database.supabase_client import SupabaseClientWrapper
except ImportError:
    # Fallback to relative import if running as module
    from ...database.supabase_client import SupabaseClientWrapper

logger = logging.getLogger(__name__)

class MaterialsDatabase:
    """Helper class for querying materials database."""
    
    def __init__(self, use_api: bool = True, api_key: str = None):
        self.use_api = use_api
        # Initialize Supabase
        self.supabase = SupabaseClientWrapper()
        
        # Initialize External API
        if use_api:
            self.api = UnifiedMaterialsAPI(api_key)
        else:
            self.api = None
            
    def _query_supabase(self, table: str, query: Dict[str, Any]) -> Optional[Dict]:
        """Helper to query Supabase."""
        if not self.supabase.enabled: return None
        try:
            params = self.supabase.client.table(table).select("*")
            for k, v in query.items():
                params = params.eq(k, v)
            res = params.execute()
            if res.data: return res.data[0]
        except Exception as e:
            logger.warning(f"Supabase query query error ({table}): {e}")
        return None

    def get_element(self, symbol: str) -> Optional[Dict]:
        """Get element by symbol."""
        # 1. Try Supabase (Source of Truth)
        sb_res = self._query_supabase("elements", {"symbol": symbol})
        if sb_res: return {**sb_res, "_source": "supabase"}

        # 2. Try External API
        if self.use_api and self.api:
            api_data = self.api.get_element_data(symbol)
            if api_data: return {
                "symbol": symbol,
                "name": api_data.get("formula_pretty", symbol),
                "density": api_data.get("density"),
                "_source": "api"
            }
        return None
    
    def get_alloy(self, alloy_id: str) -> Optional[Dict]:
        """Get alloy by ID."""
        # 1. Supabase
        sb_res = self._query_supabase("alloys", {"id": alloy_id})
        if sb_res: 
             # Ensure composition is dict
             if isinstance(sb_res.get("composition"), str):
                 sb_res["composition"] = json.loads(sb_res["composition"])
             return {**sb_res, "_source": "supabase"}
        return None
    
    def get_compound(self, compound_id: str) -> Optional[Dict]:
        """Get compound by ID."""
        sb_res = self._query_supabase("compounds", {"id": compound_id})
        if sb_res: return {**sb_res, "_source": "supabase"}
        return None

    def search_alloys(self, category: str = None, min_strength: float = None) -> List[Dict]:
        """Search alloys."""
        results = []
        # Supabase
        if self.supabase.enabled:
            try:
                base = self.supabase.client.table("alloys").select("*")
                if category: base = base.eq("category", category)
                if min_strength: base = base.gte("yield_strength", min_strength)
                res = base.execute()
                for item in res.data:
                    if isinstance(item.get("composition"), str):
                        item["composition"] = json.loads(item["composition"])
                    item["_source"] = "supabase"
                results = res.data
            except Exception as e:
                logger.warning(f"Supabase search error: {e}")
        return results
    
    def find_material(self, name: str) -> Optional[Dict]:
        """Find material by name."""
        # Element
        elem = self.get_element(name)
        if elem: return {"type": "element", "data": elem}
        # Alloy
        alloy = self.get_alloy(name)
        if alloy: return {"type": "alloy", "data": alloy}
        # Compound
        compound = self.get_compound(name)
        if compound: return {"type": "compound", "data": compound}
        
        # Fuzzy Search Supabase
        if self.supabase.enabled:
            try:
                # Basic fuzzy search approach
                res = self.supabase.client.table("alloys").select("*").ilike("name", f"%{name}%").execute()
                if res.data:
                    d = res.data[0]
                    if isinstance(d.get("composition"), str): d["composition"] = json.loads(d["composition"])
                    d["_source"] = "supabase_fuzzy"
                    return {"type": "alloy", "data": d}
            except: pass
        
        return None

    def get_property_at_temperature(self, material_id: str, property_name: str, temperature_k: float) -> Optional[float]:
        """
        Get material property at specific temperature.
        Interpolates linearly.
        """
        points = []
        
        # 1. Supabase Properties
        if self.supabase.enabled:
            try:
                res = self.supabase.client.table("properties") \
                    .select("temperature, property_value") \
                    .eq("material_id", material_id) \
                    .eq("property_name", property_name) \
                    .order("temperature") \
                    .execute()
                points = [(r['temperature'], r['property_value']) for r in res.data]
            except: pass

        if not points:
             # Static Failover
             mat = self.find_material(material_id)
             if mat and isinstance(mat.get('data'), dict):
                  val = mat['data'].get(property_name)
                  if val is not None: return float(val)
             return None
                
        # Interpolation Logic
        for t, v in points:
            if abs(t - temperature_k) < 0.01: return v
                
        if temperature_k <= points[0][0]: return points[0][1]
        if temperature_k >= points[-1][0]: return points[-1][1]
            
        for i in range(len(points) - 1):
            t1, v1 = points[i]
            t2, v2 = points[i+1]
            if t1 < temperature_k < t2:
                return v1 + (temperature_k - t1) * (v2 - v1) / (t2 - t1)
        return None

    def get_monomers(self) -> Dict[str, Any]:
        """Fetch all monomers (for PolymersAdapter)."""
        monomers = {}
        # 1. Supabase
        if self.supabase.enabled:
             try:
                 rows = self.supabase.fetch_table("monomers")
                 if rows:
                     for r in rows:
                         monomers[r["id"]] = r # Store full row to support flexible properties
                     return monomers
             except: pass
        return monomers

    def get_ballistic_threats(self) -> Dict[str, Any]:
        """Fetch ballistic threats (for PolymersAdapter)."""
        threats = {}
        # 1. Supabase
        if self.supabase.enabled:
             try:
                 rows = self.supabase.fetch_table("ballistic_threats")
                 if rows:
                     for r in rows:
                        threats[r["id"]] = {"mass_g": r["mass_g"], "velocity_mps": r["velocity_mps"]}
                     return threats
             except: pass
        return threats

    def get_kinetics(self, material_family: str) -> Optional[Dict[str, float]]:
        """Fetch reaction kinetics parameters for ChemistryAgent."""
        if self.supabase.enabled:
            try:
                res = self._query_supabase("kinetics", {"material_family": material_family})
                if res: return res
            except: pass
        return None
