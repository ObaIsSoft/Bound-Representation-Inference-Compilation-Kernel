
from typing import Dict, Any, List, Optional
import logging
import json
from database.supabase_client import SupabaseClientWrapper
from models.component import Component, ComponentInstance

logger = logging.getLogger(__name__)

class ComponentAgent:
    """
    Component Agent - Parts Selection & Universal Catalog.
    
    Selects COTS (Commercial Off-The-Shelf) components from the Universal Catalog.
    Supports Goal-Oriented Search (requirements -> components).
    """
    
    def __init__(self):
        self.name = "ComponentAgent"
        self.db = SupabaseClientWrapper()

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute component selection based on requirements.
        
        Args:
            params: {
                "category": str (optional),
                "requirements": Dict[str, Any], # e.g. {"min_power_w": 100, "max_cost": 50}
                "limit": int,
                "volatility": float # 0.0-1.0 for stochastic instantiation
            }
        """
        category = params.get("category")
        requirements = params.get("requirements", params.get("specs", {})) # Support legacy 'specs' key
        limit = params.get("limit", 5)
        volatility = params.get("volatility", 0.0)
        
        logs = [f"[COMPONENT] Search Requirements: {requirements} (Category: {category or 'ALL'})"]
        
        # 1. Fetch Candidates (Broad Filter)
        candidates = self._fetch_candidates(category)
        if not candidates:
            return {
                "selection": [],
                "count": 0,
                "logs": logs + ["[COMPONENT] No candidates found in catalog."]
            }
            
        logs.append(f"[COMPONENT] Found {len(candidates)} candidates. Filtering...")
        
        # 2. Filter & Instantiate (Deep Filter)
        valid_matches = []
        for comp_model in candidates:
            # Instantiate with 0 volatility for filtering (Nominal values)
            # We filter on nominal, but return stochastic instances if requested?
            # Usually we filter on nominal to ensure fit.
            nominal_instance = comp_model.instantiate(volatility=0.0)
            
            if self._satisfies_requirements(nominal_instance, requirements):
                # If valid, create the final instance (potentially stochastic)
                final_instance = comp_model.instantiate(volatility=volatility)
                valid_matches.append(final_instance)
        
        # 3. Sort / Rank? (Simple truncation for now)
        selection = valid_matches[:limit]
        
        logs.append(f"[COMPONENT] Selected {len(selection)} components.")
        
        return {
            "selection": [self._serialize_instance(inst) for inst in selection],
            "count": len(selection),
            "logs": logs
        }

    def _fetch_candidates(self, category: Optional[str]) -> List[Component]:
        """Fetch raw rows from Supabase and convert to Component models."""
        if not self.db.enabled:
            return []
            
        try:
            # Basic select
            query = self.db.client.table("components").select("*")
            if category:
                query = query.eq("category", category)
                
            res = query.execute()
            return [Component(row) for row in res.data]
        except Exception as e:
            logger.error(f"Supabase fetch error: {e}")
            return []

    def _satisfies_requirements(self, instance: ComponentInstance, reqs: Dict[str, Any]) -> bool:
        """Check if an instance meets all requirements."""
        for key, target_val in reqs.items():
            # Handle min/max constraints
            if key.startswith("min_"):
                attr = key[4:]
                val = self._get_attr_recursive(instance, attr)
                if val is None or float(val) < float(target_val): return False
                
            elif key.startswith("max_"):
                attr = key[4:]
                val = self._get_attr_recursive(instance, attr)
                if val is None or float(val) > float(target_val): return False
                
            elif key.startswith("eq_") or key == "name": # Exact match
                attr = key[3:] if key.startswith("eq_") else key
                val = self._get_attr_recursive(instance, attr)
                if str(val) != str(target_val): return False
                
            else:
                # Default to exact match or 'contains' for strings?
                # For now assume min requirement if numeric? No, usually constraints are explicit.
                # If implicit key (e.g. "kv"), assume it's a target? No, filtering needs operators.
                # Legacy behavior: keys in 'specs' were often min/max logic handled by prefix.
                # If no prefix, check equality.
                val = self._get_attr_recursive(instance, key)
                if val != target_val: return False
                
        return True

    def _get_attr_recursive(self, instance: ComponentInstance, key: str) -> Any:
        """Helper to find attribute in instance fields or specs dict."""
        # 1. Try direct attributes (mass_g, cost_usd)
        if hasattr(instance, key):
            return getattr(instance, key)
            
        # 2. Try specs dictionary
        if key in instance.specs:
            return instance.specs[key]
            
        # 3. Try parsing nested keys? (future)
        return None

    def _serialize_instance(self, instance: ComponentInstance) -> Dict:
        """Convert objects back to dict for generic agent response."""
        return {
            "id": instance.catalog_id,
            "name": instance.name,
            "category": instance.category,
            "mass_g": instance.mass_g,
            "cost_usd": instance.cost_usd,
            "specs": instance.specs,
            "geometry_def": instance.geometry_def
        }
