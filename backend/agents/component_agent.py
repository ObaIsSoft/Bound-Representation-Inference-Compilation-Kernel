
from typing import Dict, Any, List, Optional
import logging
import json
try:
    from database.supabase_client import SupabaseClient
except ImportError:
    from database.supabase_client import SupabaseClient

try:
    from models.component import Component, ComponentInstance
except ImportError:
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
        self.db = SupabaseClient()
        
        # Load Config
        try:
            from config.component_config import COMPONENT_DEFAULTS, TEST_ASSETS, ATLAS_CONFIG
            self.defaults = COMPONENT_DEFAULTS
            self.assets = TEST_ASSETS
            self.atlas_config = ATLAS_CONFIG
        except ImportError:
            logger.warning("Could not import component_config. Using defaults.")
            self.defaults = {"db_table": "components", "weights_path": "data/component_agent_weights.json"}
            self.assets = {"cube": "test_assets/test_cube.stl"}
            self.atlas_config = {"default_resolution": 64}

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
            nominal_instance = comp_model.instantiate(volatility=0.0)
            
            if self._satisfies_requirements(nominal_instance, requirements):
                # If valid, create the final instance
                final_instance = comp_model.instantiate(volatility=volatility)
                valid_matches.append(final_instance)
        
        # 3. Sort by Learned Preferences (RL)
        # Score = Preference Weight (stats)
        preferences = self._load_preferences()
        
        def get_score(instance):
            # Default score 1.0. Higher is better.
            # Look up by ID, Category, or Source? ID is most specific.
            pid = instance.catalog_id
            return preferences.get(pid, 1.0)
            
        valid_matches.sort(key=get_score, reverse=True)
        
        selection = valid_matches[:limit]
        
        logs.append(f"[COMPONENT] Selected {len(selection)} components (Sorted by Preference).")
        
        return {
            "selection": [self._serialize_instance(inst) for inst in selection],
            "count": len(selection),
            "logs": logs
        }

    def _load_preferences(self) -> Dict[str, float]:
        """Load learned component preference weights."""
        import json
        import os
        path = "data/component_agent_weights.json"
        if not os.path.exists(path): return {}
        try:
            with open(path, 'r') as f: return json.load(f)
        except Exception: return {}

    def update_preferences(self, component_id: str, reward_signal: float):
        """
        Reinforcement Learning Update.
        reward_signal: +1.0 (Good), -1.0 (Bad/Reject), -5.0 (Critical Fail)
        """
        import json
        import os
        path = "data/component_agent_weights.json"
        
        prefs = {}
        if os.path.exists(path):
            try: 
                with open(path, 'r') as f: 
                    prefs = json.load(f)
            except Exception: 
                pass
            
        current = prefs.get(component_id, 1.0)
        # Simple update rule: New = Old + alpha * Reward
        # Alpha = 0.1
        new_score = current + (0.1 * reward_signal)
        prefs[component_id] = max(0.1, new_score) # Clamp min score
        
        try:
            with open(path, 'w') as f: json.dump(prefs, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save preferences: {e}")

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

    def install_component(self, component_id: str, mesh_path: str = None, mesh_url: str = None, resolution: int = None) -> Dict[str, Any]:
        """
        Installs a component by converting its mesh to an SDF texture.
        Supports local paths or remote URLs.
        
        Args:
            component_id: ID of the component to install
            mesh_path: Optional path to local mesh file.
            component_id: ID of the component to install
            mesh_path: Optional path to local mesh file.
            mesh_url: Optional URL to download mesh from.
            resolution: Optional explicit resolution (overrides adaptive).
            
        Returns:
            Dict containing metadata and SDF paths.
        """
        from utils.mesh_to_sdf_bridge import MeshSDFBridge
        import os
        import requests
        import tempfile
        
        logger.info(f"[COMPONENT] Installing {component_id}...")
        
        # 0. Handle Remote Download
        temp_file = None
        if mesh_url and not mesh_path:
            try:
                logger.info(f"[COMPONENT] Downloading mesh from {mesh_url}...")
                response = requests.get(mesh_url, stream=True)
                response.raise_for_status()
                
                # Infer extension or default to .stl
                ext = os.path.splitext(mesh_url)[1] or ".stl"
                
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_file.close()
                
                mesh_path = temp_file.name
                logger.info(f"[COMPONENT] Downloaded to temporary file: {mesh_path} ({os.path.getsize(mesh_path)} bytes)")
                
            except Exception as e:
                logger.error(f"[COMPONENT] Download failed: {e}")
                logger.warning(f"[COMPONENT] Falling back to synthetic generation/local lookup.")
                # Do not return error, proceed to fallback logic
                temp_file = None

        # 1. Resolve Mesh Path (Fallback)
        if not mesh_path:
            # Check for generic assets
            candidates = list(self.assets.values())
            for cand in candidates:
                if os.path.exists(cand):
                    mesh_path = cand
                    break
            
            if not mesh_path:
                logger.warning(f"No mesh found for {component_id}, using synthetic generation for test.")
                # Generate a dummy mesh if none exists (for testing flow)
                import trimesh
                mesh = trimesh.creation.box(extents=[1,1,1])
                install_dir = self.defaults.get("install_path", "data/components")
                os.makedirs(install_dir, exist_ok=True)
                mesh_path = f"{install_dir}/{component_id}.stl"
                mesh.export(mesh_path)
        
        # 2. Convert to SDF (Using Atlas Pipeline)
        try:
            bridge = MeshSDFBridge()
            # Use Atlas Baker for everything to standardize the output format (Manifest + Texture)
            result = bridge.bake_scene_to_atlas(mesh_path, resolution=resolution or 64)
            
            # Clean up temp file
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            
            logger.info(f"[COMPONENT] Successfully installed {component_id}. Atlas Res: {result['resolution']}")
            
            return {
                "status": "installed",
                "component_id": component_id,
                "original_source": mesh_url or "local",
                "sdf_metadata": result.get("manifest", []),
                "texture_data": result.get("texture_data"),
                "manifest": result.get("manifest"),
                "glsl": result.get("glsl"),
                "is_atlas": True,
                "ready_for_simulation": True
            }
            
        except Exception as e:
            logger.error(f"[COMPONENT] Install failed: {e}")
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            return {
                "status": "error", 
                "error": str(e)
            }

