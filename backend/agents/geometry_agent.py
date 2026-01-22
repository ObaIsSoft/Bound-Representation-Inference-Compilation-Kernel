import os
import re
import logging
import math
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class GeometryAgent:
    """
    Generates KCL code and 3D geometry from design intent.
    Features:
    - Parametric Generation (Rules-based)
    - Procedural Component Library ("Digital Twin" parts)
    - Multi-Regime Sizing (Aerial, Ground, Space)
    - KittyCAD Zoo Integration (Optional)
    """
    
    # --- Physics / Engineering Constants ---
    
    # Aerial Constants
    ENERGY_DENSITY_KG_M3 = 1500.0  # Approx density of energy storage
    STRUCTURE_MASS_RATIO = 1.5     # Structure mass relative to payload
    DEFAULT_PAYLOAD_MASS_KG = 0.5 
    
    # Ground Constants
    LINEAR_DENSITY_KG_M = 20.0   # kg per meter of length for generic chassis
    DEFAULT_VEHICLE_LENGTH_M = 1.0
    
    # Space Constants
    UNIT_MASS_LIMIT_KG = 1.33 # Standard Unit mass limit
    DEFAULT_UNIT_SIZE = 1

    def __init__(self):
        self.zoo_client = None
        # Try initializing Zoo client if token exists
        token = os.getenv("ZOO_API_TOKEN") or os.getenv("KITTYCAD_API_TOKEN")
        if token:
            try:
                from kittycad.client import Client
                self.zoo_client = Client(token=token)
                logger.info("Geometry Agent connected to Zoo (KittyCAD)")
            except ImportError:
                logger.warning("kittycad library not installed.")
            except Exception as e:
                logger.warning(f"Failed to init Zoo: {e}")

    def run(self, params: Dict[str, Any], intent: str, environment: Dict[str, Any] = None, ldp_instructions: List[Dict] = None) -> Dict[str, Any]:
        """
        Main execution entry point.
        """
        if environment is None: environment = {}
        if ldp_instructions: 
             # Merge into params if passed separately, ensuring logic below sees it
             params["ldp_instructions"] = ldp_instructions
        
        # 1. Recursive ISA Scope Resolution
        # If running in a focused pod, pull its constraints (Source of Truth)
        pod_id = params.get("pod_id")
        if pod_id:
            try:
                from core.system_registry import get_system_resolver
                resolver = get_system_resolver()
                
                # Helper to find pod by ID (TODO: Add get_pod_by_id to Resolver)
                def find_pod(root, target_id):
                    if root.id == target_id: return root
                    for sub in root.sub_pods.values():
                        f = find_pod(sub, target_id)
                        if f: return f
                    return None
                
                pod = find_pod(resolver.root, pod_id)
                if pod:
                    logger.info(f"GeometryAgent: SCOPED EXECUTION -> {pod.name}")
                    # Merge constraints into params (High Priority)
                    # This allows the Pod's state to drive the geometry
                    params.update(pod.constraints)
                    
                    # Also set a context flag
                    params["context_name"] = pod.name
                    
            except ImportError:
                 logger.warning("System Registry not available for Scoped Geometry.")

        # 2. Determine Generation Mode
        mode = "parametric" # Default
        if "generative" in intent.lower() or (not params and not pod_id):
            mode = "generative"
            # If no params provided at all, we might switch to a generative stub
            # OR we populate default params based on regime to avoid "magic numbers" in code logic
 
        # 2. Logic Branching
        kcl_code = ""
        glsl_code = "" # New Output
        include_scad = False

        if mode == "parametric":
            # --- HWC KERNEL INTEGRATION ---
            from hwc_kernel import HighFidelityKernel, OpType
            hwc = HighFidelityKernel()
            
            # New Step: Check for LDP Instructions
            # If present, use the Hardware Compiler Loop
            ldp_instructions = params.get("ldp_instructions", [])
            if ldp_instructions:
                logger.info(f"GeometryAgent: Driving HWC via {len(ldp_instructions)} LDP Instructions")
                geometry_tree = self._run_hardware_compiler_loop(ldp_instructions)
            else:
                # Fallback to Legacy Heuristic
                regime = environment.get("regime", "AERIAL")
                geometry_tree = self._estimate_geometry_tree(regime, params)
            
            # Compile ISA
            isa = hwc.synthesize_isa("current_project", [], geometry_tree)
            
            # Transpile to GLSL (Frontend View)
            glsl_code = hwc.to_glsl(isa)
            
            # Transpile to KCL (Manufacturing)
            kcl_code = hwc.to_kcl(isa)
            
        else:
            # Generative Fallback (LLM Stub)
            kcl_code = self.generate_kcl_from_prompt(intent)
            geometry_tree = [{"id": "gen_1", "type": "generative_mesh", "mass_kg": 1.0}]

        # 3. Append High-Fidelity Components (Procedural Library)
        components_to_render = params.get("components", [])
        if components_to_render:
            kcl_code = self._append_component_placeholders(kcl_code, components_to_render)

        # 4. Compile (Zoo or Stub)
        gltf_data = None
        if self.zoo_client:
            zoo_res = self._run_zoo(kcl_code)
            if zoo_res.get("success"):
                gltf_data = zoo_res.get("gltf_data")

        # 5. Validation (Manifold Agent)
        from agents.manifold_agent import ManifoldAgent
        manifold_agent = ManifoldAgent()
        manifold_res = manifold_agent.run({"geometry_tree": geometry_tree})
        
        validation = manifold_res.get("validation", {})
        if not validation.get("is_watertight", True):
            logger.warning(f"Geometry not watertight: {validation.get('issues')}")

        return {
            "kcl_code": kcl_code,
            "glsl_code": glsl_code,
            "hwc_isa": isa if mode == "parametric" else {},
            "geometry_tree": geometry_tree,
            "gltf_data": gltf_data,
            "validation_logs": validation.get("logs", [])
        }

    def _estimate_geometry_tree(self, regime, params):
        """Stub for parametric estimation."""
        return [{"id": "main_body", "type": "box", "dims": [10,10,10]}]

    def _run_hardware_compiler_loop(self, instructions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        The Hardware Compiler Loop.
        Translates Logic Kernel instructions into HWC Geometry Primitives.
        """
        tree = []
        import math
        
        for instr in instructions:
            handler = instr.get("handler")
            val = instr.get("params") # Resolved value
            node_id = instr.get("id")
            
            if handler == "generate_propeller_volume":
                # handler for THRUST_REQ_N
                # Thrust (N) -> Prop Diameter check
                # T ~ D^2 * v^2 ... Heuristic: D (m) = sqrt(T / 50) roughly
                thrust = float(val)
                dia = math.sqrt(thrust / 10.0) * 0.2
                if dia < 0.1: dia = 0.1
                
                # Add Propeller Discs (Visualized as Cylinders)
                tree.append({
                    "id": f"prop_{node_id}",
                    "type": "cylinder",
                    "params": {"radius": dia/2.0, "height": 0.02},
                    "transform": {"translate": [0.2, 0.2, 0]} # Placeholder Layout
                })
                 # Symmetry? For now just one block representing volume
                
            elif handler == "resize_battery_bay":
                # handler for BATTERY_CAP_WH
                # Energy Density ~ 250 Wh/L => 1 Wh = 1/250 L = 4 cc.
                cap_wh = float(val)
                vol_liters = cap_wh / 250.0
                # Cube root for dims
                side = (vol_liters / 1000.0) ** (1/3)
                
                tree.append({
                    "id": f"battery_{node_id}",
                    "type": "box",
                    "params": {"length": side, "width": side, "height": side},
                    "transform": {"translate": [0, 0, -0.05]}
                })
                
            elif handler == "reinforce_structure":
                 # Thickness mod?
                 pass
            
            # Default Fallback?
            
        # Ensure at least one body exists
        if not tree:
            tree.append({"id": "fallback_core", "type": "sphere", "params": {"radius": 0.5}})
            
        return tree
        
    def _generate_parametric_kcl(self, tree):
        """Stub for KCL generation."""
        return "// Generated KCL"
        
    def _run_zoo(self, kcl):
        """Stub for Zoo execution."""
        return {"success": True, "gltf_data": b"fake_gltf"}

    def perform_mesh_boolean(self, mesh_a, mesh_b, operation: str = "DIFFERENCE") -> Any:
        """
        Executes a Boolean operation (A op B) on two Trimesh objects.
        Strategy:
        1. Try fast exact boolean (Manifold/Blender/SCAD backend).
        2. Fallback to VMK/SDF Voxelization (Guaranteed result).
        """
        import trimesh
        import numpy as np
        
        # 1. Fast Path (Exact)
        op_upper = operation.upper()
        try:
            if op_upper == "DIFFERENCE":
                result = trimesh.boolean.difference([mesh_a, mesh_b])
            elif op_upper == "UNION":
                result = trimesh.boolean.union([mesh_a, mesh_b])
            elif op_upper == "INTERSECTION":
                result = trimesh.boolean.intersection([mesh_a, mesh_b])
            else:
                raise ValueError(f"Unknown Op: {operation}")
                
            if result.is_volume:
                return result
        except Exception as e:
            logger.warning(f"Fast Boolean Failed: {e}. Switching to SDF Fallback.")
            
        # 2. SDF Fallback (VMK Logic)
        # 2. SDF Fallback (VMK Logic)
        settings = self._load_kernel_settings()
        resolution = settings.get("sdf_resolution", 64)
        return self._sdf_boolean(mesh_a, mesh_b, operation, resolution)

    def get_composite_sdf(self, geometry_tree: List[Dict[str, Any]]) -> Any:
        """
        Returns a callable SDF function `f(points) -> distances` 
        representing the composite geometry tree.
        Used for Marching Cubes export.
        """
        import numpy as np
        from utils.sdf_mesher import sdf_box, sdf_sphere
        
        # Helper: Resolve params
        def resolve_dims(part):
            p = part.get("params", {})
            ptype = part.get("type", "box")
            
            if ptype == "box":
                # Convert to half-extents
                l = p.get("length", 1.0)
                w = p.get("width", 1.0)
                h = p.get("height", 1.0)
                return np.array([l/2, w/2, h/2])
            elif ptype == "sphere":
                return p.get("radius", 1.0)
            return np.array([0.5, 0.5, 0.5])

        # Closure for the composite function
        def composite_func(points):
            # Start with "empty" space (infinite distance)
            d_min = np.full(points.shape[0], 1e9)
            
            for part in geometry_tree:
                ptype = part.get("type", "box")
                dims = resolve_dims(part)
                
                # Assume origin-centered for now (TODO: Apply transforms)
                # In real Composite SDF, we'd subtract center from points
                center = np.array(part.get("transform", {}).get("translate", [0,0,0]))
                p_local = points - center
                
                if ptype == "box" or ptype == "plate":
                    d = sdf_box(p_local, dims)
                elif ptype == "sphere":
                    d = sdf_sphere(p_local, dims)
                else:
                    d = sdf_box(p_local, np.array([0.5, 0.5, 0.5])) # Fallback
                
                # Apply Boolean Operation
                op = part.get("operation", "UNION").upper()
                
                if op == "UNION":
                    d_min = np.minimum(d_min, d)
                elif op == "DIFFERENCE" or op == "SUBTRACT":
                    d_min = np.maximum(d_min, -d)
                elif op == "INTERSECTION":
                    d_min = np.maximum(d_min, d)
                else:
                    d_min = np.minimum(d_min, d) # Default to Union
                
            return d_min
            
        return composite_func

    def _sdf_boolean(self, mesh_a, mesh_b, operation: str, res: int = 64) -> Any:
        """
        SDF-based Boolean.
        """
        import trimesh
        from skimage import measure
        import numpy as np
        
        # Combined bounds
        bounds = np.vstack((mesh_a.bounds, mesh_b.bounds))
        min_p = np.min(bounds, axis=0) - 2.0
        max_p = np.max(bounds, axis=0) + 2.0
        
        x = np.linspace(min_p[0], max_p[0], res)
        y = np.linspace(min_p[1], max_p[1], res)
        z = np.linspace(min_p[2], max_p[2], res)
        
        # Create Grid Points
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        pts = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
        
        # Verify meshes are watertight for SDF
        if not mesh_a.is_watertight: trimesh.repair.fill_holes(mesh_a)
        if not mesh_b.is_watertight: trimesh.repair.fill_holes(mesh_b)

        # Compute Signed Distance (Negative inside)
        sdf_a_flat = trimesh.proximity.signed_distance(mesh_a, pts)
        sdf_b_flat = trimesh.proximity.signed_distance(mesh_b, pts)
        
        op_upper = operation.upper()
        sdf_comb = sdf_a_flat 
        
        if op_upper == "DIFFERENCE":
             sdf_comb = np.maximum(sdf_a_flat, -sdf_b_flat)
        elif op_upper == "UNION":
             sdf_comb = np.minimum(sdf_a_flat, sdf_b_flat) 
        elif op_upper == "INTERSECTION":
             sdf_comb = np.maximum(sdf_a_flat, sdf_b_flat)
        else:
             logger.warning(f"Unknown Boolean Op: {operation}, defaulting to Union")
             sdf_comb = np.minimum(sdf_a_flat, sdf_b_flat)
             
        sdf_grid = sdf_comb.reshape(res, res, res)
        
        # Reconstruct
        spacing = (
            (max_p[0]-min_p[0])/(res-1),
            (max_p[1]-min_p[1])/(res-1),
            (max_p[2]-min_p[2])/(res-1)
        )
        
        try:
            verts, faces, normals, values = measure.marching_cubes(sdf_grid, level=0.0, spacing=spacing)
            verts += min_p # Offset
            
            new_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            new_mesh.process() # Cleanup
            return new_mesh
            
        except Exception as e:
            logger.error(f"SDF Reconstruction Failed: {e}")
            return None

    def _load_kernel_settings(self) -> Dict:
        import json
        path = "data/geometry_agent_weights.json"
        if not os.path.exists(path): return {"sdf_resolution": 64}
        try:
            with open(path, 'r') as f: return json.load(f)
        except: return {"sdf_resolution": 64}

    def update_kernel_settings(self, action: str):
        """Evolve kernel settings based on critic feedback."""
        settings = self._load_kernel_settings()
        res = settings.get("sdf_resolution", 64)
        
        if action == "INCREASE_RESOLUTION":
            res = min(256, int(res * 1.5))
            logger.info(f"GeometryAgent: Increasing SDF Resolution to {res} (Quality Boost)")
        elif action == "DECREASE_RESOLUTION":
            res = max(32, int(res * 0.8))
            logger.info(f"GeometryAgent: Decreasing SDF Resolution to {res} (Speed Boost)")
            
        settings["sdf_resolution"] = res
        
        import json
        with open("data/geometry_agent_weights.json", 'w') as f:
            json.dump(settings, f, indent=2)


    # --- Regime-Aware Sizing ---

    def _estimate_geometry_tree(self, regime: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Estimates component tree based on physics/regime OR explicit constraints.
        """
        tree = []
        name = params.get("context_name", "main_body")
        
        # 1. Explicit Constraints (Recursive ISA Priority)
        # If the pod has explicit geometry vars, use them directly.
        if "length" in params and "width" in params and "height" in params:
             tree.append({
                "id": f"{name}_geometry",
                "type": "box",
                "params": {
                    "length": float(params["length"]), 
                    "width": float(params["width"]), 
                    "height": float(params["height"])
                },
                "mass_kg": float(params.get("mass_budget", 1.0))
            })
             return tree
        
        # 2. Heuristics (Fallback)
        if regime == "AERIAL":
            # Generic Physics-based sizing for Aerial Vehicles
            # Volume based on Energy Density + Payload
            payload_mass_kg = params.get("payload_mass", self.DEFAULT_PAYLOAD_MASS_KG)
            
            # Estimate Volume needed for avionics/battery
            vol = payload_mass_kg / self.ENERGY_DENSITY_KG_M3
            
            # Radius of sphere/fuselage with that volume * packing factor
            # V = 4/3 * pi * r^3  =>  r = (3V / 4pi)^(1/3)
            core_radius = max(0.1, (vol * 3 / (4 * math.pi))**(1/3) * 2.0)
            
            tree.append({
                "id": "fuselage_core",
                "type": "sphere", # Generic containment
                "params": {"radius": core_radius},
                "blend": 0.0
            })
            
        elif regime == "GROUND":
            # Sizing for Ground Vehicles
            length = params.get("length", self.DEFAULT_VEHICLE_LENGTH_M)
            tree.append({
                "id": "main_chassis",
                "type": "box",
                "mass_kg": self.LINEAR_DENSITY_KG_M * length, 
                "params": {"length": length, "width": length * 0.6, "height": 0.3}
            })
            
        elif regime == "SPACE":
            # Satellite logic
            u_size = params.get("units", self.DEFAULT_UNIT_SIZE)
            tree.append({
                "id": "satellite_bus",
                "type": "box",
                "mass_kg": self.UNIT_MASS_LIMIT_KG * u_size,
                "params": {"length": 0.1, "width": 0.1, "height": 0.1 * u_size}
            })
            
        else:
            # Fallback
            tree.append({"id": "default_part", "type": "box", "mass_kg": 1.0, "params": {"width": 100, "length": 100, "thickness": 5}})

        return tree

    # --- KCL Generation ---

    def _generate_parametric_kcl(self, tree: List[Dict[str, Any]]) -> str:
        """
        Converts the abstract geometry tree into concrete KCL code.
        """
        kcl = "// BRICK OS Parametric Generation\n\n"
        
        for part in tree:
            p = part.get("params", {})
            ptype = part.get("type", "box")
            
            if ptype == "structure" or ptype == "plate" or ptype == "box":
                # Convert to KCL Plate/Box (units: mm)
                # Ensure we handle potentially missing params gracefully or assume 0
                w = p.get("width", p.get("radius", 0.05) * 2) * 1000 
                l = p.get("length", p.get("radius", 0.05) * 2) * 1000
                h = p.get("height", p.get("thickness", 0.05)) * 1000
                
                kcl += f"""
                const part_{part['id']} = startSketchOn('XY')
                    |> startProfileAt([0,0], %)
                    |> line([{w}, 0], %)
                    |> line([0, {l}], %)
                    |> line([- {w}, 0], %)
                    |> close(%)
                    |> extrude({h}, %)
                \n"""
        return kcl

    def generate_kcl_from_prompt(self, user_intent: str) -> str:
        """
        Generates raw KCL code using the LLM (Stub for now).
        """
        # In Phase 4, this calls the LLM Provider
        return f"// Generative KCL Stub for: {user_intent}\n// TODO: Connect LLM Provider"

    # --- Procedural Library ---

    def _append_component_placeholders(self, kcl_code: str, components: List[Dict[str, Any]]) -> str:
        """
        Appends High-Fidelity Procedural KCL from DB.
        """
        import sqlite3
        conn = sqlite3.connect("data/materials.db")
        cur = conn.cursor()
        
        cur.execute("SELECT kcl_source FROM kcl_templates")
        rows = cur.fetchall()
        
        procedural_lib = "\n// PROCEDURAL LIBRARY (DB)\n"
        for row in rows:
            procedural_lib += row[0] + "\n"
            
        kcl_code += "\n" + procedural_lib
        conn.close()
        return kcl_code

    def _run_zoo(self, kcl_code: str) -> Dict[str, Any]:
        """Compile KCL via Zoo API."""
        try:
            from kittycad.models.file_export_format import FileExportFormat
            from kittycad.models.file_import_format import FileImportFormat
            
            result = self.zoo_client.file.create_file_conversion(
                body=kcl_code.encode('utf-8'),
                src_format=FileImportFormat.KCL,
                output_format=FileExportFormat.GLTF,
            )
            return {"success": True, "gltf_data": result.body}
        except Exception as e:
            logger.error(f"Zoo Compilation Failed: {e}")
            return {"success": False, "error": str(e)}
