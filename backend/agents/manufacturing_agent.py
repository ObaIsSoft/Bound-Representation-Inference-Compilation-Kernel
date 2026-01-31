from typing import Dict, Any, List
import math

class ManufacturingAgent:
    """
    Generates Bill of Materials (BOM) and estimates manufacturing costs
    based on the design geometry and material.
    """
    
    # --- Economic Constants ---
    HOURLY_MACHINING_RATE_USD = 50.0 
    SETUP_COST_USD = 100.0
    MACHINING_TIME_PER_KG_HOUR = 1.0 
    
    def __init__(self):
        self.name = "ManufacturingAgent"
        
        # Phase 10 Optimization: Use Physics Kernel Singletons
        from backend.physics.kernel import get_physics_kernel
        self.physics = get_physics_kernel()
        
        # 1. Materials DB (Shared)
        # Access via Materials Domain to share the connection
        if hasattr(self.physics.domains["materials"], "materials_db"):
             self.api = self.physics.domains["materials"].materials_db
        else:
             # Fallback if kernel init failed or partial
             from backend.materials.materials_db import MaterialsDatabase
             self.api = MaterialsDatabase()
        
        # 2. Oracles
        try:
            from agents.materials_oracle.materials_oracle import MaterialsOracle
            self.materials_oracle = MaterialsOracle()
            self.has_oracles = True
        except ImportError:
            self.materials_oracle = None
            self.has_oracles = False
            
        # 3. Manufacturing Surrogate (Shared via SurrogateManager)
        self.surrogate_manager = self.physics.intelligence["surrogate_manager"]
        if self.surrogate_manager.has_model("manufacturing_surrogate"):
             # Wrapper to expose expected methods if needed, or just use manager directly
             # ManufacturingAgent expects 'self.surrogate.predict_defect_probability'
             # We can map it or grab the model instance (less clean but fast)
             self.surrogate = self.surrogate_manager.surrogates["manufacturing_surrogate"]["model"]
             self.has_surrogate = True
        else:
             self.surrogate = None
             self.has_surrogate = False
             # Try lazily loading if not found in manager (redundancy)
             try:
                 from backend.models.manufacturing_surrogate import ManufacturingSurrogate
                 self.surrogate = ManufacturingSurrogate()
                 self.has_surrogate = True
             except ImportError:
                 pass

    def _get_material_data(self, material_name: str) -> Dict[str, Any]:
        """Fetch material costing data from DB."""
        # Defaults
        defaults = {"density": 2700, "cost_per_kg": 2.50, "machining_factor": 1.0}
        #hardcoded
        result = self.api.find_material(material_name)
        if not result:
            return defaults
            
        # Try to find an alloy match
        found = result['data']
        return {
            "density": found.get("density", defaults["density"]),
            "cost_per_kg": found.get("cost_per_kg", defaults["cost_per_kg"]),
            "machining_factor": found.get("machining_factor", defaults["machining_factor"])
        }

    def _calculate_volume(self, node: Dict[str, Any]) -> float:
        """Estimate volume in m^3 based on params."""
        params = node.get("params", {})
        shape_type = node.get("type", "unknown").lower()
        scale = 1e-9 # mm^3 to m^3
        
        if shape_type in ["box", "plate", "structure"]:
            # Check for width/length OR length/width/height mixing
            w = params.get("width", params.get("radius", 0)*2)
            l = params.get("length", params.get("radius", 0)*2)
            h = params.get("thickness", params.get("height", 0))
            return (w * l * h) * scale
            
        elif shape_type == "cylinder":
            r = params.get("radius", 0)
            h = params.get("height", params.get("length", 0))
            return (math.pi * r**2 * h) * scale
            
        return 0.001 # Fallback volume

    def run(self, geometry_tree: List[Dict[str, Any]], material: str, pod_id: str = None) -> Dict[str, Any]:
        """
        Analyze geometry and material to produce manufacturing data.
        If pod_id is provided, aggregates BOM from sub-assemblies (Recursive ISA).
        """
        mat_data = self._get_material_data(material)
        density = mat_data["density"]
        base_cost = mat_data["cost_per_kg"]
        machining_factor = mat_data["machining_factor"]
        
        total_cost = 0.0
        total_mass = 0.0
        bom_items = []
        
        # 1. Local Geometry Analysis
        if geometry_tree:
            for i, part in enumerate(geometry_tree):
                # Mass Calculation
                if "mass_kg" in part:
                    mass = part["mass_kg"]
                else:
                    vol = self._calculate_volume(part)
                    mass = vol * density
                
                # Cost Calculation
                material_cost = mass * base_cost
                est_hours = mass * self.MACHINING_TIME_PER_KG_HOUR * machining_factor
                machining_cost = (self.HOURLY_MACHINING_RATE_USD * est_hours) + self.SETUP_COST_USD
                
                part_total = material_cost + machining_cost
                
                total_mass += mass
                total_cost += part_total
                
                bom_items.append({
                    "id": part.get("id", f"part-{i+1}"),
                    "name": part.get("type", "Component").title(),
                    "material": material,
                    "process": "CNC Milling" if machining_factor > 1 else "Injection Molding",
                    "mass_kg": round(mass, 3),
                    "cost": round(part_total, 2),
                    "type": "part"
                })

        # 2. Recursive Sub-Assembly Aggregation
        if pod_id:
            try:
                from core.system_registry import get_system_resolver
                resolver = get_system_resolver()
                # Simple BFS/DFS to find pod
                def find_pod(root, target_id):
                    if root.id == target_id: return root
                    for sub in root.sub_pods.values():
                        f = find_pod(sub, target_id)
                        if f: return f
                    return None
                
                current_pod = find_pod(resolver.root, pod_id)
                
                if current_pod:
                    for sub_key, sub_pod in current_pod.sub_pods.items():
                        # ... sub-pod logic ...
                        sub_cost = sub_pod.exports.get("cost", 0.0)
                        sub_mass = sub_pod.exports.get("mass", 0.0)
                        
                        bom_items.append({
                            "id": sub_pod.id,
                            "name": f"{sub_pod.name} (Assembly)",
                            "material": "Mixed",
                            "process": "Assembly",
                            "mass_kg": round(sub_mass, 3),
                            "cost": round(sub_cost, 2),
                            "type": "assembly"
                        })
                        
                        total_cost += sub_cost
                        total_mass += sub_mass

                    # [PHASE 22/23] Handle Linked Components (Internal Files)
                    if current_pod.is_folder_linked:
                        for comp in current_pod.linked_components:
                            if not comp.get("active", True): continue
                            
                            comp_id = comp.get("id")
                            comp_path = comp.get("path")
                            
                            # Heuristic costing for linked files if not already simulated
                            # In real loop, these would have their own computed constraints
                            c_mass = current_pod.constraints.get(f"{comp_id}_mass", 0.1)
                            c_cost = current_pod.constraints.get(f"{comp_id}_cost", 10.0)
                            
                            bom_items.append({
                                "id": f"{current_pod.id}_{comp_id}",
                                "name": f"{comp_id} (Part: {comp_path})",
                                "material": material,
                                "process": "Hardware Compilation",
                                "mass_kg": round(c_mass, 3),
                                "cost": round(c_cost, 2),
                                "type": "linked_component"
                            })
                            total_mass += c_mass
                            total_cost += c_cost

                    # [NEW] Commit to System Registry (Recursive ISA State)
                    # This allows 'converge_up' to work for parents.
                    current_pod.exports["mass"] = total_mass
                    current_pod.exports["cost"] = total_cost
                    current_pod.is_dirty = False 
                        
            except ImportError:
                pass # Registry not ready or circular import

        return {
            "components": bom_items,
            "bom_analysis": {
                "total_cost": round(total_cost, 2),
                "currency": "USD",
                "lead_time_days": 7 + (len(bom_items) * 1),
                "manufacturability_score": 0.95 if machining_factor < 4 else 0.60
            }
        }

    def verify_toolpath_accuracy(self, toolpaths: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verify toolpaths for collisions and gouging using VMK.
        
        Args:
            toolpaths: List of VMKInstruction dicts
        """
        try:
            from vmk_kernel import SymbolicMachiningKernel, ToolProfile, VMKInstruction
            import numpy as np
        except ImportError:
            return {"verified": False, "error": "VMK not available"}
            
        # Initialize Kernel
        kernel = SymbolicMachiningKernel(stock_dims=[100, 100, 100])
        
        # Tools tracking
        tools = {}
        
        gouges = []
        
        # We need to simulate the process sequentially
        for i, op_data in enumerate(toolpaths):
            tool_id = op_data.get("tool_id")
            
            # Register tool if new (assuming simple default)
            if tool_id not in tools:
                tool = ToolProfile(id=tool_id, radius=0.5, type="BALL") # Default
                kernel.register_tool(tool)
                tools[tool_id] = tool
            
            op = VMKInstruction(**op_data)
            
            # Check for Rapid Gouging (G0)
            # If this move is a rapid (usually marked, here we assume all are cuts for MVP)
            # But we can check if the START point of a cut is DEEP inside material.
            # If we plunge into material without a ramp, that's a crash.
            
            start_pt = np.array(op.path[0])
            sdf_start = kernel.get_sdf(start_pt)
            
            # If SDF < -depth_of_cut, we are crashing
            # Simple heuristic: if SDF < -0.1mm, fail
            if sdf_start < -0.1:
                gouges.append(f"Op {i}: Plunge collision at {start_pt}, Depth={abs(sdf_start):.3f}mm")
            
            # Execute logic (remove material)
            kernel.execute_gcode(op)
            
        return {
            "verified": len(gouges) == 0,
            "collisions": gouges,
            "safety_score": 100 if len(gouges) == 0 else 0
        }

    def analyze_material_processing_oracle(self, params: dict) -> dict:
        """Analyze material processing using Materials Oracle"""
        if not self.has_oracles:
            return {"status": "error", "message": "Oracles not available"}
        
        # Use appropriate domain based on material type
        domain = params.get("domain", "METALLURGY")  # METALLURGY, CERAMICS, COMPOSITES
        
        return self.materials_oracle.solve(
            query="Manufacturing process analysis",
            domain=domain,
            params=params
        )
    
    def analyze_tribology_oracle(self, params: dict) -> dict:
        """Analyze wear and friction using Materials Oracle (TRIBOLOGY)"""
        if not self.has_oracles:
            return {"status": "error", "message": "Oracles not available"}
        
        return self.materials_oracle.solve(
            query="Tribology analysis",
            domain="TRIBOLOGY",
            params=params
        )

    def critique_sketch(self, sketch_points: List[Dict[str, float]]) -> List[Dict[str, str]]:
        """
        Critique the user's sketch strokes for manufacturability.
        Rules:
        1. Small Radius: If curves are too tight for standard CNC tools.
        2. Thin Walls: (Heuristic from proximity of points) - Logic for later.
        
        Args:
            sketch_points: List of vectors {x, y, z, ...}
            
        Returns:
            List of critique messages {level: 'INFO|WARN|ERR', message: '...'}
        """
        critiques = []
        
        # 1. Curvature / Radius Analysis
        # Simplest heuristic: Check distance between consecutive points if they represent a curve.
        # If we have 3 points A, B, C, we can est curvature.
        # But `sketch_points` here might be raw strokes.
        # Assuming sketch_points contains 'radius' if it's a capsule/tube stroke.
        
        # Check explicit radius if available (Capsule Sketching)
        min_cnc_radius = 1.0 # mm (Standard smallest tool 2mm dia)
        
        # If sketch is just raw points, we can't easily do radius without fitting.
        # But if the 'sketch' passed here is actually the 'geometry_sketch' (Capsules), we can check 'radius'.
        
        for i, point in enumerate(sketch_points):
            # Check for 'radius' property (Capsule/Tube sketches)
            r = point.get("radius")
            
            # If r is None, maybe it's in the param dict?
            if r is None and "params" in point:
                 r = point.get("params", {}).get("radius")
                 
            if r is not None:
                # Convert to mm (assuming system units are meters if small, or check convention)
                # Convention: System is usually Meters. 1mm = 0.001m.
                # If r > 0.5 (500mm), it's probably meters. If r=1.0, maybe mm?
                # Let's assume input is standard meters.
                r_mm = r * 1000.0
                
                if r_mm < min_cnc_radius:
                    critiques.append({
                        "level": "WARN",
                        "agent": "Manufacturing",
                        "message": f"Radius {r_mm:.2f}mm is too small for standard CNC. Increase to >{min_cnc_radius}mm?"
                    })
                    
            # 2. Neural Defect Prediction (Deep Evolution)
            if self.has_surrogate:
                # Heuristic Features extraction
                # Ideally we'd scan the whole stroke. For now, use the point radius/position.
                # Feature Vec: [Radius, Aspect(dummy), Complexity(dummy), Undercuts(dummy)]
                if r is not None:
                     r_mm = r * 1000.0
                     # Predict Probability
                     prob = self.surrogate.predict_defect_probability(r_mm, 2.0, 10.0, 0.0)
                     if prob > 0.7:
                         critiques.append({
                             "level": "WARN",
                             "agent": "Manufacturing (Neural)",
                             "message": f"Neural Predictor detects high defect probability ({prob:.2f}) for this geometry."
                         })

                    
        return critiques

    def predict_defects(self, geometry_features: List[float]) -> float:
        """
        Predict defect probability using learned surrogate.
        Args:
            geometry_features: [Radius(mm), AspectRatio, Complexity, Undercuts]
        """
        if not self.has_surrogate:
            return 0.0
        return self.surrogate.predict_defect_probability(*geometry_features)

    def evolve(self, training_data: List[Any]) -> Dict[str, Any]:
        """
        Deep Evolution Trigger.
        Train the surrogate on real/simulated defect outcomes.
        Args:
            training_data: List of (features, label) tuples.
        """
        if not self.has_surrogate:
            return {"status": "error", "message": "No surrogate"}
            
        import numpy as np
        total_loss = 0
        count = 0
        
        for x, y in training_data:
            loss = self.surrogate.train_step(np.array(x), np.array(y))
            total_loss += loss
            count += 1
            
        self.surrogate.save()
        return {
            "status": "evolved",
            "avg_loss": total_loss / max(1, count),
            "epochs": self.surrogate.trained_epochs
        }


