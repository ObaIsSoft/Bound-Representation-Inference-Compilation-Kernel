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
        
        # Initialize Oracles for manufacturing analysis
        try:
            from agents.materials_oracle.materials_oracle import MaterialsOracle
            self.materials_oracle = MaterialsOracle()
            self.has_oracles = True
        except ImportError:
            self.materials_oracle = None
            self.has_oracles = False
        try:
            from materials.materials_db import MaterialsDatabase
        except ImportError:
            from backend.materials.materials_db import MaterialsDatabase
        self.api = MaterialsDatabase()

    def _get_material_data(self, material_name: str) -> Dict[str, Any]:
        """Fetch material costing data from DB."""
        # Defaults
        defaults = {"density": 2700, "cost_per_kg": 2.50, "machining_factor": 1.0}
        
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
                        # We treat sub-pods as "Assemblies" in the BOM
                        # Ideally, we'd recursively call ManufacturingAgent on them,
                        # but for now we assume they have 'exports' populated or we just list them.
                        # Phase 10: We list them as Line Items.
                        
                        # Check if sub-pod has exported cost/mass (cached from previous runs)
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

                    # [NEW] Commit to System Registry (Recursive ISA State)
                    # This allows 'converge_up' to work for parents.
                    current_pod.exports["mass"] = total_mass
                    current_pod.exports["cost"] = total_cost
                    current_pod.is_dirty = False # Mark as clean/computed
                        
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
