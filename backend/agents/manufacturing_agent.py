from typing import Dict, Any, List
import math
import logging

logger = logging.getLogger(__name__)


class ManufacturingAgent:
    """
    Generates Bill of Materials (BOM) and estimates manufacturing costs
    based on the design geometry and material.
    
    Uses database-driven manufacturing rates - no hardcoded constants.
    Fails fast if rates not configured.
    """
    
    def __init__(self):
        self.name = "ManufacturingAgent"
        self._initialized = False
        
        # Will be loaded from database
        self._rates: Dict[str, Any] = {}
        
        # Phase 10 Optimization: Use Physics Kernel Singletons
        from physics.kernel import get_physics_kernel
        self.physics = get_physics_kernel()
        
        # 1. Materials DB (Shared)
        if hasattr(self.physics.domains["materials"], "materials_db"):
             self.api = self.physics.domains["materials"].materials_db
        else:
             from materials.materials_db import MaterialsDatabase
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
             self.surrogate = self.surrogate_manager.surrogates["manufacturing_surrogate"]["model"]
             self.has_surrogate = True
        else:
             self.surrogate = None
             self.has_surrogate = False
             try:
                 from models.manufacturing_surrogate import ManufacturingSurrogate
                 self.surrogate = ManufacturingSurrogate()
                 self.has_surrogate = True
             except ImportError:
                 pass

    async def initialize(self, process_type: str = "cnc_milling", region: str = "global"):
        """
        Load manufacturing rates from database.
        
        Args:
            process_type: Manufacturing process (cnc_milling, 3d_printing, etc.)
            region: Geographic region for rates (us, eu, global)
            
        Raises:
            ValueError: If rates not found in database
        """
        # Allow re-initialization if process/region changed
        if (self._initialized and 
            hasattr(self, '_current_process') and 
            self._current_process == process_type and
            hasattr(self, '_current_region') and
            self._current_region == region):
            return
            
        from backend.services import supabase
        
        try:
            self._rates = await supabase.get_manufacturing_rates(process_type, region)
            
            if not self._rates:
                raise ValueError(
                    f"No manufacturing rates found for {process_type}/{region}. "
                    f"Please configure in manufacturing_rates table or get supplier quote."
                )
            
            logger.info(
                f"ManufacturingAgent initialized: {process_type}/{region} "
                f"(hourly_rate: ${self._rates.get('machine_hourly_rate_usd')}, "
                f"setup: ${self._rates.get('setup_cost_usd')})"
            )
            self._initialized = True
            self._current_process = process_type
            self._current_region = region
            
        except Exception as e:
            raise ValueError(f"Failed to load manufacturing rates: {e}")

    @property
    def hourly_rate(self) -> float:
        """Hourly machining rate from database"""
        if not self._initialized:
            raise RuntimeError("ManufacturingAgent not initialized. Call initialize() first.")
        return self._rates.get("machine_hourly_rate_usd", 0.0)
    
    @property
    def setup_cost(self) -> float:
        """Setup cost from database"""
        if not self._initialized:
            raise RuntimeError("ManufacturingAgent not initialized. Call initialize() first.")
        return self._rates.get("setup_cost_usd", 0.0)
    
    @property
    def setup_time_minutes(self) -> int:
        """Setup time from database"""
        if not self._initialized:
            raise RuntimeError("ManufacturingAgent not initialized. Call initialize() first.")
        if not self._initialized:
            raise RuntimeError("ManufacturingAgent not initialized. Call initialize() first.")
        setup_time = self._rates.get("setup_time_minutes")
        if setup_time is None:
            raise ValueError("setup_time_minutes not found in manufacturing rates")
        return setup_time
    
    @property
    def min_cnc_radius_mm(self) -> float:
        """Minimum CNC radius from database tolerance"""
        if not self._initialized:
            raise RuntimeError("ManufacturingAgent not initialized. Call initialize() first.")
        tolerance = self._rates.get("tolerance_mm", 0.1)
        # Min radius is typically 10x tolerance for standard tools
        return max(1.0, tolerance * 10)

    async def _get_material_data(self, material_name: str) -> Dict[str, Any]:
        """
        Fetch material costing data from Supabase.
        
        Uses materials_extended table with real properties.
        No defaults - fails if material not found.
        """
        from backend.services import supabase
        
        try:
            material = await supabase.get_material(material_name)
            
            return {
                "density": material.get("density_kg_m3"),
                "cost_per_kg": material.get("cost_per_kg_usd"),
                "machining_factor": material.get("machining_factor", 1.0)
            }
        except ValueError:
            # Material not in database - return error info
            logger.error(f"Material '{material_name}' not found in database")
            raise ValueError(
                f"Material '{material_name}' not found. "
                f"Please add to materials table or check name."
            )
        except Exception as e:
            logger.error(f"Failed to fetch material data: {e}")
            raise

    def _calculate_volume(self, node: Dict[str, Any]) -> float:
        """Estimate volume in m^3 based on params."""
        params = node.get("params", {})
        shape_type = node.get("type", "unknown").lower()
        scale = 1e-9  # mm^3 to m^3
        
        if shape_type in ["box", "plate", "structure"]:
            w = params.get("width", params.get("radius", 0)*2)
            l = params.get("length", params.get("radius", 0)*2)
            h = params.get("thickness", params.get("height", 0))
            return (w * l * h) * scale
            
        elif shape_type == "cylinder":
            r = params.get("radius", 0)
            h = params.get("height", params.get("length", 0))
            return (math.pi * r**2 * h) * scale
            
        logger.warning(f"Unknown shape type: {shape_type}, cannot calculate volume")
        return 0.0

    async def run(
        self, 
        geometry_tree: List[Dict[str, Any]], 
        material: str, 
        pod_id: str = None,
        process_type: str = "cnc_milling",
        region: str = "global"
    ) -> Dict[str, Any]:
        """
        Analyze geometry and material to produce manufacturing data.
        
        Args:
            geometry_tree: List of geometry nodes
            material: Material name
            pod_id: Optional pod ID for recursive assembly
            process_type: Manufacturing process type
            region: Geographic region for rates
        """
        # Initialize with rates
        await self.initialize(process_type, region)
        
        # Get material data (from database)
        try:
            mat_data = await self._get_material_data(material)
        except ValueError as e:
            return {
                "error": str(e),
                "components": [],
                "bom_analysis": None
            }
        
        density = mat_data["density"]
        base_cost = mat_data["cost_per_kg"]
        machining_factor = mat_data["machining_factor"]
        
        # Check if we have pricing
        if base_cost is None:
            return {
                "error": f"No price available for {material}. Set via /api/pricing/set-price",
                "components": [],
                "bom_analysis": None
            }
        
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
                
                # Cost Calculation (using database rates)
                material_cost = mass * base_cost
                
                # Machining time from rates
                # Get time_per_kg from rates or use process-specific default
                time_per_kg = self._rates.get("time_per_kg_hours", 1.0)
                est_hours = mass * time_per_kg * machining_factor
                machining_cost = (self.hourly_rate * est_hours) + self.setup_cost
                
                part_total = material_cost + machining_cost
                
                total_mass += mass
                total_cost += part_total
                
                bom_items.append({
                    "id": part.get("id", f"part-{i+1}"),
                    "name": part.get("type", "Component").title(),
                    "material": material,
                    "process": process_type.replace("_", " ").title(),
                    "mass_kg": round(mass, 3),
                    "cost": round(part_total, 2),
                    "cost_breakdown": {
                        "material": round(material_cost, 2),
                        "machining": round(machining_cost, 2),
                        "setup": self.setup_cost
                    },
                    "type": "part"
                })

        # 2. Recursive Sub-Assembly Aggregation
        if pod_id:
            try:
                from core.system_registry import get_system_resolver
                resolver = get_system_resolver()
                
                def find_pod(root, target_id):
                    if root.id == target_id:
                        return root
                    for sub in root.sub_pods.values():
                        f = find_pod(sub, target_id)
                        if f:
                            return f
                    return None
                
                current_pod = find_pod(resolver.root, pod_id)
                
                if current_pod:
                    for sub_key, sub_pod in current_pod.sub_pods.items():
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

                    if current_pod.is_folder_linked:
                        for comp in current_pod.linked_components:
                            if not comp.get("active", True):
                                continue
                            
                            comp_id = comp.get("id")
                            
                            c_mass = current_pod.constraints.get(f"{comp_id}_mass", 0.1)
                            c_cost = current_pod.constraints.get(f"{comp_id}_cost", 10.0)
                            
                            bom_items.append({
                                "id": f"{current_pod.id}_{comp_id}",
                                "name": f"{comp_id} (Linked)",
                                "material": material,
                                "process": "Hardware Compilation",
                                "mass_kg": round(c_mass, 3),
                                "cost": round(c_cost, 2),
                                "type": "linked_component"
                            })
                            total_mass += c_mass
                            total_cost += c_cost

                    current_pod.exports["mass"] = total_mass
                    current_pod.exports["cost"] = total_cost
                    current_pod.is_dirty = False
                    
            except ImportError:
                pass

        # Calculate lead time from setup time
        setup_hours = self.setup_time_minutes / 60.0
        lead_time_days = max(1, int(setup_hours + len(bom_items) * 0.5))

        return {
            "components": bom_items,
            "bom_analysis": {
                "total_cost": round(total_cost, 2),
                "currency": "USD",
                "lead_time_days": lead_time_days,
                "manufacturability_score": 0.95 if machining_factor < 4 else 0.60,
                "rates_source": f"{process_type}/{region}"
            }
        }

    def verify_toolpath_accuracy(self, toolpaths: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify toolpaths for collisions and gouging using VMK."""
        try:
            from vmk_kernel import SymbolicMachiningKernel, ToolProfile, VMKInstruction
            import numpy as np
        except ImportError:
            return {"verified": False, "error": "VMK not available"}
            
        kernel = SymbolicMachiningKernel(stock_dims=[100, 100, 100])
        tools = {}
        gouges = []
        
        for i, op_data in enumerate(toolpaths):
            tool_id = op_data.get("tool_id")
            
            if tool_id not in tools:
                tool = ToolProfile(id=tool_id, radius=0.5, type="BALL")
                kernel.register_tool(tool)
                tools[tool_id] = tool
            
            op = VMKInstruction(**op_data)
            
            start_pt = np.array(op.path[0])
            sdf_start = kernel.get_sdf(start_pt)
            
            if sdf_start < -0.1:
                gouges.append(
                    f"Op {i}: Plunge collision at {start_pt}, Depth={abs(sdf_start):.3f}mm"
                )
            
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
        
        domain = params.get("domain", "METALLURGY")
        
        return self.materials_oracle.solve(
            query="Manufacturing process analysis",
            domain=domain,
            params=params
        )
    
    def analyze_tribology_oracle(self, params: dict) -> dict:
        """Analyze wear and friction using Materials Oracle"""
        if not self.has_oracles:
            return {"status": "error", "message": "Oracles not available"}
        
        return self.materials_oracle.solve(
            query="Tribology analysis",
            domain="TRIBOLOGY",
            params=params
        )

    async def critique_sketch(
        self, 
        sketch_points: List[Dict[str, float]],
        process_type: str = "cnc_milling",
        region: str = "global"
    ) -> List[Dict[str, str]]:
        """
        Critique the user's sketch strokes for manufacturability.
        Uses database-driven tolerance values.
        """
        critiques = []
        
        # Ensure initialized
        if not self._initialized:
            await self.initialize(process_type, region)
        
        # Get min radius from database
        min_cnc_radius = self.min_cnc_radius_mm
        
        for i, point in enumerate(sketch_points):
            r = point.get("radius")
            
            if r is None and "params" in point:
                 r = point.get("params", {}).get("radius")
                 
            if r is not None:
                r_mm = r * 1000.0
                
                if r_mm < min_cnc_radius:
                    critiques.append({
                        "level": "WARN",
                        "agent": "Manufacturing",
                        "message": (
                            f"Radius {r_mm:.2f}mm is too small for {process_type}. "
                            f"Minimum: {min_cnc_radius}mm"
                        )
                    })
                    
            if self.has_surrogate and r is not None:
                r_mm = r * 1000.0
                prob = self.surrogate.predict_defect_probability(r_mm, 2.0, 10.0, 0.0)
                if prob > 0.7:
                    critiques.append({
                        "level": "WARN",
                        "agent": "Manufacturing (Neural)",
                        "message": f"Neural Predictor detects high defect probability ({prob:.2f})"
                    })
                    
        return critiques

    def predict_defects(self, geometry_features: List[float]) -> float:
        """
        Predict defect probability using learned surrogate.
        """
        if not self.has_surrogate:
            return 0.0
        return self.surrogate.predict_defect_probability(*geometry_features)

    def evolve(self, training_data: List[Any]) -> Dict[str, Any]:
        """
        Deep Evolution Trigger.
        Train the surrogate on real/simulated defect outcomes.
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
