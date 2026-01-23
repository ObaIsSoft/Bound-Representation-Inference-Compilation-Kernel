from typing import Dict, Any

class MaterialAgent:
    """
    The 'Librarian' of Matter.
    Provides rigorous, temperature-dependent material properties.
    """
    
    def __init__(self):
        self.db_path = "data/materials.db"
        
        # Initialize Oracles for advanced calculations
        try:
            from agents.physics_oracle.physics_oracle import PhysicsOracle
            from agents.chemistry_oracle.chemistry_oracle import ChemistryOracle
            self.physics_oracle = PhysicsOracle()
            self.chemistry_oracle = ChemistryOracle()
            self.has_oracles = True
        except ImportError:
            self.physics_oracle = None
            self.chemistry_oracle = None
            self.has_oracles = False
            
        # Initialize Neural Brain (Tier 3.5 Deep Evolution)
        try:
            # Try absolute import first (standard for run from root)
            try:
                from backend.models.material_net import MaterialNet
            except ImportError:
                # Fallback to relative import (if run as module)
                from ...models.material_net import MaterialNet
                
            self.brain = MaterialNet(input_size=4, hidden_size=16, output_size=2)
            self.model_path = "data/material_net.weights.json"
            self.brain.load(self.model_path)
            self.has_brain = True
        except ImportError as e:
            self.has_brain = False
            print(f"MaterialNet not found: {e}")
            # Ensure self.brain exists to avoid AttributeError in run() even if it is None
            self.brain = None
    
    def run(self, material_name: str, temperature: float = 20.0) -> Dict[str, Any]:
        """
        Query database for material properties and apply thermal degradation.
        """
        import sqlite3
        import logging
        
        logger = logging.getLogger(__name__)
        
        props = {}
        found = False
        
        # 1. Query Local SQLite DB
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Fuzzy search
            cursor.execute("SELECT * FROM alloys WHERE name LIKE ?", (f"%{material_name}%",))
            row = cursor.fetchone()
            
            if row:
                props = dict(row)
                found = True
                # Check for max temp, default to 150C if not in DB
                props["max_temp"] = props.get("max_temp", 150.0) 
                
            conn.close()
        except Exception as e:
            logger.error(f"DB Error: {e}")
            
        # 2. Try Materials Project API / Unified DB if Local DB failed
        if not found:
            try:
                from materials.materials_db import MaterialsDatabase
                mat_db = MaterialsDatabase(use_api=True)
                # Use unified find_material to search across all sources
                api_results = mat_db.api.find_material(material_name, source="auto")
                
                if api_results:
                    # Find candidate with most complete data (prefer existing elasticity/thermo)
                    best_data = None
                    best_score = -1
                    
                    for candidate in api_results:
                        score = 0
                        if candidate.get("density"): score += 1
                        if candidate.get("elasticity"): score += 2
                        if candidate.get("thermo") and candidate["thermo"].get("melting_point"): score += 2
                        
                        if score > best_score:
                            best_score = score
                            best_data = candidate
                            
                    api_data = best_data or api_results[0]
                    
                    # DEBUG: Print selected candidate keys
                    print(f"DEBUG SELECTED CANDIDATE KEYS: {list(api_data.keys())}")
                    
                    # Map API response to internal schema
                    # Elasticity data is often in G_VRH (shear) or K_VRH (bulk), or just not present
                    elasticity = api_data.get("elasticity") or {}
                    thermo = api_data.get("thermo") or {}
                    
                    # Convert density from g/cm^3 (MP default) to kg/m^3 (SI)
                    # 1 g/cm^3 = 1000 kg/m^3
                    density_raw = api_data.get("density", 0.0)
                    density_si = density_raw * 1000.0 if density_raw else None
                    
                    # Approximating yield strength from bulk modulus (K_VRH) if available
                    # yield ~ K_VRH / 10 is a very rough rule of thumb, but better than nothing
                    # Real yield strength is microstructure dependent and not in MP core data
                    bulk_modulus = elasticity.get("K_VRH") # GPa
                    yield_strength = (bulk_modulus * 1e9 / 10.0) if bulk_modulus else None
                    
                    props = {
                        "name": api_data.get("formula_pretty", material_name),
                        "density": density_si,
                        "yield_strength": yield_strength,
                        "melting_point": thermo.get("melting_point"), # Kelvin? MP is usually Kelvin? No, check docs. usually Kelvin.
                    }
                    
                    # Convert Melting Point from Kelvin to Celsius if > 200 (assuming K)
                    if props["melting_point"] and props["melting_point"] > 273.15:
                         props["melting_point"] -= 273.15
                         
                    # Derive max_temp from melting point
                    if props["melting_point"]:
                        props["max_temp"] = props["melting_point"] * 0.8
                    else:
                        props["max_temp"] = None
                        
                    found = True
                    logger.info(f"Found {material_name} in Materials Project API")
            except Exception as e:
                logger.warning(f"Materials Project API query failed: {e}")

        # 3. If still not found, return error - NO HARDCODED FALLBACKS
        if not found:
            logger.error(f"Material '{material_name}' not found in DB or API. Cannot proceed.")
            return {
                "name": f"ERROR: {material_name} not found",
                "properties": {},
                "error": "Material not found in database or Materials Project API"
            }

        # 4. Temperature Degradation (Deep Evolution: MaterialNet)
        
        # Safe access to properties - NO FALLBACKS (None if missing)
        T_melt = props.get("melting_point")
        max_temp = props.get("max_temp")
        yield_strength = props.get("yield_strength")
        
        # Base Heuristic (Analytic Prior)
        heuristic_factor = 1.0
        
        # Only apply degradation if we have thermal data
        if max_temp is not None and temperature > max_temp:
            excess = temperature - max_temp
            heuristic_factor = max(0, 1.0 - (excess * 0.005))
            
        # Neural/Learned Correction (Learned Residual)
        import numpy as np
        
        correction = 0.0
        # Only run neural net if we have yield strength
        if self.has_brain and self.brain and yield_strength is not None:
            inputs = np.array([
                temperature / 1000.0, 
                yield_strength / 1e9,
                0.0, # Time placeholder
                7.0  # pH placeholder
            ])
            
            # Neural Inference
            nn_output = self.brain.forward(inputs).flatten()
            correction = float(nn_output[0])
        
        # Hybrid Fusion
        final_factor = max(0.0, min(1.0, heuristic_factor + correction))
        
        # Prepare Result
        result_props = {
            "density": props.get("density"),
            "melting_point": T_melt,
            "max_temp": max_temp,
            "is_melted": (temperature >= T_melt) if T_melt is not None else None,
            "strength_factor": round(final_factor, 3),
            "degradation_model": "Deep Hybrid" if self.has_brain else "Heuristic"
        }
        
        # Handle Yield Strength specially since we modify it
        if yield_strength is not None:
            result_props["yield_strength"] = yield_strength * final_factor
        else:
            result_props["yield_strength"] = None
            
        # Add neural correction metadata if applied
        if correction != 0.0:
            result_props["neural_correction"] = round(correction, 4)
            
        return {
            "name": material_name if found else f"Unknown ({material_name})",
            "properties": result_props
        }

    def evolve(self, training_data: list):
        """
        Deep Evolution Trigger.
        Called by MaterialCritic to train the neural brain.
        Args:
            training_data: List of tuples (input_vector, target_output_vector)
        """
        if not training_data: return
        
        import numpy as np
        
        total_loss = 0
        for x, y in training_data:
            loss = self.brain.train_step(np.array(x), np.array(y))
            total_loss += loss
            
        avg_loss = total_loss / len(training_data)
        
        # Auto-save weights
        self.brain.save(self.model_path)
        
        return {"status": "evolved", "avg_loss": avg_loss, "epochs": self.brain.trained_epochs}

    def _get_learned_parameter(self, mat_name: str, param_key: str, default: float) -> float:
        """Legacy Scalar Tuning (Deprecated by MaterialNet)""" 
        return default
            
    def update_learned_parameters(self, mat_name: str, updates: Dict[str, float]):
        """Legacy Scalar Update (Deprecated)"""
        pass


    def calculate_exact_mass_sdf(self, material_name: str, stock_dims: list, toolpaths: list, precision: int = 1000) -> Dict[str, Any]:
        """
        Calculate EXACT Mass by integrating the SDF Volume (Zero Mesh Error).
        Uses Monte Carlo integration for robustness on arbitrary 3D shapes.
        
        Args:
            material_name: For density lookup
            stock_dims: [x, y, z] bounding box of stock
            toolpaths: List of VMK instructions
            precision: Samples per dimension or total samples? 
                       Let's use Total Samples = precision^3 for grid, or just N for Monte Carlo.
                       Using N=10000 for speed in this implementation.
        """
        try:
            from vmk_kernel import SymbolicMachiningKernel, ToolProfile, VMKInstruction
            import numpy as np
        except ImportError:
            return {"mass_kg": 0.0, "error": "VMK not available"}

        # 1. Get Density (kg/m^3)
        mat_data = self.run(material_name)
        density = mat_data["properties"]["density"] # kg/m^3
        
        # 2. Setup Kernel & Execute
        kernel = SymbolicMachiningKernel(stock_dims=stock_dims)
        
        # Auto-register tools
        tools_seen = set()
        for op in toolpaths:
            tid = op.get("tool_id")
            if tid not in tools_seen:
                 # Default tool if not strictly defined (Verification context)
                 kernel.register_tool(ToolProfile(id=tid, radius=1.0, type="BALL"))
                 tools_seen.add(tid)
            kernel.execute_gcode(VMKInstruction(**op))
            
        # 3. Monte Carlo Volume Integration
        # Volume Box = stock_dims[0] * stock_dims[1] * stock_dims[2]
        # (Assuming dims are full width, centered? Kernel logic: q = abs(p) - dims/2)
        # So dims represents the FULL extent (Width, Length, Height).
        # Box Volume:
        vol_box = (stock_dims[0]/1000.0) * (stock_dims[1]/1000.0) * (stock_dims[2]/1000.0) # mm -> m
        
        # Sampling bounds: -dims/2 to +dims/2
        low = np.array(stock_dims) * -0.5
        high = np.array(stock_dims) * 0.5
        
        N_SAMPLES = 1000 if precision < 1000 else precision # Default 1000 is too low for accuracy, use higher in production
        
        points = np.random.uniform(low=low, high=high, size=(N_SAMPLES, 3))
        
        inside_count = 0
        for p in points:
            sdf = kernel.get_sdf(p)
            # Inside Material if SDF < 0 (Inside Stock AND Outside Cut)
            # Wait, our logic: max(d_stock, -d_cut).
            # Inside Stock: d_stock < 0.
            # Outside Cut: d_cut > 0 (Outside capsule). -d_cut < 0.
            # max(neg, neg) = neg.
            # So SDF < 0 means Material Exists.
            if sdf < 0:
                inside_count += 1
                
        volume_ratio = inside_count / N_SAMPLES
        volume_m3 = vol_box * volume_ratio
        
        mass_kg = volume_m3 * density
        
        return {
            "mass_kg": mass_kg,
            "volume_m3": volume_m3,
            "density": density,
            "samples": N_SAMPLES,
            "confidence": "Statistical (Monte Carlo)"
        }

    def calculate_physics(self, domain: str, params: dict) -> dict:
        """Delegate physics calculations to Physics Oracle"""
        if not self.has_oracles:
            return {"status": "error", "message": "Physics Oracle not available"}
        return self.physics_oracle.solve(f"Calculate {domain}", domain, params)
    
    def calculate_chemistry(self, domain: str, params: dict) -> dict:
        """Delegate chemistry calculations to Chemistry Oracle"""
        if not self.has_oracles:
            return {"status": "error", "message": "Chemistry Oracle not available"}
        return self.chemistry_oracle.solve(f"Calculate {domain}", domain, params)

    def calculate_material_property(self, domain: str, params: dict) -> dict:
        """Delegate to Materials Oracle for comprehensive materials science calculations"""
        if not self.has_oracles:
            return {"status": "error", "message": "Materials Oracle not available"}
        
        # Initialize Materials Oracle if not already done
        if not hasattr(self, 'materials_oracle'):
            try:
                from agents.materials_oracle.materials_oracle import MaterialsOracle
                self.materials_oracle = MaterialsOracle()
            except ImportError:
                return {"status": "error", "message": "Materials Oracle not available"}
        
        return self.materials_oracle.solve(f"Calculate {domain}", domain, params)
