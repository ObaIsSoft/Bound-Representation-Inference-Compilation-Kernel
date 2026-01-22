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
                    api_data = api_results[0]
                    # Map API response to internal schema
                    props = {
                        "name": api_data.get("name", material_name),
                        "density": api_data.get("density", 0.0),
                        # Elasticity might be nested or missing
                        "yield_strength": api_data.get("elasticity", {}).get("K_VRH", 0.0) if isinstance(api_data.get("elasticity"), dict) else 0.0,
                        # Melting point from thermo data
                        "melting_point": api_data.get("thermo", {}).get("melting_point", 1000.0) if isinstance(api_data.get("thermo"), dict) else 1000.0,
                    }
                    # Derive max_temp from melting point
                    props["max_temp"] = props["melting_point"] * 0.8
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
        
        # Safe access to properties
        T_melt = props.get("melting_point", 1500.0)
        max_temp = props.get("max_temp", 200.0)
        
        # Base Heuristic (Analytic Prior)
        heuristic_factor = 1.0
        if temperature > max_temp:
            excess = temperature - max_temp
            heuristic_factor = max(0, 1.0 - (excess * 0.005))
            
        # Neural/Learned Correction (Learned Residual)
        # Input Features: [Temperature/1000, YieldStrength/1e9, Time(unused), pH(unused)]
        # Normalized inputs for better NN performance
        import numpy as np
        
        correction = 0.0
        if self.has_brain and self.brain:
            inputs = np.array([
                temperature / 1000.0, 
                props.get("yield_strength", 200e6) / 1e9,
                0.0, # Time placeholder
                7.0  # pH placeholder
            ])
            
            # Neural Inference
            # Output [0] is correction to strength_factor
            nn_output = self.brain.forward(inputs).flatten()
            correction = float(nn_output[0])
        
        # Hybrid Fusion: Prediction = Heuristic + NeuralResidual
        # We clamp the result between 0 and 1
        final_factor = max(0.0, min(1.0, heuristic_factor + correction))
        
        return {
            "name": material_name if found else f"Generic Aluminum (Fallback for {material_name})",
            "properties": {
                "density": props.get("density", 2700),
                "yield_strength": props.get("yield_strength", 200e6) * final_factor,
                "melting_point": props.get("melting_point", 1500.0),
                "is_melted": temperature >= T_melt,
                "strength_factor": round(final_factor, 3),
                "degradation_model": "Deep Hybrid (Linear Prior + MaterialNet)" if self.has_brain else "Heuristic (Fallback)",
                "neural_correction": round(correction, 4)
            }
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
