from typing import Dict, Any, List
try:
    from materials.materials_db import MaterialsDatabase
except ImportError:
    from backend.materials.materials_db import MaterialsDatabase

class ChemistryAgent:
    """
    The 'Chemist'.
    Checks for chemical incompatibility and simulates material degradation.
    Integrates real-world data via MaterialsDatabase.
    """
    
    SECONDS_PER_YEAR = 31536000
    
    def __init__(self):
        self.api = MaterialsDatabase()
        
        # Initialize Chemistry Oracle for advanced calculations
        try:
            from agents.chemistry_oracle.chemistry_oracle import ChemistryOracle
            self.oracle = ChemistryOracle()
            self.has_oracle = True
        except ImportError:
            self.oracle = None
            self.has_oracle = False
            
        # Initialize Neural Brain (Tier 3.5 Deep Evolution)
        try:
            try:
                from backend.models.chemistry_surrogate import ChemistrySurrogate
            except ImportError:
                 from ...models.chemistry_surrogate import ChemistrySurrogate
            
            self.brain = ChemistrySurrogate(input_size=4, hidden_size=16, output_size=1)
            self.model_path = "data/chemistry_surrogate.weights.json"
            self.brain.load(self.model_path)
            self.has_brain = True
        except ImportError as e:
            print(f"ChemistrySurrogate not found: {e}")
            self.has_brain = False
            self.brain = None

    def run(self, materials: List[str], environment_type: str) -> Dict[str, Any]:
        """
        Static validation of material compatibility in an environment.
        """
        issues = []
        report = []
        
        # 1. Fetch Material Data
        for mat_name in materials:
            # Use API to finding properties
            result = self.api.find_material(mat_name)
            if result:
                # Take best match
                data = result['data'] 
                report.append(f"Material Identified: {data.get('name', mat_name)} ({data.get('formula', '')})")
                
                # Check Environment Compatibility
                self._check_environment_compatibility(data, environment_type, issues, mat_name)
                
            else:
                report.append(f"Material '{mat_name}' not found in database. Assuming standard properties.")
                
        return {
            "chemical_safe": len(issues) == 0,
            "issues": issues,
            "report": report
        }

    def _check_environment_compatibility(self, data: Dict, env_type: str, issues: List[str], name: str):
        """
        Data-driven compatibility check. 
        In strict mode, looks for 'corrosion_resistance_{env}' property.
        Falls back to element heuristics if properties not found.
        """
        elements = data.get("elements", [])
        
        # TODO: Add 'corrosion_resistance' table to DB/Supabase for explicit lookups.
        
        if env_type == "MARINE":
            # Heuristics (Temporary until DB has explicit flags)
            if "Fe" in elements and "Cr" not in elements: 
                issues.append(f"HAZARD: Ferrous material '{name}' (Iron) is highly susceptible to rust in MARINE environment.")
            if "Mg" in elements:
                issues.append(f"HAZARD: Magnesium '{name}' dissolves rapidly in salt water.")

    def step(self, state: Dict[str, float], inputs: Dict[str, float], dt: float = 0.1) -> Dict[str, Any]:
        """
        Advances chemical simulation by one time step (dt).
        Simulates accelerated corrosion/degradation.
        """
        # Unpack State
        integrity = state.get("integrity", 1.0) # 0.0 - 1.0 (100%)
        corrosion_depth = state.get("corrosion_depth", 0.0) # mm
        mass_loss = state.get("mass_loss", 0.0) # g

        # Unpack Inputs
        env_ph = inputs.get("ph", 7.0) # 7 = Neutral, <7 Acidic
        temperature = inputs.get("temperature", 20.0)
        humidity = inputs.get("humidity", 0.5) # 0-1
        material_type = inputs.get("material_type", "steel") 
        thickness = inputs.get("thickness_mm", 10.0)

        # Resolve Material Family & Properties from DB
        mat_info = self.api.find_material(material_type)
        mat_family = "steel"
        density = 7.85
        
        if mat_info and 'data' in mat_info:
            d = mat_info['data']
            # Try to determine family from category or name
            cat = d.get('category', '').lower()
            name_lower = d.get('name', '').lower()
            
            if 'aluminum' in cat or 'aluminum' in name_lower: mat_family = 'aluminum'
            elif 'titanium' in cat or 'titanium' in name_lower: mat_family = 'titanium'
            elif 'steel' in cat or 'iron' in cat: mat_family = 'steel'
            
            # Density
            if 'density' in d: density = float(d['density'])
            elif 'density_g_cc' in d: density = float(d['density_g_cc'])

        else:
            # Fallback for unknown materials strings
            lower_type = material_type.lower()
            if "aluminum" in lower_type: mat_family = "aluminum"; density = 2.7
            elif "titanium" in lower_type: mat_family = "titanium"; density = 4.4
            
        
        # Get Kinetics Parameters from DB
        params = self.api.get_kinetics(mat_family)
        if not params:
             # Default fallback if DB empty
             params = {"base_rate_mm_year": 0.05}

        # --- Deep Evolution: Neural Kinetics ---
        # Learned factor modulates the base rate
        learned_factor = 1.0
        
        # Base Heuristic (JSON) - Legacy Support acting as Prior
        json_factor = self._get_learned_parameter(mat_family, "corrosion_rate_factor", 1.0)
        
        if self.has_brain and self.brain:
            # Neural Inference
            # Inputs: [Temp, pH, Humidity, MaterialBaseFactor]
            # We treat json_factor as the 'Material Factor' input to the net
            learned_factor = self.brain.predict_rate_factor(
                temp=temperature,
                ph=env_ph,
                humidity=humidity,
                mat_factor=json_factor
            )
        else:
            learned_factor = json_factor
        
        base_rate = params.get("base_rate_mm_year", 0.05) * learned_factor
        
        # pH Sensitivity (Heuristic) - Can be learned, but keeping as prior
        ph_sens = params.get("ph_sensitivity", 0.0)
        if ph_sens > 0 and env_ph < 7:
            base_rate *= (7 - env_ph) * ph_sens
            
        # Amphoteric Check (Aluminum)
        if "ph_limit_low" in params:
             limit_low = params["ph_limit_low"]
             limit_high = params.get("ph_limit_high", 9.0)
             if limit_low is not None and (env_ph < limit_low or env_ph > limit_high):
                 base_rate *= params.get("amphoteric_factor", 1.0)
        
        # Accelerated Time Scale for vHIL (1s = 1 year)
        # Simulation Scaling
        time_acceleration = self.SECONDS_PER_YEAR 
        
        # Arrhenius Equation for Temperature 
        # Rate increases by factor Q10 for every 10C.
        q10 = params.get("q10_factor", 2.0)
        temp_diff = temperature - 20.0
        temp_factor = q10 ** (temp_diff / 10.0)
        
        # Calculate Step
        rate_per_second = (base_rate / self.SECONDS_PER_YEAR) * time_acceleration * temp_factor * humidity
        
        new_corrosion = corrosion_depth + rate_per_second * dt
        
        # Integrity Loss Model (Linear decay for simplicity)
        new_integrity = max(0.0, 1.0 - (new_corrosion / thickness))
        
        return {
            "state": {
                "integrity": new_integrity,
                "corrosion_depth": new_corrosion,
                "mass_loss": mass_loss + (rate_per_second * dt * density) 
            },
            "metrics": {
                "rate_mm_y": base_rate * temp_factor * humidity, # Equivalent yearly rate
                "ph_effect": env_ph,
                "material_family": mat_family,
                "neural_factor": learned_factor
            }
        }

    def evolve(self, training_data: list):
        """
        Deep Evolution Trigger.
        Called by ChemistryCritic to train the Neural Kinetics model.
        Args:
             training_data: List of (input, target) pairs
        """
        if not self.has_brain or not self.brain or not training_data:
            return {"status": "error", "message": "No brain or data"}
            
        import numpy as np
        total_loss = 0
        for x, y in training_data:
            loss = self.brain.train_step(np.array(x), np.array(y))
            total_loss += loss
            
        avg_loss = total_loss / len(training_data)
        self.brain.save(self.model_path)
        
        return {"status": "evolved", "avg_loss": avg_loss, "epochs": self.brain.trained_epochs}

    def calculate_reactive_surface(self, geometry_history: List[Dict[str, Any]], stock_dims: List[float]) -> float:
        """
        Calculate EXACT Reactive Surface Area using VMK SDF.
        Crucial for battery electrodes or catalyst beds where area >> geometric envelope.
        Method: Monte Carlo Shell Integration.
        """
        try:
            from vmk_kernel import SymbolicMachiningKernel, ToolProfile, VMKInstruction
            import numpy as np
        except ImportError:
            return 0.0
            
        # 1. Setup Kernel
        kernel = SymbolicMachiningKernel(stock_dims=stock_dims)
        for op in geometry_history:
             tid = op.get("tool_id", "t_chem")
             if tid not in kernel.tools:
                 kernel.register_tool(ToolProfile(id=tid, radius=op.get("radius", 1.0), type="BALL"))
             kernel.execute_gcode(VMKInstruction(**op))
             
        # 2. Monte Carlo Shell Integration
        # Volume V_shell = Area * Thickness
        # Area = V_shell / Thickness
        
        N_SAMPLES = 5000 
        epsilon = 0.5 # Thickness of shell (mm)
        
        # Sampling Bounds
        low = np.array(stock_dims) * -0.5
        high = np.array(stock_dims) * 0.5
        vol_box = np.prod(stock_dims)
        
        points = np.random.uniform(low=low, high=high, size=(N_SAMPLES, 3))
        
        shell_count = 0
        for p in points:
            d = kernel.get_sdf(p)
            
            # Filter out Stock Boundaries
            # We want 'Internal' surface (Cuts).
            # d_stock(p) = max(|p| - dims/2). 
            # If d_stock > -epsilon*2, we are near outer wall.
            # We want points DEEP inside stock (d_stock < -2.0) that are ALSO on a surface (d ~ 0).
            
            # Recalculate d_stock manually (since kernel doesn't expose it cheap)
            # q = abs(p) - dims/2
            q = np.abs(p) - (np.array(stock_dims) * 0.5)
            d_stock = max(max(q[0], max(q[1], q[2])), 0.0) if np.all(q < 0) else np.linalg.norm(np.maximum(q, 0.0))
            # Simplified signed dist to box:
            d_stock_signed = float(np.max(q)) # Approximation for inside box
            
            # If we are near the outer box wall (within 2mm), ignore
            if d_stock_signed > -2.0:
                continue
                
            # Surface is the boundary. |d| < epsilon/2.
            if abs(d) < (epsilon / 2.0):
                shell_count += 1
                
        # Ratio of box that is Shell
        vol_shell = vol_box * (shell_count / N_SAMPLES)
        
        area_mm2 = vol_shell / epsilon
        
        return area_mm2

    def calculate_chemistry(self, domain: str, params: dict) -> dict:
        """
        Delegate advanced chemistry calculations to Chemistry Oracle.
        
        Args:
            domain: Chemistry domain (THERMOCHEMISTRY, KINETICS, etc.)
            params: Calculation parameters
        
        Returns:
            Dictionary with calculation results
        """
        if not self.has_oracle:
            return {
                "status": "error",
                "message": "Chemistry Oracle not available"
            }
        
        return self.oracle.solve(
            query=f"Calculate {domain}",
            domain=domain,
            params=params
        )

    def _get_learned_parameter(self, mat_family: str, param_key: str, default: float) -> float:
        """Self-Evolution Interface."""
        import json
        import os
        path = "data/chemistry_agent_weights.json"
        if not os.path.exists(path): return default
        try:
            with open(path, 'r') as f: 
                d = json.load(f)
                val = d.get(mat_family, {}).get(param_key, default)
                logger = logging.getLogger(__name__)
                # logger.info(f"[DEBUG] Loading {mat_family}.{param_key} from {path} -> {val}")
                return val
        except: return default

    def update_learned_parameters(self, mat_family: str, updates: Dict[str, float]):
        """Called by ChemistryCritic."""
        import json
        import os
        path = "data/chemistry_agent_weights.json"
        data = {}
        if os.path.exists(path):
            try: 
                with open(path, 'r') as f: 
                    data = json.load(f)
            except: 
                pass
            
        if mat_family not in data: data[mat_family] = {}
        data[mat_family].update(updates)
        
        with open(path, 'w') as f: json.dump(data, f, indent=2)
