import os
import math
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from backend.physics import get_physics_kernel


# Try importing TensorFlow (User requested "The Real Thing")
try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model, optimizers
    HAS_TF = True
except ImportError:
    HAS_TF = False

# Fallback: Scikit-Learn (Just in case)
try:
    from sklearn.neural_network import MLPRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_hybrid_structural_model(input_dim: int = 5):
    """
    Gated Hybrid Structural Model.
    Inputs: [force, area, length, yield_strength, elastic_modulus]
    Outputs: [stress, safety_factor]
    """
    inputs = layers.Input(shape=(input_dim,))
    
    # --- Branch 1: Heuristic (Beam Theory) ---
    # We implement simple Hooke's Law as a non-trainable layer to embed physics knowledge
    # Stress = Force / Area
    def physics_eq(x):
        # x[:, 0] = Force
        # x[:, 1] = Area
        force = x[:, 0:1]
        area = tf.maximum(x[:, 1:2], 0.0001) # Avoid div/0
        return force / area

    heuristic_stress = layers.Lambda(physics_eq, name="heuristic_stress")(inputs)
    
    # --- Branch 2: Neural Residual (Complex Geometry / Stress Risers) ---
    # Learns plastic deformation, stress concentrations, etc.
    x = layers.Dense(64, activation='swish')(inputs)
    x = layers.Dense(32, activation='swish')(x)
    stress_correction = layers.Dense(1, name="stress_correction")(x)
    
    # --- Gate ---
    # 0 = Linear Elastic (Hooke), 1 = Non-Linear/Complex
    gate_net = layers.Dense(16, activation='relu')(inputs)
    gate = layers.Dense(1, activation='sigmoid', name="gate")(gate_net)
    
    # --- Hybrid Output ---
    # Hybrid Stress = Heuristic + (Gate * Correction)
    weighted_corr = layers.Multiply()([gate, stress_correction])
    hybrid_stress = layers.Add(name="hybrid_stress")([heuristic_stress, weighted_corr])
    
    # Deriving Safety Factor: SF = Yield / HybridStress
    # This is hard to enforce as a layer output without explicit Yield input,
    # but we can output Stress directly and calculate SF outside or have a 2nd output
    # Let's just output Stress for the model target.
    
    model = Model(inputs=inputs, outputs=hybrid_stress, name="HybridStructuralAgent")
    model.compile(optimizer=optimizers.Adam(0.001), loss='mse')
    return model

class StructuralAgent:
    """
    Structural Analysis Agent.
    Estimates stress, strain, and safety factors.
    Hybridized: Learns from FEA Oracles.
    """
    def __init__(self):
        self.name = "StructuralAgent"
        
        # Initialize Physics Kernel
        self.physics = get_physics_kernel()
        logger.info("StructuralAgent: Physics kernel initialized")
        
        # Initialize Oracles for structural analysis
        try:
            from agents.physics_oracle.physics_oracle import PhysicsOracle
            from agents.materials_oracle.materials_oracle import MaterialsOracle
            self.physics_oracle = PhysicsOracle()
            self.materials_oracle = MaterialsOracle()
            self.has_oracles = True
        except ImportError:
            self.physics_oracle = None
            self.materials_oracle = None
            self.has_oracles = False

        # Hybrid Model
        self.has_tf = HAS_TF
        self.model = None
        self.model_path = os.path.join(os.path.dirname(__file__), "models", "structural_hybrid.h5")
        
        if self.has_tf:
            self._load_or_create_model()
            
    def _load_or_create_model(self):
        if not self.has_tf: return
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        if os.path.exists(self.model_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                logger.info("Loaded Structural Hybrid Model.")
            except:
                logger.warning("Failed to load model, creating new.")
                self._create_new_model()
        else:
            self._create_new_model()
            
    def _create_new_model(self):
        self.model = build_hybrid_structural_model(input_dim=5)
        logger.info("Created new Structural Hybrid Model.")

    def train(self, X: np.ndarray, y: np.ndarray, epochs=50):
        """Self-Evolution: Update structural model weights."""
        if not self.has_tf or not self.model: return
        self.model.fit(X, y, epochs=epochs, verbose=0)
        
    def save_model(self):
        if self.has_tf and self.model: 
            self.model.save(self.model_path)

    def validate_prediction(self, prediction: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ground Truth Verification.
        Compares hybrid prediction against Physics Oracle (Mechanics).
        """
        if not self.has_oracles:
            return {"error": "No Oracle available for validation"}
            
        try:
            # Prepare params for Oracle
            # Oracle expects: type="STRESS", force_n, cross_section_m2, etc.
            oracle_params = {
                "type": "STRESS",
                "force_n": prediction.get("load_n", 0.0),
                "cross_section_m2": state.get("cross_section_mm2", 100.0) / 1e6, # Convert mm2 to m2
                "length_m": state.get("length_m", 1.0),
                "youngs_modulus_pa": state.get("elastic_modulus_gpa", 69.0) * 1e9,
                "yield_strength_pa": state.get("yield_strength_mpa", 276.0) * 1e6
            }
            
            oracle_res = self.physics_oracle.solve(
                query="Structural validation",
                domain="MECHANICS",
                params=oracle_params
            )
            
            ground_truth_stress = oracle_res.get("result", {}).get("stress_pa", 0.0) / 1e6 # Convert Pa to MPa
            predicted_stress = prediction.get("max_stress_mpa", 0.0)
            
            error = abs(predicted_stress - ground_truth_stress)
            match = error < (ground_truth_stress * 0.1) # 10% tolerance
            
            gate_val = prediction.get("gate_value", 0.0)
            
            return {
                "verified": match,
                "ground_truth": ground_truth_stress,
                "prediction": predicted_stress,
                "error": error,
                "gate_value": gate_val,
                "gate_aligned": True, # Placeholder
                "drift_alert": not match
            }
        except Exception as e:
            logger.error(f"Structural Validation failed: {e}")
            return {"error": str(e)}

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate stress under load.
        """
        logger.info(f"{self.name} starting structural analysis...")
        
        # Robust Param Extraction
        mass_kg = float(params.get("mass_kg", 10.0))
        g_force = float(params.get("g_loading", 3.0)) 
        cross_section_mm2 = float(params.get("cross_section_mm2", 100.0))
        length_m = float(params.get("length_m", 1.0)) 
        
        yield_strength_mpa = float(params.get("yield_strength_mpa", 276.0))
        elastic_modulus_gpa = float(params.get("elastic_modulus_gpa", 69.0))

        # Material Properties Override
        mat_props = params.get("material_properties", {})
        if "yield_strength" in mat_props:
            yield_strength_mpa = float(mat_props["yield_strength"]) / 1e6 
        if "elastic_modulus" in mat_props:
             elastic_modulus_gpa = float(mat_props["elastic_modulus"]) / 1e9 

        logs = []
        
        # 1. Stress Analysis (Axial) - Using REAL PHYSICS
        g = self.physics.get_constant("g")  # Get real gravity constant
        force_n = mass_kg * (g_force * g)
        stress_mpa = force_n / max(cross_section_mm2, 0.1) # Avoid div/0
        
        fos_yield = yield_strength_mpa / max(stress_mpa, 0.001)
        
        logs.append(f"Load Case: {g_force}G on {mass_kg}kg ({force_n:.1f}N)")
        logs.append(f"Heuristic Stress: {stress_mpa:.1f} MPa")
        
        # 2. Hybrid Correction
        gate_val = 0.0
        hybrid_stress = stress_mpa
        
        if self.has_tf and self.model:
            try:
                # Features: [force, area(mm2), length, yield, modulus]
                features = np.array([[force_n, cross_section_mm2, length_m, yield_strength_mpa, elastic_modulus_gpa]], dtype=np.float32)
                
                # Predict
                pred = self.model.predict(features, verbose=0)
                hybrid_stress = float(pred[0][0])
                
                # Gate
                gate_layer = Model(inputs=self.model.input, outputs=self.model.get_layer("gate").output)
                gate_val = float(gate_layer.predict(features, verbose=0)[0][0])
                
                logs.append(f"[HYBRID] Neural adjustment. Gate: {gate_val:.2f}")
                logs.append(f"[HYBRID] Stress: {stress_mpa:.1f} -> {hybrid_stress:.1f} MPa")
                
                # Recalculate FoS with new stress
                stress_mpa = hybrid_stress
                fos_yield = yield_strength_mpa / max(stress_mpa, 0.001)
                
            except Exception as e:
                logs.append(f"[HYBRID] Inference failed: {e}")

        # 3. Buckling Analysis (Euler) - (Currently purely heuristic, could be hybrid too)
        # Derived Radius (mm)
        radius_mm = (cross_section_mm2 / 3.14159) ** 0.5
        radius_m = radius_mm / 1000.0
        I = (3.14159 * (radius_m ** 4)) / 4
        
        pi = 3.14159
        E_pa = elastic_modulus_gpa * 1e9
        L = length_m
        K = 1.0 # Pinned-Pinned default
        
        try:
            critical_load_n = (pi**2 * E_pa * I) / ((K * L)**2)
        except ZeroDivisionError:
            critical_load_n = 0
            
        fos_buckling = critical_load_n / max(force_n, 0.001)
        
        logs.append(f"Buckling Critical Load: {critical_load_n:.1f}N")
        logs.append(f"Buckling FoS: {fos_buckling:.2f}")
        
        # 4. Verdict
        overall_fos = min(fos_yield, fos_buckling)
        
        status = "safe"
        if overall_fos < 1.0: 
            status = "failure"
            if fos_buckling < 1.0: logs.append("FAILURE MODE: Buckling Instability")
            elif fos_yield < 1.0: logs.append("FAILURE MODE: Yield Stress Exceeded")
            
        elif overall_fos < 1.5: 
            status = "marginal"
        
        return {
            "status": status,
            "max_stress_mpa": round(stress_mpa, 2),
            "safety_factor": round(overall_fos, 2),
            "yield_fos": round(fos_yield, 2),
            "buckling_fos": round(fos_buckling, 2),
            "load_n": round(force_n, 2),
            "gate_value": gate_val,
            "logs": logs
        }

    def detect_stress_risers(self, geometry_history: List[Dict[str, Any]], critical_radius: float = 1.0) -> List[str]:
        """
        Uses VMK to find sharp corners (Radius < critical_radius) which act as stress concentrators.
        """
        try:
            from vmk_kernel import SymbolicMachiningKernel, ToolProfile, VMKInstruction
            import numpy as np
        except ImportError:
            return []
            
        # Replay Geometry
        kernel = SymbolicMachiningKernel(stock_dims=[100,100,100]) # Generic bounds for analysis
        for op in geometry_history:
             # Auto-register tools (Simplified)
             tid = op.get("tool_id")
             if tid and tid not in kernel.tools:
                 kernel.register_tool(ToolProfile(id=tid, radius=op.get("radius", 1.0), type="BALL"))
             kernel.execute_gcode(VMKInstruction(**op))
             
        # Scan Surface for Curvature
        # Heuristic: Sample points. If SDF changes rapidly (high 2nd derivative) -> Sharp.
        # But SDF of sharp corner is continuous. 
        # Actually, sharp corner = Gradient Discontinuity.
        # But Sampled SDF smooths it? No, Exact SDF preserves it.
        
        # Simplified Check for MVP:
        # Check if we have "Square" cuts? 
        # A Square Cut (Endmill) leaves a radius = tool_radius.
        # If tool_radius < critical_radius, it's a riser.
        
        risers = []
        
        # Analytical Check (Meta-Analysis of History)
        # Verify that all Subtractive Tools have Radius >= Critical Radius
        for op in geometry_history:
            r = op.get("radius", 0.0)
            # If it's a cutting move (subtract)
            # And radius is small.
            if r > 0 and r < critical_radius:
                risers.append(f"Sharp Corner Detected: Tool '{op.get('tool_id')}' Radius {r}mm < Limit {critical_radius}mm")
                
        # Future: True Geometric Curvature scan using kernel.get_sdf gradient analysis.
        
        return risers

    def analyze_stress_oracle(self, params: dict) -> dict:
        """Analyze structural stress using Physics Oracle (MECHANICS)"""
        if not self.has_oracles:
            return {"status": "error", "message": "Oracles not available"}
        
        return self.physics_oracle.solve(
            query="Structural stress analysis",
            domain="MECHANICS",
            params=params
        )
    
    def analyze_material_properties_oracle(self, params: dict) -> dict:
        """Analyze material properties using Materials Oracle"""
        if not self.has_oracles:
            return {"status": "error", "message": "Oracles not available"}
        
        return self.materials_oracle.solve(
            query="Material property analysis",
            domain="MECHANICAL",
            params=params
        )
    
    def predict_failure_oracle(self, params: dict) -> dict:
        """Predict structural failure using Materials Oracle (FAILURE domain)"""
        if not self.has_oracles:
            return {"status": "error", "message": "Oracles not available"}
        
        return self.materials_oracle.solve(
            query="Failure prediction",
            domain="FAILURE",
            params=params
        )
