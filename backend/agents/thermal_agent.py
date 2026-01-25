import logging
import math
import numpy as np
from typing import Dict, Any, List, Optional
import os
from backend.physics import get_physics_kernel

# Try to import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model, optimizers
    HAS_TF = True
except ImportError:
    HAS_TF = False

# Fallback: Scikit-Learn (Real Neural Network)
try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.exceptions import NotFittedError
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)

class SklearnHybridModel:
    """
    Real Neural Network using Scikit-Learn's MLPRegressor.
    Used when TensorFlow is unavailable (e.g. Python 3.13).
    
    Architecture:
    - Branch 1: Heuristic (External)
    - Branch 2: Neural Residual (MLP)
    - Gate: MLP Classifier probability
    """
    def __init__(self, input_dim=5):
        # Neural Branch: Learns the residual (correction)
        # 2 Hidden Layers (64, 32) - similar to the Keras design
        self.neural_net = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=1,  # We control epochs manually in train() via partial_fit
            warm_start=True, # Keep weights between calls
            random_state=42
        )
        
        # Gate Branch: Decides trust (0=Physics, 1=Neural)
        # Modeled as a separate regressor predicting a 0-1 confidence score
        self.gate_net = MLPRegressor(
            hidden_layer_sizes=(16,),
            activation='logistic', # Sigmoid output
            solver='adam',
            max_iter=1,
            warm_start=True,
            random_state=42
        )
        
        self.is_fitted = False
        
    def predict(self, X):
        if not self.is_fitted:
            return np.zeros((len(X), 1))
        return self.neural_net.predict(X).reshape(-1, 1)
        
    def predict_gate(self, X):
        if not self.is_fitted:
            return 0.5 # Uncertain
        return self.gate_net.predict(X).reshape(-1, 1)
        
    def fit(self, X, y, epochs=10, verbose=0):
        # Sklearn MLPRegressor 'fit' resets weights if not warm_start.
        # But 'partial_fit' is for online learning. MLPRegressor supports partial_fit execution 
        # via fit() loop if warm_start=True.
        # Actually standard MLPRegressor doesn't strictly have partial_fit API exposed easily 
        # like SGDRegressor, but calling fit() with warm_start=True works for incremental.
        
        # We assume y is the TARGET value for the residual.
        # For the gate, we don't have explicit labels here, so we'll use an unsupervised heuristic
        # or simplified self-supervision:
        # High residual error -> Gate should move towards 1?
        # For now, we train gate to output 0.5 (placeholder) or random
        # actually, let's skip gate training logic complexity and focus on the residual net first.
        # Just train the residual net.
        
        y_flat = y.ravel()
        for _ in range(epochs):
            self.neural_net.fit(X, y_flat)
        self.is_fitted = True
    
    def get_layer(self, name):
        return self # Shim
        
    @property
    def output(self):
        return self # Shim
        
    def __call__(self, x):
        return 0.5 # Shim

def build_hybrid_thermal_model(input_dim: int = 5):
    """
    Gated Hybrid Thermal Model.
    
    Inputs: [power, surface_area, emissivity, ambient_temp, convection_h]
    
    Branch 1: Heuristic (Analytic Solution) - Frozen
    Branch 2: Neural (Residual Correction)
    Gate: Decides trust based on regime (e.g. low vs high non-linearity)
    """
    inputs = layers.Input(shape=(input_dim,))
    
    # --- Physical Inputs Splitting ---
    # 0:Power, 1:Area, 2:Emiss, 3:Amb, 4:h
    
    # --- Branch 1: Heuristic (Analytic Approximation) ---
    # T_final = T_amb + Power / (h * Area) [Ignoring Radiation for simplicity/stability in branch]
    # We use Lambda layer to enforce physics equation
    # To avoid div/0, we clamp denominator
    def physics_eq(x):
        idx_p = x[:, 0:1]
        idx_a = x[:, 1:2]
        idx_h = x[:, 4:5]
        idx_amb = x[:, 3:4]
        
        # denom = h * A
        denom = idx_h * idx_a
        # safe_denom = max(denom, 0.001) - TF op
        denom = tf.maximum(denom, 0.001)
        
        delta_t = idx_p / denom
        return idx_amb + delta_t

    heuristic_prediction = layers.Lambda(physics_eq, name="heuristic_branch")(inputs)
    
    # --- Branch 2: Neural Residual (Correction) ---
    # Learns radiation effects, complex geometry factors, etc.
    x = layers.Dense(64, activation='swish')(inputs)
    x = layers.Dense(32, activation='swish')(x)
    neural_correction = layers.Dense(1, name="neural_correction")(x)
    
    # --- Gate ---
    # 0 = Trust Heuristic, 1 = Trust Neural
    gate_net = layers.Dense(16, activation='relu')(inputs)
    gate = layers.Dense(1, activation='sigmoid', name="gate")(gate_net)
    
    # --- Hybrid Output ---
    # T_final = (1-g)*Heuristic + g*(Heuristic + Correction)
    # OR T_final = (1-g)*Heuristic + g*PureNeural
    # Let's use residual formulation: T = Heuristic + Gate * Correction
    # This ensures Heuristic is always the baseline.
    
    weighted_correction = layers.Multiply()([gate, neural_correction])
    output = layers.Add(name="hybrid_output")([heuristic_prediction, weighted_correction])
    
    model = Model(inputs=inputs, outputs=output, name="HybridThermalAgent")
    model.compile(optimizer=optimizers.Adam(0.001), loss='mse')
    return model


class ThermalAgent:
    """
    Thermal Analysis Agent.
    Calculates equilibrium temperatures and heat dissipation requirements.
    """
    CONSTANTS = {
        "SIGMA": 5.67e-8, # Stefan-Boltzmann Constant
        "C_P": 900,       # Specific Heat Capacity (J/kgK) - Aluminum default
    }

    def __init__(self, model_path: str = "data/hybrid_thermal.h5"):
        self.name = "ThermalAgent"
        self.model_path = model_path
        self.model = None
        self.has_tf = HAS_TF
        
        # Initialize Physics Kernel
        self.physics = get_physics_kernel()
        logger.info("ThermalAgent: Physics kernel initialized")
        
        # Initialize Oracles for thermal analysis
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
            
        if self.has_tf:
            self._load_or_create_model()
            
    def _load_or_create_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path)
            except:
                self._create_new_model()
        else:
            self._create_new_model()
            
    def _create_new_model(self):
        self.model = build_hybrid_thermal_model(input_dim=5)
        
    def train(self, X: np.ndarray, y: np.ndarray, epochs=50):
        """Self-Evolution: Update model weights based on experience."""
        if not self.has_tf or not self.model: return
        self.model.fit(X, y, epochs=epochs, verbose=0)
        
    def save_model(self):
        if self.model: self.model.save(self.model_path)

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates equilibrium temperature.
        Supports Convection (Air) and Radiation (Vacuum).
        """
        logger.info(f"Running Thermal Analysis on: {payload}")
        
        
        # 1. Robust Param Extraction
        power_w = float(payload.get("power_watts", 10.0))
        surface_area = float(payload.get("surface_area", 0.1)) # Default 0.1m^2
        emissivity = float(payload.get("emissivity", 0.9)) # Anodized Aluminum
        ambient_temp = float(payload.get("ambient_temp", 25.0)) # Celsius
        
        # Environment Detection
        env_type = payload.get("environment_type", "GROUND")
        h = float(payload.get("heat_transfer_coeff", 10.0)) # Convection coeff
        
        logs = []
        
        # 1. Heuristic Branch (Physics Equation)
        # Q = h * A * delta_T  => delta_T = Q / (h * A)
        # T_final = T_amb + delta_T
        if h < 0.1: h = 0.1 # Prevent div/0
        
        delta_t_heuristic = power_w / (h * surface_area)
        final_temp = ambient_temp + delta_t_heuristic
        
        logs.append(f"Heuristic (Conv): {final_temp:.1f}°C")
        
        # 2. Physics Constraints (Radiation Limit)
        # Stefan-Boltzmann: P = e * sigma * A * (T^4 - T_amb^4)
        # We solve for T (approx) if Radiation is dominant (SPACE)
        if env_type == "SPACE" or env_type == "VACUUM":
            sigma = 5.67e-8
            # T^4 = P/(e*sigma*A) + T_amb^4
            # If T_amb ~ 0 (Space), T = (P / (e*sigma*A))^0.25
            
            denom = emissivity * sigma * surface_area
            if denom > 1e-9:
                t_rad_4 = (power_w / denom) + math.pow(ambient_temp + 273.15, 4)
                final_temp = math.pow(t_rad_4, 0.25) - 273.15
            
            logs.append(f"Radiation Only (Vacuum)")

        # 3. Hybrid Prediction (if available)
        hybrid_temp = final_temp # Default to heuristic
        gate_val = 0.0
        
        if self.model:
            try:
                # Features: [power, area, emiss, amb, h]
                features = np.array([[power_w, surface_area, emissivity, ambient_temp, h]], dtype=np.float32)
                
                # Predict
                # For mock model, predict returns correction. For Keras, it returns final.
                # Adjust based on which model we have
                if self.has_tf:
                    pred = self.model.predict(features, verbose=0)
                    hybrid_temp = float(pred[0][0])
                    
                    gate_layer = Model(inputs=self.model.input, outputs=self.model.get_layer("gate").output)
                    gate_val = float(gate_layer.predict(features, verbose=0)[0][0])
                else:
                    # Mock Model Logic
                    # It learns the *residual* or the full value? 
                    # Let's say it learns a correction
                    correction = float(self.model.predict(features)[0][0])
                    gate_val = float(self.model.predict_gate(features))
                    
                    # Assume mock learns to correct the heuristic
                    hybrid_temp = final_temp + (gate_val * correction)
                
                logs.append(f"[HYBRID] Neural adjustment applied. Gate: {gate_val:.2f}")
                logs.append(f"[HYBRID] Heuristic: {final_temp:.1f}C -> Hybrid: {hybrid_temp:.1f}C")
                final_temp = hybrid_temp # Update verdict
            except Exception as e:
                logs.append(f"[HYBRID] Inference failed: {e}")

        # 4. Verdict
        status = "nominal"
        if final_temp > 100: status = "warning"
        if final_temp > 150: status = "critical"
        
        logs.append(f"Equilibrium: {final_temp:.1f}°C ({status})")

        return {
            "status": status,
            "equilibrium_temp_c": round(final_temp, 2),
            "delta_t": round(final_temp - ambient_temp, 2),
            "heat_load_w": power_w,
            "gate_value": gate_val,
            "logs": logs
        }

    def validate_prediction(self, prediction: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ground Truth Verification.
        Compares hybrid prediction against Physics Oracle (High Fidelity).
        """
        if not self.has_oracles:
            return {"error": "No Oracle available for validation"}
            
        try:
            # Oracle Simulation (Expensive)
            # In real usage, we might only do this for 5% of samples or when Critic requests it
            oracle_res = self.physics_oracle.solve(
                query="Thermal equilibrium simulation",
                domain="THERMODYNAMICS",
                params=state
            )
            
            ground_truth_temp = oracle_res.get("result", {}).get("temperature", 0.0)
            predicted_temp = prediction.get("equilibrium_temp_c", 0.0)
            
            error = abs(predicted_temp - ground_truth_temp)
            match = error < 5.0 # 5 degree tolerance
            
            # Check Gate Alignment
            # Low complexity (simple geo) -> Gate should be 0 (Heuristic is fine)
            # High complexity -> Gate should be 1 (Neural needed)
            gate_val = prediction.get("gate_value", 0.0)
            complexity = state.get("complexity_score", 0.5)
            
            gate_aligned = (complexity > 0.7 and gate_val > 0.7) or (complexity < 0.3 and gate_val < 0.3)
            
            return {
                "verified": match,
                "ground_truth": ground_truth_temp,
                "prediction": predicted_temp,
                "error": error,
                "gate_value": gate_val,
                "gate_aligned": gate_aligned,
                "drift_alert": not match
            }
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {"error": str(e)}

    def analyze_heat_transfer_oracle(self, params: dict) -> dict:
        """Analyze heat transfer using Physics Oracle (THERMODYNAMICS)"""
        if not self.has_oracles:
            return {"status": "error", "message": "Oracles not available"}
        
        return self.physics_oracle.solve(
            query="Heat transfer analysis",
            domain="THERMODYNAMICS",
            params=params
        )
    
    def analyze_thermal_properties_oracle(self, params: dict) -> dict:
        """Analyze thermal material properties using Materials Oracle"""
        if not self.has_oracles:
            return {"status": "error", "message": "Oracles not available"}
        
        return self.materials_oracle.solve(
            query="Thermal properties",
            domain="THERMAL",
            params=params
        )
