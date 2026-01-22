import os
import numpy as np
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Try to import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model, optimizers
    HAS_TF = True
except ImportError:
    HAS_TF = False
    logger.warning("TensorFlow not available. SurrogateAgent will run in fallback mode.")

def build_rigorous_hybrid_agent(input_dim):
    """
    Gated Hybrid Agent Architecture.
    
    Combines:
    - Physics Branch (F = m * a) - frozen heuristic
    - Neural Branch - learns residuals (turbulence, friction, etc.)
    - Gate - meta-cognitive decision: when to trust physics vs neural
    
    Args:
        input_dim: Number of input features
        
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=(input_dim,))

    # --- Heuristic Branch (Frozen/Static) ---
    # F = m * a where inputs[0] is mass, inputs[1] is acceleration
    physics_truth = layers.Lambda(lambda x: x[:, 0:1] * x[:, 1:2], name="physics_branch")(inputs)

    # --- Neural Branch (Adaptive) ---
    # Learns the 'Residuals' (e.g., friction, drag, turbulence, noise)
    neural_res = layers.Dense(128, activation='swish', name="neural_layer1")(inputs)
    neural_res = layers.Dense(64, activation='swish', name="neural_layer2")(neural_res)
    intuition = layers.Dense(1, name="intuition")(neural_res)

    # --- The Gating Branch (The Meta-Cognitive Head) ---
    # Determines the weight of the neural branch vs the physics branch
    # Low speeds → gate ≈ 0 (trust physics)
    # High speeds with turbulence → gate ≈ 1 (trust neural)
    gate_logic = layers.Dense(32, activation='relu', name="gate_logic")(inputs)
    gate = layers.Dense(1, activation='sigmoid', name="gate")(gate_logic)

    # --- The Final Weighted Sum ---
    # Total = (1-gate)*Physics + gate*Intuition
    inv_gate = layers.Lambda(lambda x: 1.0 - x, name="inv_gate")(gate)
    
    physics_part = layers.Multiply(name="physics_weighted")([physics_truth, inv_gate])
    intuition_part = layers.Multiply(name="intuition_weighted")([intuition, gate])
    
    output = layers.Add(name="hybrid_output")([physics_part, intuition_part])

    model = Model(inputs=inputs, outputs=output, name="GatedHybridAgent")
    
    # Using Huber Loss to prevent outliers from disrupting the learning of residuals
    model.compile(optimizer=optimizers.Adam(0.001), loss='huber')
    
    return model


class SurrogateAgent:
    """
    Neural Surrogate Agent (The "Gated Hybrid Oracle").
    
    Uses a Gated Hybrid Model that combines:
    1. Physics heuristics (F = m*a) for low-speed, well-understood regimes
    2. Neural intuition for complex residuals (turbulence, friction)
    3. Meta-cognitive gate to decide when to trust each
    
    Acts as a 'Fast Filter' to reject obviously bad designs before expensive simulation.
    """
    def __init__(self, model_path: str = "data/hybrid_surrogate.h5"):
        self.name = "SurrogateAgent"
        self.model_path = model_path
        self.model = None
        self.has_tf = HAS_TF
        
        if self.has_tf:
            self._load_or_create_model()
        else:
            logger.warning(f"{self.name}: TensorFlow not available. Running in passthrough mode.")

    def _load_or_create_model(self):
        """Loads the trained model or creates a new one."""
        if os.path.exists(self.model_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                logger.info(f"{self.name} loaded trained hybrid model from {self.model_path}")
            except Exception as e:
                logger.error(f"{self.name} failed to load model: {e}")
                logger.info("Creating new hybrid model...")
                self._create_new_model()
        else:
            logger.info(f"{self.name} no model found at {self.model_path}. Creating new model...")
            self._create_new_model()
    
    def _create_new_model(self):
        """Creates a new untrained hybrid model."""
        # Default: 2 features (mass, acceleration)
        self.model = build_rigorous_hybrid_agent(input_dim=2)
        logger.info(f"{self.name} created new gated hybrid model (untrained)")

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts physics outcomes based on state.
        Returns prediction, gate value, and confidence.
        """
        if not self.has_tf or not self.model:
            return {"status": "skipped", "reason": "No TensorFlow or model"}

        # 1. Extract Features
        # Feature 1: Mass (kg)
        geo_tree = state.get("geometry_tree", [])
        mass = sum(p.get("mass_kg", 0.0) for p in geo_tree)
        
        # Feature 2: Acceleration (m/s^2) - from environment or assume standard
        env = state.get("environment", {})
        acceleration = env.get("gravity", 9.81)  # Default to Earth gravity
        
        # Features Vector: [mass, acceleration]
        features = np.array([[mass, acceleration]], dtype=np.float32)
        
        # 2. Predict
        try:
            # Main prediction
            prediction = self.model.predict(features, verbose=0)
            pred_force = float(prediction[0][0])
            
            # Extract gate value (meta-cognitive decision)
            gate_layer = self.model.get_layer("gate")
            gate_model = Model(inputs=self.model.input, outputs=gate_layer.output)
            gate_value = float(gate_model.predict(features, verbose=0)[0][0])
            
            # Interpret gate
            # gate ≈ 0 → trusting physics (F=ma)
            # gate ≈ 1 → trusting neural intuition
            confidence = abs(0.5 - gate_value) * 2  # Distance from uncertain (0.5)
            
            # Safety heuristic: if force is unreasonably high, flag
            is_safe_likely = pred_force < (mass * 100)  # 100 m/s^2 threshold
            
            return {
                "status": "predicted",
                "predicted_force_N": round(pred_force, 2),
                "gate_value": round(gate_value, 3),
                "confidence": round(confidence, 2),
                "gate_interpretation": self._interpret_gate(gate_value),
                "recommendation": "PROCEED" if is_safe_likely else "REJECT",
                "logs": [
                    f"Hybrid prediction: Force={pred_force:.2f}N",
                    f"Gate: {gate_value:.3f} ({self._interpret_gate(gate_value)})",
                    f"Confidence: {confidence:.2f}"
                ]
            }
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _interpret_gate(self, gate_value: float) -> str:
        """Interpret what the gate is telling us."""
        if gate_value < 0.3:
            return "TRUSTING_PHYSICS (low-speed regime)"
        elif gate_value > 0.7:
            return "TRUSTING_NEURAL (complex regime)"
        else:
            return "UNCERTAIN (mixed regime)"

    def validate_prediction(self, prediction: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ground Truth Verification.
        Compares hybrid prediction against actual physics simulation.
        
        This is critical for:
        - Detecting model drift
        - Validating gate alignment
        - Active learning (finding where model needs improvement)
        """
        try:
            from agents.physics_agent import PhysicsAgent
            
            phys = PhysicsAgent()
            
            # Simple validation: check if prediction is close to actual physics
            # In real system, would run full VMK simulation
            env = state.get("environment", {})
            geo = state.get("geometry_tree", [])
            params = state.get("design_parameters", {})
            
            # Run actual physics
            phys_result = phys.run(env, geo, params)
            
            # Compare predictions
            # Simplified: just check if both agree on safety
            pred_safe = prediction.get("recommendation") == "PROCEED"
            actually_safe = phys_result["validation_flags"]["physics_safe"]
            
            match = pred_safe == actually_safe
            
            # Gate alignment check
            gate_value = prediction.get("gate_value", 0.5)
            velocity = env.get("velocity", [0, 0, 0])
            speed = np.linalg.norm(velocity) if isinstance(velocity, (list, np.ndarray)) else 0
            
            # Expected gate behavior
            expected_gate = "low" if speed < 10 else "high" if speed > 50 else "medium"
            actual_gate = "low" if gate_value < 0.3 else "high" if gate_value > 0.7 else "medium"
            
            gate_aligned = (expected_gate == actual_gate) or expected_gate == "medium"
            
            return {
                "verified": match,
                "ground_truth": "SAFE" if actually_safe else "UNSAFE",
                "prediction": "SAFE" if pred_safe else "UNSAFE",
                "gate_value": gate_value,
                "gate_aligned": gate_aligned,
                "speed": speed,
                "drift_alert": not match or not gate_aligned
            }
            
        except ImportError:
            return {"error": "Physics Agent Unavailable"}
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {"error": f"Validation Failed: {e}"}
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32):
        """
        Train the hybrid model on new data.
        
        Args:
            X: Features [mass, acceleration]
            y: Target forces
            epochs: Training epochs
            batch_size: Batch size
        """
        if not self.has_tf or not self.model:
            logger.error("Cannot train: TensorFlow or model not available")
            return {"status": "error", "error": "TensorFlow not available"}
        
        try:
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=0
            )
            
            final_loss = history.history['loss'][-1]
            
            return {
                "status": "trained",
                "epochs": epochs,
                "final_loss": float(final_loss),
                "history": {
                    "loss": [float(x) for x in history.history['loss']],
                    "val_loss": [float(x) for x in history.history.get('val_loss', [])]
                }
            }
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def save_model(self, path: str = None):
        """Save the trained model."""
        if not self.has_tf or not self.model:
            return
        
        save_path = path or self.model_path
        try:
            self.model.save(save_path)
            logger.info(f"Model saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
