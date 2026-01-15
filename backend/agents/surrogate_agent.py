import os
import joblib
import numpy as np
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class SurrogateAgent:
    """
    Neural Surrogate Agent (The "Oracle").
    Uses a trained Neural Network (MLP) to predict physics outcomes instantly.
    Acts as a 'Fast Filter' to reject obviously bad designs before expensive simulation.
    """
    def __init__(self, model_path: str = "data/physics_surrogate.pkl"):
        self.name = "SurrogateAgent"
        self.model_path = model_path
        self.model = None
        self.scaler = None
        
        self._load_model()

    def _load_model(self):
        """Loads the trained model and scaler from disk."""
        if os.path.exists(self.model_path):
            try:
                data = joblib.load(self.model_path)
                self.model = data["model"]
                self.scaler = data["scaler"]
                self.feature_names = data["features"]
                logger.info(f"{self.name} loaded trained surrogate model.")
            except Exception as e:
                logger.error(f"{self.name} failed to load model: {e}")
        else:
            logger.warning(f"{self.name} no model found at {self.model_path}. Running in passthrough mode.")

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts physics outcomes based on state.
        Returns a 'confidence' score and predicted values.
        """
        if not self.model:
            return {"status": "skipped", "reason": "No model loaded"}

        # 1. Extract Features (Must match Training Script)
        # Features: [geometry_mass, cost_estimate] (Simplified for MVP)
        # We need to robustly extract these.
        # Note: In a real system, we'd have a shared FeatureExtractor class.
        
        # Mass
        geo_tree = state.get("geometry_tree", [])
        mass = 0.0
        for p in geo_tree:
             mass += p.get("mass_kg", 0.0)
             
        # Cost
        bom = state.get("bom_analysis", {})
        cost = bom.get("total_cost_currency", 0.0)
        
        # Features Vector
        features = np.array([[mass, cost]])
        
        # 2. Scale & Predict
        try:
            X_scaled = self.scaler.transform(features)
            # Output: [thrust_req, physics_safe (as float 0-1)]
            prediction = self.model.predict(X_scaled)
            pred_thrust = prediction[0][0]
            pred_safe_score = prediction[0][1]
            
            is_safe_likely = pred_safe_score > 0.5
            
            return {
                "status": "predicted",
                "predicted_thrust_N": round(pred_thrust, 2),
                "predicted_safety_score": round(pred_safe_score, 2),
                "recommendation": "PROCEED" if is_safe_likely else "REJECT",
                "logs": [f"Surrogate predicts Thrust={pred_thrust:.2f}N, Safe={pred_safe_score:.2f}"]
            }
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {"status": "error", "error": str(e)}

    def validate_prediction(self, prediction: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ground Truth Verification.
        Runs the actual Physics/VMK simulation to verify if the Surrogate's prediction was accurate.
        Used for Active Learning (finding where model drift is high).
        """
        try:
            from agents.physics_agent import PhysicsAgent
            # Assuming physics agent can run a quick check
            phys = PhysicsAgent()
            
            # Ground Truth Check: Collision
            # The surrogate predicts "Safety Score". 
            # If Safe > 0.5, it predicts NO collision.
            # Let's verify with VMK Collision check.
            
            # Extract geometry position from state (Simplified)
            pos = state.get("position", [0,0,0])
            
            # Run VMK Check
            # We need terrain for this. Assuming state has it or we default to empty.
            # In Phase 2 test, PhysicsAgent used terrain map.
            terrain = state.get("context", {}).get("terrain", {"obstacles": []})
            
            # Passing radius 1.0 (default for surrogate checking point safety)
            gt = phys.check_collision_sdf(pos, 1.0, terrain)
            
            is_actually_safe = not gt["is_underground"]
            predicted_safe = prediction.get("predicted_safety_score", 0.0) > 0.5
            
            match = is_actually_safe == predicted_safe
            
            return {
                "verified": match,
                "ground_truth": "SAFE" if is_actually_safe else "COLLISION",
                "prediction": "SAFE" if predicted_safe else "COLLISION",
                "sdf_value": gt["sdf"],
                "drift_alert": not match
            }
            
        except ImportError:
            return {"error": "Physics Agent Unavailable"}
        except Exception as e:
            return {"error": f"Validation Failed: {e}"}

