"""
Surrogate Manager

Manages neural physics surrogate models for fast approximations.
"""

import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class SurrogateManager:
    """
    Manages loading and execution of neural physics surrogates.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the surrogate manager.
        
        Args:
            data_dir: Directory containing surrogate model files
        """
        self.data_dir = data_dir
        self.surrogates = {}
        self._load_available_surrogates()
    
    def _load_available_surrogates(self):
        """Load all available surrogate models from data directory"""
        if not os.path.exists(self.data_dir):
            logger.warning(f"Data directory not found: {self.data_dir}")
            return
        
        # Look for .weights.json files
        try:
            for filename in os.listdir(self.data_dir):
                if filename.endswith(".weights.json"):
                    domain = filename.replace(".weights.json", "")
                    path = os.path.join(self.data_dir, filename)
                    
                    # Instantiate correct model class
                    model = None
                    try:
                        if "physics" in domain:
                            try:
                                from models.physics_surrogate import PhysicsSurrogate
                            except ImportError:
                                from ...models.physics_surrogate import PhysicsSurrogate
                            model = PhysicsSurrogate(input_size=5, hidden_size=32, output_size=2)
                        else:
                            try:
                                from models.material_net import MaterialNet
                            except ImportError:
                                from models.material_net import MaterialNet
                            model = MaterialNet(input_size=5, output_size=1)
                        
                        # Load weights
                        model.load(path)
                        
                        self.surrogates[domain] = {
                            "path": path,
                            "loaded": True,
                            "model": model
                        }
                        logger.info(f"Loaded surrogate model for {domain}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to load surrogate for {domain}: {e}")

            logger.info(f"Found {len(self.surrogates)} surrogate models: {list(self.surrogates.keys())}")
        except Exception as e:
            logger.error(f"Error loading surrogates: {e}")
            
    def has_model(self, domain: str) -> bool:
        """
        Check if a surrogate model exists for a domain.
        
        Args:
            domain: Physics domain (e.g., "physics", "structural", "thermal")
        
        Returns:
            True if model exists
        """
        return domain in self.surrogates
            
    def predict(self, domain: str, params: Any) -> Dict[str, Any]:
        """
        Run prediction using surrogate model.
        
        Args:
            domain: Physics domain (e.g., "physics_surrogate")
            params: Input parameters (numpy array or list)
        
        Returns:
            Prediction result with confidence
        """
        if not self.has_model(domain):
            # Try fuzzy match (e.g. "physics" -> "physics_surrogate")
            matches = [k for k in self.surrogates.keys() if domain in k]
            if matches:
                domain = matches[0]
            else:
                raise ValueError(f"No surrogate model for domain: {domain}")
        
        model = self.surrogates[domain]["model"]
        
        # Run prediction
        try:
            import numpy as np
            inputs = np.array(params)
            
            # Use predict_with_uncertainty if available
            if hasattr(model, "predict_with_uncertainty"):
                pred, confidence = model.predict_with_uncertainty(inputs)
                return {
                    "result": pred.tolist() if isinstance(pred, np.ndarray) else pred,
                    "confidence": float(confidence),
                    "method": "neural_surrogate_uncertainty"
                }
            else:
                pred = model.forward(inputs)
                return {
                    "result": pred.tolist() if isinstance(pred, np.ndarray) else pred,
                    "confidence": 1.0, 
                    "method": "neural_surrogate_standard"
                }
                
        except Exception as e:
            logger.error(f"Surrogate prediction failed: {e}")
            return {"error": str(e), "confidence": 0.0}
    
    def get_available_domains(self) -> list:
        """
        Get list of domains with available surrogate models.
        
        Returns:
            List of domain names
        """
        return list(self.surrogates.keys())
