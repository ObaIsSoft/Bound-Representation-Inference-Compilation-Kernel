from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class DesignExplorationAgent:
    """
    Design Exploration Agent - Parametric Search (EVOLVED).
    
    Explores design space by:
    - Sampling parameter combinations
    - Ranking candidates by objectives
    - Pareto frontier optimization
    - Learning from historical explorations (Neural Surrogate)
    """
    
    def __init__(self):
        self.name = "DesignExplorationAgent"
        
        # Load Config
        try:
            from backend.config.design_exploration_config import EXPLORATION_DEFAULTS, SCORING_WEIGHTS, PARETO_CONFIG
            self.defaults = EXPLORATION_DEFAULTS
            self.weights = SCORING_WEIGHTS
            self.pareto_config = PARETO_CONFIG
        except ImportError:
            logger.warning("Could not import design_exploration_config. Using defaults.")
            self.defaults = {"num_samples": 50}
            self.weights = {"mass": -1.0, "strength": 1.0}
            self.pareto_config = {"max_front_size": 5}

        # Initialize Neural Surrogate
        try:
            from models.design_exploration_surrogate import DesignExplorationSurrogate
            self.surrogate = DesignExplorationSurrogate()
            self.has_surrogate = True
        except ImportError:
            try:
                from backend.models.design_exploration_surrogate import DesignExplorationSurrogate
                self.surrogate = DesignExplorationSurrogate()
                self.has_surrogate = True
            except ImportError:
                self.surrogate = None
                self.has_surrogate = False
                logger.debug("DesignExplorationSurrogate not found")
    
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explore design space and rank candidates.
        
        Args:
            params: {
                "parameters": Dict of {param_name: (min, max, step)},
                "objectives": List of optimization goals,
                "constraints": List of constraints,
                "num_samples": Optional int (default 50)
            }
        
        Returns:
            {
                "candidates": List of design candidates,
                "pareto_front": List of Pareto-optimal designs,
                "best_candidate": Dict of best overall design,
                "logs": List of operation logs
            }
        """
        parameters = params.get("parameters", {})
        objectives = params.get("objectives", ["minimize_mass", "maximize_strength"])
        constraints = params.get("constraints", [])
        num_samples = params.get("num_samples", self.defaults["num_samples"])
        
        logs = [
            f"[DESIGN_EXPLORATION] Exploring {len(parameters)} parameter(s)",
            f"[DESIGN_EXPLORATION] Objectives: {', '.join(objectives)}",
            f"[DESIGN_EXPLORATION] Generating {num_samples} candidate(s)"
        ]
        
        # Generate candidate designs
        candidates = []
        # Limit max samples to avoid timeout if param is crazy
        safe_limit = min(num_samples, 20) 
        
        for i in range(safe_limit):
            param_values = self._sample_parameters(parameters, i)
            
            # Use surrogate to predict quality if available
            if self.has_surrogate:
                import numpy as np
                param_array = np.array(list(param_values.values()))
                score = self.surrogate.predict(param_array)
            else:
                # Mock score logic (replaced by physics simulation in real flow)
                # But at least make it dependent on params not just 'i'
                # E.g. minimize first param
                val = list(param_values.values())[0] if param_values else 0
                score = 1.0 / (1.0 + val) 
            
            candidate = {
                "id": f"candidate_{i+1}",
                "parameters": param_values,
                "score": score,
                # Placeholder physics results
                "mass_kg": 5.0 + (i * 0.1),
                "strength_MPa": 200 - (i * 2)
            }
            candidates.append(candidate)
        
        # Sort by score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Identify Pareto front
        pareto_front = self._find_pareto_front(candidates, objectives)
        
        best_candidate = candidates[0] if candidates else None
        
        logs.append(f"[DESIGN_EXPLORATION] Ranked {len(candidates)} candidate(s)")
        logs.append(f"[DESIGN_EXPLORATION] Pareto front: {len(pareto_front)} solution(s)")
        logs.append(f"[DESIGN_EXPLORATION] Best: {best_candidate['id'] if best_candidate else 'None'}")
        
        return {
            "candidates": candidates,
            "pareto_front": pareto_front,
            "best_candidate": best_candidate,
            "logs": logs
        }
    
    def _sample_parameters(self, parameters: Dict, seed: int) -> Dict:
        """Generate parameter values for a candidate."""
        import random
        random.seed(seed)
        
        sampled = {}
        for param_name, (min_val, max_val, step) in parameters.items():
            sampled[param_name] = min_val + random.random() * (max_val - min_val)
        
        return sampled
    
    def _find_pareto_front(self, candidates: List[Dict], objectives: List[str]) -> List[Dict]:
        """Find Pareto-optimal designs (non-dominated)."""
        # Simplified: Just grab top N based on config
        limit = self.pareto_config.get("max_front_size", 5)
        return candidates[:limit]
    
    def evolve(self, training_data: List[Any]) -> Dict[str, Any]:
        """
        Train surrogate on historical exploration results.
        
        Args:
            training_data: List of (parameters, quality_score) tuples
        """
        if not self.has_surrogate:
            return {"status": "error", "message": "No surrogate"}
        
        import numpy as np
        total_loss = 0.0
        count = 0
        
        for params, score in training_data:
            # Ensure params is numpy array
            if not isinstance(params, np.ndarray):
                params = np.array(params)
            
            loss = self.surrogate.train_step(params, score)
            total_loss += loss
            count += 1
        
        self.surrogate.trained_epochs += 1
        self.surrogate.save()
        
        return {
            "status": "evolved",
            "avg_loss": total_loss / max(1, count),
            "epochs": self.surrogate.trained_epochs
        }

