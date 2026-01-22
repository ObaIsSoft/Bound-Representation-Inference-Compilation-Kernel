import logging
import random
import math
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from pydantic import BaseModel
import numpy as np

logger = logging.getLogger(__name__)

class OptimizationStrategy(str, Enum):
    GRADIENT_DESCENT = "GRADIENT_DESCENT"
    GENETIC_ALGORITHM = "GENETIC_ALGORITHM"
    SIMULATED_ANNEALING = "SIMULATED_ANNEALING"

class ObjectiveFunction(BaseModel):
    id: str
    target: str # 'MINIMIZE' | 'MAXIMIZE'
    metric: str # 'DRAG', 'MASS', 'STRESS'

class StrategySelector:
    """
    Meta-Learner: Decides WHICH optimization strategy to use.
    Uses a simplified Contextual Bandit approach.
    Context: [Problem Size, Constraints, Nonlinearity Score]
    """
    def __init__(self):
        # Maps strategy -> weight (success probability)
        # Initial priors: Gradient is fast (default), Genetic is robust
        self.weights = {
            OptimizationStrategy.GRADIENT_DESCENT: 0.6,
            OptimizationStrategy.GENETIC_ALGORITHM: 0.2,
            OptimizationStrategy.SIMULATED_ANNEALING: 0.2
        }
        self.history = []

    def select_strategy(self, context: Dict[str, Any]) -> OptimizationStrategy:
        """
        Epsilon-Greedy selection based on context.
        """
        # Exploration: 20% chance to try random strategy (Boosted for Phase 3 Verification)
        if random.random() < 0.2:
            return random.choice(list(OptimizationStrategy))
            
        # Exploitation: Heuristic overrides
        # If problem is highly constrained or non-differentiable -> Genetic
        if context.get("constraints_count", 0) > 10 or context.get("is_discrete", False):
            return OptimizationStrategy.GENETIC_ALGORITHM
            
        # Default: Pick highest weight
        return max(self.weights, key=self.weights.get)

    def update_policy(self, strategy: OptimizationStrategy, success: bool, efficiency: float):
        """
        Self-Evolution: Logic to update weights based on outcome.
        """
        reward = 1.0 if success else -0.5
        reward += efficiency * 0.1 # Bonus for speed
        
        # Simple exponential moving average update
        lr = 0.1
        self.weights[strategy] = (1 - lr) * self.weights[strategy] + lr * reward
        self.history.append({"strategy": strategy, "reward": reward})
        logger.info(f"[META-LEARNING] Updated {strategy} weight to {self.weights[strategy]:.2f}")

class OptimizationAgent:
    """
    Optimization Agent - Ares-Class Multi-Modal Solver.
    
    Evolution Capabilities:
    1. Strategy Selection (Meta-Learning)
    2. Hyperparameter Tuning (Self-Adjustment)
    """
    
    def __init__(self):
        self.name = "OptimizationAgent"
        self.selector = StrategySelector()
        self.step_size = 0.01 
        
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point.
        """
        isa_state = params.get("isa_state", {})
        obj_dict = params.get("objective", {"target": "MINIMIZE", "metric": "MASS"})
        objective = ObjectiveFunction(**obj_dict)
        
        # 1. Context Extraction (for Meta-Learner)
        constraints = isa_state.get('constraints', {})
        context = {
            "num_params": len(constraints),
            "constraints_count": sum(1 for v in constraints.values() if v.get('locked')),
            "metric": objective.metric
        }
        
        # 2. Strategy Selection
        strategy = self.selector.select_strategy(context)
        logger.info(f"[OPTIMIZATION] Strategy Selected: {strategy.value}")
        
        mutation_log = []
        success = False
        
        # 3. Execution (Polymorphic)
        try:
            if strategy == OptimizationStrategy.GRADIENT_DESCENT:
                gradients = self.compute_gradient(isa_state, objective)
                modified_state = self.evolve_geometry_gradient(isa_state, gradients, objective, mutation_log)
                success = True # Gradient always "runs", success checked by improvement downstream
                
            elif strategy == OptimizationStrategy.GENETIC_ALGORITHM:
                modified_state = self.evolve_geometry_genetic(isa_state, objective, mutation_log)
                success = True
                
            elif strategy == OptimizationStrategy.SIMULATED_ANNEALING:
                modified_state = self.evolve_geometry_annealing(isa_state, objective, mutation_log)
                success = True
                
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            modified_state = isa_state
            success = False
            
        # 4. Self-Evolution (Update Policy)
        # Note: True "success" comes from the Critic telling us if result improved. 
        # Here we just record runtime success. The global Critic loop handles long-term updates.
        self.selector.update_policy(strategy, success, efficiency=1.0) # Placeholder efficiency
        
        return {
            "success": success,
            "strategy_used": strategy.value,
            "original_state": params.get("isa_state", {}),
            "optimized_state": modified_state,
            "mutations": mutation_log
        }

    # --- STRATEGY 1: GRADIENT DESCENT ---
    def compute_gradient(self, isa_state: Dict[str, Any], objective: ObjectiveFunction) -> Dict[str, float]:
        """Calculates sensitivity gradient (J_perturbed - J) / epsilon"""
        gradients = {}
        constraints = isa_state.get('constraints', {})
        
        for node_id, node in constraints.items():
            val_obj = node.get('val', {})
            if not isinstance(val_obj, dict) or 'value' not in val_obj or node.get('locked', False):
                continue
                
            current_val = float(val_obj['value'])
            
            j_initial = self._query_surrogate_model(objective.metric, isa_state)
            
            val_obj['value'] = current_val + self.step_size
            j_perturbed = self._query_surrogate_model(objective.metric, isa_state)
            val_obj['value'] = current_val # Restore
            
            sensitivity = (j_perturbed - j_initial) / self.step_size
            if abs(sensitivity) > 1e-6:
                gradients[node_id] = sensitivity
                
        return gradients

    def evolve_geometry_gradient(self, isa_state: Dict[str, Any], gradients: Dict[str, float], objective: ObjectiveFunction, log: List[str]) -> Dict[str, Any]:
        """Standard Gradient Descent Step"""
        alpha = 0.1
        direction = -1.0 if objective.target == 'MINIMIZE' else 1.0
        import copy
        new_state = copy.deepcopy(isa_state)
        
        for node_id, grad in gradients.items():
            mutation = direction * grad * alpha
            current_val = new_state['constraints'][node_id]['val']['value']
            new_val = max(0.001, current_val + mutation)
            new_state['constraints'][node_id]['val']['value'] = new_val
            log.append(f"Gradient Step {node_id}: {current_val:.4f} -> {new_val:.4f}")
            
        return new_state

    # --- STRATEGY 2: GENETIC ALGORITHM ---
    def evolve_geometry_genetic(self, isa_state: Dict[str, Any], objective: ObjectiveFunction, log: List[str]) -> Dict[str, Any]:
        """
        Population-based optimization.
        Good for non-convex or discrete problems.
        """
        import copy
        constraints = isa_state.get('constraints', {})
        param_keys = [k for k, v in constraints.items() if not v.get('locked') and 'val' in v]
        
        if not param_keys: return isa_state
        
        # 1. Generate Population (Mutations)
        population = []
        pop_size = 10
        best_score = float('inf') if objective.target == 'MINIMIZE' else float('-inf')
        best_state = copy.deepcopy(isa_state)
        
        current_score = self._query_surrogate_model(objective.metric, isa_state)
        log.append(f"GA Baseline: {current_score:.4f}")
        
        for i in range(pop_size):
            candidate = copy.deepcopy(isa_state)
            # Mutate random parameter
            key = random.choice(param_keys)
            curr = candidate['constraints'][key]['val']['value']
            mutation = random.gauss(0, 0.2) * curr # 20% variance
            candidate['constraints'][key]['val']['value'] = max(0.001, curr + mutation)
            
            score = self._query_surrogate_model(objective.metric, candidate)
            
            improved = (objective.target == 'MINIMIZE' and score < best_score) or \
                       (objective.target == 'MAXIMIZE' and score > best_score)
                       
            if improved:
                best_score = score
                best_state = candidate
                log.append(f"GA Gen 1 Winner: {key} -> {candidate['constraints'][key]['val']['value']:.4f} (Score: {score:.4f})")
                
        return best_state

    # --- STRATEGY 3: SIMULATED ANNEALING ---
    def evolve_geometry_annealing(self, isa_state: Dict[str, Any], objective: ObjectiveFunction, log: List[str]) -> Dict[str, Any]:
        """
        Probabilistic search to escape local optima.
        """
        import copy
        current_state = copy.deepcopy(isa_state)
        current_score = self._query_surrogate_model(objective.metric, current_state)
        
        T = 1.0 # Temperature
        cooling_rate = 0.95
        steps = 20
        
        constraints = isa_state.get('constraints', {})
        param_keys = [k for k, v in constraints.items() if not v.get('locked') and 'val' in v]
        
        if not param_keys: return isa_state
        
        for i in range(steps):
            # Neighbor
            candidate = copy.deepcopy(current_state)
            key = random.choice(param_keys)
            val = candidate['constraints'][key]['val']['value']
            candidate['constraints'][key]['val']['value'] = max(0.001, val + random.gauss(0, 0.05))
            
            candidate_score = self._query_surrogate_model(objective.metric, candidate)
            
            delta = candidate_score - current_score
            if objective.target == 'MAXIMIZE': delta = -delta
            
            # Acceptance Probability
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_state = candidate
                current_score = candidate_score
                log.append(f"SA Step {i}: Accepted (Score: {current_score:.4f}, T: {T:.2f})")
            
            T *= cooling_rate
            
        return current_state

    def _query_surrogate_model(self, metric: str, state: Dict[str, Any]) -> float:
        """
        Fast Heuristic Surrogate for physical properties.
        (Same as before, simplified for this refactor demo)
        """
        constraints = state.get('constraints', {})
        radius = 0.5
        width = 1.0
        length = 1.0
        
        for k, v in constraints.items():
            val = v.get('val', {}).get('value', 0)
            if 'rad' in k.lower(): radius = val
            if 'width' in k.lower(): width = val
            
        if metric == 'MASS': return length * width * radius
        if metric == 'DRAG': return width * radius * 1.2
        if metric == 'STRESS': return 1.0 / max(0.001, radius * width)
        return 0.0

    # --- GEOMETRIC EVOLUTION (Smart Snap) ---
    def optimize_sketch_curve(self, points: List[List[float]], objective: ObjectiveFunction) -> List[List[float]]:
        """
        Adjoint-style "Smart Snap" for sketch curves.
        Adjusts point positions to minimize an objective (e.g. curvature energy, drag).
        """
        # Convert to numpy for vector math
        try:
            current_points = np.array(points)
        except:
            return points # Fallback if malformed
            
        iterations = 20
        learning_rate = 0.05
        
        for i in range(iterations):
            gradients = np.zeros_like(current_points)
            
            # Finite Difference for each coordinate of each point
            # (Simplified for performance - effectively a local smoothing + physics pull)
            for idx in range(1, len(current_points) - 1): # Anchor endpoints
                original_pos = current_points[idx].copy()
                
                # 1. Physics Pull (Mock Adjoint)
                # If Drag, pull towards flow lines (e.g. align with X axis)
                if objective.metric == 'DRAG':
                    # Heuristic: Minimize Y/Z deviation from flow (X-axis)
                    # J = y^2 + z^2
                    gradients[idx][1] += 2 * original_pos[1] * 0.1 # dJ/dy
                    gradients[idx][2] += 2 * original_pos[2] * 0.1 # dJ/dz
                    
                # 2. Smoothness (Curvature Energy)
                # Minimize distance to average of neighbors (Laplacian smoothing)
                # J = ||p_i - (p_{i-1} + p_{i+1})/2||^2
                neighbor_avg = (current_points[idx-1] + current_points[idx+1]) * 0.5
                smooth_grad = (original_pos - neighbor_avg)
                gradients[idx] += smooth_grad * 0.5 # Weight for smoothness
                
            # Apply update
            current_points -= learning_rate * gradients
            
        return current_points.tolist()

