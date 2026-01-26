import random
import math
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Union

logger = logging.getLogger(__name__)

# --- Symbolic Mathematics Engine ---

class OpNode:
    """A node in the equation tree."""
    def __init__(self, val: Union[str, float, int], left: 'OpNode' = None, right: 'OpNode' = None, is_leaf: bool = False):
        self.val = val
        self.left = left
        self.right = right
        self.is_leaf = is_leaf

    def evaluate(self, features: Dict[str, float]) -> float:
        if self.is_leaf:
            if isinstance(self.val, str):
                return features.get(self.val, 0.0)
            return float(self.val)
        
        # Operators
        l = self.left.evaluate(features) if self.left else 0.0
        r = self.right.evaluate(features) if self.right else 0.0
        
        try:
            if self.val == '+': return l + r
            if self.val == '-': return l - r
            if self.val == '*': return l * r
            if self.val == '/': return l / r if abs(r) > 1e-6 else 1e6
            if self.val == 'sin': return math.sin(l)
            if self.val == 'cos': return math.cos(l)
            if self.val == 'exp': return math.exp(min(l, 10.0)) # Clip for overflow
            if self.val == 'log': return math.log(abs(l) + 1e-6)
        except Exception:
            return 0.0
        return 0.0

    def __str__(self):
        if self.is_leaf:
            return str(self.val)
        if self.val in ['sin', 'cos', 'exp', 'log']:
            return f"{self.val}({self.left})"
        return f"({self.left} {self.val} {self.right})"

class SymbolicRegressor:
    """
    Mini-Genetic Programming Engine for Symbolic Regression.
    """
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.operators = ['+', '-', '*', '/'] # Basic arithmetic
        self.unary_operators = ['sin', 'exp'] # Complexity
        
    def generate_random_expr(self, depth: int = 2) -> OpNode:
        if depth == 0 or random.random() < 0.3:
            # Leaf: Feature or Constant
            if random.random() < 0.7:
                return OpNode(val=random.choice(self.feature_names), is_leaf=True)
            else:
                return OpNode(val=round(random.uniform(-5, 5), 2), is_leaf=True)
        
        # Operator
        op = random.choice(self.operators)
        return OpNode(val=op, left=self.generate_random_expr(depth-1), right=self.generate_random_expr(depth-1))

    def fit(self, X: List[Dict[str, float]], y: List[float], generations: int = 20, pop_size: int = 50) -> str:
        """
        Evolves an equation to fit y = f(X).
        """
        population = [self.generate_random_expr(depth=random.randint(1, 4)) for _ in range(pop_size)]
        
        best_expr = None
        best_error = float('inf')
        
        for gen in range(generations):
            # Evaluate
            scored_pop = []
            for expr in population:
                error = 0.0
                for i, row in enumerate(X):
                    pred = expr.evaluate(row)
                    error += (pred - y[i]) ** 2
                mse = error / len(X)
                scored_pop.append((expr, mse))
                
                if mse < best_error:
                    best_error = mse
                    best_expr = expr
            
            # Selection (Tournament)
            new_pop = []
            scored_pop.sort(key=lambda x: x[1])
            elites = [x[0] for x in scored_pop[:5]]
            new_pop.extend(elites)
            
            while len(new_pop) < pop_size:
                parent = random.choice(elites) # Simplify selection for MVP
                child = self._mutate(parent)
                new_pop.append(child)
            
            population = new_pop
            
        return str(best_expr), best_error

    def _mutate(self, node: OpNode) -> OpNode:
        # Simple subtree mutation
        if random.random() < 0.2:
            return self.generate_random_expr(depth=2)
        
        new_node = OpNode(node.val, node.left, node.right, node.is_leaf)
        if not node.is_leaf:
            if node.left: new_node.left = self._mutate(node.left)
            if node.right: new_node.right = self._mutate(node.right)
        return new_node

class ScientistAgent:
    """
    The Observer.
    Analyzes simulation logs to discover physical laws.
    """
    def __init__(self):
        self.name = "Scientist"
        
    def discover_law(self, data_records: List[Dict[str, Any]], target_variable: str) -> str:
        """
        Input: List of dicts (e.g. [{'mass': 10, 'force': 5, 'failure': 1}]).
        Output: "failure = 0.5 * force / mass"
        """
        if not data_records: return "No Data"
        
        # 1. Prepare Data
        keys = [k for k in data_records[0].keys() if k != target_variable and isinstance(data_records[0][k], (int, float))]
        X = []
        y = []
        
        for r in data_records:
            X.append({k: float(r[k]) for k in keys})
            y.append(float(r[target_variable]))
            
        # 2. Run Regression
        regressor = SymbolicRegressor(feature_names=keys)
        equation, error = regressor.fit(X, y)
        
        return f"{target_variable} ~= {equation} (MSE: {error:.4f})"
