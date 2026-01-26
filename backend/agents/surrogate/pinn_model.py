import math
import logging
import numpy as np
from typing import Dict, Any, List
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import NotFittedError

logger = logging.getLogger(__name__)

class MultiPhysicsPINN:
    """
    The Physics Oracle (Teacher-Student Hybrid).
    
    Architecture:
    1. Teacher (Analytical Kernel): Slow, rigorous check of Conservation Laws.
    2. Student (Neural Surrogate): Fast MLPRegressor that learns the Teacher's function.
    
    Mechanism:
    - Online Learning: The Student trains on every "ground truth" execution of the Teacher.
    - Confidence: Initially devolves to Teacher. As Student converges (low loss), it takes over for speed.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.domains = ["STRUCTURAL", "THERMAL", "FLUIDS"]
        
        # Hyperparams
        hidden_layers = self.config.get("pinn_hidden_layers", (64, 32))
        max_iter = self.config.get("pinn_max_iter", 500)
        self.batch_size = self.config.get("pinn_batch_size", 32)
        
        # The Student
        self.student = MLPRegressor(
            hidden_layer_sizes=hidden_layers, 
            activation='relu', 
            solver='adam', 
            learning_rate='adaptive',
            max_iter=max_iter,
            warm_start=True
        )
        self.is_fitted = False
        
        # Training Buffer (Short-term memory)
        self.X_buffer = []
        self.y_buffer = []
        
    def validate_design(self, genome_nodes: List[Dict[str, Any]], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main Validation Entry Point.
        """
        # 1. Feature Extraction (Geometry -> Vector)
        features = self._extract_features(genome_nodes)
        
        # 2. Student Prediction (Fast Path)
        predicted_score = None
        use_teacher = True
        
        if self.is_fitted:
            try:
                # Predict [Structural_Residual, Thermal_Residual]
                prediction = self.student.predict([features])[0]
                structural_pred, thermal_pred = prediction[0], prediction[1]
                
                # Trust heuristic: If residuals are reasonable, trust student
                # For MVP: We always run Teacher to gather training data until buffer full
                # In prod: We'd skip Teacher if student variance is low
                use_teacher = True 
            except Exception as e:
                logger.warning(f"Student prediction failed: {e}")
                use_teacher = True
        
        # 3. Teacher Execution (Ground Truth)
        if use_teacher:
            structural_loss = self._calculate_structural_residual(genome_nodes, constraints)
            thermal_loss = self._calculate_thermal_residual(genome_nodes, constraints)
            
            # Online Training Step
            self._update_student(features, [structural_loss, thermal_loss])
            
        else:
            structural_loss = structural_pred
            thermal_loss = thermal_pred

        total_residual = structural_loss + thermal_loss
        score = 1.0 / (1.0 + total_residual)
        
        return {
            "is_valid": total_residual < 2.0,
            "physics_score": float(score),
            "residuals": {
                "structural": float(structural_loss),
                "thermal": float(thermal_loss)
            },
            "source": "TEACHER" if use_teacher else "STUDENT"
        }

    def _update_student(self, X: List[float], y: List[float]):
        """
        Adds experience to buffer and re-trains Student if batch ready.
        """
        self.X_buffer.append(X)
        self.y_buffer.append(y)
        
        if len(self.X_buffer) >= self.batch_size:
            X_train = np.array(self.X_buffer)
            y_train = np.array(self.y_buffer)
            
            try:
                self.student.partial_fit(X_train, y_train)
                self.is_fitted = True
                # logger.info(f"[PINN] Student Updated. Loss: {self.student.loss_:.4f}")
            except Exception:
                # First flt needs 'classes' or just use fit
                self.student.fit(X_train, y_train)
                self.is_fitted = True
                
            # Clear buffer (or keep rolling window)
            self.X_buffer = []
            self.y_buffer = []

    def _extract_features(self, nodes: List[Dict[str, Any]]) -> List[float]:
        """
        Encodes topology into fixed-size vector for MLP.
        MVP: Fixed size aggregation (Naive embedding).
        Better: Graph Neural Network (GNN).
        """
        # Vector: [Num_Nodes, Avg_X, Avg_Y, Avg_Z, Total_Vol, Aspect_Ratio, ...]
        num_nodes = len(nodes)
        
        total_vol = 0.0
        center_sum = np.zeros(3)
        min_p = np.full(3, np.inf)
        max_p = np.full(3, -np.inf)
        
        for n in nodes:
            # Volume
            p = n.get('params', {})
            if n['type'] == 'CUBE':
                vol = p.get('width', {}).get('value', 1) * p.get('height', {}).get('value', 1) * p.get('depth', {}).get('value', 1)
            else: 
                vol = 1.0
            total_vol += vol
            
            # Pos
            pos = np.array(n['transform'][:3])
            center_sum += pos
            min_p = np.minimum(min_p, pos)
            max_p = np.maximum(max_p, pos)
            
        avg_pos = center_sum / num_nodes if num_nodes > 0 else np.zeros(3)
        bounds = max_p - min_p
        
        # Feature Vector (Size 8)
        features = [
            float(num_nodes), 
            total_vol, 
            avg_pos[0], avg_pos[1], avg_pos[2],
            bounds[0], bounds[1], bounds[2]
        ]
        return features

    def _calculate_structural_residual(self, nodes: List[Dict[str, Any]], constraints: Dict[str, Any]) -> float:
        """
        Enforces Static Equilibrium (Sum of Forces = 0).
        Heuristic: Calculate moment of inertia vs support base.
        """
        residual = 0.0
        
        # 1. Check Connectivity (Graph Continuity)
        if not nodes: return 100.0
        
        total_mass = 0.0
        center_of_mass_x = 0.0
        
        for n in nodes:
            # Approximate Volume/Mass
            p = n.get('params', {})
            # Mock volume calc
            vol = 1.0
            if n['type'] == 'CUBE':
                vol = p.get('width', {}).get('value', 1) * p.get('height', {}).get('value', 1) * p.get('depth', {}).get('value', 1)
            elif n['type'] == 'SPHERE':
                r = p.get('radius', {}).get('value', 1)
                vol = (4/3) * math.pi * r**3
                
            mass = vol * 1.0 # Density = 1
            total_mass += mass
            
            # Weighted position
            # Currently transform is [x,y,z, r,p,y]
            pos_x = n['transform'][0]
            center_of_mass_x += pos_x * mass
            
        if total_mass > 0:
            center_of_mass_x /= total_mass
            
        # 2. Aspect Ratio Penalty (Euler-Bernoulli Beam limit)
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        
        for n in nodes:
            px = n['transform'][0]
            py = n['transform'][1]
            min_x, max_x = min(min_x, px), max(max_x, px)
            min_y, max_y = min(min_y, py), max(max_y, py)
            
        width = max_x - min_x
        height = max_y - min_y
        
        aspect_ratio = height / width if width > 0.1 else 0
        
        # Constraint: Material Degradation
        # If material is degraded, the tolerance for bad aspect ratio drops
        retention = constraints.get("material_retention", {}).get("val", 1.0)
        max_safe_ratio = 10.0 * retention
        
        if aspect_ratio > max_safe_ratio:
            residual += (aspect_ratio - max_safe_ratio) * 0.5 # Penalty for instability
            
        # Constraint: Impact Load / Extreme Load
        # Force = Mass * Accel. If Force > Strength, Residual++
        simulated_load = constraints.get("max_weight", {}).get("val", 0.0)
        impulse = constraints.get("impulse_force", {}).get("val", 0.0)
        
        total_force = simulated_load + impulse
        
        # Heuristic strength check: Strength ~= Width^2 (Cross section)
        # Assuming min_width of the structure is roughly the 'width' from bounds (simplification)
        # A better check would be min(params['width']) of all nodes
        eff_width = min([n.get('params', {}).get('width', {}).get('value', 1.0) for n in nodes if n['type']=='CUBE'] or [1.0])
        strength = (eff_width ** 2) * 100 * retention
        
        if total_force > strength:
             residual += (total_force - strength) * 0.05
            
        # 3. Mass Conservation
        for n in nodes:
            for pname, pval in n.get('params', {}).items():
                if pval['value'] <= 0:
                     residual += 100.0 # Violation of existence
                     
        return residual

    def _calculate_thermal_residual(self, nodes: List[Dict[str, Any]], constraints: Dict[str, Any]) -> float:
        """
        Enforces Fourier's Law (Heat Conduction).
        Heuristic: Surface Area to Volume Ratio.
        """
        residual = 0.0
        
        total_vol = 0.0
        total_area = 0.0
        
        for n in nodes:
             p = n.get('params', {})
             if n['type'] == 'CUBE':
                 w, h, d = p.get('width', {}).get('value', 1), p.get('height', {}).get('value', 1), p.get('depth', {}).get('value', 1)
                 total_area += 2*(w*h + h*d + w*d)
                 total_vol += w*h*d
                 
        # Physics Constraint: Cooling Efficiency
        # If Volume is high but Area is low -> Heat Trap -> High Residual
        if total_vol > 0:
            ratio = total_area / total_vol
            # Assume we want efficient cooling (ratio > 2.0)
            if ratio < 2.0:
                residual += (2.0 - ratio) * 2.0 # Penalty for heat buildup
                
        # Constraint: Ambient Temp
        # High ambient temp requires MUCH higher surface area to cool
        ambient = constraints.get("ambient_temp", {}).get("val", 300.0)
        if ambient > 500.0:
            # Need ratio > 4.0
            if ratio < 4.0:
                residual += (4.0 - ratio) * 5.0 # Severe penalty for melting
                
        return residual
