"""
FIX-103, FIX-104, FIX-105: Advanced Structural Mechanics

Stress concentration factors, failure criteria, and safety factors.
"""

import numpy as np
from typing import Dict, Tuple, Literal, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class StressState:
    """3D stress state tensor"""
    sigma_x: float = 0.0
    sigma_y: float = 0.0
    sigma_z: float = 0.0
    tau_xy: float = 0.0
    tau_yz: float = 0.0
    tau_zx: float = 0.0
    
    def to_tensor(self) -> np.ndarray:
        """Convert to 3x3 tensor"""
        return np.array([
            [self.sigma_x, self.tau_xy, self.tau_zx],
            [self.tau_xy, self.sigma_y, self.tau_yz],
            [self.tau_zx, self.tau_yz, self.sigma_z]
        ])
    
    @classmethod
    def from_tensor(cls, tensor: np.ndarray) -> 'StressState':
        """Create from 3x3 tensor"""
        return cls(
            sigma_x=tensor[0, 0],
            sigma_y=tensor[1, 1],
            sigma_z=tensor[2, 2],
            tau_xy=tensor[0, 1],
            tau_yz=tensor[1, 2],
            tau_zx=tensor[2, 0]
        )
    
    def principal_stresses(self) -> Tuple[float, float, float]:
        """Calculate principal stresses"""
        eigenvalues = np.linalg.eigvals(self.to_tensor())
        return tuple(sorted(eigenvalues, reverse=True))


@dataclass
class FailureResult:
    """Result of failure analysis"""
    failure_criterion: str
    equivalent_stress: float
    yield_strength: float
    safety_factor: float
    margin_of_safety: float
    failure_mode: str
    is_safe: bool


class AdvancedStructures:
    """
    Advanced structural analysis with:
    - Stress concentration factors (Kt)
    - Failure criteria (Von Mises, Tresca)
    - Comprehensive safety factor calculation
    """
    
    # Stress concentration factors for common geometries
    # Source: Peterson's Stress Concentration Factors, Pilkey's Formulas
    STRESS_CONCENTRATION_FACTORS = {
        # Circular hole in infinite plate under tension
        "circular_hole": {
            "kt_max": 3.0,  # Theoretical max at hole edge
            "description": "Circular hole in infinite plate, uniaxial tension"
        },
        # Elliptical hole (semi-axes a, b)
        "elliptical_hole": {
            "formula": "kt = 1 + 2*(a/b)",
            "description": "Elliptical hole, major axis perpendicular to load"
        },
        # U-notch
        "u_notch": {
            "formula": "kt_approx",
            "description": "U-shaped notch"
        },
        # V-notch
        "v_notch": {
            "formula": "kt_complex",
            "description": "V-shaped notch"
        },
        # Shoulder fillet (circular)
        "shoulder_fillet": {
            "formula": "kt_peterson",
            "description": "Stepped bar with shoulder fillet"
        },
        # Groove
        "groove": {
            "formula": "kt_groove",
            "description": "Circumferential groove in shaft"
        },
        # Keyseat
        "keyseat": {
            "kt_approx": 2.0,
            "description": "Keyway/keyseat in shaft"
        },
        # Thread (metric)
        "metric_thread": {
            "kt_range": (2.2, 3.8),
            "description": "Standard metric thread"
        },
        # Thread (UNF - fine)
        "unf_thread": {
            "kt_range": (2.8, 5.0),
            "description": "UNF fine thread"
        }
    }
    
    def __init__(self):
        pass
    
    # -------------------------------------------------------------------------
    # Stress Concentration Factors (FIX-103)
    # -------------------------------------------------------------------------
    
    def calculate_stress_concentration_factor(
        self,
        geometry: Literal["circular_hole", "elliptical_hole", "shoulder_fillet",
                          "groove", "keyseat", "thread"],
        dimensions: Dict[str, float],
        load_type: Literal["tension", "bending", "torsion"] = "tension"
    ) -> float:
        """
        Calculate stress concentration factor (Kt).
        
        FIX-103: Implements proper stress concentration factors instead of
        ignoring geometric discontinuities.
        
        Args:
            geometry: Type of geometric discontinuity
            dimensions: Geometry dimensions
                - For circular_hole: {} (Kt=3 always for infinite plate)
                - For elliptical_hole: {"a": major_axis, "b": minor_axis}
                - For shoulder_fillet: {"D": large_d, "d": small_d, "r": fillet_radius}
                - For groove: {"d": diameter, "r": groove_radius}
            load_type: Type of loading
            
        Returns:
            Stress concentration factor Kt
        """
        if geometry == "circular_hole":
            return self._kt_circular_hole(load_type)
        
        elif geometry == "elliptical_hole":
            a = dimensions.get("a", 1.0)
            b = dimensions.get("b", 0.5)
            return self._kt_elliptical_hole(a, b)
        
        elif geometry == "shoulder_fillet":
            D = dimensions.get("D", 2.0)
            d = dimensions.get("d", 1.0)
            r = dimensions.get("r", 0.1)
            return self._kt_shoulder_fillet(D, d, r, load_type)
        
        elif geometry == "groove":
            d = dimensions.get("d", 1.0)
            r = dimensions.get("r", 0.1)
            return self._kt_groove(d, r, load_type)
        
        elif geometry == "keyseat":
            # Approximate range for keyseats
            return 2.0 if load_type == "torsion" else 2.5
        
        elif geometry == "thread":
            # Return conservative value for threads
            return 3.0 if load_type == "tension" else 2.5
        
        else:
            logger.warning(f"Unknown geometry '{geometry}', returning Kt=1.0")
            return 1.0
    
    def _kt_circular_hole(self, load_type: str) -> float:
        """Kt for circular hole in infinite plate"""
        if load_type == "tension":
            return 3.0
        elif load_type == "bending":
            return 2.0  # Lower for bending
        elif load_type == "torsion":
            return 4.0  # Higher for shear
        return 3.0
    
    def _kt_elliptical_hole(self, a: float, b: float) -> float:
        """
        Kt for elliptical hole (Inglis solution).
        Kt = 1 + 2*(a/b) where a is semi-axis perpendicular to load
        """
        if b <= 0:
            raise ValueError("Minor axis must be positive")
        return 1.0 + 2.0 * (a / b)
    
    def _kt_shoulder_fillet(self, D: float, d: float, r: float, 
                            load_type: str) -> float:
        """
        Kt for shoulder fillet using Peterson's charts.
        Approximate formula from Pilkey.
        """
        if r <= 0:
            return float('inf')  # Sharp corner
        
        # Geometric ratios
        D_d_ratio = D / d
        r_d_ratio = r / d
        
        if load_type == "tension":
            # Approximate formula for tension
            c1 = 0.0
            if D_d_ratio <= 1.5:
                c1 = 0.5
            elif D_d_ratio <= 2.0:
                c1 = 0.6
            else:
                c1 = 0.7
            
            kt = 1.0 + c1 / np.sqrt(r_d_ratio)
            return min(kt, 5.0)  # Cap at reasonable max
        
        elif load_type == "bending":
            # Bending typically lower Kt than tension
            kt = 1.0 + 0.4 / np.sqrt(r_d_ratio)
            return min(kt, 4.0)
        
        elif load_type == "torsion":
            # Torsion Kt
            kt = 1.0 + 0.6 / np.sqrt(r_d_ratio)
            return min(kt, 4.5)
        
        return 1.0
    
    def _kt_groove(self, d: float, r: float, load_type: str) -> float:
        """Kt for circumferential groove in shaft"""
        if r <= 0:
            return float('inf')
        
        r_d_ratio = r / d
        
        if load_type == "tension":
            kt = 1.0 + 0.5 / np.sqrt(r_d_ratio)
        elif load_type == "bending":
            kt = 1.0 + 0.35 / np.sqrt(r_d_ratio)
        elif load_type == "torsion":
            kt = 1.0 + 0.4 / np.sqrt(r_d_ratio)
        else:
            kt = 1.0
        
        return min(kt, 5.0)
    
    def apply_stress_concentration(
        self,
        nominal_stress: float,
        geometry: str,
        dimensions: Dict[str, float],
        load_type: str = "tension"
    ) -> Dict[str, float]:
        """
        Apply stress concentration to get maximum stress.
        
        Args:
            nominal_stress: Nominal stress (MPa or Pa)
            geometry: Geometric discontinuity type
            dimensions: Geometry dimensions
            load_type: Loading type
            
        Returns:
            Dictionary with nominal stress, max stress, and Kt
        """
        kt = self.calculate_stress_concentration_factor(
            geometry, dimensions, load_type
        )
        
        max_stress = nominal_stress * kt
        
        return {
            "nominal_stress": nominal_stress,
            "stress_concentration_factor": kt,
            "maximum_stress": max_stress,
            "geometry": geometry,
            "load_type": load_type
        }
    
    # -------------------------------------------------------------------------
    # Failure Criteria (FIX-104)
    # -------------------------------------------------------------------------
    
    def von_mises_stress(self, stress_state: StressState) -> float:
        """
        Calculate Von Mises equivalent stress.
        
        FIX-104: Implements Von Mises yield criterion.
        
        σ_vm = sqrt(0.5 * [(σ1-σ2)² + (σ2-σ3)² + (σ3-σ1)²])
        
        Or in terms of components:
        σ_vm = sqrt(0.5 * [(σx-σy)² + (σy-σz)² + (σz-σx)² + 6(τxy² + τyz² + τzx²)])
        """
        s = stress_state
        
        von_mises = np.sqrt(
            0.5 * (
                (s.sigma_x - s.sigma_y)**2 +
                (s.sigma_y - s.sigma_z)**2 +
                (s.sigma_z - s.sigma_x)**2 +
                6 * (s.tau_xy**2 + s.tau_yz**2 + s.tau_zx**2)
            )
        )
        
        return von_mises
    
    def tresca_stress(self, stress_state: StressState) -> float:
        """
        Calculate Tresca equivalent stress (maximum shear stress).
        
        σ_tresca = max(|σ1-σ2|, |σ2-σ3|, |σ3-σ1|) / 2
        """
        s1, s2, s3 = stress_state.principal_stresses()
        
        shear_stresses = [
            abs(s1 - s2) / 2.0,
            abs(s2 - s3) / 2.0,
            abs(s3 - s1) / 2.0
        ]
        
        return max(shear_stresses) * 2.0  # Return as normal stress equivalent
    
    def calculate_failure_criterion(
        self,
        stress_state: StressState,
        yield_strength: float,
        ultimate_strength: float = None,
        criterion: Literal["von_mises", "tresca", "rankine", 
                          "coulomb", "drucker_prager"] = "von_mises"
    ) -> FailureResult:
        """
        Calculate failure criterion and safety factor.
        
        FIX-104: Comprehensive failure criteria implementation.
        
        Args:
            stress_state: 3D stress state
            yield_strength: Material yield strength (MPa or Pa)
            ultimate_strength: Ultimate tensile strength [optional]
            criterion: Failure criterion to use
            
        Returns:
            Failure analysis result
        """
        if criterion == "von_mises":
            equiv_stress = self.von_mises_stress(stress_state)
            failure_mode = "ductile_yielding"
            
        elif criterion == "tresca":
            equiv_stress = self.tresca_stress(stress_state)
            failure_mode = "maximum_shear"
            
        elif criterion == "rankine":
            # Maximum principal stress (for brittle materials)
            s1, s2, s3 = stress_state.principal_stresses()
            equiv_stress = max(abs(s1), abs(s2), abs(s3))
            failure_mode = "brittle_fracture"
            
        elif criterion == "coulomb":
            # Coulomb-Mohr for brittle materials
            s1, s2, s3 = stress_state.principal_stresses()
            # Simplified Coulomb
            equiv_stress = abs(s1 - s3)
            failure_mode = "coulomb_mohr"
            
        else:
            raise ValueError(f"Unknown failure criterion: {criterion}")
        
        # Calculate safety factor
        if equiv_stress <= 0:
            safety_factor = float('inf')
        else:
            safety_factor = yield_strength / equiv_stress
        
        # Margin of safety
        if safety_factor == float('inf'):
            margin = float('inf')
        else:
            margin = safety_factor - 1.0
        
        # Determine if safe
        is_safe = safety_factor >= 1.0
        
        return FailureResult(
            failure_criterion=criterion,
            equivalent_stress=equiv_stress,
            yield_strength=yield_strength,
            safety_factor=safety_factor,
            margin_of_safety=margin,
            failure_mode=failure_mode,
            is_safe=is_safe
        )
    
    def calculate_safety_factor(
        self,
        applied_stress: float,
        yield_strength: float,
        stress_concentration: float = 1.0,
        load_factor: float = 1.0,
        material_factor: float = 1.0,
        required_fos: float = 1.5
    ) -> Dict[str, float]:
        """
        Calculate comprehensive safety factor with all considerations.
        
        FIX-105: Comprehensive safety factor calculation.
        
        Args:
            applied_stress: Applied nominal stress
            yield_strength: Material yield strength
            stress_concentration: Kt factor
            load_factor: Uncertainty in loads (>1 increases safety)
            material_factor: Material uncertainty (>1 increases safety)
            required_fos: Minimum required safety factor
            
        Returns:
            Dictionary with various safety metrics
        """
        # Maximum stress with concentration
        max_stress = applied_stress * stress_concentration
        
        # Basic safety factor
        if max_stress <= 0:
            basic_fos = float('inf')
        else:
            basic_fos = yield_strength / max_stress
        
        # Factored safety factor (conservative)
        # FOS_effective = (Sy / σ_max) / (load_factor * material_factor)
        if basic_fos == float('inf'):
            effective_fos = float('inf')
        else:
            effective_fos = basic_fos / (load_factor * material_factor)
        
        # Margin of safety
        if effective_fos == float('inf'):
            margin = float('inf')
        else:
            margin = effective_fos - 1.0
        
        # Reserve factor (how much more load can be applied)
        if effective_fos == float('inf'):
            reserve = float('inf')
        else:
            reserve = effective_fos
        
        # Check against required
        is_adequate = effective_fos >= required_fos
        
        return {
            "basic_safety_factor": basic_fos,
            "effective_safety_factor": effective_fos,
            "margin_of_safety": margin,
            "reserve_factor": reserve,
            "required_fos": required_fos,
            "is_adequate": is_adequate,
            "stress_concentration": stress_concentration,
            "load_factor": load_factor,
            "material_factor": material_factor,
            "nominal_stress": applied_stress,
            "maximum_stress": max_stress
        }
    
    # -------------------------------------------------------------------------
    # Combined Analysis
    # -------------------------------------------------------------------------
    
    def analyze_structural_safety(
        self,
        force: float,
        area: float,
        yield_strength: float,
        geometry: str = None,
        dimensions: Dict[str, float] = None,
        stress_state: StressState = None,
        load_factor: float = 1.0,
        required_fos: float = 1.5
    ) -> Dict[str, any]:
        """
        Comprehensive structural safety analysis.
        
        Combines stress concentration, failure criteria, and safety factors.
        
        Args:
            force: Applied force
            area: Cross-sectional area
            yield_strength: Material yield strength
            geometry: Geometric discontinuity [optional]
            dimensions: Geometry dimensions [optional]
            stress_state: Full stress state [optional]
            load_factor: Load uncertainty factor
            required_fos: Required safety factor
            
        Returns:
            Complete safety analysis results
        """
        # Calculate nominal stress
        if area <= 0:
            raise ValueError("Area must be positive")
        nominal_stress = force / area
        
        results = {
            "input": {
                "force": force,
                "area": area,
                "nominal_stress": nominal_stress,
                "yield_strength": yield_strength
            }
        }
        
        # Apply stress concentration if geometry provided
        if geometry and dimensions:
            sc_result = self.apply_stress_concentration(
                nominal_stress, geometry, dimensions
            )
            results["stress_concentration"] = sc_result
            max_stress = sc_result["maximum_stress"]
        else:
            max_stress = nominal_stress
            results["stress_concentration"] = None
        
        # Failure criterion analysis
        if stress_state is None:
            # Uniaxial stress state
            stress_state = StressState(sigma_x=max_stress)
        
        failure_result = self.calculate_failure_criterion(
            stress_state, yield_strength, criterion="von_mises"
        )
        results["failure_analysis"] = failure_result
        
        # Safety factor calculation
        kt = results["stress_concentration"]["stress_concentration_factor"] \
             if results["stress_concentration"] else 1.0
        
        safety_result = self.calculate_safety_factor(
            nominal_stress, yield_strength, kt, load_factor, 
            required_fos=required_fos
        )
        results["safety_factor"] = safety_result
        
        # Overall safety assessment
        results["is_safe"] = (
            failure_result.is_safe and 
            safety_result["is_adequate"]
        )
        
        return results


# Convenience functions
def von_mises_stress(sigma_x: float = 0, sigma_y: float = 0, sigma_z: float = 0,
                     tau_xy: float = 0, tau_yz: float = 0, tau_zx: float = 0) -> float:
    """Calculate Von Mises stress from components"""
    structs = AdvancedStructures()
    state = StressState(sigma_x, sigma_y, sigma_z, tau_xy, tau_yz, tau_zx)
    return structs.von_mises_stress(state)


def calculate_safety_factor(stress: float, yield_strength: float, 
                            kt: float = 1.0) -> float:
    """Calculate basic safety factor"""
    structs = AdvancedStructures()
    result = structs.calculate_safety_factor(stress, yield_strength, kt)
    return result["effective_safety_factor"]
