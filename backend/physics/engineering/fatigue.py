"""
FIX-106: Fatigue Analysis

S-N curve analysis, fatigue life prediction, and cumulative damage.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MaterialSNCurve:
    """S-N curve parameters for a material"""
    material_name: str
    # S-N curve in form: N = C * S^(-b) or log-log form
    fatigue_strength_coefficient: float  # σ'f (MPa)
    fatigue_strength_exponent: float     # b
    fatigue_ductility_coefficient: float # ε'f
    fatigue_ductility_exponent: float    # c
    # Endurance limit parameters
    endurance_limit: float               # σe (MPa)
    endurance_limit_cycles: int          # Ne (typically 10^6 or 10^7)
    # Static properties
    ultimate_strength: float             # Su (MPa)
    yield_strength: float                # Sy (MPa)
    elastic_modulus: float               # E (GPa)
    
    def get_cycles_to_failure(self, stress_amplitude: float) -> float:
        """Calculate cycles to failure for given stress amplitude"""
        if stress_amplitude <= 0:
            return float('inf')
        
        # Below endurance limit (infinite life)
        if stress_amplitude < self.endurance_limit:
            return float('inf')
        
        # Basquin equation: σa = σ'f * (2N)^b
        # Solve for N: N = 0.5 * (σa/σ'f)^(1/b)
        # Note: b is negative, so exponent is negative
        cycles = 0.5 * (stress_amplitude / self.fatigue_strength_coefficient) ** (1.0 / self.fatigue_strength_exponent)
        
        return max(1.0, cycles)


class FatigueAnalyzer:
    """
    Fatigue analysis for mechanical components.
    
    Implements:
    - S-N curve analysis
    - Cumulative damage (Miner's rule)
    - Mean stress effects (Goodman, Gerber, Soderberg)
    - Stress concentration effects
    """
    
    # Standard material S-N data
    # Sources: Shigley's Mechanical Engineering Design, ASM Handbook
    MATERIAL_DATABASE = {
        "steel_1045": MaterialSNCurve(
            material_name="1045 Steel",
            fatigue_strength_coefficient=1000.0,  # MPa
            fatigue_strength_exponent=-0.095,
            fatigue_ductility_coefficient=0.6,
            fatigue_ductility_exponent=-0.6,
            endurance_limit=310.0,
            endurance_limit_cycles=1_000_000,
            ultimate_strength=625.0,
            yield_strength=530.0,
            elastic_modulus=205.0
        ),
        "steel_4140": MaterialSNCurve(
            material_name="4140 Steel (Q&T)",
            fatigue_strength_coefficient=1400.0,
            fatigue_strength_exponent=-0.085,
            fatigue_ductility_coefficient=0.5,
            fatigue_ductility_exponent=-0.55,
            endurance_limit=420.0,
            endurance_limit_cycles=1_000_000,
            ultimate_strength=950.0,
            yield_strength=850.0,
            elastic_modulus=205.0
        ),
        "aluminum_6061_t6": MaterialSNCurve(
            material_name="6061-T6 Aluminum",
            fatigue_strength_coefficient=600.0,
            fatigue_strength_exponent=-0.110,
            fatigue_ductility_coefficient=0.35,
            fatigue_ductility_exponent=-0.65,
            endurance_limit=95.0,  # No true endurance for aluminum
            endurance_limit_cycles=500_000_000,
            ultimate_strength=310.0,
            yield_strength=276.0,
            elastic_modulus=69.0
        ),
        "aluminum_7075_t6": MaterialSNCurve(
            material_name="7075-T6 Aluminum",
            fatigue_strength_coefficient=800.0,
            fatigue_strength_exponent=-0.120,
            fatigue_ductility_coefficient=0.25,
            fatigue_ductility_exponent=-0.70,
            endurance_limit=130.0,
            endurance_limit_cycles=500_000_000,
            ultimate_strength=572.0,
            yield_strength=503.0,
            elastic_modulus=71.7
        ),
        "titanium_ti6al4v": MaterialSNCurve(
            material_name="Ti-6Al-4V",
            fatigue_strength_coefficient=1400.0,
            fatigue_strength_exponent=-0.08,
            fatigue_ductility_coefficient=0.4,
            fatigue_ductility_exponent=-0.55,
            endurance_limit=520.0,
            endurance_limit_cycles=10_000_000,
            ultimate_strength=950.0,
            yield_strength=880.0,
            elastic_modulus=114.0
        )
    }
    
    def __init__(self):
        self.materials = self.MATERIAL_DATABASE.copy()
    
    def estimate_sn_from_ultimate(
        self,
        ultimate_strength: float,
        material_type: Literal["steel", "aluminum", "titanium"] = "steel"
    ) -> MaterialSNCurve:
        """
        Estimate S-N curve from ultimate tensile strength.
        
        Uses empirical relationships from Shigley.
        """
        if material_type == "steel":
            # For steels: σ'f ≈ 1.0 * Su, σe ≈ 0.5 * Su (for Su < 1400 MPa)
            endurance = min(0.5 * ultimate_strength, 700.0)
            return MaterialSNCurve(
                material_name=f"Estimated Steel (Su={ultimate_strength} MPa)",
                fatigue_strength_coefficient=ultimate_strength,
                fatigue_strength_exponent=-0.085,
                fatigue_ductility_coefficient=0.6,
                fatigue_ductility_exponent=-0.6,
                endurance_limit=endurance,
                endurance_limit_cycles=1_000_000,
                ultimate_strength=ultimate_strength,
                yield_strength=0.8 * ultimate_strength,  # Estimate
                elastic_modulus=205.0
            )
        
        elif material_type == "aluminum":
            # For aluminum: no true endurance limit
            return MaterialSNCurve(
                material_name=f"Estimated Aluminum (Su={ultimate_strength} MPa)",
                fatigue_strength_coefficient=0.9 * ultimate_strength,
                fatigue_strength_exponent=-0.110,
                fatigue_ductility_coefficient=0.35,
                fatigue_ductility_exponent=-0.65,
                endurance_limit=0.3 * ultimate_strength,  # At 5e8 cycles
                endurance_limit_cycles=500_000_000,
                ultimate_strength=ultimate_strength,
                yield_strength=0.7 * ultimate_strength,
                elastic_modulus=69.0
            )
        
        elif material_type == "titanium":
            return MaterialSNCurve(
                material_name=f"Estimated Titanium (Su={ultimate_strength} MPa)",
                fatigue_strength_coefficient=1.1 * ultimate_strength,
                fatigue_strength_exponent=-0.08,
                fatigue_ductility_coefficient=0.4,
                fatigue_ductility_exponent=-0.55,
                endurance_limit=0.55 * ultimate_strength,
                endurance_limit_cycles=10_000_000,
                ultimate_strength=ultimate_strength,
                yield_strength=0.9 * ultimate_strength,
                elastic_modulus=114.0
            )
        
        else:
            raise ValueError(f"Unknown material type: {material_type}")
    
    def apply_mean_stress_correction(
        self,
        stress_amplitude: float,
        mean_stress: float,
        ultimate_strength: float,
        yield_strength: float,
        method: Literal["goodman", "gerber", "soderberg", "morrow"] = "goodman"
    ) -> float:
        """
        Apply mean stress correction to get equivalent completely reversed stress.
        
        Args:
            stress_amplitude: Alternating stress component (σa)
            mean_stress: Mean stress component (σm)
            ultimate_strength: Ultimate tensile strength (Su)
            yield_strength: Yield strength (Sy)
            method: Mean stress correction method
            
        Returns:
            Equivalent completely reversed stress amplitude
        """
        if method == "goodman":
            # Goodman: σa / σar + σm / Su = 1
            # σar = σa / (1 - σm/Su)
            if mean_stress >= ultimate_strength:
                return float('inf')
            return stress_amplitude / (1.0 - mean_stress / ultimate_strength)
        
        elif method == "gerber":
            # Gerber: σa / σar + (σm / Su)² = 1
            # σar = σa / (1 - (σm/Su)²)
            ratio = mean_stress / ultimate_strength
            return stress_amplitude / (1.0 - ratio**2)
        
        elif method == "soderberg":
            # Soderberg: σa / σar + σm / Sy = 1 (conservative)
            if mean_stress >= yield_strength:
                return float('inf')
            return stress_amplitude / (1.0 - mean_stress / yield_strength)
        
        elif method == "morrow":
            # Morrow: uses true fracture strength
            # Simplified here to use ultimate strength
            sigma_f_prime = 1.1 * ultimate_strength  # Estimate
            return stress_amplitude / (1.0 - mean_stress / sigma_f_prime)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def calculate_cycles_to_failure(
        self,
        stress_amplitude: float,
        material: str or MaterialSNCurve,
        mean_stress: float = 0.0,
        mean_stress_method: str = "goodman",
        surface_factor: float = 1.0,
        size_factor: float = 1.0,
        loading_factor: float = 1.0,
        temperature_factor: float = 1.0,
        reliability_factor: float = 0.814  # 99% reliability
    ) -> Dict[str, float]:
        """
        Calculate fatigue life with all corrections.
        
        Args:
            stress_amplitude: Applied stress amplitude (MPa)
            material: Material name or SN curve
            mean_stress: Mean stress (MPa)
            mean_stress_method: Goodman, Gerber, etc.
            surface_factor: Ka (surface finish)
            size_factor: Kb (size effect)
            loading_factor: Kc (loading type)
            temperature_factor: Kd
            reliability_factor: Ke
            
        Returns:
            Dictionary with cycles to failure and analysis details
        """
        # Get material data
        if isinstance(material, str):
            if material not in self.materials:
                # Estimate from ultimate strength if provided
                raise ValueError(f"Unknown material: {material}")
            mat = self.materials[material]
        else:
            mat = material
        
        # Apply mean stress correction
        if mean_stress != 0:
            equiv_stress = self.apply_mean_stress_correction(
                stress_amplitude, mean_stress,
                mat.ultimate_strength, mat.yield_strength,
                mean_stress_method
            )
        else:
            equiv_stress = stress_amplitude
        
        # Apply Marin factors (corrections)
        # Se = Ka * Kb * Kc * Kd * Ke * Se'
        correction_factor = (
            surface_factor * size_factor * loading_factor * 
            temperature_factor * reliability_factor
        )
        
        # Adjusted endurance limit
        adjusted_endurance = mat.endurance_limit * correction_factor
        
        # Adjusted fatigue strength coefficient
        adjusted_fatigue_strength = mat.fatigue_strength_coefficient * correction_factor
        
        # Calculate cycles to failure
        if equiv_stress <= adjusted_endurance:
            cycles = float('inf')
            infinite_life = True
        else:
            # Basquin equation
            cycles = 0.5 * (equiv_stress / adjusted_fatigue_strength) ** (1.0 / mat.fatigue_strength_exponent)
            infinite_life = False
        
        return {
            "cycles_to_failure": cycles,
            "infinite_life": infinite_life,
            "equivalent_stress": equiv_stress,
            "applied_stress_amplitude": stress_amplitude,
            "mean_stress": mean_stress,
            "mean_stress_method": mean_stress_method,
            "correction_factor": correction_factor,
            "adjusted_endurance_limit": adjusted_endurance,
            "material": mat.material_name,
            "stress_ratio": (mean_stress - stress_amplitude) / (mean_stress + stress_amplitude) if (mean_stress + stress_amplitude) != 0 else -1
        }
    
    def calculate_miners_rule(
        self,
        stress_blocks: List[Dict[str, float]],
        material: str or MaterialSNCurve
    ) -> Dict[str, float]:
        """
        Calculate cumulative damage using Miner's rule.
        
        D = Σ(ni/Ni)
        Failure when D >= 1
        
        Args:
            stress_blocks: List of dicts with:
                - stress_amplitude: Stress for this block
                - mean_stress: Mean stress for this block
                - cycles: Number of cycles at this stress
            material: Material name or SN curve
            
        Returns:
            Dictionary with damage calculation results
        """
        total_damage = 0.0
        block_results = []
        
        for block in stress_blocks:
            # Get cycles to failure for this stress level
            result = self.calculate_cycles_to_failure(
                block["stress_amplitude"],
                material,
                block.get("mean_stress", 0.0)
            )
            
            n_i = block["cycles"]
            N_i = result["cycles_to_failure"]
            
            if N_i == float('inf'):
                damage = 0.0
            else:
                damage = n_i / N_i
            
            total_damage += damage
            
            block_results.append({
                "stress": block["stress_amplitude"],
                "cycles_applied": n_i,
                "cycles_to_failure": N_i,
                "damage": damage,
                "infinite_life": result["infinite_life"]
            })
        
        # Safety factor on life
        if total_damage > 0:
            life_factor = 1.0 / total_damage
        else:
            life_factor = float('inf')
        
        return {
            "total_damage": total_damage,
            "damage_ratio": total_damage,
            "life_factor": life_factor,
            "predicted_cycles": life_factor * sum(b["cycles"] for b in stress_blocks),
            "has_failed": total_damage >= 1.0,
            "block_details": block_results,
            "remaining_life": max(0.0, 1.0 - total_damage)
        }
    
    def calculate_stress_concentration_fatigue(
        self,
        kt: float,
        notch_radius: float,
        ultimate_strength: float,
        method: Literal["neuber", "peterson"] = "peterson"
    ) -> float:
        """
        Calculate fatigue stress concentration factor (Kf).
        
        Args:
            kt: Theoretical stress concentration factor
            notch_radius: Notch radius (mm)
            ultimate_strength: Ultimate strength (MPa)
            method: Notch sensitivity method
            
        Returns:
            Fatigue stress concentration factor Kf
        """
        if method == "peterson":
            # Peterson's notch sensitivity
            # q = 1 / (1 + a/r) where a is Peterson's constant
            # For steels: a ≈ 0.025 * (2079 / Su)^1.8 (Su in MPa)
            a = 0.025 * (2079 / ultimate_strength) ** 1.8  # mm
            notch_sensitivity = 1.0 / (1.0 + a / notch_radius)
            
        elif method == "neuber":
            # Neuber's notch sensitivity
            # q = 1 / (1 + sqrt(a'/r))
            # For steels: a' ≈ 0.33 * (Su/1000)^1.5 (Su in MPa, a' in mm)
            a_prime = 0.33 * (ultimate_strength / 1000) ** 1.5
            notch_sensitivity = 1.0 / (1.0 + np.sqrt(a_prime / notch_radius))
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Kf = 1 + q * (Kt - 1)
        kf = 1.0 + notch_sensitivity * (kt - 1.0)
        
        return kf
    
    def generate_sn_curve(
        self,
        material: str or MaterialSNCurve,
        stress_range: Tuple[float, float] = (1000.0, 10.0),
        num_points: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Generate S-N curve data for plotting.
        
        Args:
            material: Material name or SN curve
            stress_range: (max_stress, min_stress) in MPa
            num_points: Number of points
            
        Returns:
            Dictionary with stress and cycles arrays
        """
        if isinstance(material, str):
            mat = self.materials[material]
        else:
            mat = material
        
        stresses = np.linspace(stress_range[0], stress_range[1], num_points)
        cycles = []
        
        for s in stresses:
            result = self.calculate_cycles_to_failure(s, mat)
            cycles.append(result["cycles_to_failure"])
        
        return {
            "stress": stresses,
            "cycles": np.array(cycles),
            "endurance_limit": mat.endurance_limit,
            "material": mat.material_name
        }


# Convenience functions
def calculate_fatigue_life(
    stress_amplitude: float,
    ultimate_strength: float,
    cycles_at_1000: float = 1000.0,
    endurance_limit: float = None,
    mean_stress: float = 0.0
) -> float:
    """
    Simplified fatigue life calculation.
    
    Uses simplified S-N curve from ultimate strength.
    """
    analyzer = FatigueAnalyzer()
    
    # Estimate material properties
    material = analyzer.estimate_sn_from_ultimate(ultimate_strength)
    
    result = analyzer.calculate_cycles_to_failure(
        stress_amplitude, material, mean_stress
    )
    
    return result["cycles_to_failure"]
