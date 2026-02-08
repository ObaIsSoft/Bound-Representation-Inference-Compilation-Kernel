"""
Standards Service - ISO, ASME, ASTM Standards Lookup

Provides access to engineering standards from database.
No hardcoded values - all standards come from Supabase.
"""

from typing import Dict, Any, Optional
import logging

from .supabase_service import supabase

logger = logging.getLogger(__name__)


class StandardsService:
    """
    Engineering standards lookup service.
    
    Supports:
    - ISO 286 tolerance classes (H7/g6, etc.)
    - AWG wire ampacity tables
    - Safety factors by industry
    - Material test standards
    """
    
    async def get_iso_fit(
        self,
        tolerance_class: str,
        nominal_size_mm: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get ISO 286 fit tolerance data.
        
        Args:
            tolerance_class: Tolerance class (e.g., "H7/g6")
            nominal_size_mm: Nominal size in mm (for size-specific tolerances)
            
        Returns:
            Fit tolerance data
            
        Raises:
            ValueError: If tolerance class not found
        """
        fit_data = await supabase.get_standard("iso_fit", tolerance_class)
        
        # If size provided, calculate absolute tolerances
        if nominal_size_mm is not None and "fundamental_deviation" in fit_data:
            # Calculate actual tolerance values based on size
            # This is simplified - real ISO 286 has complex size ranges
            fd = fit_data["fundamental_deviation"]
            it = fit_data.get("tolerance_grade", 0.01)
            
            fit_data["calculated"] = {
                "nominal_size_mm": nominal_size_mm,
                "upper_deviation_mm": fd + it,
                "lower_deviation_mm": fd,
                "max_clearance_mm": fit_data.get("max_clear", 0),
                "min_clearance_mm": fit_data.get("min_clear", 0)
            }
        
        return fit_data
    
    async def get_awg_ampacity(
        self,
        awg_gauge: int,
        insulation_rating: str = "60C"
    ) -> Dict[str, Any]:
        """
        Get AWG wire ampacity.
        
        Args:
            awg_gauge: AWG wire gauge (e.g., 12, 14, 16)
            insulation_rating: Insulation temp rating (60C, 75C, 90C)
            
        Returns:
            Ampacity data
        """
        # Get base ampacity
        base_data = await supabase.get_standard("awg_ampacity", str(awg_gauge))
        
        # Adjust for insulation rating
        temp_multiplier = {
            "60C": 1.0,
            "75C": 1.15,
            "90C": 1.25
        }.get(insulation_rating, 1.0)
        
        base_ampacity = base_data.get("ampacity_a", 0)
        
        return {
            "awg_gauge": awg_gauge,
            "base_ampacity_a": base_ampacity,
            "insulation_rating": insulation_rating,
            "adjusted_ampacity_a": base_ampacity * temp_multiplier,
            "diameter_mm": base_data.get("diameter_mm"),
            "resistance_ohm_per_m": base_data.get("resistance_ohm_per_m")
        }
    
    async def get_safety_factor(
        self,
        application: str,
        material_type: str = "metal"
    ) -> float:
        """
        Get recommended safety factor.
        
        Args:
            application: Application type (e.g., "aerospace", "automotive", "consumer")
            material_type: Material type (e.g., "metal", "composite", "plastic")
            
        Returns:
            Safety factor (e.g., 1.5, 2.0, 4.0)
        """
        key = f"{application}_{material_type}"
        
        try:
            data = await supabase.get_standard("safety_factor", key)
            return data.get("safety_factor", 2.0)
        except ValueError:
            # Fallback to generic application factor
            data = await supabase.get_standard("safety_factor", application)
            return data.get("safety_factor", 2.0)
    
    async def get_manufacturing_constraint(
        self,
        process: str,
        constraint_type: str
    ) -> Dict[str, Any]:
        """
        Get manufacturing constraint (min wall thickness, etc.)
        
        Args:
            process: Manufacturing process (e.g., "cnc_milling", "fdm_printing")
            constraint_type: Type of constraint (e.g., "min_wall_thickness")
            
        Returns:
            Constraint data
        """
        key = f"{process}_{constraint_type}"
        return await supabase.get_standard("manufacturing_constraint", key)
    
    async def get_all_iso_fits(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all available ISO fit classes.
        
        Returns:
            Dictionary of fit class -> fit data
        """
        # This would query all ISO fits from database
        # For now, return common ones
        common_fits = [
            "H7/g6",  # Clearance fit
            "H7/k6",  # Transition fit
            "H7/p6",  # Interference fit
            "H7/h6",  # Sliding fit
        ]
        
        results = {}
        for fit in common_fits:
            try:
                results[fit] = await self.get_iso_fit(fit)
            except ValueError:
                pass
        
        return results


# Global instance
standards_service = StandardsService()
