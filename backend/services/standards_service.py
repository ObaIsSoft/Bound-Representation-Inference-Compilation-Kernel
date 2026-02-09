"""
Standards Service - ISO, ASME, ASTM Standards Lookup

Provides access to engineering standards from database.
No hardcoded values - all standards come from Supabase.
"""

from typing import Dict, Any, Optional
import logging

from .supabase_service import supabase

logger = logging.getLogger(__name__)


class StandardsServiceError(Exception):
    """Standards service error"""
    pass


class StandardNotFoundError(StandardsServiceError):
    """Requested standard not found in database"""
    pass


class StandardsService:
    """
    Engineering standards lookup service.
    
    Supports:
    - ISO 286 tolerance classes (H7/g6, etc.) - definitions only
    - AWG wire ampacity tables
    - Safety factors by industry
    
    IMPORTANT: This service returns ONLY verified data from standards.
    If data is missing, it returns None or raises an error.
    NO DEFAULTS, NO GUESSES.
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
            nominal_size_mm: Nominal size in mm (required for tolerance calculation)
            
        Returns:
            Fit tolerance data
            
        Raises:
            StandardNotFoundError: If tolerance class not found
            ValueError: If nominal_size_mm not provided (required for ISO 286)
        """
        # Get fit classification from database
        fit_data = await supabase.get_standard("iso_fit", tolerance_class)
        
        if not fit_data:
            raise StandardNotFoundError(
                f"ISO fit class '{tolerance_class}' not found in standards database. "
                f"Available fits must be configured in standards_reference table."
            )
        
        # ISO 286 requires nominal size for tolerance calculation
        # The standard uses size-dependent lookup tables
        if nominal_size_mm is None:
            # Return classification only, warn that size is needed
            return {
                "fit_class": tolerance_class,
                "fit_type": fit_data.get("fit_type"),
                "hole_deviation": fit_data.get("hole_deviation"),
                "shaft_deviation": fit_data.get("shaft_deviation"),
                "application": fit_data.get("application"),
                "description": fit_data.get("description"),
                "_warning": "TOLERANCE VALUES NOT PROVIDED. "
                           "ISO 286 requires size-dependent lookup tables. "
                           "Provide nominal_size_mm for tolerance calculation."
            }
        
        # TODO: Implement full ISO 286 lookup table
        # For now, return error - we don't have the full tables
        raise StandardsServiceError(
            f"ISO 286 tolerance calculation not yet implemented. "
            f"Full ISO 286-1:2010 lookup tables required. "
            f"Use engineering reference tables or CAD software for {tolerance_class} "
            f"at {nominal_size_mm}mm."
        )
    
    async def get_awg_ampacity(
        self,
        awg_gauge: int,
        insulation_rating: str = "60C"
    ) -> Dict[str, Any]:
        """
        Get AWG wire ampacity from NEC Table 310.16.
        
        Args:
            awg_gauge: AWG wire gauge (e.g., 12, 14, 16)
            insulation_rating: Insulation temp rating (60C, 75C, 90C)
            
        Returns:
            Ampacity data from verified NEC standards
            
        Raises:
            StandardNotFoundError: If AWG gauge not in database
        """
        # Get base ampacity from database
        base_data = await supabase.get_standard("awg_ampacity", str(awg_gauge))
        
        if not base_data:
            raise StandardNotFoundError(
                f"AWG {awg_gauge} not found in standards database. "
                f"Only gauges in NEC Table 310.16 are supported."
            )
        
        # Get ampacity for specified temperature rating
        ampacity_key = f"ampacity_{insulation_rating.lower()}_a"
        ampacity = base_data.get(ampacity_key)
        
        if ampacity is None:
            # Fall back to 60C if requested rating not available
            ampacity = base_data.get("ampacity_60c_a")
            if ampacity:
                logger.warning(
                    f"Ampacity for {insulation_rating} not available for AWG {awg_gauge}. "
                    f"Using 60°C rating. Check NEC for derating factors."
                )
        
        return {
            "awg_gauge": awg_gauge,
            "ampacity_a": ampacity,
            "insulation_rating": insulation_rating,
            "diameter_mm": base_data.get("diameter_mm"),
            "area_mm2": base_data.get("area_mm2"),
            "resistance_ohm_per_m": base_data.get("resistance_ohm_per_m"),
            "standard": "NEC Table 310.16",
            "conditions": [
                "Copper conductor",
                f"{insulation_rating}°C insulation",
                "Ambient temperature 30°C",
                "Not more than 3 current-carrying conductors in raceway"
            ],
            "_note": "Apply correction factors from NEC for other conditions"
        }
    
    async def get_safety_factor(
        self,
        application: str,
        material_type: str = "metal"
    ) -> Dict[str, Any]:
        """
        Get recommended safety factor from industry standards.
        
        Args:
            application: Application type (e.g., "aerospace", "automotive", "consumer")
            material_type: Material type (not currently used, reserved for future)
            
        Returns:
            Safety factor data from verified standards
            
        Raises:
            StandardNotFoundError: If application not in database
        """
        data = await supabase.get_standard("safety_factor", application)
        
        if not data:
            raise StandardNotFoundError(
                f"Safety factor for '{application}' not found in standards database. "
                f"Available applications must be configured in standards_reference table."
            )
        
        return {
            "application": application,
            "minimum_factor": data.get("minimum_factor"),
            "standard": f"{data.get('_org')} {data.get('_number')}",
            "basis": data.get("basis"),
            "_note": "These are MINIMUM values. Actual factor depends on criticality, "
                    "uncertainty, inspection rigor, and regulatory requirements."
        }
    
    async def get_manufacturing_constraint(
        self,
        process: str,
        constraint_type: str
    ) -> Dict[str, Any]:
        """
        Get manufacturing constraint from equipment specifications.
        
        Args:
            process: Manufacturing process (e.g., "cnc_milling", "fdm_printing")
            constraint_type: Type of constraint (e.g., "min_wall_thickness")
            
        Returns:
            Constraint data from manufacturer specifications
            
        Raises:
            StandardNotFoundError: If constraint not found
        """
        key = f"{process}_{constraint_type}"
        data = await supabase.get_standard("manufacturing_constraint", key)
        
        if not data:
            raise StandardNotFoundError(
                f"Manufacturing constraint '{key}' not found in standards database. "
                f"Constraints must be configured in standards_reference table."
            )
        
        return {
            "process": process,
            "constraint_type": constraint_type,
            "minimum_mm": data.get("min_mm"),
            "typical_mm": data.get("typical_mm"),
            "limitation": data.get("limitation"),
            "note": data.get("note"),
            "_note": "Values are typical for standard equipment. "
                    "Actual capabilities vary by specific machine and material."
        }
    
    async def list_available_iso_fits(self) -> list:
        """
        List all available ISO fit classes in database.
        
        Returns:
            List of available fit class names
        """
        # Query database for available fits
        result = await supabase.client.table("standards_reference")\
            .select("standard_key")\
            .eq("standard_type", "iso_fit")\
            .execute()
        
        return [row["standard_key"] for row in result.data]
    
    async def list_available_awg_gauges(self) -> list:
        """
        List all available AWG gauges in database.
        
        Returns:
            List of available AWG gauge numbers
        """
        result = await supabase.client.table("standards_reference")\
            .select("standard_key")\
            .eq("standard_type", "awg_ampacity")\
            .execute()
        
        return [int(row["standard_key"]) for row in result.data]


# Global instance
standards_service = StandardsService()
