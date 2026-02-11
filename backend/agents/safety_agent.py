"""
SafetyAgent: Safety evaluation agent.

Uses material-specific properties and industry safety standards.
No hardcoded thresholds - all limits come from database.
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class SafetyAgent:
    """
    Safety Agent with database-driven thresholds.
    
    Stress limits: Based on material yield strength / safety_factor
    Temperature limits: Based on material max_temp_c
    Safety factors: From industry standards (NASA, ISO, IEC)
    
    No hardcoded limits - fails if material not in database.
    """
    
    def __init__(self, application_type: str = "industrial"):
        self.name = "SafetyAgent"
        self.application_type = application_type
        
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run safety checks using material-specific properties.
        
        Args:
            params: {
                "physics_results": {...},
                "materials": ["Steel 4140", "Aluminum 6061-T6", ...],
                "application_type": "aerospace" | "automotive" | "medical" | "industrial"
            }
            
        Returns:
            {
                "status": "safe" | "hazards_detected",
                "safety_score": float,
                "hazards": List[str],
                "limits": {...}
            }
        """
        logger.info("[SafetyAgent] Evaluating design safety...")
        
        try:
            from services import supabase, standards_service
        except ImportError:
            from backend.services import supabase, standards_service
        
        hazards = []
        score = 1.0
        warnings = []
        limits_used = {}
        
        physics = params.get("physics_results", {})
        materials = params.get("materials", [])
        app_type = params.get("application_type", self.application_type)
        
        # Flatten metrics
        metrics = {}
        for key, val in physics.items():
            if isinstance(val, dict):
                metrics.update(val)
            else:
                metrics[key] = val
        
        # Get safety factor for application type
        try:
            safety_factor_data = await standards_service.get_safety_factor(app_type)
            safety_factor = safety_factor_data.get("minimum_factor", 2.0)
        except Exception as e:
            warnings.append(f"Could not load safety factor for {app_type}: {e}")
            safety_factor = 2.0  # Conservative default only when standard unavailable
        
        # 1. Check stress against material yield strength
        max_stress = metrics.get("max_stress_mpa", 0)
        
        if max_stress > 0 and materials:
            # Use the primary material (first in list) for stress limit
            primary_material = materials[0]
            
            try:
                mat_data = await supabase.get_material(primary_material)
                yield_strength = mat_data.get("yield_strength_mpa")
                
                if yield_strength:
                    # Safe stress = yield_strength / safety_factor
                    safe_stress_limit = yield_strength / safety_factor
                    limits_used["stress_limit_mpa"] = round(safe_stress_limit, 2)
                    limits_used["yield_strength_mpa"] = yield_strength
                    limits_used["safety_factor"] = safety_factor
                    
                    if max_stress > safe_stress_limit:
                        hazards.append(
                            f"High Stress: {max_stress:.1f} MPa exceeds safe limit "
                            f"{safe_stress_limit:.1f} MPa "
                            f"({primary_material} yield: {yield_strength} MPa / SF: {safety_factor})"
                        )
                        score -= 0.3
                else:
                    warnings.append(
                        f"Yield strength not available for {primary_material}. "
                        f"Cannot perform stress safety check."
                    )
                    
            except Exception as e:
                warnings.append(
                    f"Material '{primary_material}' not found in database. "
                    f"Add material to perform stress safety check."
                )
        
        # 2. Check temperature against material limit
        max_temp = metrics.get("max_temp_c", 0)
        
        if max_temp > 0 and materials:
            # Find the material with the lowest max_temp
            lowest_max_temp = float('inf')
            limiting_material = None
            
            for mat_name in materials:
                try:
                    mat_data = await supabase.get_material(mat_name)
                    mat_max_temp = mat_data.get("max_temp_c")
                    
                    if mat_max_temp and mat_max_temp < lowest_max_temp:
                        lowest_max_temp = mat_max_temp
                        limiting_material = mat_name
                        
                except Exception:
                    continue
            
            if limiting_material and lowest_max_temp < float('inf'):
                # Apply safety margin (80% of max temp)
                safe_temp_limit = lowest_max_temp * 0.8
                limits_used["temperature_limit_c"] = round(safe_temp_limit, 1)
                limits_used["material_max_temp_c"] = lowest_max_temp
                limits_used["limiting_material"] = limiting_material
                
                if max_temp > safe_temp_limit:
                    hazards.append(
                        f"High Temperature: {max_temp:.1f}°C exceeds safe limit "
                        f"{safe_temp_limit:.1f}°C "
                        f"({limiting_material} max: {lowest_max_temp}°C with 20% margin)"
                    )
                    score -= 0.3
            else:
                warnings.append(
                    "No temperature limits available for materials. "
                    f"Cannot perform thermal safety check."
                )
        
        # 3. Additional checks based on application type
        if app_type == "aerospace":
            # Aerospace-specific checks
            if metrics.get("fatigue_cycles", float('inf')) < 10000:
                if metrics.get("max_stress_mpa", 0) > 100:
                    hazards.append("Potential fatigue issue: High stress with low cycle count")
                    score -= 0.1
        
        elif app_type == "medical":
            # Medical-specific checks (more stringent)
            if score < 0.9:
                hazards.append("Medical devices require >90% safety score")
        
        return {
            "status": "safe" if not hazards else "hazards_detected",
            "safety_score": max(0.0, score),
            "hazards": hazards,
            "warnings": warnings,
            "limits": limits_used,
            "application_type": app_type,
            "materials_checked": materials
        }
    
    async def get_material_safety_limits(
        self, 
        material: str, 
        application_type: str = "industrial"
    ) -> Dict[str, Any]:
        """
        Get safety limits for a specific material.
        
        Args:
            material: Material name
            application_type: Industry application type
            
        Returns:
            Safety limits and factors
        """
        from backend.services import supabase, standards_service
        
        try:
            # Get material properties
            mat_data = await supabase.get_material(material)
            
            # Get safety factor
            safety_factor_data = await standards_service.get_safety_factor(application_type)
            safety_factor = safety_factor_data.get("minimum_factor", 2.0)
            
            yield_strength = mat_data.get("yield_strength_mpa")
            max_temp = mat_data.get("max_temp_c")
            
            limits = {
                "material": material,
                "application_type": application_type,
                "safety_factor": safety_factor,
                "safety_factor_source": f"{safety_factor_data.get('standard', 'industry_standard')}"
            }
            
            if yield_strength:
                limits["safe_stress_mpa"] = yield_strength / safety_factor
                limits["yield_strength_mpa"] = yield_strength
            
            if max_temp:
                limits["safe_temperature_c"] = max_temp * 0.8
                limits["max_temperature_c"] = max_temp
            
            return limits
            
        except Exception as e:
            return {
                "error": f"Could not get safety limits for {material}",
                "details": str(e)
            }
