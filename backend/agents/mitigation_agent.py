from typing import Dict, Any, List
import logging
import re

logger = logging.getLogger(__name__)

class MitigationAgent:
    """
    Mitigation Agent - Quantitative Fix Proposer.
    
    Uses physics data and error strings to calculate specific dimensional changes.
    Example: "Yield Exceeded by 20%" -> "Increase Thickness by 20% * SafetyFactor"
    """
    
    def __init__(self):
        self.name = "MitigationAgent"
        
        # Tier 5: Failure Surrogate
        try:
            from models.mitigation_surrogate import FailureSurrogate
            self.surrogate = FailureSurrogate()
            self.use_surrogate = True
        except ImportError:
            self.surrogate = None
            self.use_surrogate = False
    
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Propose quantitative fixes.
        """
        errors = params.get("errors", [])
        physics_data = params.get("physics_data", {}) # e.g. {max_stress: 350, yield: 276}
        geometry_tree = params.get("geometry_tree", [])
        
        logs = [f"[MITIGATION] Analyzing {len(errors)} errors with quantitative data"]
        fixes = []
        
        for error in errors:
            # 1. Stress / Yield Parsing
            # Try to find numeric patterns if physics_data isn't fully providing it
            if "yield" in error.lower() or "stress" in error.lower():
                fix = self._calculate_stress_fix(error, physics_data)
                if fix: fixes.append(fix)
                
            # 2. Buckling
            elif "buckling" in error.lower():
                 fix = self._calculate_buckling_fix(physics_data)
                 if fix: fixes.append(fix)
                 
            # 3. Generic/Other (Fallback to pattern match)
            else:
                 fixes.append({"description": f"Investigate: {error}", "priority": "low"})
        
        logs.append(f"[MITIGATION] Generated {len(fixes)} quantitative fixes")
        
        # Tier 5: Neural Risk Assessment
        if self.use_surrogate and self.surrogate:
            risk_fixes = self._analyze_neural_risk(physics_data)
            if risk_fixes:
                fixes.extend(risk_fixes)
                logs.append(f"[MITIGATION] Neural Surrogate added {len(risk_fixes)} risk-based fixes")
                
        return {"fixes": fixes, "logs": logs}

    def _calculate_stress_fix(self, error_msg: str, physics_data: Dict) -> Dict:
        """Calculate required area increase for stress failure."""
        
        # Get data from physics payload if available
        max_stress = physics_data.get("max_stress_mpa")
        yield_str = physics_data.get("yield_strength_mpa") # fixed typo
        
        # If missing, try regex from error string (e.g., "Stress 350 > Yield 276")
        if not max_stress or not yield_str:
            nums = re.findall(r"\d+\.?\d*", error_msg)
            if len(nums) >= 2:
                # heuristic: larger is likely stress if it failed
                vals = sorted([float(n) for n in nums])
                yield_str = vals[0]
                max_stress = vals[-1]
        
        if max_stress and yield_str and max_stress > yield_str:
            ratio = max_stress / yield_str
            target_safety_factor = 1.5
            required_increase = (ratio * target_safety_factor) - 1.0
            
            return {
                "type": "geometry",
                "description": f"Increase load-bearing cross-section area by {required_increase:.0%}",
                "action": f"Scale thickness or width by {1.0 + required_increase:.2f}x",
                "technical_basis": f"Current Stress ({max_stress}MPa) > Yield ({yield_str}MPa). Target FoS: 1.5",
                "priority": "critical"
            }
        return None

    def _calculate_buckling_fix(self, physics_data: Dict) -> Dict:
        """Calculate inertia increase for buckling."""
        # Buckling load P_cr is proportional to Area^2 (for solid circle) approximately
        # or Radius^4. 
        # Increase radius is most effective.
        return {
            "type": "geometry",
            "description": "Increase cross-section radius to improve Moment of Inertia",
            "action": "Increase radius by 15-20%",
            "technical_basis": "Buckling is invalid; Moment of Inertia (I) scales with r^4",
            "priority": "critical"
        }

    def _analyze_neural_risk(self, physics_data: Dict) -> List[Dict]:
        """
        Use FailureSurrogate to detect hidden risks (fatigue, creep) 
        missed by simple limit checks.
        """
        fixes = []
        
        # Extract features
        max_stress = float(physics_data.get("max_stress_mpa", 0.0))
        yield_str = float(physics_data.get("yield_strength_mpa", 1.0))
        stress_ratio = max_stress / max(0.1, yield_str)
        
        temp_c = float(physics_data.get("temperature_c", 25.0))
        cycles = float(physics_data.get("cycles", 1e4))
        corrosion = float(physics_data.get("corrosion_index", 0.0))
        
        # Predict Probability
        try:
            prob = self.surrogate.predict_risk(stress_ratio, temp_c, cycles, corrosion)
            
            RISK_THRESHOLD = 0.05 # 5% probability
            
            if prob > RISK_THRESHOLD:
                fixes.append({
                    "type": "reliability",
                    "description": f"High Failure Probability ({prob:.1%}) detected by Neural Net",
                    "action": "Reduce Operating Temp or Increase Area",
                    "technical_basis": f"Combined Risk: StressRatio={stress_ratio:.2f}, Cycles={cycles:.0e}, Temp={temp_c}C",
                    "priority": "high",
                    "source": "surrogate"
                })
        except Exception as e:
            logger.warning(f"Risk prediction failed: {e}")
            
        return fixes

    def verify_critical_dimensions(self, verification_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify critical dimensions using VMK.
        Enforces 1nm tolerance (1e-6 mm).
        
        Args:
            verification_plan: {
                "toolpaths": [VMKInstruction...],
                "check_points": [
                    {"coord": [x,y,z], "expected_surface": True}
                ]
            }
        """
        try:
            from vmk_kernel import SymbolicMachiningKernel, ToolProfile, VMKInstruction
            import numpy as np
        except ImportError:
            return {"verified": False, "error": "VMK not available"}

        # 1. Replay Manufacturing (Symbolic)
        kernel = SymbolicMachiningKernel(stock_dims=[100, 100, 100])
        
        # Register default tools if needed or assume plan provides them
        # For simplicity, we assume a standard toolset for verification
        # kernel.register_tool(ToolProfile(id="verify_tool", radius=0.1, type="BALL"))
        
        registered_tools = set()

        for op in verification_plan.get("toolpaths", []):
            tid = op.get("tool_id", "default")
            if tid not in registered_tools:
                # In a real system, we'd look up the tool spec. 
                # Here we default to 0.1mm radius for specific verification logic if unknown.
                kernel.register_tool(ToolProfile(id=tid, radius=0.1, type="BALL"))
                registered_tools.add(tid)
                
            kernel.execute_gcode(VMKInstruction(**op))
            
        # 2. Verify Points
        failures = []
        max_error = 0.0
        
        for pt in verification_plan.get("check_points", []):
            coords = np.array(pt["coord"])
            sdf = kernel.get_sdf(coords)
            
            # If expected_surface=True, SDF should be 0.0 (+/- tolerance)
            # 1nm = 1e-6 mm
            TOLERANCE_MM = 1e-6 
            
            if abs(sdf) > TOLERANCE_MM:
                failures.append({
                    "point": pt["coord"],
                    "deviation_mm": float(sdf),
                    "allowed_mm": TOLERANCE_MM
                })
                max_error = max(max_error, abs(sdf))
        
        return {
            "verified": len(failures) == 0,
            "max_deviation_nm": max_error * 1e6,
            "failures": failures,
            "verification_engine": "VMK Symbolic Kernel"
        }
