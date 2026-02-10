import numpy as np
from typing import Dict, List, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)

class OracleCritic:
    """
    Universal Critic for ALL Oracle systems.
    
    Monitors:
    - PhysicsOracle (circuit, thermodynamics, mechanics, electromagnetism)
    - ChemistryOracle (thermochemistry, kinetics, electrochemistry)
    - MaterialsOracle (crystal structure, phase diagrams, mechanical properties)
    - ElectronicsOracle (signal integrity, EMI, power electronics)
    
    The oracles are the "truth engines" - if they drift, everything fails.
    
    Thresholds loaded from Supabase critic_thresholds table.
    """
    
    def __init__(self, window_size: int = None, vehicle_type: str = "default"):
        self._vehicle_type = vehicle_type
        self._thresholds_loaded = False
        self._thresholds = {}
        
        # These will be loaded from Supabase if None
        self._window_size = window_size
        
        # Observation history per oracle type
        self.oracle_calls = {
            "physics": deque(maxlen=window_size or 100),
            "chemistry": deque(maxlen=window_size or 100),
            "materials": deque(maxlen=window_size or 100),
            "electronics": deque(maxlen=window_size or 100)
        }
        
        # Error tracking
        self.calculation_errors = {
            "physics": [],
            "chemistry": [],
            "materials": [],
            "electronics": []
        }
        
        # Conservation law violations (CRITICAL for physics)
        self.conservation_violations = []
        
        # Metrics
        self.total_evaluations = 0
        self.oracle_failures = 0
        
    async def _load_thresholds(self):
        """Load thresholds from Supabase if not provided."""
        if self._thresholds_loaded:
            return
            
        try:
            from backend.services import supabase
            self._thresholds = await supabase.get_critic_thresholds("OracleCritic", self._vehicle_type)
            
            if self._window_size is None:
                self._window_size = self._thresholds.get("window_size", 100)
                
            # Update deque sizes if changed
            for key in self.oracle_calls:
                if len(self.oracle_calls[key]) != self._window_size:
                    self.oracle_calls[key] = deque(self.oracle_calls[key], maxlen=self._window_size)
                
            self._thresholds_loaded = True
            logger.info(f"OracleCritic thresholds loaded for {self._vehicle_type}")
        except Exception as e:
            logger.warning(f"Could not load thresholds from Supabase: {e}. Using defaults.")
            if self._window_size is None:
                self._window_size = 100
            self._thresholds = self._default_thresholds()
            self._thresholds_loaded = True
    
    def _default_thresholds(self) -> Dict:
        """Default thresholds if Supabase unavailable."""
        return {
            "ground_truth_error_threshold": 0.1,
            "energy_conservation_tolerance": 0.01,
            "current_tolerance": 1e-6,
            "mass_tolerance": 1e-6,
            "nuclear_tolerance": 0.01,
            "orbital_tolerance": 0.001,
            "error_rate_threshold": 0.1,
            "error_rate_evolve": 0.15,
            "min_calls_for_analysis": 10,
            "min_calls_for_evolve": 20,
        }
        
    @property
    def window_size(self) -> int:
        return self._window_size
        
    async def observe(self,
                oracle_type: str,  # "physics", "chemistry", "materials", "electronics"
                domain: str,  # "CIRCUIT", "THERMOCHEMISTRY", etc.
                input_params: Dict,
                oracle_output: Dict,
                ground_truth: Dict = None,
                validation_result: Dict = None):
        """
        Record an oracle calculation.
        
        Args:
            oracle_type: Which oracle was called
            domain: Calculation domain
            input_params: Input parameters
            oracle_output: Oracle's calculation result
            ground_truth: Known correct answer (from literature/experiments)
            validation_result: Independent validation (e.g., FEA vs oracle)
        """
        try:
            self.total_evaluations += 1
            
            if oracle_type not in self.oracle_calls:
                logger.warning(f"Unknown oracle type: {oracle_type}")
                return
            
            observation = {
                "domain": domain,
                "inputs": input_params,
                "output": oracle_output,
                "timestamp": self.total_evaluations
            }
            
            self.oracle_calls[oracle_type].append(observation)
            
            # Check for errors in oracle output
            if oracle_output.get("status") == "error":
                self.oracle_failures += 1
                error_msg = oracle_output.get("error", "Unknown error")
                self.calculation_errors[oracle_type].append({
                    "domain": domain,
                    "error": error_msg,
                    "inputs": input_params
                })
            
            # Validate against ground truth
            if ground_truth:
                error = await self._compute_validation_error(oracle_output, ground_truth)
                if error:
                    self.calculation_errors[oracle_type].append({
                        "domain": domain,
                        "error": f"Ground truth mismatch: {error}",
                        "inputs": input_params
                    })
            
            # Check conservation laws for physics oracle
            if oracle_type == "physics":
                violation = await self._check_conservation_laws(domain, oracle_output)
                if violation:
                    self.conservation_violations.append({
                        "domain": domain,
                        "violation": violation,
                        "output": oracle_output
                    })
        except Exception as e:
            logger.error(f"Error in observe: {e}")
    
    async def _compute_validation_error(self, output: Dict, ground_truth: Dict) -> str:
        """Compare oracle output to known ground truth."""
        try:
            gt_threshold = self._thresholds.get("ground_truth_error_threshold", 0.1)
            
            # Look for numeric results
            for key in ground_truth.keys():
                if key in output:
                    try:
                        oracle_val = float(output[key])
                        truth_val = float(ground_truth[key])
                        
                        rel_error = abs(oracle_val - truth_val) / abs(truth_val) if truth_val != 0 else abs(oracle_val - truth_val)
                        
                        if rel_error > gt_threshold:  # >10% error
                            return f"{key}: {rel_error:.1%} error (oracle={oracle_val:.4f}, truth={truth_val:.4f})"
                    except (ValueError, TypeError):
                        continue
        except Exception as e:
            logger.error(f"Error computing validation error: {e}")
        return None
    
    async def _check_conservation_laws(self, domain: str, output: Dict) -> str:
        """Validate conservation laws for physics calculations."""
        try:
            # Energy conservation
            if domain == "THERMODYNAMICS":
                energy_in = output.get("energy_in", 0)
                energy_out = output.get("energy_out", 0)
                energy_stored = output.get("energy_stored", 0)
                
                balance = energy_in - energy_out - energy_stored
                tolerance = self._thresholds.get("energy_conservation_tolerance", 0.01)
                if abs(balance) > tolerance * energy_in:  # >1% imbalance
                    return f"Energy not conserved: ΔE = {balance:.4f} J ({balance/energy_in*100:.1f}%)"
            
            # Charge conservation (circuits)
            if domain == "CIRCUIT":
                current_in = output.get("current_in", 0)
                current_out = output.get("current_out", 0)
                
                current_tol = self._thresholds.get("current_tolerance", 1e-6)
                if abs(current_in - current_out) > current_tol:
                    return f"Kirchhoff's Current Law violated: I_in={current_in}, I_out={current_out}"
            
            # Mass conservation (chemistry)
            if domain == "REACTION":
                mass_reactants = output.get("mass_reactants", 0)
                mass_products = output.get("mass_products", 0)
                
                mass_tol = self._thresholds.get("mass_tolerance", 1e-6)
                if abs(mass_reactants - mass_products) > mass_tol:
                    return f"Mass not conserved: Δm = {mass_reactants - mass_products:.6f} g"
                    
            # Nuclear Physics (Mass-Energy Equivalence)
            if domain == "NUCLEAR":
                mass_in = output.get("mass_in", 0)
                mass_out = output.get("mass_out", 0)
                energy_out = output.get("energy_out", 0)
                c = 299792458
                
                # Mass Defect
                dm = mass_in - mass_out
                expected_energy = dm * (c**2)
                
                # Allow 1% error margin for simulation approx
                nuc_tol = self._thresholds.get("nuclear_tolerance", 0.01)
                if abs(energy_out - expected_energy) > abs(nuc_tol * expected_energy):
                     return f"Einstein Violation: Energy output mismatch. Expected {expected_energy:.2e}J, Got {energy_out:.2e}J"
                     
            # Astrophysics (Orbital Mechanics)
            if domain == "ASTROPHYSICS":
                k1 = output.get("kinetic_energy_initial", 0)
                u1 = output.get("potential_energy_initial", 0)
                k2 = output.get("kinetic_energy_final", 0)
                u2 = output.get("potential_energy_final", 0)
                
                e1 = k1 + u1
                e2 = k2 + u2
                
                orb_tol = self._thresholds.get("orbital_tolerance", 0.001)
                # If no non-conservative forces (thrust/drag), E must be constant
                if not output.get("has_thrust", False) and not output.get("has_drag", False):
                    if abs(e2 - e1) > abs(orb_tol * e1): # 0.1% tolerance
                        return f"Orbital Energy Drift: ΔE = {e2 - e1:.2e} J"
        except Exception as e:
            logger.error(f"Error checking conservation laws: {e}")

        return None
    
    async def analyze(self) -> Dict:
        """Analyze all oracle performance."""
        await self._load_thresholds()
        
        try:
            min_calls = self._thresholds.get("min_calls_for_analysis", 10)
            if self.total_evaluations < min_calls:
                return {
                    "status": "insufficient_data",
                    "observations": self.total_evaluations
                }
            
            # Per-oracle statistics
            oracle_stats = {}
            for oracle_type, calls in self.oracle_calls.items():
                if not calls:
                    continue
                
                total_calls = len(calls)
                errors = len(self.calculation_errors[oracle_type])
                error_rate = errors / total_calls if total_calls > 0 else 0
                
                # Domain distribution
                domains = {}
                for call in calls:
                    domain = call["domain"]
                    domains[domain] = domains.get(domain, 0) + 1
                
                oracle_stats[oracle_type] = {
                    "total_calls": total_calls,
                    "errors": errors,
                    "error_rate": error_rate,
                    "domains": domains
                }
            
            # Conservation violations
            physics_calls = len(self.oracle_calls.get("physics", []))
            conservation_violation_rate = len(self.conservation_violations) / max(1, physics_calls)
            
            # Failure mode detection
            failure_modes = self._detect_oracle_failure_modes(oracle_stats, conservation_violation_rate)
            
            # Recommendations
            recommendations = self._generate_oracle_recommendations(oracle_stats, failure_modes)
            
            return {
                "timestamp": self.total_evaluations,
                "oracle_stats": oracle_stats,
                "conservation_violations": len(self.conservation_violations),
                "conservation_violation_rate": conservation_violation_rate,
                "total_oracle_failures": self.oracle_failures,
                "failure_modes": failure_modes,
                "recommendations": recommendations,
                "confidence": min(1.0, self.total_evaluations / 100)
            }
        except Exception as e:
            logger.error(f"Error in analyze: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": self.total_evaluations
            }
    
    def _detect_oracle_failure_modes(self, oracle_stats: Dict, conservation_violation_rate: float) -> List[str]:
        """Identify oracle-specific failure patterns."""
        failures = []
        
        try:
            err_threshold = self._thresholds.get("error_rate_threshold", 0.1)
            
            # FAILURE 1: Conservation law violations (CRITICAL)
            if len(self.conservation_violations) > 0:
                failures.append(f"⚠️ CRITICAL: {len(self.conservation_violations)} conservation law violations detected")
                for violation in self.conservation_violations[:3]:  # Show first 3
                    failures.append(f"  → {violation['domain']}: {violation['violation']}")
            
            # FAILURE 2: High error rates per oracle
            for oracle_type, stats in oracle_stats.items():
                if stats["error_rate"] > err_threshold:
                    failures.append(f"{oracle_type.capitalize()}Oracle: {stats['error_rate']:.0%} error rate (>{stats['errors']} failures)")
            
            # FAILURE 3: Unused oracles (no calls)
            for oracle_type in self.oracle_calls.keys():
                if oracle_type not in oracle_stats or oracle_stats[oracle_type]["total_calls"] == 0:
                    failures.append(f"{oracle_type.capitalize()}Oracle: Not being used (0 calls)")
            
            # FAILURE 4: Domain coverage gaps
            for oracle_type, stats in oracle_stats.items():
                domains = stats.get("domains", {})
                if len(domains) == 1 and stats["total_calls"] > 10:
                    single_domain = list(domains.keys())[0]
                    failures.append(f"{oracle_type.capitalize()}Oracle: Only used for {single_domain} (limited coverage)")
        except Exception as e:
            logger.error(f"Error in failure mode detection: {e}")
        
        return failures
    
    def _generate_oracle_recommendations(self, oracle_stats: Dict, failure_modes: List[str]) -> List[str]:
        """Generate actionable recommendations for oracles."""
        recs = []
        
        try:
            # Conservation violation recommendations
            if len(self.conservation_violations) > 0:
                recs.append("🚨 FIX CONSERVATION LAWS: Fundamental physics violations detected")
                recs.append("🔍 AUDIT CALCULATIONS: Review numerical methods and floating-point precision")
                recs.append("📚 VALIDATE AGAINST LITERATURE: Compare results to known physics problems")
            
            # Error rate recommendations
            err_warn = self._thresholds.get("error_rate_threshold", 0.1)
            for oracle_type, stats in oracle_stats.items():
                if stats["error_rate"] > err_warn / 2:  # 0.05
                    recs.append(f"🔧 {oracle_type.upper()}: Error rate {stats['error_rate']:.0%} - review implementation")
            
            # Unused oracle recommendations
            for oracle_type in self.oracle_calls.keys():
                if oracle_type not in oracle_stats or oracle_stats.get(oracle_type, {}).get("total_calls", 0) == 0:
                    recs.append(f"💡 {oracle_type.upper()}: Not being utilized - integrate into agent workflows")
            
            # Coverage recommendations
            for oracle_type, stats in oracle_stats.items():
                domains = stats.get("domains", {})
                if len(domains) < 3 and stats["total_calls"] > 20:
                    recs.append(f"📊 {oracle_type.upper()}: Expand domain coverage (currently only {len(domains)} domains)")
            
            # Ground truth collection
            total_errors_with_validation = sum(
                1 for oracle_type in self.calculation_errors 
                for err in self.calculation_errors[oracle_type] 
                if "Ground truth" in err.get("error", "")
            )
            if total_errors_with_validation < 10:
                recs.append("💾 COLLECT GROUND TRUTH: Need more validated test cases (<10 samples)")
            
            if not recs:
                recs.append("✅ NOMINAL: All oracles performing within parameters")
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recs
    
    async def should_evolve(self) -> Tuple[bool, str, str]:
        """Decide if any oracle needs correction."""
        await self._load_thresholds()
        
        try:
            min_calls = self._thresholds.get("min_calls_for_evolve", 20)
            if self.total_evaluations < min_calls:
                return False, "Insufficient data", None
            
            report = await self.analyze()
            
            # CRITICAL: Conservation law violations = fundamental bug
            if len(self.conservation_violations) > 0:
                return True, f"CRITICAL: {len(self.conservation_violations)} conservation violations", "FIX_PHYSICS_ENGINE"
            
            # High error rate in any oracle
            err_evolve = self._thresholds.get("error_rate_evolve", 0.15)
            for oracle_type, stats in report.get("oracle_stats", {}).items():
                if stats["error_rate"] > err_evolve:
                    return True, f"{oracle_type}Oracle error rate: {stats['error_rate']:.0%}", "DEBUG_ORACLE_IMPLEMENTATION"
        except Exception as e:
            logger.error(f"Error in should_evolve: {e}")
            return False, f"Error: {e}", None
        
        return False, "Oracles within acceptable parameters", None
    
    def export_report(self, filepath: str):
        """Export analysis to JSON."""
        import asyncio
        import json
        try:
            report = asyncio.run(self.analyze())
            
            # Add detailed error logs
            report["detailed_errors"] = self.calculation_errors
            report["conservation_violations_detail"] = self.conservation_violations
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
