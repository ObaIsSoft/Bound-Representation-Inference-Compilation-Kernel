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
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
        # Observation history per oracle type
        self.oracle_calls = {
            "physics": deque(maxlen=window_size),
            "chemistry": deque(maxlen=window_size),
            "materials": deque(maxlen=window_size),
            "electronics": deque(maxlen=window_size)
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
        
    def observe(self,
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
            error = self._compute_validation_error(oracle_output, ground_truth)
            if error:
                self.calculation_errors[oracle_type].append({
                    "domain": domain,
                    "error": f"Ground truth mismatch: {error}",
                    "inputs": input_params
                })
        
        # Check conservation laws for physics oracle
        if oracle_type == "physics":
            violation = self._check_conservation_laws(domain, oracle_output)
            if violation:
                self.conservation_violations.append({
                    "domain": domain,
                    "violation": violation,
                    "output": oracle_output
                })
    
    def _compute_validation_error(self, output: Dict, ground_truth: Dict) -> str:
        """Compare oracle output to known ground truth."""
        # Look for numeric results
        for key in ground_truth.keys():
            if key in output:
                try:
                    oracle_val = float(output[key])
                    truth_val = float(ground_truth[key])
                    
                    rel_error = abs(oracle_val - truth_val) / abs(truth_val) if truth_val != 0 else abs(oracle_val - truth_val)
                    
                    if rel_error > 0.1:  # >10% error
                        return f"{key}: {rel_error:.1%} error (oracle={oracle_val:.4f}, truth={truth_val:.4f})"
                except (ValueError, TypeError):
                    continue
        return None
    
    def _check_conservation_laws(self, domain: str, output: Dict) -> str:
        """Validate conservation laws for physics calculations."""
        # Energy conservation
        if domain == "THERMODYNAMICS":
            energy_in = output.get("energy_in", 0)
            energy_out = output.get("energy_out", 0)
            energy_stored = output.get("energy_stored", 0)
            
            balance = energy_in - energy_out - energy_stored
            if abs(balance) > 0.01 * energy_in:  # >1% imbalance
                return f"Energy not conserved: ΔE = {balance:.4f} J ({balance/energy_in*100:.1f}%)"
        
        # Charge conservation (circuits)
        if domain == "CIRCUIT":
            current_in = output.get("current_in", 0)
            current_out = output.get("current_out", 0)
            
            if abs(current_in - current_out) > 1e-6:
                return f"Kirchhoff's Current Law violated: I_in={current_in}, I_out={current_out}"
        
        # Mass conservation (chemistry)
        if domain == "REACTION":
            mass_reactants = output.get("mass_reactants", 0)
            mass_products = output.get("mass_products", 0)
            
            if abs(mass_reactants - mass_products) > 1e-6:
                return f"Mass not conserved: Δm = {mass_reactants - mass_products:.6f} g"
                
        # Nuclear Physics (Mass-Energy Equivalence)
        if domain == "NUCLEAR":
            # E = mc^2
            # Input mass vs Output mass + Energy
            mass_in = output.get("mass_in", 0)
            mass_out = output.get("mass_out", 0)
            energy_out = output.get("energy_out", 0)
            c = 299792458
            
            # Mass Defect
            dm = mass_in - mass_out
            expected_energy = dm * (c**2)
            
            # Allow 1% error margin for simulation approx
            if abs(energy_out - expected_energy) > abs(0.01 * expected_energy):
                 return f"Einstein Violation: Energy output mismatch. Expected {expected_energy:.2e}J, Got {energy_out:.2e}J"
                 
        # Astrophysics (Orbital Mechanics)
        if domain == "ASTROPHYSICS":
            # Vis-viva equation check: v^2 = GM(2/r - 1/a)
            # Or simpler: Total Mechanical Energy Conservation (E = K + U = constant)
            # K = 0.5 * m * v^2
            # U = -G * M * m / r
            
            k1 = output.get("kinetic_energy_initial", 0)
            u1 = output.get("potential_energy_initial", 0)
            k2 = output.get("kinetic_energy_final", 0)
            u2 = output.get("potential_energy_final", 0)
            
            e1 = k1 + u1
            e2 = k2 + u2
            
            # If no non-conservative forces (thrust/drag), E must be constant
            if not output.get("has_thrust", False) and not output.get("has_drag", False):
                if abs(e2 - e1) > abs(0.001 * e1): # 0.1% tolerance
                    return f"Orbital Energy Drift: ΔE = {e2 - e1:.2e} J"

        return None
    
    def analyze(self) -> Dict:
        """Analyze all oracle performance."""
        if self.total_evaluations < 10:
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
        conservation_violation_rate = len(self.conservation_violations) / max(1, len(self.oracle_calls["physics"]))
        
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
    
    def _detect_oracle_failure_modes(self, oracle_stats: Dict, conservation_violation_rate: float) -> List[str]:
        """Identify oracle-specific failure patterns."""
        failures = []
        
        # FAILURE 1: Conservation law violations (CRITICAL)
        if len(self.conservation_violations) > 0:
            failures.append(f"⚠️ CRITICAL: {len(self.conservation_violations)} conservation law violations detected")
            for violation in self.conservation_violations[:3]:  # Show first 3
                failures.append(f"  → {violation['domain']}: {violation['violation']}")
        
        # FAILURE 2: High error rates per oracle
        for oracle_type, stats in oracle_stats.items():
            if stats["error_rate"] > 0.1:
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
        
        return failures
    
    def _generate_oracle_recommendations(self, oracle_stats: Dict, failure_modes: List[str]) -> List[str]:
        """Generate actionable recommendations for oracles."""
        recs = []
        
        # Conservation violation recommendations
        if len(self.conservation_violations) > 0:
            recs.append("🚨 FIX CONSERVATION LAWS: Fundamental physics violations detected")
            recs.append("🔍 AUDIT CALCULATIONS: Review numerical methods and floating-point precision")
            recs.append("📚 VALIDATE AGAINST LITERATURE: Compare results to known physics problems")
        
        # Error rate recommendations
        for oracle_type, stats in oracle_stats.items():
            if stats["error_rate"] > 0.05:
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
        
        return recs
    
    def should_evolve(self) -> Tuple[bool, str, str]:
        """Decide if any oracle needs correction."""
        if self.total_evaluations < 20:
            return False, "Insufficient data", None
        
        report = self.analyze()
        
        # CRITICAL: Conservation law violations = fundamental bug
        if len(self.conservation_violations) > 0:
            return True, f"CRITICAL: {len(self.conservation_violations)} conservation violations", "FIX_PHYSICS_ENGINE"
        
        # High error rate in any oracle
        for oracle_type, stats in report["oracle_stats"].items():
            if stats["error_rate"] > 0.15:
                return True, f"{oracle_type}Oracle error rate: {stats['error_rate']:.0%}", "DEBUG_ORACLE_IMPLEMENTATION"
        
        return False, "Oracles within acceptable parameters", None
    
    def export_report(self, filepath: str):
        """Export analysis to JSON."""
        import json
        report = self.analyze()
        
        # Add detailed error logs
        report["detailed_errors"] = self.calculation_errors
        report["conservation_violations_detail"] = self.conservation_violations
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
