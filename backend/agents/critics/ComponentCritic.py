import numpy as np
from typing import Dict, List, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)

class ComponentCritic:
    """
    Critic for ComponentAgent.
    
    Monitors:
    - Component selection quality (cost, specs, availability)
    - Requirement satisfaction accuracy
    - Catalog coverage
    - Component installation success rate
    - Selection diversity
    
    Thresholds loaded from Supabase critic_thresholds table.
    """
    
    def __init__(self, window_size: int = None, vehicle_type: str = "default"):
        self._vehicle_type = vehicle_type
        self._thresholds_loaded = False
        self._thresholds = {}
        
        # These will be loaded from Supabase if None
        self._window_size = window_size
        
        # Observation history
        self.selection_history = deque(maxlen=window_size or 100)
        self.requirement_history = deque(maxlen=window_size or 100)
        self.installation_history = deque(maxlen=window_size or 100)
        self.component_ids_used = deque(maxlen=window_size or 100)
        
        # Metrics
        self.zero_results_count = 0
        self.over_spec_count = 0  # Selected component far exceeds requirements
        self.installation_failures = 0
        self.user_rejections = 0
        self.total_evaluations = 0
        
    async def _load_thresholds(self):
        """Load thresholds from Supabase if not provided."""
        if self._thresholds_loaded:
            return
            
        try:
            from backend.services import supabase
            self._thresholds = await supabase.get_critic_thresholds("ComponentCritic", self._vehicle_type)
            
            if self._window_size is None:
                self._window_size = self._thresholds.get("window_size", 100)
                
            # Update deque sizes if changed
            if len(self.selection_history) != self._window_size:
                self.selection_history = deque(self.selection_history, maxlen=self._window_size)
                self.requirement_history = deque(self.requirement_history, maxlen=self._window_size)
                self.installation_history = deque(self.installation_history, maxlen=self._window_size)
                self.component_ids_used = deque(self.component_ids_used, maxlen=self._window_size)
                
            self._thresholds_loaded = True
            logger.info(f"ComponentCritic thresholds loaded for {self._vehicle_type}")
        except Exception as e:
            logger.warning(f"Could not load thresholds from Supabase: {e}. Using defaults.")
            if self._window_size is None:
                self._window_size = 100
            self._thresholds = self._default_thresholds()
            self._thresholds_loaded = True
    
    def _default_thresholds(self) -> Dict:
        """Default thresholds if Supabase unavailable."""
        return {
            "zero_result_rate_threshold": 0.3,
            "over_spec_rate_threshold": 0.4,
            "installation_success_threshold": 0.7,
            "user_acceptance_threshold": 0.6,
            "diversity_score_threshold": 0.3,
            "catalog_coverage_critical": 0.5,
            "over_spec_evolve_threshold": 0.5,
            "zero_result_warn_threshold": 0.2,
            "over_spec_warn_threshold": 0.3,
            "installation_warn_threshold": 0.8,
            "user_acceptance_warn_threshold": 0.7,
            "over_spec_factor": 3,
        }
    
    @property
    def window_size(self) -> int:
        return self._window_size
        
    async def observe(self,
                requirements: Dict,
                selection_output: Dict,
                installation_result: Dict = None,
                user_accepted: bool = None):
        """
        Record component agent decision.
        
        Args:
            requirements: {"min_power_w": 100, "max_cost": 50, ...}
            selection_output: Agent's component selection
            installation_result: Result of component installation (if attempted)
            user_accepted: Did user keep this component or replace it?
        """
        try:
            self.total_evaluations += 1
            
            self.requirement_history.append(requirements)
            self.selection_history.append(selection_output)
            
            # Track zero results
            selected = selection_output.get("selection", [])
            if len(selected) == 0:
                self.zero_results_count += 1
            
            # Track component IDs for diversity
            for comp in selected:
                self.component_ids_used.append(comp.get("id", "UNKNOWN"))
            
            # Installation tracking
            if installation_result:
                self.installation_history.append(installation_result)
                if installation_result.get("status") != "installed":
                    self.installation_failures += 1
            
            # User acceptance tracking
            if user_accepted is False:
                self.user_rejections += 1
            
            # Over-specification detection
            over_spec_factor = self._thresholds.get("over_spec_factor", 3)
            for comp in selected:
                specs = comp.get("specs", {})
                for req_key, req_val in requirements.items():
                    if req_key.startswith("min_"):
                        attr = req_key[4:]
                        comp_val = specs.get(attr, comp.get(attr, 0))
                        
                        # If component is over_spec_factor over requirement, flag as over-spec
                        if comp_val > req_val * over_spec_factor:
                            self.over_spec_count += 1
                            break
        except Exception as e:
            logger.error(f"Error in observe: {e}")
    
    async def analyze(self) -> Dict:
        """Analyze component agent performance."""
        await self._load_thresholds()
        
        try:
            if len(self.selection_history) < 10:
                return {
                    "status": "insufficient_data",
                    "observations": len(self.selection_history)
                }
            
            # 1. CATALOG COVERAGE
            zero_result_rate = self.zero_results_count / len(self.selection_history)
            
            # 2. SELECTION QUALITY
            total_selections = sum(len(s.get("selection", [])) for s in self.selection_history)
            avg_selections_per_query = total_selections / len(self.selection_history)
            
            # 3. OVER-SPECIFICATION RATE
            over_spec_rate = self.over_spec_count / max(1, total_selections)
            
            # 4. INSTALLATION SUCCESS
            installation_success_rate = 1.0
            if self.installation_history:
                successful = sum(1 for i in self.installation_history if i.get("status") == "installed")
                installation_success_rate = successful / len(self.installation_history)
            
            # 5. USER ACCEPTANCE
            user_acceptance_rate = 1.0 - (self.user_rejections / self.total_evaluations)
            
            # 6. COMPONENT DIVERSITY
            unique_components = len(set(self.component_ids_used))
            total_used = len(self.component_ids_used)
            diversity_score = unique_components / max(1, total_used)
            
            # 7. COST ANALYSIS
            cost_stats = self._analyze_cost_patterns()
            
            # 8. FAILURE MODE DETECTION
            failure_modes = self._detect_component_failure_modes(
                zero_result_rate, over_spec_rate, installation_success_rate, 
                user_acceptance_rate, diversity_score
            )
            
            # 9. RECOMMENDATIONS
            recommendations = self._generate_component_recommendations(
                zero_result_rate, over_spec_rate, failure_modes, 
                installation_success_rate, user_acceptance_rate
            )
            
            return {
                "timestamp": self.total_evaluations,
                "catalog_coverage": 1.0 - zero_result_rate,
                "zero_results": self.zero_results_count,
                "avg_selections_per_query": avg_selections_per_query,
                "over_spec_rate": over_spec_rate,
                "installation_success_rate": installation_success_rate,
                "installation_failures": self.installation_failures,
                "user_acceptance_rate": user_acceptance_rate,
                "user_rejections": self.user_rejections,
                "component_diversity": diversity_score,
                "unique_components_used": unique_components,
                "cost_stats": cost_stats,
                "failure_modes": failure_modes,
                "recommendations": recommendations,
                "confidence": min(1.0, len(self.selection_history) / self._window_size)
            }
        except Exception as e:
            logger.error(f"Error in analyze: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": self.total_evaluations
            }
    
    def _analyze_cost_patterns(self) -> Dict:
        """Analyze cost trends in selected components."""
        try:
            costs = []
            for selection in self.selection_history:
                for comp in selection.get("selection", []):
                    cost = comp.get("cost_usd", 0)
                    if cost > 0:
                        costs.append(cost)
            
            if not costs:
                return {}
            
            return {
                "mean_cost": float(np.mean(costs)),
                "max_cost": float(np.max(costs)),
                "total_bom_cost": float(np.sum(costs)),
                "samples": len(costs)
            }
        except Exception as e:
            logger.error(f"Error in cost analysis: {e}")
            return {}
    
    def _detect_component_failure_modes(self,
                                       zero_result_rate: float,
                                       over_spec_rate: float,
                                       installation_success_rate: float,
                                       user_acceptance_rate: float,
                                       diversity_score: float) -> List[str]:
        """Identify component-specific failure patterns."""
        failures = []
        
        try:
            # FAILURE 1: Poor catalog coverage
            zr_threshold = self._thresholds.get("zero_result_rate_threshold", 0.3)
            if zero_result_rate > zr_threshold:
                failures.append(f"Catalog gaps: {zero_result_rate:.0%} of queries return zero results")
            
            # FAILURE 2: Over-specification (wasteful)
            os_threshold = self._thresholds.get("over_spec_rate_threshold", 0.4)
            if over_spec_rate > os_threshold:
                failures.append(f"Over-specification: {over_spec_rate:.0%} of components exceed requirements by {self._thresholds.get('over_spec_factor', 3)}x")
            
            # FAILURE 3: Installation failures
            inst_threshold = self._thresholds.get("installation_success_threshold", 0.7)
            if installation_success_rate < inst_threshold:
                failures.append(f"Installation issues: {installation_success_rate:.0%} success rate")
            
            # FAILURE 4: User rejection rate high
            ua_threshold = self._thresholds.get("user_acceptance_threshold", 0.6)
            if user_acceptance_rate < ua_threshold:
                failures.append(f"Low user acceptance: {user_acceptance_rate:.0%} (users replacing selections)")
            
            # FAILURE 5: Low diversity (always selecting same components)
            div_threshold = self._thresholds.get("diversity_score_threshold", 0.3)
            if diversity_score < div_threshold and len(self.component_ids_used) > 20:
                failures.append(f"Low diversity: {diversity_score:.0%} (repeating same components)")
            
            # FAILURE 6: No selections at all
            if self.zero_results_count > len(self.selection_history) * 0.5:
                failures.append("Critical: >50% of queries fail to find components")
        except Exception as e:
            logger.error(f"Error in failure mode detection: {e}")
        
        return failures
    
    def _generate_component_recommendations(self,
                                           zero_result_rate: float,
                                           over_spec_rate: float,
                                           failure_modes: List[str],
                                           installation_success_rate: float,
                                           user_acceptance_rate: float) -> List[str]:
        """Generate actionable recommendations."""
        recs = []
        
        try:
            # Catalog recommendations
            zr_warn = self._thresholds.get("zero_result_warn_threshold", 0.2)
            if zero_result_rate > zr_warn:
                recs.append("📦 EXPAND CATALOG: High zero-result rate - add more components to database")
                recs.append("🔍 LOG MISSING SPECS: Track unfulfilled requirements to prioritize catalog expansion")
            
            # Selection strategy recommendations
            os_warn = self._thresholds.get("over_spec_warn_threshold", 0.3)
            if over_spec_rate > os_warn:
                recs.append("💰 OPTIMIZE SELECTION: Prefer components closer to requirements (reduce over-spec)")
                recs.append("⚖️ COST-AWARE RANKING: Weight selection by cost-to-spec ratio")
            
            # Installation recommendations
            inst_warn = self._thresholds.get("installation_warn_threshold", 0.8)
            if installation_success_rate < inst_warn:
                recs.append("🔧 DEBUG INSTALLATION: High failure rate in mesh-to-SDF conversion")
                recs.append("📊 VALIDATE GEOMETRY: Check mesh quality before installation")
            
            # User preference learning
            ua_warn = self._thresholds.get("user_acceptance_warn_threshold", 0.7)
            if user_acceptance_rate < ua_warn:
                recs.append("🎯 LEARN USER PREFERENCES: Track which components users keep vs replace")
                recs.append("🔄 IMPROVE RANKING: Adjust selection criteria based on user feedback")
            
            # Diversity recommendations
            unique_ratio = len(set(self.component_ids_used)) / max(1, len(self.component_ids_used))
            div_threshold = self._thresholds.get("diversity_score_threshold", 0.3)
            if unique_ratio < div_threshold:
                recs.append("🎨 INCREASE DIVERSITY: Explore more component options (not always selecting same parts)")
            
            # Data collection
            if len(self.installation_history) < 10:
                recs.append("💾 VALIDATE INSTALLATIONS: Need more installation attempts for statistical analysis")
            
            if not recs:
                recs.append("✅ NOMINAL: Component agent performing within parameters")
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recs
    
    async def should_evolve(self) -> Tuple[bool, str, str]:
        """Decide if component agent needs evolution."""
        await self._load_thresholds()
        
        try:
            if len(self.selection_history) < 20:
                return False, "Insufficient data", None
            
            report = await self.analyze()
            
            # CRITICAL: High zero-result rate (catalog incomplete)
            cat_critical = self._thresholds.get("catalog_coverage_critical", 0.5)
            if report.get("catalog_coverage", 1.0) < cat_critical:
                return True, f"Catalog coverage: {report['catalog_coverage']:.0%}", "EXPAND_COMPONENT_CATALOG"
            
            # User acceptance too low
            ua_threshold = self._thresholds.get("user_acceptance_threshold", 0.6)
            if report.get("user_acceptance_rate", 1.0) < ua_threshold:
                return True, f"User acceptance: {report['user_acceptance_rate']:.0%}", "LEARN_USER_PREFERENCES"
            
            # Installation failures
            inst_threshold = self._thresholds.get("installation_success_threshold", 0.7)
            if report.get("installation_success_rate", 1.0) < inst_threshold:
                return True, f"Installation success: {report['installation_success_rate']:.0%}", "IMPROVE_MESH_PIPELINE"
            
            # Over-specification (wasteful)
            os_evolve = self._thresholds.get("over_spec_evolve_threshold", 0.5)
            if report.get("over_spec_rate", 0) > os_evolve:
                return True, f"Over-spec rate: {report['over_spec_rate']:.0%}", "OPTIMIZE_SELECTION_RANKING"
        except Exception as e:
            logger.error(f"Error in should_evolve: {e}")
            return False, f"Error: {e}", None
        
        return False, "Agent within acceptable parameters", None
    
    def export_report(self, filepath: str):
        """Export analysis to JSON."""
        import asyncio
        import json
        try:
            report = asyncio.run(self.analyze())
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
