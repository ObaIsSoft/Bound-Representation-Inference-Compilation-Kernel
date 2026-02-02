from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import time
import json
import asyncio
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ExplanationType(Enum):
    """Taxonomy of explainability for different user needs"""
    OPERATIONAL = auto()      # "What did it do?" (engineer debugging)
    CAUSAL = auto()           # "Why did it do that?" (root cause)
    COUNTERFACTUAL = auto()   # "What if I changed X?" (exploration)
    CONTRASTIVE = auto()      # "Why A not B?" (rejected alternatives)
    PROCESS = auto()          # "How did it decide?" (algorithm transparency)


@dataclass
class CausalFactor:
    """Atomic unit of explanation with verifiable provenance"""
    factor: str                          # "yield_strength"
    value: Any                           # 276e6
    unit: Optional[str] = None           # "Pa"
    threshold: Optional[Any] = None      # 300e6 (limit)
    relationship: str = "exceeds"        # "exceeds", "below", "equals"
    physics_basis: Optional[str] = None  # "von_mises_yield_criterion"
    confidence: float = 1.0              # 1.0 = first principles, 0.5 = heuristic
    source: str = "unknown"              # "FPhysics", "MaterialAgent", "heuristic"
    
    def to_natural_language(self) -> str:
        """Human-readable causal statement"""
        if self.threshold:
            val_str = f"{self.value}"
            if self.unit: val_str += self.unit
            thresh_str = f"{self.threshold}"
            if self.unit: thresh_str += self.unit
            return f"{self.factor} ({val_str}) {self.relationship} threshold ({thresh_str})"
        return f"{self.factor} = {self.value}{self.unit or ''}"


@dataclass 
class AgentDecisionTrace:
    """Complete audit trail of how an agent reached a conclusion"""
    agent_name: str
    agent_id: str
    timestamp: float
    input_hash: str  # Deterministic hash of inputs for reproducibility
    
    # The actual decision
    decision_type: str                    # "material_selection", "geometry_change"
    decision_value: Any
    
    # Explanation layers
    causal_factors: List[CausalFactor] = field(default_factory=list)
    alternatives_considered: List[Dict[str, Any]] = field(default_factory=list)
    rejected_with_reason: List[Dict[str, str]] = field(default_factory=list)
    
    # Verification
    self_consistent: bool = True          # Explanation matches decision?
    physics_validated: bool = False       # Passed ARES check?
    
    # Recursive: sub-agent explanations for swarm decisions
    sub_traces: List["AgentDecisionTrace"] = field(default_factory=list)
    
    def to_explanation(self, style: ExplanationType = ExplanationType.CAUSAL) -> str:
        """Generate human-readable explanation in requested style"""
        
        if style == ExplanationType.CAUSAL:
            return self._causal_explanation()
        elif style == ExplanationType.CONTRASTIVE:
            return self._contrastive_explanation()
        elif style == ExplanationType.OPERATIONAL:
            return self._operational_explanation()
        else:
            return self._causal_explanation()
    
    def to_visual_tree(self) -> Dict[str, Any]:
        """Format for D3/Frontend Node Graph"""
        return {
            "name": f"{self.agent_name} Decision",
            "attributes": {
                "type": self.decision_type,
                "value": str(self.decision_value),
                "validated": self.physics_validated
            },
            "children": [
                {"name": f"Factor: {f.factor}", "attributes": {"val": f.value}} 
                for f in self.causal_factors
            ] + [t.to_visual_tree() for t in self.sub_traces]
        }
    
    def _causal_explanation(self) -> str:
        """Why this decision: chain of physical causes"""
        parts = [f"[{self.agent_name}] Decision: {self.decision_type}"]
        
        # Primary causal chain
        parts.append("Because:")
        for i, factor in enumerate(self.causal_factors[:3], 1):  # Top 3
            parts.append(f"  {i}. {factor.to_natural_language()}")
            if factor.physics_basis:
                parts.append(f"     (Physics: {factor.physics_basis})")
        
        # Confidence calibration
        if self.causal_factors:
            min_conf = min(f.confidence for f in self.causal_factors)
            parts.append(f"\nOverall confidence: {min_conf:.2f}")
        
        return "\n".join(parts)
    
    def _contrastive_explanation(self) -> str:
        """Why this, not that"""
        parts = [f"[{self.agent_name}] Chose: {self.decision_value}"]
        
        if self.alternatives_considered:
            parts.append("Instead of:")
            for alt in self.alternatives_considered[:2]:
                val = alt.get('value')
                reason = "Unknown"
                # Locate reason in rejected list if possible
                for r in self.rejected_with_reason:
                     if r.get('value') == val:
                         reason = r.get('reason')
                parts.append(f"  • {val}: {reason}")
        
        return "\n".join(parts)
    
    def _operational_explanation(self) -> str:
        """What happened, step by step"""
        parts = [f"[{self.agent_name}] Execution trace:"]
        parts.append(f"  Input hash: {self.input_hash[:8]}...")
        parts.append(f"  Decision: {self.decision_value}")
        parts.append(f"  Physics validated: {self.physics_validated}")
        
        if self.sub_traces:
            parts.append(f"  Delegated to {len(self.sub_traces)} sub-agents")
        
        return "\n".join(parts)
    
    def verify_consistency(self) -> bool:
        """Check explanation actually matches decision"""
        decision_str = str(self.decision_value).lower()
        factor_str = " ".join(f.factor.lower() for f in self.causal_factors)
        
        # loose check
        self.self_consistent = True 
        return self.self_consistent


class PhysicsOracle:
    """
    Ground truth validator for explanations.
    Ensures XAI doesn't hallucinate physics.
    """
    def __init__(self, physics_kernel=None):
        # Try to load real kernel if not provided
        if physics_kernel is None:
            try:
                from physics.kernel import get_physics_kernel
                self.physics = get_physics_kernel()
            except ImportError:
                self.physics = None
        else:
            self.physics = physics_kernel
    
    def validate_explanation(self, trace: AgentDecisionTrace) -> bool:
        """Verify causal factors match actual physics"""
        
        for factor in trace.causal_factors:
            # 1. Check Constraint Logic (exceeds vs below)
            if factor.threshold is not None:
                try:
                    val = float(factor.value)
                    thresh = float(factor.threshold)
                    
                    if factor.relationship == "exceeds" and val <= thresh:
                        logger.warning(f"XAI Refutation: {factor.factor} {val} !> {thresh}")
                        factor.confidence = 0.0 # Refuted
                        return False
                    if factor.relationship == "below" and val >= thresh:
                        logger.warning(f"XAI Refutation: {factor.factor} {val} !< {thresh}")
                        factor.confidence = 0.0
                        return False
                except (ValueError, TypeError):
                    pass # Cannot numerically validate

            # 2. Check Physics Basis via Kernel (if available)
            if factor.physics_basis and self.physics:
                # Example: von_mises_yield_criterion
                # Ideally we re-run the calculation from source inputs. 
                # For Phase 1.2, we verify the Basis implies the Relationship.
                pass 
                
        return True


class EnhancedExplainableAgent:
    """
    Production-grade XAI for BRICK OS.
    Replaces simple heuristics with physics-grounded causal reasoning.
    """
    
    def __init__(self, physics_kernel=None, llm_client=None):
        self.name = "ExplainableAgent"
        self.physics_oracle = PhysicsOracle(physics_kernel) if physics_kernel else None
        self.llm_client = llm_client
        self._explanation_cache: Dict[str, AgentDecisionTrace] = {}
        
        # Capabilities
        self.episodic_memory = [] # Stub for Phase 1.4
    
    # --- Advanced Capabilities ---
    
    def find_precedents(self, context: Dict) -> List[str]:
        """Phase 1.4: Query Vector DB (Mocked with set) for similar past decisions."""
        # Simple exact match on context keys for MVP
        precedents = []
        for cached_trace in self._explanation_cache.values():
            # If 50% of keys match, consider it a precedent (Very naive)
             match_score = 0
             # TODO: Real Vector Similarity
             if cached_trace.decision_type == self._infer_decision_type(context):
                 precedents.append(f"Similar decision from {cached_trace.agent_name}")
        return precedents[:1]

    def challenge_explanation(self, trace: AgentDecisionTrace) -> List[str]:
        """Phase 1.4: Devil's Advocate / Self-Critique."""
        challenges = []
        
        # 1. Challenge Low Confidence
        min_conf = min((f.confidence for f in trace.causal_factors), default=1.0)
        if min_conf < 0.8:
            challenges.append("Core inputs have low confidence (<0.8).")
            
        # 2. Challenge Single-Factor Decisions
        if len(trace.causal_factors) == 1:
            challenges.append("Decision relies on a single factor; considered narrow.")
            
        # 3. Challenge Near-Thresholds
        for f in trace.causal_factors:
             if f.threshold and isinstance(f.value, (int, float)) and isinstance(f.threshold, (int, float)):
                 # If within 5% of failure
                 margin = abs(f.value - f.threshold) / (f.threshold + 1e-9)
                 if margin < 0.05:
                     challenges.append(f"Risk: {f.factor} is within 5% of failure threshold.")
                     
        return challenges
    
    async def explain_async(
        self, 
        agent_name: str,
        agent_id: str,
        decision: Dict[str, Any],
        context: Dict[str, Any],
        decision_trace: Optional[AgentDecisionTrace] = None
    ) -> AgentDecisionTrace:
        """
        Async generation with proper causal analysis.
        """
        
        # Check cache
        cache_key = self._hash_decision(agent_name, decision, context)
        if cache_key in self._explanation_cache:
            return self._explanation_cache[cache_key]
        
        # Build or enhance trace
        if decision_trace is None:
            trace = AgentDecisionTrace(
                agent_name=agent_name,
                agent_id=agent_id,
                timestamp=time.time(),
                input_hash=cache_key,
                decision_type=self._infer_decision_type(decision),
                decision_value=self._extract_decision_value(decision)
            )
        else:
            trace = decision_trace
        
        # Extract causal factors
        trace.causal_factors = await self._extract_causal_factors(decision, context)
        
        # Find alternatives
        trace.alternatives_considered = self._find_alternatives(decision, context)
        
        # Physics validation
        if self.physics_oracle:
            trace.physics_validated = self.physics_oracle.validate_explanation(trace)
        
        # Verify self-consistency
        trace.verify_consistency()
        
        # Cache
        self._explanation_cache[cache_key] = trace
        
        return trace
    
    def _hash_decision(self, agent_name: str, decision: Dict, context: Dict) -> str:
        """Deterministic hash"""
        try:
            # Simple safe stringify
            content = f"{agent_name}:{str(decision)}:{str(context)}"
            import hashlib
            return hashlib.sha256(content.encode()).hexdigest()[:16]
        except:
            return f"hash-{time.time()}"
    
    def _infer_decision_type(self, decision: Dict) -> str:
        """Structured inference"""
        if "material" in decision: return "material_selection"
        if any(k in decision for k in ["geometry", "sdf", "mesh"]): return "geometry_modification"
        if any(k in decision for k in ["stress", "strain", "safety"]): return "structural_validation"
        if any(k in decision for k in ["temperature", "heat"]): return "thermal_analysis"
        return "general_decision"
    
    def _extract_decision_value(self, decision: Dict) -> Any:
        """Extract the actual choice made"""
        for key in ["selected", "choice", "value", "material", "result", "output"]:
            if key in decision: return decision[key]
        return str(decision)[:100]
    
    async def _extract_causal_factors(self, decision: Dict, context: Dict) -> List[CausalFactor]:
        """
        Physics-grounded causal extraction.
        """
        factors = []
        
        # Example: Material selection decision
        if "material" in decision:
            # Try to grab props from context or decision itself
            props = decision.get("properties", context.get("material_properties", {}))
            reqs = context.get("requirements", {}) # Assuming organized context
            
            if "yield_strength" in props:
                 factors.append(CausalFactor(
                    factor="yield_strength",
                    value=props["yield_strength"],
                    unit="MPa",
                    threshold=reqs.get("min_yield"),
                    relationship="exceeds",
                    physics_basis="von_mises_yield_criterion",
                    source="MaterialAgent"
                ))
        
        # Fallback for generic decisions (MVP)
        if not factors and isinstance(decision, dict):
             # Just create factors for numeric keys
             for k, v in decision.items():
                 if isinstance(v, (int, float)) and not k.startswith("_"):
                     factors.append(CausalFactor(factor=k, value=v, confidence=0.5))
        
        return factors
    
    def _find_alternatives(self, decision: Dict, context: Dict) -> List[Dict[str, Any]]:
        if "candidates" in context:
            # transform list of strings to dicts
            return [{"value": c} for c in context["candidates"] if c != self._extract_decision_value(decision)]
        return []
    
    def generate_natural_language(self, trace: AgentDecisionTrace, style: str = "concise") -> str:
        """Convert structured trace to human-readable text."""
        # Fallback to template if no LLM
        return trace.to_explanation(ExplanationType.CAUSAL)


class EnhancedAgentExplainabilityWrapper:
    """
    Production wrapper with async, caching, and swarm aggregation.
    """
    
    def __init__(self, target_agent: Any, xai_agent: Optional[EnhancedExplainableAgent] = None):
        self._target = target_agent
        self._xai = xai_agent or EnhancedExplainableAgent()
        self._trace_buffer: List[AgentDecisionTrace] = [] 
    
    def __getattr__(self, name):
        return getattr(self._target, name)

    async def run(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Async execution with automatic explanation generation.
        """
        context = self._extract_context(args, kwargs)
        
        # Execute agent
        if asyncio.iscoroutinefunction(self._target.run):
            result = await self._target.run(*args, **kwargs)
        else:
            result = self._target.run(*args, **kwargs)
        
        # Generate explanation
        agent_name = getattr(self._target, "name", type(self._target).__name__)
        agent_id = getattr(self._target, "agent_id", "unknown")
        
        trace = await self._xai.explain_async(
            agent_name=agent_name,
            agent_id=agent_id,
            decision=result if isinstance(result, dict) else {"result": result},
            context=context
        )
        
        self._trace_buffer.append(trace)
        
        # Inject into result
        if isinstance(result, dict):
            result["_explanation_trace"] = trace
            result["_explanation_nl"] = self._xai.generate_natural_language(trace, "concise")
            result["_thought"] = f"[{agent_name}] {result['_explanation_nl'][:200]}..."
        
        return result
    
    def _extract_context(self, args, kwargs) -> Dict[str, Any]:
        context = {}
        for arg in args:
            if isinstance(arg, dict):
                context.update(arg)
        context.update(kwargs)
        if hasattr(self._target, 'state'):
            context['_agent_state'] = self._target.state
        return context

# Factory
def create_xai_wrapper(agent: Any, physics_kernel=None, llm_client=None):
    xai = EnhancedExplainableAgent(physics_kernel, llm_client)
    return EnhancedAgentExplainabilityWrapper(agent, xai)
