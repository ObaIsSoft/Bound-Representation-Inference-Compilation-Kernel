"""
Input Classifier

Routes user inputs to appropriate execution strategies.
Classifies intent and determines whether to use RLM, which nodes to invoke,
and how to structure the execution.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum
import json

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Types of user intents"""
    NEW_DESIGN = "new_design"              # Starting fresh design
    CONSTRAINT_CHANGE = "constraint_change"  # Modifying existing constraints
    MATERIAL_VARIANT = "material_variant"    # Asking about different materials
    COMPARATIVE = "comparative"              # Compare multiple options
    FEATURE_ADD = "feature_add"              # Add features to existing design
    EXPLANATION = "explanation"              # "Why", "explain", "how come"
    NARROWING = "narrowing"                  # Tightening constraints
    CLARIFICATION = "clarification"          # Asking for clarification
    GREETING = "greeting"                    # Hello, thanks, etc.
    UNKNOWN = "unknown"


@dataclass
class ExecutionStrategy:
    """Defines how to execute for a given intent"""
    
    # Whether to use recursive decomposition
    use_rlm: bool = True
    
    # Which nodes to invoke (empty = auto-decompose)
    nodes: List[str] = None
    
    # Parallel execution groups (within group = parallel, between = sequential)
    parallel_groups: List[List[str]] = None
    
    # For refinements: only re-run affected nodes
    affected_by: str = None
    
    # For memory-only queries
    use_memory_only: bool = False
    
    # Delta mode optimization
    use_delta: bool = False
    
    def __post_init__(self):
        if self.nodes is None:
            self.nodes = []
        if self.parallel_groups is None:
            self.parallel_groups = []


class InputClassifier:
    """
    Classifies user input and selects execution strategy.
    
    Uses both rule-based and LLM-based classification for speed and accuracy.
    """
    
    # Rule-based patterns for fast classification
    PATTERNS = {
        IntentType.GREETING: [
            "hello", "hi", "hey", "good morning", "good afternoon",
            "thanks", "thank you", "bye", "goodbye"
        ],
        IntentType.EXPLANATION: [
            "why", "explain", "how come", "what do you mean",
            "can you explain", "tell me why"
        ],
        IntentType.COMPARATIVE: [
            "compare", "versus", "vs", "difference between",
            "which is better", "pros and cons"
        ],
        IntentType.CONSTRAINT_CHANGE: [
            "make it", "change", "instead", "different",
            "lighter", "heavier", "smaller", "larger",
            "cheaper", "faster", "stronger"
        ],
        IntentType.MATERIAL_VARIANT: [
            "use ", "switch to", "what about", "instead of",
            "material", "aluminum", "titanium", "steel", "carbon"
        ],
        IntentType.FEATURE_ADD: [
            "add ", "include", "put in", "mounting", "hole",
            "feature", "attachment", "connector"
        ],
        IntentType.NARROWING: [
            "under ", "less than", "maximum", "at most",
            "within", "budget", "deadline", "by "
        ],
    }
    
    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider
        
        # Build reverse lookup for patterns
        self._pattern_lookup = {}
        for intent, patterns in self.PATTERNS.items():
            for pattern in patterns:
                self._pattern_lookup[pattern.lower()] = intent
    
    async def classify(
        self,
        user_input: str,
        session_context: Dict[str, Any],
        conversation_history: Optional[List[Dict]] = None
    ) -> tuple[IntentType, ExecutionStrategy]:
        """
        Classify user input and return execution strategy.
        
        Returns:
            Tuple of (intent, strategy)
        """
        input_lower = user_input.lower()
        
        # Step 1: Fast rule-based classification
        rule_intent = self._rule_classify(input_lower)
        if rule_intent:
            logger.info(f"Rule-based classification: {rule_intent.value}")
            strategy = self._get_strategy(rule_intent, session_context)
            return rule_intent, strategy
        
        # Step 2: Context-aware heuristics
        heuristic_intent = self._heuristic_classify(input_lower, session_context)
        if heuristic_intent:
            logger.info(f"Heuristic classification: {heuristic_intent.value}")
            strategy = self._get_strategy(heuristic_intent, session_context)
            return heuristic_intent, strategy
        
        # Step 3: LLM-based classification (for complex cases)
        if self.llm_provider:
            llm_intent = await self._llm_classify(
                user_input, session_context, conversation_history
            )
            logger.info(f"LLM classification: {llm_intent.value}")
            strategy = self._get_strategy(llm_intent, session_context)
            return llm_intent, strategy
        
        # Default: treat as new design
        logger.info("Default classification: new_design")
        return IntentType.NEW_DESIGN, self._get_strategy(
            IntentType.NEW_DESIGN, session_context
        )
    
    def _rule_classify(self, input_lower: str) -> Optional[IntentType]:
        """Fast pattern matching"""
        for pattern, intent in self._pattern_lookup.items():
            if pattern in input_lower:
                return intent
        return None
    
    def _heuristic_classify(
        self,
        input_lower: str,
        session_context: Dict[str, Any]
    ) -> Optional[IntentType]:
        """Context-aware heuristics"""
        
        # If no design context exists, must be new design
        if not session_context.get("mission"):
            if any(word in input_lower for word in [
                "design", "make", "create", "build", "need"
            ]):
                return IntentType.NEW_DESIGN
        
        # If design exists and input is short, likely clarification
        if session_context.get("mission") and len(input_lower.split()) < 5:
            return IntentType.CLARIFICATION
        
        # Complex descriptions suggest new design or major change
        word_count = len(input_lower.split())
        if word_count > 15:
            complexity_score = self._calculate_complexity(input_lower)
            if complexity_score > 3:
                return IntentType.NEW_DESIGN
        
        return None
    
    def _calculate_complexity(self, text: str) -> int:
        """Calculate complexity score for text"""
        score = 0
        indicators = [
            " and ", ",", " with ", " that ", " which ",
            "optimize", "compare", "analyze", "calculate",
            "if ", "when ", "unless ", "depending"
        ]
        for indicator in indicators:
            if indicator in text:
                score += 1
        return score
    
    async def _llm_classify(
        self,
        user_input: str,
        session_context: Dict[str, Any],
        conversation_history: Optional[List[Dict]]
    ) -> IntentType:
        """Use LLM for complex classification"""
        
        history_str = ""
        if conversation_history:
            recent = conversation_history[-3:]
            history_str = json.dumps(recent, default=str)
        
        prompt = f"""Classify the user intent based on input and context.

User Input: "{user_input}"

Current Design Context:
{json.dumps(session_context, indent=2, default=str)}

Recent Conversation:
{history_str}

Intent Types:
- NEW_DESIGN: Starting a fresh design
- CONSTRAINT_CHANGE: Modifying existing design constraints
- MATERIAL_VARIANT: Asking about different materials
- COMPARATIVE: Compare multiple options
- FEATURE_ADD: Add features to existing design
- EXPLANATION: Ask why or how
- NARROWING: Tightening constraints (budget, time)
- CLARIFICATION: Short response needing clarification
- GREETING: Social greeting

Return JSON: {{"intent": "TYPE", "confidence": 0.0-1.0, "reasoning": "..."}}"""
        
        try:
            # TODO: Call actual LLM
            # For now, return new_design
            return IntentType.NEW_DESIGN
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return IntentType.UNKNOWN
    
    def _get_strategy(
        self,
        intent: IntentType,
        session_context: Dict[str, Any]
    ) -> ExecutionStrategy:
        """Get execution strategy for intent"""
        
        strategies = {
            IntentType.NEW_DESIGN: ExecutionStrategy(
                use_rlm=True,
                nodes=["DiscoveryRecursiveNode", "GeometryRecursiveNode", 
                       "MaterialRecursiveNode", "CostRecursiveNode"],
                parallel_groups=[
                    ["DiscoveryRecursiveNode"],
                    ["GeometryRecursiveNode", "MaterialRecursiveNode"],
                    ["CostRecursiveNode"]
                ]
            ),
            
            IntentType.CONSTRAINT_CHANGE: ExecutionStrategy(
                use_rlm=True,
                nodes=["GeometryRecursiveNode", "CostRecursiveNode"],
                affected_by="constraints",
                use_delta=True
            ),
            
            IntentType.MATERIAL_VARIANT: ExecutionStrategy(
                use_rlm=True,
                nodes=["MaterialRecursiveNode", "GeometryRecursiveNode",
                       "CostRecursiveNode", "SafetyRecursiveNode"],
                parallel_groups=[
                    ["MaterialRecursiveNode", "SafetyRecursiveNode"],
                    ["GeometryRecursiveNode", "CostRecursiveNode"]
                ]
            ),
            
            IntentType.COMPARATIVE: ExecutionStrategy(
                use_rlm=True,
                # Nodes determined dynamically based on variants
                parallel_groups=[]  # Will fork branches
            ),
            
            IntentType.FEATURE_ADD: ExecutionStrategy(
                use_rlm=True,
                nodes=["GeometryRecursiveNode", "CostRecursiveNode"],
                use_delta=True,
                affected_by="features"
            ),
            
            IntentType.EXPLANATION: ExecutionStrategy(
                use_rlm=False,
                use_memory_only=True
            ),
            
            IntentType.NARROWING: ExecutionStrategy(
                use_rlm=True,
                nodes=["CostRecursiveNode", "MaterialRecursiveNode"],
                affected_by="constraints"
            ),
            
            IntentType.CLARIFICATION: ExecutionStrategy(
                use_rlm=False
            ),
            
            IntentType.GREETING: ExecutionStrategy(
                use_rlm=False
            ),
            
            IntentType.UNKNOWN: ExecutionStrategy(
                use_rlm=True  # Use RLM to figure it out
            ),
        }
        
        return strategies.get(intent, ExecutionStrategy(use_rlm=True))
    
    def should_use_rlm(self, user_input: str, context: Dict[str, Any]) -> bool:
        """
        Quick check if input likely needs recursive processing.
        
        Used for fast-path routing without full classification.
        """
        input_lower = user_input.lower()
        
        # High complexity indicators
        indicators = [
            len(user_input.split()) > 10,
            " and " in input_lower,
            "compare" in input_lower,
            "optimize" in input_lower,
            "design" in input_lower,
            "if " in input_lower,
            context.get("has_multiple_constraints", False),
            context.get("requires_analysis", False)
        ]
        
        return sum(indicators) >= 2
