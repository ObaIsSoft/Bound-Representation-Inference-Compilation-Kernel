"""
Recursive Task Executor

Manages the recursive decomposition of complex tasks into sub-tasks,
their execution (sequential or parallel), and synthesis of results.

Key features:
- Max recursion depth enforcement
- Cost budget tracking
- Parallel execution for independent sub-tasks
- Dependency resolution
- Result synthesis with grounded generation
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime
from enum import Enum
import json

from .base_node import BaseRecursiveNode, NodeResult, NodeContext, ExecutionMode

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SubTask:
    """
    Represents a single sub-task in the recursive decomposition.
    """
    # Identification
    id: str
    node_type: str
    
    # Input
    context: NodeContext
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    
    # Execution
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0  # Higher = execute first
    
    # Results
    result: Optional[NodeResult] = None
    error: Optional[str] = None
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    def can_execute(self, completed_task_ids: Set[str]) -> bool:
        """Check if all dependencies are satisfied"""
        return all(dep in completed_task_ids for dep in self.depends_on)


@dataclass
class ExecutionTrace:
    """
    Complete trace of recursive execution for debugging and explainability.
    """
    root_task: str
    max_depth: int
    tasks: Dict[str, SubTask] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    
    # Cost tracking
    total_tokens_used: int = 0
    total_execution_time_ms: int = 0
    
    # Synthesis
    synthesis_prompt: Optional[str] = None
    synthesis_result: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_task": self.root_task,
            "max_depth": self.max_depth,
            "task_count": len(self.tasks),
            "execution_order": self.execution_order,
            "total_tokens": self.total_tokens_used,
            "total_time_ms": self.total_execution_time_ms,
        }
    
    def get_execution_tree(self) -> str:
        """Generate ASCII tree of execution for debugging"""
        lines = ["Execution Tree:"]
        for task_id in self.execution_order:
            task = self.tasks[task_id]
            indent = "  " * task.context.depth
            status_icon = "✓" if task.status == TaskStatus.COMPLETED else "✗"
            lines.append(f"{indent}{status_icon} {task.node_type} ({task_id})")
        return "\n".join(lines)


@dataclass
class ExecutionResult:
    """
    Final result of recursive task execution.
    """
    success: bool
    response: str
    
    # Structured data
    facts: Dict[str, Any] = field(default_factory=dict)
    
    # Execution metadata
    trace: Optional[ExecutionTrace] = None
    tokens_used: int = 0
    execution_time_ms: int = 0
    
    # Error handling
    error: Optional[str] = None
    fallback_used: bool = False


class RecursiveTaskExecutor:
    """
    Main executor for recursive task decomposition and execution.
    
    Usage:
        executor = RecursiveTaskExecutor(
            node_registry=node_registry,
            max_depth=3,
            cost_budget=4000
        )
        
        result = await executor.execute(
            user_input="Design a drone frame",
            session_context=scene_context
        )
    """
    
    def __init__(
        self,
        node_registry: Dict[str, BaseRecursiveNode],
        max_depth: int = 3,
        cost_budget: int = 4000,
        max_parallel_tasks: int = 5,
        enable_caching: bool = True
    ):
        self.node_registry = node_registry
        self.max_depth = max_depth
        self.cost_budget = cost_budget
        self.max_parallel_tasks = max_parallel_tasks
        self.enable_caching = enable_caching
        
        # Tracking
        self.tokens_used = 0
        self._execution_trace: Optional[ExecutionTrace] = None
        
        logger.info(f"RLM Executor initialized (max_depth={max_depth}, "
                   f"budget={cost_budget})")
    
    async def execute(
        self,
        user_input: str,
        session_context: Dict[str, Any],
        session_id: str,
        intent: Optional[str] = None,
        strategy: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute user input through recursive decomposition.
        
        This is the main entry point for RLM execution.
        """
        start_time = datetime.now()
        self.tokens_used = 0
        
        try:
            # Step 1: Decompose into sub-tasks
            logger.info(f"Decomposing input: {user_input[:50]}...")
            sub_tasks = await self._decompose(
                user_input=user_input,
                session_context=session_context,
                session_id=session_id,
                intent=intent,
                strategy=strategy
            )
            
            if not sub_tasks:
                # No decomposition needed - single pass
                logger.info("No decomposition needed, single-pass execution")
                return await self._single_pass_execute(
                    user_input, session_context
                )
            
            # Step 2: Execute sub-tasks
            logger.info(f"Executing {len(sub_tasks)} sub-tasks")
            results = await self._execute_subtasks(sub_tasks)
            
            # Step 3: Synthesize results
            logger.info("Synthesizing results")
            response = await self._synthesize(results, user_input)
            
            # Step 4: Build execution trace
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            trace = ExecutionTrace(
                root_task=user_input[:100],
                max_depth=self.max_depth,
                tasks=sub_tasks,
                total_tokens_used=self.tokens_used,
                total_execution_time_ms=int(execution_time)
            )
            
            # Extract facts for scene context
            facts = self._extract_facts(results)
            
            return ExecutionResult(
                success=True,
                response=response,
                facts=facts,
                trace=trace,
                tokens_used=self.tokens_used,
                execution_time_ms=int(execution_time)
            )
            
        except Exception as e:
            logger.error(f"RLM execution failed: {e}")
            return ExecutionResult(
                success=False,
                response="I encountered an error processing your request.",
                error=str(e),
                fallback_used=True
            )
    
    async def _decompose(
        self,
        user_input: str,
        session_context: Dict[str, Any],
        session_id: str,
        intent: Optional[str] = None,
        strategy: Optional[str] = None
    ) -> Dict[str, SubTask]:
        """
        Decompose user input into sub-tasks.
        
        Uses LLM to break down complex queries into executable units.
        """
        # Build decomposition prompt
        prompt = self._build_decomposition_prompt(
            user_input=user_input,
            session_context=session_context,
            intent=intent,
            strategy=strategy
        )
        
        # Call LLM for decomposition
        # TODO: Replace with actual LLM call
        decomposition = await self._call_decomposition_llm(prompt)
        
        # Create sub-tasks from decomposition
        sub_tasks = {}
        for i, task_def in enumerate(decomposition.get("tasks", [])):
            task_id = f"task_{i}_{datetime.now().strftime('%H%M%S')}"
            
            # Create node context
            node_context = NodeContext(
                session_id=session_id,
                turn_id=task_id,
                scene_context=session_context.copy(),
                depth=1,
                constraints=task_def.get("constraints", {}),
                requirements=task_def.get("requirements", {})
            )
            
            sub_task = SubTask(
                id=task_id,
                node_type=task_def["node_type"],
                context=node_context,
                params=task_def.get("params", {}),
                depends_on=task_def.get("depends_on", []),
                priority=task_def.get("priority", 0)
            )
            
            sub_tasks[task_id] = sub_task
        
        return sub_tasks
    
    def _build_decomposition_prompt(
        self,
        user_input: str,
        session_context: Dict[str, Any],
        intent: Optional[str],
        strategy: Optional[str]
    ) -> str:
        """Build prompt for decomposition LLM"""
        
        available_nodes = list(self.node_registry.keys())
        
        prompt = f"""You are a task decomposition expert for a hardware design system.

User Input: "{user_input}"

Current Design Context:
{json.dumps(session_context, indent=2, default=str)}

Intent: {intent or "unknown"}
Strategy: {strategy or "default"}

Available Node Types:
{chr(10).join(f"- {node}: {self.node_registry[node].DESCRIPTION}" for node in available_nodes)}

Decompose the user input into sub-tasks. Each sub-task should:
1. Use exactly one node type from the available list
2. Have clear, specific parameters
3. Declare dependencies on other tasks if needed
4. Be independently executable

Return JSON:
{{
  "tasks": [
    {{
      "node_type": "GeometryNode",
      "params": {{"operation": "calculate_mass", "material": "aluminum"}},
      "depends_on": [],
      "constraints": {{"max_mass_kg": 2.0}},
      "priority": 1
    }}
  ],
  "reasoning": "Why this decomposition was chosen"
}}

Rules:
- Max 5 sub-tasks
- Only use available node types
- Declare all dependencies explicitly
- Higher priority = execute first"""
        
        return prompt
    
    async def _call_decomposition_llm(self, prompt: str) -> Dict[str, Any]:
        """
        Call LLM to decompose task.
        
        TODO: Replace with actual LLM integration
        """
        # Placeholder - would call actual LLM
        # For now, return a simple decomposition based on intent
        
        return {
            "tasks": [
                {
                    "node_type": "DiscoveryRecursiveNode",
                    "params": {"extract_requirements": True},
                    "depends_on": [],
                    "priority": 3
                },
                {
                    "node_type": "GeometryRecursiveNode",
                    "params": {"calculate_envelope": True},
                    "depends_on": [],
                    "priority": 2
                },
                {
                    "node_type": "MaterialRecursiveNode",
                    "params": {"select_material": True},
                    "depends_on": [],
                    "priority": 2
                },
                {
                    "node_type": "CostRecursiveNode",
                    "params": {"estimate_cost": True},
                    "depends_on": ["MaterialRecursiveNode", "GeometryRecursiveNode"],
                    "priority": 1
                }
            ],
            "reasoning": "Standard design workflow: gather requirements, calculate geometry, select material, then estimate cost"
        }
    
    async def _execute_subtasks(
        self,
        sub_tasks: Dict[str, SubTask]
    ) -> List[NodeResult]:
        """
        Execute sub-tasks respecting dependencies.
        
        Supports parallel execution for independent tasks.
        """
        results = []
        completed_tasks: Set[str] = set()
        pending_tasks = set(sub_tasks.keys())
        
        while pending_tasks:
            # Find tasks that can execute now
            executable = [
                task_id for task_id in pending_tasks
                if sub_tasks[task_id].can_execute(completed_tasks)
            ]
            
            if not executable:
                # Deadlock or missing dependencies
                logger.error(f"Cannot execute remaining tasks: {pending_tasks}")
                break
            
            # Sort by priority (highest first)
            executable.sort(
                key=lambda tid: sub_tasks[tid].priority,
                reverse=True
            )
            
            # Limit parallel execution
            batch = executable[:self.max_parallel_tasks]
            
            # Execute batch
            batch_tasks = [sub_tasks[tid] for tid in batch]
            batch_results = await asyncio.gather(*[
                self._execute_single_task(task)
                for task in batch_tasks
            ])
            
            # Process results
            for task_id, result in zip(batch, batch_results):
                sub_tasks[task_id].result = result
                sub_tasks[task_id].status = (
                    TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
                )
                sub_tasks[task_id].completed_at = datetime.now().isoformat()
                
                results.append(result)
                completed_tasks.add(task_id)
                pending_tasks.remove(task_id)
                
                # Track costs
                self.tokens_used += result.tokens_used
                
                # Check budget
                if self.tokens_used > self.cost_budget:
                    logger.warning(f"Cost budget exceeded: {self.tokens_used}")
                    # Cancel remaining tasks
                    for tid in pending_tasks:
                        sub_tasks[tid].status = TaskStatus.CANCELLED
                    break
        
        return results
    
    async def _execute_single_task(self, task: SubTask) -> NodeResult:
        """Execute a single sub-task"""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now().isoformat()
        
        node = self.node_registry.get(task.node_type)
        if not node:
            return NodeResult(
                node_type=task.node_type,
                node_id=task.id,
                success=False,
                error_message=f"Unknown node type: {task.node_type}"
            )
        
        try:
            result = await node.run(task.context)
            return result
        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            return NodeResult(
                node_type=task.node_type,
                node_id=task.id,
                success=False,
                error_message=str(e)
            )
    
    async def _synthesize(
        self,
        results: List[NodeResult],
        original_input: str
    ) -> str:
        """
        Synthesize sub-task results into coherent response.
        
        Uses grounded generation - every claim tied to node results.
        """
        # Build synthesis prompt
        facts_text = "\n".join([
            r.to_synthesis_format() for r in results if r.success
        ])
        
        prompt = f"""You are a hardware design assistant. Synthesize the following analysis into a clear, helpful response.

User Request: "{original_input}"

Analysis Results:
{facts_text}

Instructions:
1. Address the user's request directly
2. Reference specific data points from the analysis
3. Explain trade-offs if multiple options exist
4. Be concise but thorough
5. If there were errors, explain what we can and cannot determine

Response:"""
        
        # TODO: Call LLM for synthesis
        # For now, return placeholder
        return f"Based on my analysis:\n\n{facts_text}"
    
    def _extract_facts(self, results: List[NodeResult]) -> Dict[str, Any]:
        """Extract key facts from results for scene context"""
        facts = {}
        
        for result in results:
            if result.success:
                # Merge result data into facts
                # Use node type as namespace to avoid collisions
                for key, value in result.data.items():
                    facts[f"{result.node_type}.{key}"] = value
        
        return facts
    
    async def _single_pass_execute(
        self,
        user_input: str,
        session_context: Dict[str, Any]
    ) -> ExecutionResult:
        """Fallback to single-pass execution"""
        return ExecutionResult(
            success=True,
            response=f"Single-pass response for: {user_input}",
            fallback_used=True
        )
