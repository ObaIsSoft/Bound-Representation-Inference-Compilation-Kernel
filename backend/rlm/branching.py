"""
Conversation Branching for Design Variants

Enables "what-if" exploration by creating isolated branches for design variants.
Branches can be compared, merged, or discarded without affecting the main design.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from enum import Enum
import copy
import json

logger = logging.getLogger(__name__)


class BranchStatus(Enum):
    """Status of a conversation branch"""
    ACTIVE = "active"           # Currently being worked on
    COMPLETED = "completed"     # Analysis complete
    MERGED = "merged"          # Merged back to parent
    DISCARDED = "discarded"    # Abandoned
    CONFLICT = "conflict"      # Merge conflict


@dataclass
class DesignVariant:
    """
    Represents a design variant within a branch.
    """
    # Variant identification
    variant_id: str
    name: str
    description: str
    
    # Design parameters that differ from parent
    parameter_changes: Dict[str, Any] = field(default_factory=dict)
    
    # Results from analysis
    results: Dict[str, Any] = field(default_factory=dict)
    
    # Comparison metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "name": self.name,
            "description": self.description,
            "parameter_changes": self.parameter_changes,
            "metrics": self.metrics
        }


@dataclass
class ConversationBranch:
    """
    Represents a branched conversation for exploring design variants.
    """
    # Identification
    branch_id: str
    parent_id: str
    session_id: str  # Root session
    
    # Content
    name: str
    description: str
    
    # State
    context: Dict[str, Any] = field(default_factory=dict)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    
    # Variant being explored
    variant: Optional[DesignVariant] = None
    
    # Status
    status: BranchStatus = BranchStatus.ACTIVE
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    merged_at: Optional[str] = None
    
    def get_context_diff(self, parent_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get differences from parent context"""
        diff = {}
        for key, value in self.context.items():
            if parent_context.get(key) != value:
                diff[key] = {"from": parent_context.get(key), "to": value}
        return diff


class BranchManager:
    """
    Manages conversation branching for design variant exploration.
    
    Usage:
        manager = BranchManager()
        
        # Create branch for material variant
        branch = await manager.create_branch(
            parent_session="main_session",
            name="Titanium Variant",
            parameter_changes={"material": "titanium_grade5"}
        )
        
        # Work in branch
        result = await agent.run(params, session_id=branch.branch_id)
        
        # Compare with main
        comparison = manager.compare_branches("main_session", branch.branch_id)
        
        # Merge if desired
        await manager.merge_branch(branch.branch_id, "main_session")
    """
    
    def __init__(self):
        self._branches: Dict[str, ConversationBranch] = {}
        self._parent_children: Dict[str, Set[str]] = {}  # parent_id -> set of child branch_ids
        
        logger.info("BranchManager initialized")
    
    async def create_branch(
        self,
        parent_session: str,
        parent_context: Dict[str, Any],
        name: str,
        description: str = "",
        parameter_changes: Optional[Dict[str, Any]] = None
    ) -> ConversationBranch:
        """
        Create a new branch from a parent session.
        
        Args:
            parent_session: ID of parent session to branch from
            parent_context: Context dictionary to copy
            name: Human-readable name for the branch
            description: Description of what this branch explores
            parameter_changes: Parameters that differ from parent
            
        Returns:
            New ConversationBranch
        """
        branch_id = f"branch-{parent_session}-{int(datetime.now().timestamp())}"
        
        # Create variant specification
        variant = DesignVariant(
            variant_id=f"variant-{branch_id}",
            name=name,
            description=description,
            parameter_changes=parameter_changes or {}
        )
        
        # Deep copy parent context
        branched_context = copy.deepcopy(parent_context)
        
        # Apply parameter changes
        if parameter_changes:
            branched_context.update(parameter_changes)
            branched_context["is_variant"] = True
            branched_context["variant_of"] = parent_session
        
        # Create branch
        branch = ConversationBranch(
            branch_id=branch_id,
            parent_id=parent_session,
            session_id=parent_session,  # Root remains the same
            name=name,
            description=description,
            context=branched_context,
            variant=variant
        )
        
        # Track branch
        self._branches[branch_id] = branch
        
        if parent_session not in self._parent_children:
            self._parent_children[parent_session] = set()
        self._parent_children[parent_session].add(branch_id)
        
        logger.info(f"Created branch {branch_id} from {parent_session}: {name}")
        
        return branch
    
    async def create_comparison_branches(
        self,
        parent_session: str,
        parent_context: Dict[str, Any],
        variants: List[Dict[str, Any]]
    ) -> List[ConversationBranch]:
        """
        Create multiple branches for comparative analysis.
        
        Args:
            parent_session: Parent session ID
            parent_context: Parent context to copy
            variants: List of variant specifications
                Each variant: {"name": "...", "parameters": {...}}
                
        Returns:
            List of created branches
        """
        branches = []
        
        for variant_spec in variants:
            branch = await self.create_branch(
                parent_session=parent_session,
                parent_context=parent_context,
                name=variant_spec["name"],
                description=variant_spec.get("description", ""),
                parameter_changes=variant_spec.get("parameters", {})
            )
            branches.append(branch)
        
        logger.info(f"Created {len(branches)} comparison branches from {parent_session}")
        
        return branches
    
    def get_branch(self, branch_id: str) -> Optional[ConversationBranch]:
        """Get branch by ID"""
        return self._branches.get(branch_id)
    
    def get_child_branches(self, parent_id: str) -> List[ConversationBranch]:
        """Get all child branches of a parent"""
        child_ids = self._parent_children.get(parent_id, set())
        return [self._branches[bid] for bid in child_ids if bid in self._branches]
    
    def update_branch_results(
        self,
        branch_id: str,
        results: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None
    ):
        """Update branch with analysis results"""
        branch = self._branches.get(branch_id)
        if not branch:
            logger.warning(f"Branch {branch_id} not found")
            return
        
        if branch.variant:
            branch.variant.results = results
            if metrics:
                branch.variant.metrics.update(metrics)
        
        branch.status = BranchStatus.COMPLETED
        branch.completed_at = datetime.now().isoformat()
        
        logger.info(f"Updated branch {branch_id} with results")
    
    def compare_branches(
        self,
        parent_id: str,
        branch_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare branches against parent and each other.
        
        Returns:
            Comparison report with metrics table
        """
        if branch_ids is None:
            # Get all children of parent
            branch_ids = list(self._parent_children.get(parent_id, set()))
        
        branches = [self._branches.get(bid) for bid in branch_ids if bid in self._branches]
        
        if not branches:
            return {"error": "No branches to compare"}
        
        # Build comparison table
        comparison = {
            "parent_id": parent_id,
            "branches_compared": len(branches),
            "comparison_table": [],
            "trade_offs": []
        }
        
        for branch in branches:
            if not branch.variant:
                continue
            
            entry = {
                "branch_id": branch.branch_id,
                "name": branch.name,
                "parameter_changes": branch.variant.parameter_changes,
                "metrics": branch.variant.metrics,
                "status": branch.status.value
            }
            comparison["comparison_table"].append(entry)
        
        # Identify trade-offs
        metrics_keys = set()
        for branch in branches:
            if branch.variant:
                metrics_keys.update(branch.variant.metrics.keys())
        
        for metric in metrics_keys:
            values = []
            for branch in branches:
                if branch.variant and metric in branch.variant.metrics:
                    values.append((branch.name, branch.variant.metrics[metric]))
            
            if len(values) > 1:
                # Sort to find best
                values.sort(key=lambda x: x[1])
                comparison["trade_offs"].append({
                    "metric": metric,
                    "best": values[0],
                    "worst": values[-1],
                    "range": values[-1][1] - values[0][1] if values else 0
                })
        
        return comparison
    
    async def merge_branch(
        self,
        branch_id: str,
        target_session: str,
        conflict_resolution: str = "branch_wins"  # or "target_wins", "manual"
    ) -> Dict[str, Any]:
        """
        Merge a branch back into target session.
        
        Args:
            branch_id: Branch to merge
            target_session: Target session ID
            conflict_resolution: How to handle conflicts
            
        Returns:
            Merge result with applied changes
        """
        branch = self._branches.get(branch_id)
        if not branch:
            return {"error": f"Branch {branch_id} not found"}
        
        if branch.status != BranchStatus.COMPLETED:
            return {"error": f"Branch {branch_id} is not completed"}
        
        # Calculate changes to apply
        changes = branch.variant.parameter_changes if branch.variant else {}
        
        merge_result = {
            "branch_id": branch_id,
            "target_session": target_session,
            "changes_applied": changes,
            "conflict_resolution": conflict_resolution,
            "merged_at": datetime.now().isoformat()
        }
        
        # Update branch status
        branch.status = BranchStatus.MERGED
        branch.merged_at = merge_result["merged_at"]
        
        logger.info(f"Merged branch {branch_id} into {target_session}")
        
        return merge_result
    
    async def discard_branch(self, branch_id: str, reason: str = ""):
        """Discard a branch without merging"""
        branch = self._branches.get(branch_id)
        if branch:
            branch.status = BranchStatus.DISCARDED
            logger.info(f"Discarded branch {branch_id}: {reason}")
    
    def get_branch_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of all branches for a session"""
        child_ids = self._parent_children.get(session_id, set())
        
        summary = {
            "parent_session": session_id,
            "total_branches": len(child_ids),
            "branches": []
        }
        
        for bid in child_ids:
            branch = self._branches.get(bid)
            if branch:
                summary["branches"].append({
                    "branch_id": bid,
                    "name": branch.name,
                    "status": branch.status.value,
                    "created_at": branch.created_at,
                    "completed_at": branch.completed_at,
                    "parameter_changes": branch.variant.parameter_changes if branch.variant else {}
                })
        
        return summary
    
    def clear_session_branches(self, session_id: str):
        """Clear all branches for a session"""
        child_ids = self._parent_children.get(session_id, set())
        
        for bid in child_ids:
            if bid in self._branches:
                del self._branches[bid]
        
        if session_id in self._parent_children:
            del self._parent_children[session_id]
        
        logger.info(f"Cleared {len(child_ids)} branches for session {session_id}")


# Convenience functions for common patterns

async def create_material_variant_branches(
    branch_manager: BranchManager,
    parent_session: str,
    parent_context: Dict[str, Any],
    materials: List[str]
) -> List[ConversationBranch]:
    """
    Create branches for comparing different materials.
    
    Args:
        branch_manager: BranchManager instance
        parent_session: Parent session ID
        parent_context: Parent context
        materials: List of materials to compare
        
    Returns:
        List of branches (one per material)
    """
    variants = []
    for material in materials:
        variants.append({
            "name": f"{material.title()} Variant",
            "description": f"Design using {material}",
            "parameters": {"material": material}
        })
    
    return await branch_manager.create_comparison_branches(
        parent_session=parent_session,
        parent_context=parent_context,
        variants=variants
    )


async def create_process_variant_branches(
    branch_manager: BranchManager,
    parent_session: str,
    parent_context: Dict[str, Any],
    processes: List[str]
) -> List[ConversationBranch]:
    """
    Create branches for comparing manufacturing processes.
    
    Args:
        branch_manager: BranchManager instance
        parent_session: Parent session ID
        parent_context: Parent context
        processes: List of processes to compare (e.g., ["CNC", "3D_print", "sheet_metal"])
        
    Returns:
        List of branches (one per process)
    """
    variants = []
    for process in processes:
        variants.append({
            "name": f"{process.replace('_', ' ').title()} Process",
            "description": f"Manufacturing via {process}",
            "parameters": {"manufacturing_process": process}
        })
    
    return await branch_manager.create_comparison_branches(
        parent_session=parent_session,
        parent_context=parent_context,
        variants=variants
    )
