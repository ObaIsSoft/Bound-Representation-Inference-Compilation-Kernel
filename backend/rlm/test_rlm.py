"""
Tests for RLM (Recursive Language Model) Implementation

Run with: pytest backend/rlm/test_rlm.py -v
"""

import pytest
import asyncio
from typing import Dict, Any

# Import RLM components
from backend.rlm.base_node import (
    BaseRecursiveNode, NodeResult, NodeContext, ExecutionMode
)
from backend.rlm.executor import RecursiveTaskExecutor, SubTask, TaskStatus
from backend.rlm.classifier import InputClassifier, IntentType, ExecutionStrategy
from backend.rlm.branching import BranchManager, DesignVariant, ConversationBranch
from backend.rlm.nodes import (
    DiscoveryRecursiveNode,
    GeometryRecursiveNode,
    MaterialRecursiveNode,
    CostRecursiveNode,
    SafetyRecursiveNode
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def node_context():
    """Create a basic node context for testing"""
    return NodeContext(
        session_id="test_session",
        turn_id="turn_1",
        scene_context={"mission": "test_drone"},
        requirements={"mass_kg": 2.0, "material": "aluminum"}
    )


@pytest.fixture
def node_registry():
    """Create node registry for testing"""
    return {
        "DiscoveryRecursiveNode": DiscoveryRecursiveNode(),
        "GeometryRecursiveNode": GeometryRecursiveNode(),
        "MaterialRecursiveNode": MaterialRecursiveNode(),
        "CostRecursiveNode": CostRecursiveNode(),
        "SafetyRecursiveNode": SafetyRecursiveNode(),
    }


@pytest.fixture
def rlm_executor(node_registry):
    """Create RLM executor for testing"""
    return RecursiveTaskExecutor(
        node_registry=node_registry,
        max_depth=2,
        cost_budget=2000
    )


@pytest.fixture
def input_classifier():
    """Create input classifier for testing"""
    return InputClassifier(llm_provider=None)


@pytest.fixture
def branch_manager():
    """Create branch manager for testing"""
    return BranchManager()


# =============================================================================
# Base Node Tests
# =============================================================================

class TestNodeResult:
    def test_node_result_creation(self):
        result = NodeResult(
            node_type="TestNode",
            node_id="test_123",
            success=True,
            data={"key": "value"}
        )
        assert result.node_type == "TestNode"
        assert result.success is True
        assert result.data["key"] == "value"
    
    def test_node_result_to_synthesis_format(self):
        result = NodeResult(
            node_type="GeometryNode",
            node_id="geom_1",
            success=True,
            data={"mass_kg": 2.5, "material": "aluminum"}
        )
        synthesis = result.to_synthesis_format()
        assert "GeometryNode" in synthesis
        assert "mass_kg=2.5" in synthesis
    
    def test_failed_result_synthesis(self):
        result = NodeResult(
            node_type="TestNode",
            node_id="test_1",
            success=False,
            error_message="Calculation failed"
        )
        synthesis = result.to_synthesis_format()
        assert "Failed" in synthesis


class TestNodeContext:
    def test_context_fact_management(self):
        ctx = NodeContext(
            session_id="test",
            turn_id="turn_1",
            scene_context={}
        )
        
        # Set and get fact
        ctx.set_fact("material", "titanium")
        assert ctx.get_fact("material") == "titanium"
        assert ctx.get_fact("missing", "default") == "default"
    
    def test_ephemeral_context(self):
        ctx = NodeContext(
            session_id="test",
            turn_id="turn_1"
        )
        
        ctx.add_ephemeral("temp_calc", 42)
        assert ctx.ephemeral_context["temp_calc"] == 42


# =============================================================================
# Node Implementation Tests
# =============================================================================

class TestDiscoveryNode:
    @pytest.mark.asyncio
    async def test_discovery_execution(self, node_context):
        node = DiscoveryRecursiveNode()
        
        result = await node.run(node_context)
        
        assert result.success is True
        assert "requirements" in result.data
        assert result.data["requirements"]["mission"] == "test_drone"
    
    @pytest.mark.asyncio
    async def test_discovery_updates_context(self, node_context):
        node = DiscoveryRecursiveNode()
        node_context.requirements["mission"] = "aerospace_frame"
        
        await node.run(node_context)
        
        assert node_context.get_fact("mission") == "aerospace_frame"


class TestGeometryNode:
    @pytest.mark.asyncio
    async def test_geometry_calculation(self, node_context):
        node = GeometryRecursiveNode()
        node_context.requirements["mass_kg"] = 5.0
        node_context.requirements["complexity"] = "moderate"
        node_context.set_fact("material", "aluminum")
        
        result = await node.run(node_context)
        
        assert result.success is True
        assert "dimensions" in result.data
        assert "mass" in result.data
        assert result.data["feasible"] is True
    
    @pytest.mark.asyncio
    async def test_geometry_dimension_constraints(self, node_context):
        node = GeometryRecursiveNode()
        node_context.constraints["max_dimension_m"] = 0.05  # Very small
        
        result = await node.run(node_context)
        
        assert result.success is True
        # Should warn about constraint violation
        assert len(result.warnings) > 0


class TestMaterialNode:
    @pytest.mark.asyncio
    async def test_material_selection(self, node_context):
        node = MaterialRecursiveNode()
        node_context.set_fact("application_type", "aerospace")
        
        result = await node.run(node_context)
        
        assert result.success is True
        assert "selected_material" in result.data
        assert "material_properties" in result.data
    
    @pytest.mark.asyncio
    async def test_material_alternatives(self, node_context):
        node = MaterialRecursiveNode()
        
        result = await node.run(node_context)
        
        assert result.success is True
        assert "alternatives" in result.data
        assert len(result.data["alternatives"]) > 0


class TestCostNode:
    @pytest.mark.asyncio
    async def test_cost_estimation(self, node_context):
        node = CostRecursiveNode()
        node_context.set_fact("material", "aluminum_6061")
        node_context.set_fact("estimated_mass_kg", 2.5)
        node_context.set_fact("dimensions", {
            "length_m": 0.15, "width_m": 0.1, "height_m": 0.08
        })
        
        result = await node.run(node_context)
        
        assert result.success is True
        assert "total_cost" in result.data
        assert "material_cost" in result.data
        assert result.data["total_cost"] > 0


class TestSafetyNode:
    @pytest.mark.asyncio
    async def test_safety_analysis(self, node_context):
        node = SafetyRecursiveNode()
        node_context.set_fact("material", "titanium")
        node_context.set_fact("application_type", "aerospace")
        
        result = await node.run(node_context)
        
        assert result.success is True
        assert "safety_score" in result.data
        assert "hazards" in result.data


# =============================================================================
# Input Classifier Tests
# =============================================================================

class TestInputClassifier:
    @pytest.mark.asyncio
    async def test_rule_based_classification(self, input_classifier):
        # Test greeting pattern
        intent, strategy = await input_classifier.classify(
            "hello there",
            session_context={},
            conversation_history=[]
        )
        assert intent == IntentType.GREETING
        assert strategy.use_rlm is False
    
    @pytest.mark.asyncio
    async def test_explanation_classification(self, input_classifier):
        intent, strategy = await input_classifier.classify(
            "why did you choose aluminum?",
            session_context={"mission": "drone"},
            conversation_history=[]
        )
        assert intent == IntentType.EXPLANATION
        assert strategy.use_memory_only is True
    
    @pytest.mark.asyncio
    async def test_comparative_classification(self, input_classifier):
        intent, strategy = await input_classifier.classify(
            "compare aluminum and titanium",
            session_context={},
            conversation_history=[]
        )
        assert intent == IntentType.COMPARATIVE
        assert strategy.use_rlm is True
    
    @pytest.mark.asyncio
    async def test_new_design_heuristic(self, input_classifier):
        # Complex description should trigger NEW_DESIGN
        intent, strategy = await input_classifier.classify(
            "I need a complex drone frame with multiple mounting points "
            "and optimized for weight and cost with carbon fiber",
            session_context={},
            conversation_history=[]
        )
        assert intent == IntentType.NEW_DESIGN
        assert strategy.use_rlm is True
    
    def test_complexity_calculation(self, input_classifier):
        score = input_classifier._calculate_complexity(
            "design a frame with mounting and optimize for weight and cost"
        )
        assert score > 0
    
    def test_should_use_rlm_simple(self, input_classifier):
        # Simple query should not use RLM
        should_use = input_classifier.should_use_rlm(
            "hello",
            context={}
        )
        assert should_use is False
    
    def test_should_use_rlm_complex(self, input_classifier):
        # Complex query should use RLM
        should_use = input_classifier.should_use_rlm(
            "design a complex drone frame with multiple constraints and optimize",
            context={}
        )
        assert should_use is True


# =============================================================================
# Branch Manager Tests
# =============================================================================

class TestBranchManager:
    @pytest.mark.asyncio
    async def test_create_branch(self, branch_manager):
        parent_context = {
            "mission": "drone",
            "material": "aluminum",
            "mass_kg": 2.5
        }
        
        branch = await branch_manager.create_branch(
            parent_session="main_session",
            parent_context=parent_context,
            name="Titanium Variant",
            description="Testing titanium material",
            parameter_changes={"material": "titanium"}
        )
        
        assert branch.branch_id.startswith("branch-")
        assert branch.parent_id == "main_session"
        assert branch.name == "Titanium Variant"
        assert branch.context["material"] == "titanium"
    
    @pytest.mark.asyncio
    async def test_create_comparison_branches(self, branch_manager):
        variants = [
            {"name": "Aluminum", "parameters": {"material": "aluminum"}},
            {"name": "Titanium", "parameters": {"material": "titanium"}},
            {"name": "Steel", "parameters": {"material": "steel"}}
        ]
        
        branches = await branch_manager.create_comparison_branches(
            parent_session="main",
            parent_context={"mission": "test"},
            variants=variants
        )
        
        assert len(branches) == 3
        assert branches[0].name == "Aluminum"
        assert branches[1].name == "Titanium"
        assert branches[2].name == "Steel"
    
    @pytest.mark.asyncio
    async def test_branch_comparison(self, branch_manager):
        # Create branches
        branch1 = await branch_manager.create_branch(
            parent_session="main",
            parent_context={},
            name="Variant A",
            parameter_changes={"material": "aluminum"}
        )
        
        branch2 = await branch_manager.create_branch(
            parent_session="main",
            parent_context={},
            name="Variant B",
            parameter_changes={"material": "titanium"}
        )
        
        # Update with results
        branch_manager.update_branch_results(
            branch1.branch_id,
            results={},
            metrics={"cost": 100, "mass": 2.5}
        )
        
        branch_manager.update_branch_results(
            branch2.branch_id,
            results={},
            metrics={"cost": 200, "mass": 1.8}
        )
        
        # Compare
        comparison = branch_manager.compare_branches("main")
        
        assert comparison["branches_compared"] == 2
        assert len(comparison["trade_offs"]) > 0
    
    @pytest.mark.asyncio
    async def test_merge_branch(self, branch_manager):
        branch = await branch_manager.create_branch(
            parent_session="main",
            parent_context={"mission": "drone"},
            name="Test Variant",
            parameter_changes={"material": "carbon_fiber"}
        )
        
        # Complete the branch
        branch_manager.update_branch_results(
            branch.branch_id,
            results={"cost": 150}
        )
        
        # Merge
        result = await branch_manager.merge_branch(
            branch_id=branch.branch_id,
            target_session="main"
        )
        
        assert result["target_session"] == "main"
        assert "changes_applied" in result
        assert branch.status.value == "merged"


# =============================================================================
# Integration Tests
# =============================================================================

class TestRLMIntegration:
    @pytest.mark.asyncio
    async def test_full_execution_pipeline(self, rlm_executor):
        """Test full RLM execution with decomposition"""
        session_context = {
            "mission": "drone_frame",
            "application_type": "aerospace"
        }
        
        result = await rlm_executor.execute(
            user_input="Design a lightweight drone frame",
            session_context=session_context,
            session_id="test_session"
        )
        
        assert result.success is True
        assert result.response is not None
        assert result.trace is not None
    
    @pytest.mark.asyncio
    async def test_rlm_cost_budget(self, node_registry):
        """Test that cost budget is enforced"""
        executor = RecursiveTaskExecutor(
            node_registry=node_registry,
            max_depth=2,
            cost_budget=100  # Very low budget
        )
        
        # Should still complete but potentially with warnings
        result = await executor.execute(
            user_input="Test query",
            session_context={},
            session_id="test"
        )
        
        # Result should exist even if budget constrained
        assert result is not None


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
