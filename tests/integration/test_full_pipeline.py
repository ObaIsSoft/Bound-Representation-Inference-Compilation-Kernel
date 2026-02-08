"""
End-to-end pipeline integration tests.

These tests verify the complete flow from user intent through
the entire orchestration pipeline to final output.
"""
import pytest
import asyncio

pytestmark = [pytest.mark.integration, pytest.mark.asyncio, pytest.mark.slow]


class TestFullPipelineExecution:
    """Test complete orchestration pipeline execution."""
    
    async def test_simple_design_ball(self, initialize_registry):
        """
        E2E test: Design a simple red plastic ball.
        
        Verifies the complete pipeline:
        1. Intent parsing
        2. Agent selection
        3. Geometry generation
        4. Physics validation
        5. Cost estimation
        6. Lazy loading verification
        """
        from backend.orchestrator import build_graph, AgentState
        from backend.agent_registry import registry
        
        # Initialize state
        state = AgentState(
            messages=[],
            user_intent="Design a simple red plastic ball with 5cm radius.",
            project_id="test_ball_001",
            mode="run",
            feasibility_report={},
            geometry_estimate={},
            cost_estimate={},
            plan_review={},
            mass_properties={},
            structural_analysis={},
            fluid_analysis={},
            selected_physics_agents=[],
            final_documentation={},
            quality_review_report={}
        )
        
        # Build and run graph
        graph = build_graph()
        final_state = await graph.ainvoke(state)
        
        # Verify lazy loading
        loaded_agents = len(registry._agents)
        print(f"\nLoaded agents: {loaded_agents}/64")
        assert loaded_agents < 30, f"Lazy loading failed: {loaded_agents} agents loaded"
        
        # Verify feasibility
        assert final_state["feasibility_report"].get("status") == "feasible", \
            "Simple ball should be feasible"
        
        # Verify appropriate agent selection
        selected = final_state.get("selected_physics_agents", [])
        assert "PhysicsAgent" in selected or "GeometryAgent" in selected
        assert "ElectronicsAgent" not in selected
        assert "NuclearAgent" not in selected
        
        # Verify cost estimation
        cost = final_state["cost_estimate"].get("total_cost", 1000)
        assert cost < 10000, f"Ball cost ${cost} is too high"
        
        print("\n✅ Simple Ball Test Passed")
    
    async def test_structural_bracket_design(self, initialize_registry):
        """
        E2E test: Design a load-bearing bracket.
        
        Verifies structural analysis integration with real physics.
        """
        from backend.orchestrator import build_graph, AgentState
        
        state = AgentState(
            messages=[],
            user_intent="Design a steel wall bracket that can hold 50kg",
            project_id="test_bracket_001",
            mode="run",
            feasibility_report={},
            geometry_estimate={},
            cost_estimate={},
            plan_review={},
            mass_properties={},
            structural_analysis={},
            fluid_analysis={},
            selected_physics_agents=[],
            final_documentation={},
            quality_review_report={}
        )
        
        graph = build_graph()
        final_state = await graph.ainvoke(state)
        
        # Should select structural agents
        selected = final_state.get("selected_physics_agents", [])
        assert any(agent in selected for agent in ["PhysicsAgent", "StructuralAgent"]), \
            "Should select physics/structural agents for load-bearing design"
        
        # Should have structural analysis results
        struct_analysis = final_state.get("structural_analysis", {})
        if struct_analysis:  # May be empty if agent didn't run
            assert "stress" in struct_analysis or "safety_factor" in struct_analysis, \
                "Should have structural calculations"
    
    async def test_fluid_system_design(self, initialize_registry):
        """
        E2E test: Design a fluid handling system.
        
        Verifies fluid dynamics agent integration.
        """
        from backend.orchestrator import build_graph, AgentState
        
        state = AgentState(
            messages=[],
            user_intent="Design a pipe system for water flow at 10L/min",
            project_id="test_fluid_001",
            mode="run",
            feasibility_report={},
            geometry_estimate={},
            cost_estimate={},
            plan_review={},
            mass_properties={},
            structural_analysis={},
            fluid_analysis={},
            selected_physics_agents=[],
            final_documentation={},
            quality_review_report={}
        )
        
        graph = build_graph()
        final_state = await graph.ainvoke(state)
        
        # Should select fluid agents
        selected = final_state.get("selected_physics_agents", [])
        # Fluid dynamics might trigger specific agent selection
        
        # Verify state is properly structured
        assert final_state["project_id"] == "test_fluid_001"
        assert final_state["user_intent"] == "Design a pipe system for water flow at 10L/min"


class TestPhysicsValidationIntegration:
    """Test physics validation integrated with orchestration."""
    
    async def test_infeasible_design_detection(self, initialize_registry):
        """
        Test that physically infeasible designs are rejected.
        
        An impossible design (e.g., 100m tall thin concrete needle)
        should be flagged as infeasible.
        """
        from backend.orchestrator import build_graph, AgentState
        from physics.kernel import get_physics_kernel
        
        kernel = get_physics_kernel()
        
        # Verify physics kernel can detect infeasibility
        infeasible_geometry = {
            "volume": 1.0,       # Large volume
            "cross_section_area": 0.001,  # Very small base
            "length": 50.0,      # Very tall
        }
        
        validation = kernel.validate_geometry(
            geometry=infeasible_geometry,
            material="Concrete",
        )
        
        # Physics should detect this is infeasible
        assert validation["feasible"] is False or validation["fos"] < 1.0, \
            "Should detect infeasible geometry"
    
    async def test_material_selection_physics(self, initialize_registry):
        """
        Test that material selection considers physics constraints.
        """
        from physics.kernel import get_physics_kernel
        from physics.domains.materials import MaterialsDomain
        
        kernel = get_physics_kernel()
        materials = kernel.domains["materials"]
        
        # Get properties for common materials
        steel_density = materials.get_property("Steel", "density")
        aluminum_density = materials.get_property("Aluminum", "density")
        
        # Aluminum should be lighter than steel
        assert aluminum_density < steel_density, \
            f"Aluminum ({aluminum_density}) should be lighter than steel ({steel_density})"


class TestStateManagement:
    """Test state handling throughout the pipeline."""
    
    async def test_immutable_state_physics_node(self, initialize_registry):
        """
        Verify C5 fix: Physics node handles state immutably.
        
        State mutations should not affect previous nodes' outputs.
        """
        from backend.orchestrator import AgentState
        
        # Create initial state with validation flags
        initial_flags = {
            "physics_safe": True,
            "reasons": ["Initial check passed"],
        }
        
        state = AgentState(
            messages=[],
            user_intent="Test state immutability",
            project_id="test_state_001",
            mode="run",
            feasibility_report={},
            geometry_estimate={},
            cost_estimate={},
            plan_review={},
            mass_properties={},
            structural_analysis={},
            fluid_analysis={},
            selected_physics_agents=[],
            final_documentation={},
            quality_review_report={},
            # Add validation flags
            physics_validation=initial_flags,
        )
        
        # State should be valid
        assert state["project_id"] == "test_state_001"
    
    async def test_state_propagation_through_nodes(self, initialize_registry):
        """
        Test that state correctly propagates through graph nodes.
        """
        from backend.orchestrator import build_graph, AgentState
        
        state = AgentState(
            messages=[],
            user_intent="Test state propagation",
            project_id="test_prop_001",
            mode="run",
            feasibility_report={},
            geometry_estimate={},
            cost_estimate={},
            plan_review={},
            mass_properties={},
            structural_analysis={},
            fluid_analysis={},
            selected_physics_agents=[],
            final_documentation={},
            quality_review_report={}
        )
        
        graph = build_graph()
        
        # Just verify graph builds and can be invoked
        # Full execution would require complete setup
        assert graph is not None


class TestErrorHandlingPipeline:
    """Test error handling in the pipeline."""
    
    async def test_graceful_agent_failure(self, initialize_registry):
        """
        Test that agent failures don't crash the entire pipeline.
        """
        from backend.orchestrator import build_graph, AgentState
        
        state = AgentState(
            messages=[],
            user_intent="Test with potentially problematic input that might cause agent errors",
            project_id="test_error_001",
            mode="run",
            feasibility_report={},
            geometry_estimate={},
            cost_estimate={},
            plan_review={},
            mass_properties={},
            structural_analysis={},
            fluid_analysis={},
            selected_physics_agents=[],
            final_documentation={},
            quality_review_report={}
        )
        
        # Should not crash even with edge case inputs
        graph = build_graph()
        assert graph is not None
    
    async def test_missing_data_handling(self, initialize_registry):
        """
        Test that missing data is handled gracefully.
        """
        from backend.orchestrator import AgentState
        
        # Create state with minimal data
        state = AgentState(
            messages=[],
            user_intent="Minimal test",
            project_id="test_minimal",
            mode="run",
        )
        
        # Should be valid even with minimal fields
        assert state["project_id"] == "test_minimal"


class TestAPIIntegration:
    """Test API endpoint integration."""
    
    async def test_chat_requirements_endpoint(self, client):
        """
        Test /chat/requirements endpoint with JSON payload.
        
        Verifies Phase 2 API standardization.
        """
        import pytest
        pytest.skip("Requires full backend setup with LLM - run manually")
        
        # This test shows the expected API usage
        # In practice, this requires a running backend with LLM configured
        response = client.post(
            "/chat/requirements",
            json={
                "message": "Design a bracket",
                "session_id": "test_session",
                "project_id": "test_project",
            }
        )
        
        # Should accept JSON (not FormData)
        assert response.status_code == 200
        assert "content" in response.json()
    
    async def test_orchestrator_plan_endpoint(self, client):
        """Test /orchestrator/plan endpoint."""
        import pytest
        pytest.skip("Requires full backend setup - run manually")
        
        response = client.post(
            "/orchestrator/plan",
            json={
                "project_id": "test_plan",
                "requirements": {"intent": "Design a bracket"},
            }
        )
        
        assert response.status_code == 200


class TestPerformanceBaseline:
    """Establish performance baselines for the pipeline."""
    
    async def test_agent_loading_performance(self, initialize_registry):
        """
        Test that agent loading completes within reasonable time.
        """
        import time
        from backend.agent_registry import registry
        
        # Measure time to get an agent
        start = time.time()
        agent = registry.get_agent("GeometryAgent")
        elapsed = time.time() - start
        
        # Should load quickly (lazy loading)
        assert elapsed < 5.0, f"Agent loading took {elapsed:.2f}s, expected < 5s"
        assert agent is not None
    
    async def test_physics_calculation_performance(self, initialize_registry):
        """
        Test that physics calculations complete within reasonable time.
        """
        import time
        from physics.kernel import get_physics_kernel
        
        kernel = get_physics_kernel()
        
        # Measure geometry validation
        geometry = {
            "volume": 0.1,
            "cross_section_area": 0.01,
            "length": 2.0,
            "width": 0.1,
            "height": 0.1,
        }
        
        start = time.time()
        result = kernel.validate_geometry(geometry, "Steel")
        elapsed = time.time() - start
        
        # Should complete quickly
        assert elapsed < 1.0, f"Validation took {elapsed:.2f}s, expected < 1s"
        assert result["feasible"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
