
import logging
from typing import Dict, Any
from unittest.mock import MagicMock, patch
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

# Mock agents that we don't want to actually instantiate or that might be missing
mock_registry = MagicMock()
mock_forensic_agent = MagicMock()
mock_registry.get_agent.return_value = mock_forensic_agent

# Mock the analyze_failure method result
mock_forensic_agent.analyze_failure.return_value = {
    "root_causes": ["Simulated Structural Failure"],
    "recommendations": ["Increase thickness"],
    "severity": "CRITICAL"
}

# Patch modules
with patch.dict(sys.modules, {
    "agent_registry": MagicMock(registry=mock_registry),
    "agents.environment_agent": MagicMock(),
    "agents.critics.SurrogateCritic": MagicMock(),
    "agents.critics.PhysicsCritic": MagicMock(),
    "agents.stt_agent": MagicMock(),
    "agents.geometry_agent": MagicMock(),
    "agents.physics_agent_v2": MagicMock(),
    "agents.explainable_agent": MagicMock(),
    "ldp_kernel": MagicMock(),
    "ares": MagicMock(), 
    "llm.factory": MagicMock(), 
}):
    # Import the node AFTER patching
    from orchestrator import forensic_node, check_validation, END

    def test_forensic_node_wiring():
        print("--- Testing Forensic Node & Wiring ---")
        
        # 1. Test check_validation logic for routing to forensic_node
        print("[1] Testing Conditional Edge (check_validation)...")
        
        # Case A: Validation Failed, Count < 3 -> Should go to forensic_node
        state_fail = {
            "validation_flags": {"physics_safe": False, "reasons": ["Stress too high"]},
            "iteration_count": 1
        }
        next_node = check_validation(state_fail)
        print(f"   Input: Unsafe, Count 1 -> Output: {next_node}")
        assert next_node == "forensic_node", f"Expected 'forensic_node', got '{next_node}'"
        
        # Case B: Validation Passed -> Asset Sourcing (based on code, or END if simplified)
        # The code usually sends valid -> asset_sourcing, but let's check strict logic
        state_pass = {
            "validation_flags": {"physics_safe": True},
            "iteration_count": 1
        }
        next_node_pass = check_validation(state_pass)
        print(f"   Input: Safe -> Output: {next_node_pass}")
        assert next_node_pass == END, f"Expected END (or specific node), got '{next_node_pass}'"
        
        # Case C: Max Retries Exceeded -> END
        state_max = {
            "validation_flags": {"physics_safe": False},
            "iteration_count": 3
        }
        next_node_max = check_validation(state_max)
        print(f"   Input: Unsafe, Count 3 -> Output: {next_node_max}")
        assert next_node_max == END, f"Expected END, got '{next_node_max}'"
        
        print("✅ Conditional Edge Logic Verified.")

        # 2. Test forensic_node execution
        print("\n[2] Testing Forensic Node Execution...")
        
        state_execution = {
            "validation_flags": {"physics_safe": False, "reasons": ["Overheating"]},
            "physics_predictions": {"temp": 200},
            "design_parameters": {"material": "Plastic"},
            "messages": []
        }
        
        # Run node
        result = forensic_node(state_execution)
        
        # Verify Agent Interaction
        mock_registry.get_agent.assert_called_with("ForensicAgent")
        mock_forensic_agent.analyze_failure.assert_called_once()
        
        # Verify State Update
        print(f"   Result Keys: {result.keys()}")
        assert "forensic_analysis" in result
        assert "validation_flags" in result
        
        # Verify Analysis Injection
        root_causes = result["validation_flags"]["reasons"]
        print(f"   Updated Reasons: {root_causes}")
        assert "Simulated Structural Failure" in root_causes
        
        print("✅ Forensic Node Execution Verified.")
        
    if __name__ == "__main__":
        try:
            test_forensic_node_wiring()
            print("\n🎉 ALL TESTS PASSED")
        except AssertionError as e:
            print(f"\n❌ TEST FAILED: {e}")
            exit(1)
        except Exception as e:
            print(f"\n❌ RUNTIME ERROR: {e}")
            exit(1)
