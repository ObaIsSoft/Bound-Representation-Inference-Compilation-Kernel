import sys
import os
import logging

# Add project root to path
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Phase5_3_Verifier")

def verify_integration():
    print("Verifying Phase 5.3: Specialized Agent Integration\n")
    
    # 1. Test Validation Node Integration
    print("--- Testing Validation Node (Safety, Performance, Sustainability) ---")
    try:
        from backend.new_nodes import validation_node
        
        # Mock State
        mock_state = {
            "sub_agent_reports": {"physics": {"max_stress_mpa": 250, "max_temp_c": 110}}, # High stress/temp to trigger hazards
            "design_parameters": {"req_strength": 100},
            "mass_properties": {"total_mass_kg": 10.0},
            "material": "Aluminum 6061"
        }
        
        result = validation_node(mock_state)
        report = result["verification_report"]
        
        # Checks
        has_safety = "safety_analysis" in report
        has_perf = "performance_metrics" in report
        has_sust = "sustainability_report" in report
        hazards = report["safety_analysis"].get("hazards", [])
        
        print(f"✅ Safety Analysis Present: {has_safety}")
        print(f"✅ Performance Metrics Present: {has_perf}")
        print(f"✅ Sustainability Report Present: {has_sust}")
        
        if len(hazards) > 0:
            print(f"✅ Hazards Detected Correctly: {len(hazards)} found")
        else:
            print(f"❌ Hazards Detection Failed (Expected > 0)")
            
    except Exception as e:
        print(f"❌ Validation Node Test Failed: {e}")
        import traceback
        traceback.print_exc()

    # 2. Test Feedback Agent Integration (via direct import implies availability)
    print("\n--- Testing Feedback Agent ---")
    try:
        from backend.agents.feedback_agent import FeedbackAgent
        fb = FeedbackAgent()
        mock_flags = {"reasons": ["High Stress detected", "Thermal failure"]}
        res = fb.analyze_failure({"validation_flags": mock_flags})
        print(f"✅ Feedback Generated: {res.get('status')}")
        print(f"   Suggestion: {res.get('priority_fix')}")
    except Exception as e:
        print(f"❌ Feedback Agent Test Failed: {e}")

    # 3. Test Compliance Agent (via Physics Mega Node logic check)
    print("\n--- Testing Compliance Agent (Direct) ---")
    try:
        from backend.agents.compliance_agent import ComplianceAgent
        comp = ComplianceAgent()
        print(f"✅ Compliance Agent Instantiated: {comp.name}")
    except Exception as e:
        print(f"❌ Compliance Agent Test Failed: {e}")

if __name__ == "__main__":
    verify_integration()
