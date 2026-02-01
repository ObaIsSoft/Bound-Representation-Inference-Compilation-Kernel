
import sys
import os
sys.path.insert(0, ".")

def test_diagnostic_agent():
    print("\n--- Testing DiagnosticAgent ---")
    try:
        from backend.agents.diagnostic_agent import DiagnosticAgent
        agent = DiagnosticAgent()
        
        # Test 1: Neural Prediction
        logs = [
            "Connection timed out",
            "Retrying request to api.materialsproject.org",
            "Failed to connect after 3 attempts"
        ]
        result = agent.run({"logs": logs, "error_stream": []})
        print(f"Prediction: {result['diagnosis']} (Prob: {result['root_cause_probability']:.2f})")
        
        # Test 2: Evolution
        training_data = [
            (["Out of memory", "Java heap space"], 1), # Memory = Class 1
            (["Syntax error line 10", "Unexpected token"], 2) # Logic = Class 2
        ]
        evo_result = agent.evolve(training_data)
        print(f"Evolution: {evo_result}")
        
    except Exception as e:
        print(f"DiagnosticAgent Failed: {e}")

def test_verification_agent():
    print("\n--- Testing VerificationAgent ---")
    try:
        from backend.agents.verification_agent import VerificationAgent
        agent = VerificationAgent()
        
        # Test 1: Safe G-Code
        gcode_safe = [
             {"op": "RAPID", "path": [[0,0,10], [10,10,10]]}, # Above stock
             {"op": "CUT", "path": [[10,10,10], [10,10,0]]}
        ]
        res_safe = agent.verify_safety(gcode_safe, stock_dims=[100,100,20], material="aluminum")
        print(f"Safe Check: {res_safe['verified']}")
        
        # Test 2: Unsafe G-Code (Crash)
        gcode_unsafe = [
             {"op": "RAPID", "path": [[0,0,10], [0,0,-5]]} # Rapid plunge into material!
        ]
        res_unsafe = agent.verify_safety(gcode_unsafe, stock_dims=[100,100,20], material="aluminum")
        print(f"Unsafe Check: {res_unsafe['verified']}")
        
        if not res_unsafe['verified']:
             if 'error' in res_unsafe:
                 print(f"Error: {res_unsafe['error']}")
             elif 'collisions' in res_unsafe:
                 print(f"Collisions: {res_unsafe['collisions']}")
             else:
                 print(f"Unknown verification failure: {res_unsafe}")
             
    except Exception as e:
        print(f"VerificationAgent Failed: {e}")

if __name__ == "__main__":
    test_diagnostic_agent()
    test_verification_agent()
