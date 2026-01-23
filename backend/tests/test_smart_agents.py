import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.conversational_agent import ConversationalAgent
from agents.designer_agent import DesignerAgent
from agents.mitigation_agent import MitigationAgent

def test_smart_agents():
    print("--- Testing Smart Agents ---")
    
    # 1. Conversational (Mock LLM)
    conv = ConversationalAgent() # Uses default Factory (Real LLM)
    resp = conv.run({"input_text": "Please design a high-speed drone"})
    print("\n[Conversational] Response:", resp.get("response"))
    print("[Conversational] Intent:", resp.get("intent"))
    print("[Conversational] Entities:", resp.get("entities"))
    
    # 2. Designer (Procedural)
    des = DesignerAgent()
    palette = des.run({"style": "cyberpunk", "harmony": "split", "base_color": "#00ff00"})
    print("\n[Designer] Palette:", palette.get("aesthetics"))
    
    # 3. Mitigation (Quantitative)
    mit = MitigationAgent()
    # Simulate a stress failure: Stress 300, Yield 200
    fix = mit.run({
        "errors": ["Yield Stress Exceeded (300 MPa > 200 MPa)"],
        "physics_data": {"max_stress_mpa": 300, "yield_strength_mpa": 200}
    })
    print("\n[Mitigation] Fixes:", fix.get("fixes"))

if __name__ == "__main__":
    test_smart_agents()
