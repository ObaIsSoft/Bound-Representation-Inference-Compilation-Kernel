from backend.agents.environment_agent import EnvironmentAgent
import json

def test_intent(agent, intent):
    print(f"\n--- INTENT: '{intent}' ---")
    result = agent.run(intent)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    agent = EnvironmentAgent()
    
    intents = [
        "autonomous drone on titan",
        "submarine in europa ocean",
        "orbiter around jupiter",
        "rover for mars surface",
        "solar glider on venus"
    ]
    
    for i in intents:
        test_intent(agent, i)
