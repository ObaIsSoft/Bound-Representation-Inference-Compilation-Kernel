from agents.generic_agent import GenericAgent
from orchestrator import get_agent_registry

def test_generic_agent_registration():
    print("Testing Agent Registry...")
    registry = get_agent_registry()
    
    # Test a few expected generic agents
    test_agents = ["surrogate_physics", "design_exploration", "doctor", "pvc"]
    
    for agent_id in test_agents:
        if agent_id in registry:
            agent = registry[agent_id]
            print(f"✅ Found {agent_id}: {agent.name} ({agent.role})")
            
            # Run it
            result = agent.run({"test": "param"})
            if result["status"] == "success":
                print(f"   Run Success: {result['logs'][-1]}")
            else:
                print(f"   Run Failed!")
        else:
            print(f"❌ Missing {agent_id}")

if __name__ == "__main__":
    test_generic_agent_registration()
