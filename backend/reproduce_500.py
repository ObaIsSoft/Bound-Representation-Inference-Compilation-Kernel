
import asyncio
import os
import sys

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()


async def reproduce():
    print("Attempting to reproduce 500 error...")
    
    try:
        # Import what main.py imports
        from agents.conversational_agent import ConversationalAgent
        from agents.geometry_estimator import GeometryEstimator
        from agents.cost_agent import CostAgent
        from agents.environment_agent import EnvironmentAgent
        
        print("Instantiating agents...")
        # mimic main.py global instantiation
        conversational_agent = ConversationalAgent()
        geom_estimator = GeometryEstimator()
        cost_agent = CostAgent()
        env_agent = EnvironmentAgent()
        
        print("Agents instantiated successfully.")
        
        # mimic endpoint logic
        print("Running ConvAgent chat...")
        res = await conversational_agent.chat(
            user_input="I want to design a drone",
            history=[],
            current_intent="",
            session_id="test-repro"
        )
        print(f"ConvAgent Success: {res}")
        
        updated_intent = f"I want to design a drone"
        print("Running Environment Agent...")
        env_result = env_agent.detect_environment(updated_intent)
        print(f"Env Success: {env_result}")
        
        print("Running Geometry Estimator...") 
        design_params = {"max_dim": 1.0}
        geom_result = geom_estimator.estimate(updated_intent, design_params)
        print(f"Geom Success: {geom_result}")
        
        print("Running Cost Agent...")
        cost_result = cost_agent.estimate_cost(geom_result)
        print(f"Cost Success: {cost_result}")
        
    except Exception as e:
        print(f"Caught exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(reproduce())
