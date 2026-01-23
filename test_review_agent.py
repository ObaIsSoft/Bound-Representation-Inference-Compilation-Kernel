
import sys
import os
import logging
sys.path.insert(0, 'backend')
from agents.review_agent import ReviewAgent
from llm.factory import get_llm_provider

logging.basicConfig(level=logging.INFO)

try:
    from dotenv import load_dotenv
    load_dotenv('backend/.env')
    
    print("--- ATTEMPTING TO USE GROQ provider ---")
    provider = get_llm_provider(preferred="groq")
    agent = ReviewAgent(llm_provider=provider)
    
    print("--- TESTING REVIEW AGENT (CODE REVIEW) ---")
    diff = """
    def login(user, password):
        # TODO: Fix this insecure hardcode
        if password == "admin123": 
            return True
        return False
    """
    
    result = agent.review_code(diff, context="Authentication Module")
    
    print(f"Status: {result.get('status')}")
    print(f"Approved: {result.get('approved')}")
    print(f"Issues: {result.get('issues')}")
    print(f"Score: {result.get('security_score')}")
    
    if result.get("issues"):
        print("SUCCESS: Vulnerability detected.")
    else:
        print("FAILURE: No issues found (False Negative).")

except Exception as e:
    print(f"\nCRITICAL ERROR: {e}")
    sys.exit(1)
