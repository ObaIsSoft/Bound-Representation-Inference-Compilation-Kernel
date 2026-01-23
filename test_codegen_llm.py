
import sys
import os
import logging
sys.path.insert(0, 'backend')
from agents.codegen_agent import CodegenAgent

from llm.provider import LLMProvider


import sys
import os
import logging
sys.path.insert(0, 'backend')
from agents.codegen_agent import CodegenAgent
from llm.provider import LLMProvider

logging.basicConfig(level=logging.INFO)

try:
    from dotenv import load_dotenv
    load_dotenv('backend/.env')
    from llm.factory import get_llm_provider
    
    # User Request: Use Groq
    print("--- ATTEMPTING TO USE GROQ provider ---")
    provider = get_llm_provider(preferred="groq")
    print(f"Using Provider: {provider.__class__.__name__}")
    
    agent = CodegenAgent(provider=provider)

    print("--- TESTING CODEGEN AGENT (REAL AI) ---")
    reqs = "Write a Python script to read a temperature sensor every 1 second and print the value."
    
    # This call will actually hit the LLM API
    result = agent.generate_script(reqs, language="python")

    print(f"Status: {result.get('status')}")
    print(f"Code Preview:\n{result.get('code')}")
    print(f"Validation: {result.get('validation')}")

    if result.get("status") == "success":
        print("SUCCESS: Code generated and validated.")
    else:
        print("FAILURE: Generation failed.")

except Exception as e:
    print(f"\nCRITICAL ERROR: {e}")
    print("Test failed because Real AI (Groq/OpenAI) is not available or configured.")
    print("MockDreamer has been removed, so we cannot fallback.")
    sys.exit(1)

