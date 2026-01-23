
import sys
import os
import logging
sys.path.insert(0, 'backend')
from agents.document_agent import DocumentAgent
from llm.provider import LLMProvider


try:
    from dotenv import load_dotenv
    load_dotenv('backend/.env')
    from llm.factory import get_llm_provider
    
    print("--- ATTEMPTING TO USE GROQ provider ---")
    provider = get_llm_provider(preferred="groq")
    agent = DocumentAgent(llm_provider=provider)

except Exception as e:
    print(f"\nCRITICAL ERROR: {e}")
    print("Test failed because Real AI (Groq/OpenAI) is not available.")
    sys.exit(1)

params = {
    "project_name": "Test Drone",
    "environment": {"regime": "AERO"},
    "metrics": {"max_speed": "100 km/h"}
}

result = agent.run(params)

print(f"Status: {result.get('status')}")
print(f"Doc Title: {result.get('document', {}).get('title')}")
print(f"PDF Path: {result.get('document', {}).get('pdf_path')}")

if result.get("status") == "success":
    print("SUCCESS: Plan generated.")
else:
    print("FAILURE: Plan generation failed.")
