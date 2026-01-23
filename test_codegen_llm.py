
import sys
import os
import logging
sys.path.insert(0, 'backend')
from agents.codegen_agent import CodegenAgent
from llm.mock_dreamer import MockDreamer
from llm.provider import LLMProvider

# Setup Mock Provider to avoid live calls in test if key missing,
# but try to use real one if env var exists.
class TestProvider(LLMProvider):
    def chat(self, messages):
        return """
import time
def read_sensor():
    return 42

while True:
    print(f"Sensor Value: {read_sensor()}")
    time.sleep(1)
"""

# Try real provider if available
try:
    from dotenv import load_dotenv
    load_dotenv('backend/.env')
    from llm.factory import get_llm_provider
    provider = get_llm_provider(preferred="groq")
    print(f"Using Provider: {provider.__class__.__name__}")
except:
    print("Using Test Mock Provider")
    provider = TestProvider()

logging.basicConfig(level=logging.INFO)
agent = CodegenAgent(provider=provider)

print("--- TESTING CODEGEN AGENT ---")
reqs = "Write a Python script to read a temperature sensor every 1 second and print the value."
result = agent.generate_script(reqs, language="python")

print(f"Status: {result.get('status')}")
print(f"Code Preview:\n{result.get('code')}")
print(f"Validation: {result.get('validation')}")

if result.get("status") == "success":
    print("SUCCESS: Code generated and validated.")
else:
    print("FAILURE: Generation failed.")
