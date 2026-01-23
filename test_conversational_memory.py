
import sys
import os
import logging
sys.path.insert(0, 'backend')
from agents.conversational_agent import ConversationalAgent

logging.basicConfig(level=logging.INFO)

print("--- TESTING CONVERSATIONAL MEMORY ---")

# Step 1: Clean slate
import os
if os.path.exists("data/long_term_memory.json"):
    os.remove("data/long_term_memory.json")

# Define Stub Provider to bypass Factory check
class StubProvider:
    def generate(self, *args, **kwargs): return ""
    def generate_json(self, *args, **kwargs): return {}

# Step 2: First Run (Inject details)
print("\n[Session 1] Injecting Mission...")
agent1 = ConversationalAgent(provider=StubProvider())
# Use a mock "LLM" effect by manually forcing context update if LLM fails (since we might lack keys)
# Ideally we run this with real LLM, but for stability we can verify the memory IO logic regardless.
agent1.discovery.context['mission'] = "Mars Rover" 
agent1.discovery._save_memory()
print("Saved mission: Mars Rover")

# Step 3: Second Run (New Instance)
print("\n[Session 2] Reloading Agent from disk...")
agent2 = ConversationalAgent(provider=StubProvider())
mission = agent2.discovery.context.get('mission')

print(f"Loaded Mission: {mission}")

if mission == "Mars Rover":
    print("SUCCESS: Memory persisted across instances.")
else:
    print(f"FAILURE: Memory mismatch (Got: {mission})")
