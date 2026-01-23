
import sys
import os
import logging
sys.path.insert(0, 'backend')
from agents.asset_sourcing_agent import AssetSourcingAgent

logging.basicConfig(level=logging.INFO)

print("--- TESTING ASSET SOURCING AGENT ---")
# Ensure we try to load env
try:
    from dotenv import load_dotenv
    load_dotenv('backend/.env')
except:
    pass

agent = AssetSourcingAgent()

# Query "Drone" -> Should trigger LLM smart breakdown
print("\n[Query] 'Drone'")
result = agent.run({"query": "Drone"})

print(f"Count: {result.get('count')}")
for log in result.get('logs', []):
    print(log)

if result.get('count') > 0:
    print("Assets:")
    for a in result.get('assets'):
        print(f" - {a.get('name')} ({a.get('category')})")
        
    print("SUCCESS: Intelligent sourcing triggered.")
else:
    print("RESULT: No assets found (likely API failure or Key missing).")
