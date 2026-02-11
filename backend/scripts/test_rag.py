import sys
import os
import asyncio
import logging

# Add backend to path so imports work
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from backend.agents.standards_agent import StandardsAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG_TEST")

def test_rag_flow():
    print("🚀 Testing RAG Infrastructure...")
    
    # 1. Instantiate Agent
    try:
        agent = StandardsAgent()
        print(f"✅ Agent Instantiated. Mode: {agent.mode}")
    except Exception as e:
        print(f"❌ Agent Init Failed: {e}")
        return

    # 2. Run Query
    query = "What is the safety factor for ground support equipment?"
    print(f"🔎 Querying: '{query}'")
    
    try:
        result = agent.run({"query": query})
        print(f"✅ Query Executed.")
        print(f"   Count: {result['count']}")
        print(f"   Logs: {result['logs']}")
        
        if agent.mode == "RAG" and result['count'] == 0:
            print("ℹ️  (Expected) No results found because Vector DB is empty.")
            print("    Please run 'ingest_pipeline.py' after adding PDFs.")
            
    except Exception as e:
        print(f"❌ Query Failed: {e}")

if __name__ == "__main__":
    test_rag_flow()
