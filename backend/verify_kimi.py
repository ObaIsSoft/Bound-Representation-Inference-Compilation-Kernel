import os
import sys
import json
import asyncio

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm.factory import get_llm_provider
from llm.kimi_provider import KimiProvider

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import logging
logging.basicConfig(level=logging.INFO)

async def verify_kimi():
    print("--- Kimi AI Provider Verification ---")
    
    # Check for API keys
    raw_key = os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY")
    if not raw_key:
        print("ERROR: KIMI_API_KEY or MOONSHOT_API_KEY not found in environment.")
        return

    variations = [
        ("Full Key", raw_key),
        ("Remove 'kimi-'", raw_key.replace("kimi-", "")),
        ("Just Alphanumeric", raw_key.replace("sk-kimi-", ""))
    ]

    for label, key in variations:
        print(f"\n--- Testing Variation: {label} ---")
        for base_url in ["https://api.moonshot.ai/v1", "https://api.moonshot.cn/v1"]:
            print(f"  Trying Base URL: {base_url}")
            try:
                from openai import OpenAI
                client = OpenAI(api_key=key.strip(), base_url=base_url)
                
                models = client.models.list()
                print(f"  SUCCESS with {label} @ {base_url}!")
                print(f"  Available models example: {[m.id for m in models.data[:3]]}")
                return # Stop if one works
            except Exception as e:
                print(f"  FAILED with {label} @ {base_url}: {e}")

if __name__ == "__main__":
    # Kimi provider is sync in its current implementation (following the LLMProvider base class)
    # But agents wrap it in async if needed.
    
    # Check if KIMI_API_KEY is set in current environment
    # Note: The user said "NO MOCK, USE REAL. CALLSS", implying they expect me to use real ones.
    # I don't see KIMI_API_KEY in the environment snapshot, so this might fail unless I ask for it or find it.
    
    asyncio.run(verify_kimi())
