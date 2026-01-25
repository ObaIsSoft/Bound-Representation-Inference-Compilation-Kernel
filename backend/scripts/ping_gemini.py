import sys
import os
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

from llm.gemini_provider import GeminiProvider

if __name__ == "__main__":
    print("Ping Gemini...", flush=True)
    try:
        p = GeminiProvider()
        res = p.generate("ping")
        print(f"Result: {res}", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)
