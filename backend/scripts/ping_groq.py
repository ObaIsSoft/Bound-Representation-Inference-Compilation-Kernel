import sys
import os
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

from llm.groq_provider import GroqProvider

if __name__ == "__main__":
    print("Ping Groq...", flush=True)
    try:
        p = GroqProvider()
        res = p.generate("ping")
        print(f"Result: {res}", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)
