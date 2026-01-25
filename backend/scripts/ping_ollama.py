import sys
import os
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

from llm.ollama_provider import OllamaProvider

if __name__ == "__main__":
    print("Ping Ollama...", flush=True)
    try:
        # Use llama2 as we confirmed it exists
        p = OllamaProvider(model_name="llama2")
        res = p.generate("ping")
        print(f"Result: {res}", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)
