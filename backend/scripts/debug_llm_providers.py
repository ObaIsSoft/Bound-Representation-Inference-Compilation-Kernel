import sys
import os
from dotenv import load_dotenv

# Add parent dir to path so we can import 'llm'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables explicitly
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(env_path)

from llm.groq_provider import GroqProvider
from llm.gemini_provider import GeminiProvider
from llm.ollama_provider import OllamaProvider

def test_provider(name, provider):
    print(f"\n--- Testing {name} ---")
    try:
        response = provider.generate("ping", system_prompt="Reply with 'pong'")
        print(f"Response: {response}")
        if "pong" in response.lower() or "ping" in response.lower():
            print(f"✅ {name} Initialized and Responded")
        else:
            print(f"⚠️ {name} Responded with unexpected content: {response}")
    except Exception as e:
        print(f"❌ {name} Failed: {e}")

if __name__ == "__main__":
    print("Debug: AI Service Connectivity Check")
    
    # GROQ
    try:
        groq = GroqProvider()
        test_provider("Groq", groq)
    except Exception as e:
        print(f"❌ Groq Init Failed: {e}")

    # GEMINI
    try:
        gemini = GeminiProvider()
        test_provider("Gemini", gemini)
    except Exception as e:
         print(f"❌ Gemini Init Failed: {e}")

    # OLLAMA (Using 'llama2' which is installed, overriding default 'llama3.2')
    try:
        ollama = OllamaProvider(model_name="llama2")
        test_provider("Ollama", ollama)
    except Exception as e:
         print(f"❌ Ollama Init Failed: {e}")
