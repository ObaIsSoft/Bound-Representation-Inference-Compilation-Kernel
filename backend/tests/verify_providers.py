
import sys
import os
import logging
# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from dotenv import load_dotenv
load_dotenv(override=True)

from llm.groq_provider import GroqProvider
from llm.huggingface_provider import HuggingFaceProvider

def test_providers():
    print("--- DIRECT PROVIDER VERIFICATION ---")
    
    # 1. GROQ
    print("\n[GROQ] Initializing...")
    try:
        groq = GroqProvider()
        # Verify either client exists OR rest mode is active
        if not groq.client and not groq.use_rest:
            print("❌ Groq Client failed to initialize (No SDK + No REST fallback).")
        else:
            print(f"✅ Groq initialized. (Mode: {'SDK' if groq.client else 'REST'})")
            print("   Testing generation...")
            response = groq.generate("What is 2+2? Answer with just the number.")
            print(f"   Response: {response}")
            if "4" in response:
                print("✅ Groq Generation SUCCESS")
            else:
                print(f"⚠️ Groq Generation Unexpected: {response}")
    except Exception as e:
        print(f"❌ Groq Exception: {e}")

    # 2. HUGGING FACE
    print("\n[HUGGING FACE] Initializing...")
    try:
        hf = HuggingFaceProvider()
        if not hf.client:
            print("❌ HF Client failed to initialize.")
        else:
            print("✅ HF Client initialized.")
            print("   Testing generation...")
            response = hf.generate("What is 2+2? Answer with just the number.")
            print(f"   Response: {response}")
            if response and len(response) > 0:
                print("✅ HF Generation SUCCESS")
            else:
                print("⚠️ HF Generation Output Empty")

    except Exception as e:
        print(f"❌ HF Exception: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_providers()
