
import os
import sys
from huggingface_hub import HfApi, list_models

def ping_huggingface():
    print("--- PING HUGGING FACE ---")
    try:
        # Check if we can access the API
        api = HfApi()
        
        # Helper: Try to list a common model to verify connectivity
        print("Attempting to reach Hugging Face Hub...")
        # We search for a specific popular model to keep the payload small
        models = list(list_models(limit=1, search="bert-base-uncased"))
        
        if models:
            print(f"✅ SUCCESS: Connected to Hugging Face Hub.")
            print(f"   Found model: {models[0].modelId}")
        else:
            print("⚠️ WARNING: Connected, but no models found (unexpected).")

        # Check for Token (optional but good for inference)
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
        if token:
            print("✅ Hugging Face token detected.")
            # Try a simple inference check if token exists?
            # For now, just 'ping' connectivity is enough.
        else:
            print("ℹ️ No HF_TOKEN found in env. (Rate limits may be stricter).")

    except Exception as e:
        print(f"❌ FAILURE: Could not connect to Hugging Face.")
        print(f"   Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    ping_huggingface()
