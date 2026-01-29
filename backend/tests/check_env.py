
import os
from dotenv import load_dotenv

def check_keys():
    print("--- ENV CHECK ---")
    # Force reload of .env
    load_dotenv(override=True)
    
    groq = os.getenv("GROQ_API_KEY")
    hf = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
    
    print(f"GROQ_API_KEY: {'[FOUND]' if groq else '[MISSING]'}")
    print(f"HF_TOKEN: {'[FOUND]' if hf else '[MISSING]'}")
        
    if not groq and not hf:
        print("❌ Both keys missing. Please check .env file.")
    else:
        print("✅ At least one key found.")

if __name__ == "__main__":
    check_keys()
