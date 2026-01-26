
import os
from dotenv import load_dotenv

def inspect_keys():
    print("--- KEY INSPECTION ---")
    load_dotenv(override=True)
    
    keys = {
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "HF_TOKEN": os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
    }
    
    for name, value in keys.items():
        if not value:
            print(f"❌ {name}: MISSING")
            continue
            
        print(f"🔍 {name}:")
        print(f"   Length: {len(value)}")
        print(f"   Starts with: '{value[:4]}'")
        print(f"   Ends with:   '{value[-4:]}'")
        
        problems = []
        if value.strip() != value:
            problems.append("Has leading/trailing whitespace")
        if value.startswith('"') or value.startswith("'"):
            problems.append("Cannot have quotes at start")
        if " " in value:
            problems.append("Contains spaces inside")
            
        if problems:
            print(f"   ⚠️ POTENTIAL ISSUES: {', '.join(problems)}")
        else:
            print("   ✅ Format looks clean (no whitespace/quotes)")

if __name__ == "__main__":
    inspect_keys()
