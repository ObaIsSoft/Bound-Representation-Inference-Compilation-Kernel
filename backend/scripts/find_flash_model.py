import google.generativeai as genai
import os
from dotenv import load_dotenv

# Robustly load backend/.env
cwd_env = os.path.join(os.getcwd(), "backend", ".env")
local_env = os.path.join(os.path.dirname(__file__), "../.env") # backend/scripts/../.env -> backend/.env

if os.path.exists(cwd_env):
    load_dotenv(cwd_env, override=True)
elif os.path.exists(local_env):
    load_dotenv(local_env, override=True)

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

print("Searching for 'flash' models...")
found = False
for m in genai.list_models():
    if 'flash' in m.name.lower():
        print(f"✅ Found: {m.name}")
        found = True
        
if not found:
    print("❌ No 'flash' models found.")
