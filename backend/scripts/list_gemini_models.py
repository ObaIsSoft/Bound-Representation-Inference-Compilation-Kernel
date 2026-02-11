import google.generativeai as genai
import os
from dotenv import load_dotenv

# Robustly load backend/.env
cwd_env = os.path.join(os.getcwd(), "backend", ".env")
local_env = os.path.join(os.path.dirname(__file__), "../../.env")

if os.path.exists(cwd_env):
    load_dotenv(cwd_env, override=True)
elif os.path.exists(local_env):
    load_dotenv(local_env, override=True)

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

print("Listing available Gemini models...")
for m in genai.list_models():
    if 'embedContent' in m.supported_generation_methods:
        print(f"✅ Embedding Model: {m.name}")
    elif 'generateContent' in m.supported_generation_methods:
        print(f"✨ Generation Model: {m.name}")
    else:
        print(f"❓ Other Model: {m.name}")
