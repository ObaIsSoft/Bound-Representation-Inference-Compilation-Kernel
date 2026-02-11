import google.generativeai as genai
import os
from dotenv import load_dotenv

# Robustly load backend/.env
cwd_env = os.path.join(os.getcwd(), "backend", ".env")
local_env = os.path.join(os.path.dirname(__file__), "../.env")

if os.path.exists(cwd_env):
    load_dotenv(cwd_env, override=True)
elif os.path.exists(local_env):
    load_dotenv(local_env, override=True)

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

model_name = "models/text-embedding-004"
print(f"Testing Model: {model_name}")

try:
    result = genai.embed_content(
        model=model_name,
        content="Hello world",
        task_type="retrieval_document"
    )
    vec = result['embedding']
    print(f"✅ Success! Dimension: {len(vec)}")
except Exception as e:
    print(f"❌ Error: {e}")
