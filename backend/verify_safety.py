
import sys
import os

# Mimic running from backend root
sys.path.append(os.getcwd())

try:
    from agents.safety_agent import SafetyAgent
    print("✅ SafetyAgent imported successfully")
except Exception as e:
    print(f"❌ SafetyAgent import failed: {e}")
