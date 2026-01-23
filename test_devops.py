
import sys
import os
import logging
sys.path.insert(0, 'backend')
from agents.devops_agent import DevOpsAgent

logging.basicConfig(level=logging.INFO)
print("--- TESTING DEVOPS AGENT ---")

agent = DevOpsAgent()

# 1. Health Check
print("\n[Action] Health Check")
health = agent.run({"action": "health_check"})
print(f"Health: {health}")

# 2. Dockerfile Audit (Create dummy file first)
print("\n[Action] Dockerfile Audit")
with open("test_dockerfile", "w") as f:
    f.write("FROM python:latest\nRUN pip install flask")

audit = agent.run({"action": "audit_dockerfile", "dockerfile_path": "test_dockerfile"})
print(f"Audit: {audit}")

# Cleanup
import os
if os.path.exists("test_dockerfile"):
    os.remove("test_dockerfile")

if health["status"] == "healthy" and audit["status"] == "audited":
    print("SUCCESS: DevOps agent operational.")
else:
    print("FAILURE: DevOps agent checks failed.")
