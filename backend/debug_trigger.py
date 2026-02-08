
import requests

url = "http://localhost:8000/api/orchestrator/plan"
data = {
    "user_intent": "design a drone",
    "project_id": "debug_test"
}

try:
    response = requests.post(url, data=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
