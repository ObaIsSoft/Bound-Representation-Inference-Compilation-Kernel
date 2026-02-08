import requests
import json

url = "http://localhost:8000/api/chat/requirements"

# Simulate the FormData payload
payload = {
    "message": "I want to design a drone",
    "user_intent": "design a drone",
    "conversation_history": "[]",
    "mode": "requirements_gathering",
    "ai_model": "groq",
    "session_id": "debug-session-123"
}

try:
    print(f"Sending POST to {url}...")
    response = requests.post(url, data=payload)
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Response JSON:")
        print(json.dumps(response.json(), indent=2))
    else:
        print("Error Response:")
        print(response.text)

except Exception as e:
    print(f"Request failed: {e}")
