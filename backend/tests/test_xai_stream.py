
import pytest
from fastapi.testclient import TestClient
from main import app
from xai_stream import inject_thought, clear_thoughts, get_thoughts

client = TestClient(app)

def test_thought_injection_and_retrieval():
    # 1. Clear buffer
    clear_thoughts()
    assert len(get_thoughts(clear=False)) == 0
    
    # 2. Inject thought
    inject_thought("TestAgent", "This is a test thought.")
    
    # 3. Verify via function
    thoughts = get_thoughts(clear=False)
    assert len(thoughts) == 1
    assert thoughts[0]["agent"] == "TestAgent"
    assert thoughts[0]["text"] == "This is a test thought."
    
    # 4. Verify via API (Destructive read)
    response = client.get("/api/agents/thoughts")
    assert response.status_code == 200
    data = response.json()
    assert len(data["thoughts"]) == 1
    assert data["thoughts"][0]["agent"] == "TestAgent"
    
    # 5. Verify buffer is cleared
    thoughts_after = get_thoughts(clear=False)
    assert len(thoughts_after) == 0

def test_thought_buffer_limit():
    clear_thoughts()
    # Inject 60 thoughts (limit is 50)
    for i in range(60):
        inject_thought("TestAgent", f"Thought {i}")
        
    thoughts = get_thoughts(clear=False)
    assert len(thoughts) == 50
    assert thoughts[0]["text"] == "Thought 10" # deque discards oldest
    assert thoughts[-1]["text"] == "Thought 59"
