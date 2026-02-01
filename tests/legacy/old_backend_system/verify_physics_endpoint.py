import sys
import os
sys.path.append(os.getcwd())
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_physics_validation():
    print("Testing /api/physics/validate...")
    
    payload = {
        "geometry": {
            "type": "box",
            "dims": {"length": 1.0, "width": 0.1, "height": 0.1}
        },
        "material": "Aluminum",
        "loads": {"force": 1000}
    }
    
    try:
        response = client.post("/api/physics/validate", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print("Response:", json.dumps(data, indent=2))
            
            metrics = data.get("metrics", {})
            assert "mass_kg" in metrics, "Missing mass_kg"
            assert "deflection_mm" in metrics, "Missing deflection_mm"
            assert "fos" in metrics or "stress_MPa" in metrics, "Missing FOS/Stress"
            
            print("\nSUCCESS: Physics validation endpoint returns expected structure and metrics.")
        else:
            print(f"\nFAILURE: Status {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"\nERROR: {e}")

if __name__ == "__main__":
    test_physics_validation()
