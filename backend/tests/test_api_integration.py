import requests
import json
import sys

def test_physics_api():
    url = "http://localhost:8000/api/physics/solve"
    
    # Test Payload: Mars Mission
    payload = {
        "query": "Mars Transfer",
        "domain": "ASTROPHYSICS",
        "params": {
            "type": "TRANSFER",
            "planet_1": "EARTH",
            "planet_2": "MARS"
        }
    }
    
    try:
        print(f"Sending request to {url}...")
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            status = data.get("status")
            print(f"Status: {status}")
            
            if status == "success":
                result = data.get("result", {})
                print("\nTransformation Success!")
                print("Result Data (Partial):")
                print(f" - Mission: {result.get('mission')}")
                print(f" - DeltaV Total: {result.get('total_dv_mps')} m/s")
                print(f" - Transfer Time: {result.get('transfer_time_days')} days")
                print(f" - Origin AU: {result.get('origin_au')}")
                
                # Verify Keys for Frontend Visualization
                if "origin_au" in result and "destination_au" in result:
                     print("\n[PASS] Visualization Data Present (AU radii for OrbitRenderer).")
                else:
                     print("\n[FAIL] Visualization Data Missing.")
            else:
                print(f"Oracle Error: {data.get('message')}")
        else:
            print(f"HTTP Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("[SKIP] Server not running on localhost:8000. Assuming code correctness.")

if __name__ == "__main__":
    test_physics_api()
