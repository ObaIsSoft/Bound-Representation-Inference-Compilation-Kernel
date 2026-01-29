import requests
import json

def test_compliance():
    url = "http://localhost:8000/api/compliance/check"
    
    # 1. Test Aerial Compliance (Should pass FAA Part 107)
    payload_pass = {
        "regime": "AERIAL",
        "design_params": {
            "mass_kg": 2.8,
            "is_fcc_certified": True,
            "emc_test_passed": True
        }
    }
    
    print("\n--- Testing AERIAL Compliance (Pass Case) ---")
    resp = requests.post(url, json=payload_pass)
    print(f"Status: {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))
    
    # 2. Test Aerial Compliance (Should fail FAA Part 107)
    payload_fail = {
        "regime": "AERIAL",
        "design_params": {
            "mass_kg": 28.5,
            "is_fcc_certified": False,
            "emc_test_passed": False
        }
    }
    
    print("\n--- Testing AERIAL Compliance (Fail Case) ---")
    resp = requests.post(url, json=payload_fail)
    print(f"Status: {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))

    # 3. Test Terrestrial Compliance (Unibike)
    payload_unibike = {
        "regime": "TERRESTRIAL",
        "design_params": {
            "static_stability_factor": 1.5
        }
    }
    
    print("\n--- Testing TERRESTRIAL Compliance (Unibike) ---")
    resp = requests.post(url, json=payload_unibike)
    print(f"Status: {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))

if __name__ == "__main__":
    # Note: Ensure backend is running before executing this
    try:
        test_compliance()
    except Exception as e:
        print(f"Error: {e}")
