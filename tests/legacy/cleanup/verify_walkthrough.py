
import requests
import time
import json

BASE_URL = "http://localhost:8000/api"

def print_result(step, response):
    status = "✅" if response.status_code == 200 else "❌"
    print(f"{status} {step}: {response.status_code}")
    if response.status_code != 200:
        print(f"   Error: {response.text}")
    else:
        try:
            print(f"   Response: {json.dumps(response.json(), indent=2)[:200]}...")
        except:
            print(f"   Response: {response.text[:200]}")

def run_walkthrough():
    print("🚀 Starting BRICK OS End-to-End Walkthrough...\n")

    # 1. Check System Health
    print_result("Health Check", requests.get(f"{BASE_URL}/health"))

    # 2. Add Hardware Pod (Recursive ISA)
    print("\n--- 🔨 Hardware Creation ---")
    pod_payload = {"name": "Test_Leg_Pod", "constraints": {"length": 1.2}}
    print_result("Create Pod", requests.post(f"{BASE_URL}/isa/create", json=pod_payload))
    
    # 3. View ISA Tree
    print("\n--- 🌳 ISA Tree Inspection ---")
    print_result("Get ISA Tree", requests.get(f"{BASE_URL}/isa/tree"))
    
    # 4. Run Physics Simulation
    print("\n--- 🏎️ Simulation Control ---")
    sim_payload = {"command": "START", "scenario": "thermal"}
    print_result("Start Simulation", requests.post(f"{BASE_URL}/simulation/control", json=sim_payload))
    time.sleep(1)
    sim_payload["command"] = "STOP"
    print_result("Stop Simulation", requests.post(f"{BASE_URL}/simulation/control", json=sim_payload))
    
    # 5. Export Project
    print("\n--- 📦 Export Data ---")
    export_payload = {"format": "step"}
    print_result("Export STEP", requests.post(f"{BASE_URL}/project/export", json=export_payload))
    
    # 6. Save Project Version
    print("\n--- 💾 Persistence ---")
    save_payload = {
        "data": {"manifest": {"version": "2.0"}, "geometry_tree": []},
        "filename": "walkthrough_test.brick"
    }
    print_result("Save Project", requests.post(f"{BASE_URL}/project/save", json=save_payload))

    # 7. Check Version History
    print("\n--- 📜 Version Control ---")
    print_result("Get History", requests.get(f"{BASE_URL}/version/history"))

    print("\n✅ Walkthrough Complete.")

if __name__ == "__main__":
    run_walkthrough()
