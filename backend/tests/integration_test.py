import requests
import json
import sys
import time

BASE_URL = "http://localhost:8000/api"

def log(msg, status="INFO"):
    colors = {
        "INFO": "\033[94m",
        "PASS": "\033[92m",
        "FAIL": "\033[91m",
        "RESET": "\033[0m"
    }
    print(f"{colors.get(status, '')}[{status}] {msg}{colors['RESET']}")

def test_health():
    try:
        res = requests.get(f"{BASE_URL}/health")
        if res.status_code == 200:
            log("Health check passed", "PASS")
            return True
        else:
            log(f"Health check failed: {res.status_code}", "FAIL")
            return False
    except Exception as e:
        log(f"Health check connection error: {e}", "FAIL")
        return False

def test_agent_discovery():
    try:
        res = requests.get(f"{BASE_URL}/agents")
        data = res.json()
        agents = data.get("agents", [])
        if "thermal" in agents and "cost" in agents:
             log(f"Discovered {len(agents)} agents: {', '.join(agents[:5])}...", "PASS")
             return True
        else:
             log("Agent discovery incomplete", "FAIL")
             return False
    except Exception as e:
        log(f"Agent discovery error: {e}", "FAIL")
        return False

def test_agent_execution(agent_name, payload):
    try:
        log(f"Testing {agent_name} agent...", "INFO")
        res = requests.post(f"{BASE_URL}/agents/{agent_name}/run", json=payload)
        if res.status_code == 200:
            result = res.json()
            if result.get("status") == "success":
                log(f"{agent_name} execution success", "PASS")
                # print(json.dumps(result['result'], indent=2))
                return True
            else:
                log(f"{agent_name} reported failure", "FAIL")
                return False
        else:
            log(f"{agent_name} HTTP {res.status_code}: {res.text}", "FAIL")
            return False
    except Exception as e:
        log(f"{agent_name} error: {e}", "FAIL")
        return False

def run_tests():
    log("Starting BRICK OS End-to-End Test Suite", "INFO")
    
    if not test_health():
        sys.exit(1)
        
    if not test_agent_discovery():
        sys.exit(1)
        
    # 1. Thermal Test
    test_agent_execution("thermal", {
        "power_watts": 100,
        "surface_area": 0.1,
        "ambient_temp": 25
    })
    
    # 2. Structural Test
    test_agent_execution("structural", {
        "mass_kg": 5.0,
        "g_loading": 9.0,
        "cross_section_mm2": 50.0
    })
    
    # 3. Cost Test
    test_agent_execution("cost", {
        "mass_kg": 2.5,
        "material_cost_per_kg": 15.0,
        "processing_time_hr": 4.0
    })

    log("All tests completed.", "INFO")

if __name__ == "__main__":
    run_tests()
