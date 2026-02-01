"""
Test which parameter name works for MP API fields
"""
import os
import requests
import json

# Try to look for key in environment or hardcoded (for test only)
# Assuming key is available since previous scripts ran
API_KEY = os.getenv("MATERIALS_PROJECT_API_KEY")
if not API_KEY:
    # Try looking in materials_api.py? No, just ask user if this fails.
    # But previous run debug_mp_keys.py worked so env var IS usually set in the session/context
    pass

if not API_KEY:
    print("No API Key found")
    exit(1)

base_url = "https://api.materialsproject.org/materials/core/"
headers = {"X-API-KEY": API_KEY}

# Test 1: using _fields
print("Test 1: _fields")
params = {
    "formula": "Fe2O3",
    "_limit": 1,
    "_fields": "formula_pretty,material_id,density,elasticity,thermo,structure,energy_above_hull"
}
try:
    r = requests.get(base_url, headers=headers, params=params, timeout=10)
    print(r.status_code)
    print(json.dumps(r.json()['data'][0], indent=2))
except Exception as e:
    print("Error:", e)

# Test 2: using fields (no underscore) - API v2 uses _fields usually but let's check
print("\nTest 2: fields")
params["fields"] = params.pop("_fields")
try:
    r = requests.get(base_url, headers=headers, params=params, timeout=10)
    print(r.status_code)
    try:
        print(json.dumps(r.json()['data'][0], indent=2))
    except:
        print(r.text)
except Exception as e:
    print("Error:", e)
