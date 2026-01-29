"""
LAYER 1: Test raw Materials Project API directly
This is the LOWEST level - if this hangs, the issue is with MP API or network
"""

import os
import requests
import time
import sys

# Try to load .env if dotenv is installed (common in dev environments)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY = os.getenv("MATERIALS_PROJECT_API_KEY")

if not API_KEY:
    print("ERROR: No API key found in environment!")
    # Just in case, check if it's hardcoded anywhere else or provided as arg, 
    # but strictly we rely on env var or the user needs to provide it.
    sys.exit(1)

# Test 1: Simple ping to Materials Project
print("\n" + "="*70)
print("TEST 1: Ping Materials Project API")
print("="*70)

base_url = "https://api.materialsproject.org"
headers = {"X-API-KEY": API_KEY}

print(f"Endpoint: {base_url}/materials/core/")
print("Sending request with timeout=10...")

start = time.time()
try:
    response = requests.get(
        f"{base_url}/materials/core/",
        headers=headers,
        params={"_limit": 1, "_fields": "material_id"},
        timeout=10
    )
    elapsed = time.time() - start
    print(f"✓ Response received in {elapsed:.2f}s")
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        print(f"  Data: {str(response.json())[:100]}...")
    else:
        print(f"  Error Text: {response.text}")
except requests.exceptions.Timeout:
    elapsed = time.time() - start
    print(f"✗ TIMEOUT after {elapsed:.2f}s")
except Exception as e:
    elapsed = time.time() - start
    print(f"✗ ERROR after {elapsed:.2f}s: {e}")

# Test 2: Query for Fe2O3 specifically
print("\n" + "="*70)
print("TEST 2: Query Fe2O3 from Materials Project")
print("="*70)

print("Sending request for formula=Fe2O3...")
start = time.time()
try:
    response = requests.get(
        f"{base_url}/materials/core/",
        headers=headers,
        params={
            "formula": "Fe2O3",
            "_limit": 10,
            "_fields": "formula_pretty,material_id,density,elasticity"
        },
        timeout=10
    )
    elapsed = time.time() - start
    print(f"✓ Response received in {elapsed:.2f}s")
    print(f"  Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"  Results: {len(data.get('data', []))} materials found")
        if data.get('data'):
            print(f"  First result: {data['data'][0].get('formula_pretty', 'N/A')}")
            print(f"  First result density: {data['data'][0].get('density', 'N/A')}")
    else:
        print(f"  Error Text: {response.text}")

except requests.exceptions.Timeout:
    elapsed = time.time() - start
    print(f"✗ TIMEOUT after {elapsed:.2f}s")
    print("  This is the source of the hang!")
except Exception as e:
    elapsed = time.time() - start
    print(f"✗ ERROR after {elapsed:.2f}s: {e}")

print("\n" + "="*70)
print("LAYER 1 TEST COMPLETE")
print("="*70)
