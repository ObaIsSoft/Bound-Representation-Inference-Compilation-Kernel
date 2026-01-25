"""
Test progressive OpenSCAD compilation endpoint
"""

import requests
import json

# Simple OpenSCAD code
scad_code = """
cube([10, 10, 10]);
sphere(r=5, $fn=16);
cylinder(h=15, r=3, $fn=16);
"""

print("Testing progressive compilation endpoint...")
print("=" * 60)

# Make POST request
url = "http://localhost:8000/api/openscad/compile-stream"
headers = {"Content-Type": "application/json"}
data = {"code": scad_code}

try:
    response = requests.post(url, headers=headers, json=data, stream=True)
    
    print(f"Status: {response.status_code}")
    print(f"Headers: {dict(response.headers)}")
    print("=" * 60)
    print("Streaming response:")
    print("=" * 60)
    
    # Read stream
    buffer = ""
    for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
        if chunk:
            buffer += chunk
            print(chunk, end='', flush=True)
    
    print("\n" + "=" * 60)
    print("Stream complete!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
