"""
Test progressive OpenSCAD compilation with COMPLEX assembly
"""

import requests
import json

# Complex OpenSCAD assembly with modules (like what users would paste)
scad_code = """
// Wing module
module wing(length=10, width=1, thickness=0.5) {
    cube([length, width, thickness]);
}

// Fuselage module
module fuselage(height=20, radius=2) {
    cylinder(h=height, r=radius, $fn=16);
}

// Tail module
module tail(size=1) {
    sphere(r=size, $fn=12);
}

// Engine module
module engine(length=3, radius=0.8) {
    cylinder(h=length, r=radius, $fn=8);
}

// Complete aircraft assembly
union() {
    // Main fuselage
    fuselage(height=20, radius=2);
    
    // Wings
    translate([0, -5, 10]) rotate([0, 90, 0]) wing(length=12, width=1, thickness=0.5);
    
    // Tail
    translate([0, 0, 20]) tail(size=1.5);
    
    // Engines
    translate([3, -3, 8]) rotate([0, 90, 0]) engine(length=4, radius=0.8);
    translate([-3, -3, 8]) rotate([0, 90, 0]) engine(length=4, radius=0.8);
}
"""

print("Testing COMPLEX assembly with modules...")
print("=" * 60)
print("Code:")
print(scad_code)
print("=" * 60)

# Make POST request
url = "http://localhost:8000/api/openscad/compile-stream"
headers = {"Content-Type": "application/json"}
data = {"code": scad_code}

try:
    response = requests.post(url, headers=headers, json=data, stream=True)
    
    print(f"Status: {response.status_code}")
    print("=" * 60)
    print("Streaming response:")
    print("=" * 60)
    
    # Read stream
    part_count = 0
    for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
        if chunk:
            print(chunk, end='', flush=True)
            
            # Count parts
            if 'event: part' in chunk and '"success": true' not in chunk:
                part_count += 1
    
    print("\n" + "=" * 60)
    print(f"Stream complete! Received {part_count} parts")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
