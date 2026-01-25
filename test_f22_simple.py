"""
Test F-22 Raptor model with progressive rendering
"""

import requests
import json

# Read the F-22 code from file (user will paste it)
f22_code = """
// F-22 Raptor 3D Model - SIMPLIFIED TEST VERSION
// Testing progressive rendering with realistic complexity

$fn = 20;  // Lower resolution for faster testing

module simple_wing() {
    cube([1000, 200, 50]);
}

module simple_fuselage() {
    cylinder(h=2000, r=200);
}

module simple_tail() {
    cube([100, 300, 500]);
}

// Main assembly
union() {
    simple_fuselage();
    
    translate([0, 0, 1000])
        simple_wing();
    
    translate([0, 0, 1800])
        simple_tail();
}
"""

print("Testing F-22 Raptor model (simplified)...")
print("=" * 60)

url = "http://localhost:8000/api/openscad/compile-stream"
headers = {"Content-Type": "application/json"}
data = {"code": f22_code}

try:
    response = requests.post(url, headers=headers, json=data, stream=True)
    
    print(f"Status: {response.status_code}")
    print("=" * 60)
    
    part_count = 0
    error_count = 0
    
    for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
        if chunk:
            # Count successful parts
            if 'event: part\n' in chunk and 'part_error' not in chunk:
                part_count += 1
                print(f"✓ Part {part_count} received")
            # Count errors
            elif 'event: part_error' in chunk:
                error_count += 1
                print(f"✗ Part error {error_count}")
                # Print first 300 chars of error
                if '"error"' in chunk:
                    error_start = chunk.find('"error"')
                    print(chunk[error_start:error_start+300])
    
    print("=" * 60)
    print(f"Parts: {part_count}, Errors: {error_count}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
