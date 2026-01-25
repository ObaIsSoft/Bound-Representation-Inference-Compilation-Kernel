"""
Test F-22 Raptor Simplified Model - Progressive Rendering
"""

import requests
import json
import time

# Simplified F-22 with multiple modules
f22_code = """
$fn = 20;

module wing() {
    cube([1000, 200, 50]);
}

module fuselage() {
    cylinder(h=2000, r=200);
}

module tail() {
    cube([100, 300, 500]);
}

module engine() {
    cylinder(h=400, r=80);
}

// Main assembly
union() {
    // Fuselage
    fuselage();
    
    // Wings
    translate([0, 0, 1000]) wing();
    
    // Tail
    translate([0, 0, 1800]) tail();
    
    // Engines
    translate([150, 0, 800]) rotate([90, 0, 0]) engine();
    translate([-150, 0, 800]) rotate([90, 0, 0]) engine();
}
"""

print("Testing F-22 Raptor Simplified Model")
print("=" * 60)

url = "http://localhost:8000/api/openscad/compile-stream"
headers = {"Content-Type": "application/json"}
data = {"code": f22_code}

try:
    start_time = time.time()
    response = requests.post(url, headers=headers, json=data, stream=True)
    
    print(f"Status: {response.status_code}")
    print("=" * 60)
    
    part_count = 0
    error_count = 0
    events = []
    
    for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
        if chunk:
            # Parse SSE events
            if 'event: start' in chunk:
                # Extract total_parts from data
                if '"total_parts"' in chunk:
                    import re
                    match = re.search(r'"total_parts":\s*(\d+)', chunk)
                    if match:
                        total = match.group(1)
                        print(f"✓ Started: {total} parts to compile")
            
            elif 'event: part\n' in chunk and 'part_error' not in chunk:
                part_count += 1
                # Extract part_id
                if '"part_id"' in chunk:
                    import re
                    match = re.search(r'"part_id":\s*"([^"]+)"', chunk)
                    if match:
                        part_id = match.group(1)
                        print(f"✓ Part {part_count}: {part_id}")
            
            elif 'event: part_error' in chunk:
                error_count += 1
                # Extract error message
                if '"error"' in chunk:
                    import re
                    match = re.search(r'"error":\s*"([^"]+)"', chunk)
                    if match:
                        error_msg = match.group(1)[:100]
                        print(f"✗ Error {error_count}: {error_msg}...")
            
            elif 'event: complete' in chunk:
                elapsed = time.time() - start_time
                print(f"✓ Complete in {elapsed:.2f}s")
    
    print("=" * 60)
    print(f"Results: {part_count} parts, {error_count} errors")
    
    if part_count > 0 and error_count == 0:
        print("✅ SUCCESS - All parts compiled!")
    elif part_count > 0 and error_count > 0:
        print("⚠️  PARTIAL - Some parts failed")
    else:
        print("❌ FAILED - No parts compiled")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
