
import os
import subprocess
import tempfile

def test_openscad():
    scad_code = "difference() { cube([40, 60, 30], center=true); for(x=[-10, 10]) translate([x, 0, 0]) cylinder(h=40, r=8, center=true); }"
    scad_path = "test.scad"
    stl_path = "test.stl"
    
    with open(scad_path, "w") as f:
        f.write(scad_code)
        
    cmd = [
        "/usr/local/bin/openscad",
        "--export-format", "binstl",
        "-o", stl_path,
        scad_path
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    env = os.environ.copy()
    env['DISPLAY'] = ''
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            env=env
        )
        print("Return Code:", result.returncode)
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        
        if os.path.exists(stl_path):
            print("SUCCESS: STL created.")
            os.remove(stl_path)
        else:
            print("FAILURE: STL not created.")
            
    except Exception as e:
        print("EXCEPTION:", e)
    finally:
        if os.path.exists(scad_path):
            os.remove(scad_path)

if __name__ == "__main__":
    test_openscad()
