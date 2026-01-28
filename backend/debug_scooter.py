
import subprocess
import os
import tempfile
import sys

# The user's scooter code
SCOOTER_CODE = """
$fn = 60;
DECK_LEN = 550;
DECK_W = 160;
WHEEL_D = 215;
RAKE_ANGLE = 20;
STEM_HEIGHT = 950;

module e_scooter_full() {
    deck_assembly();
    steering_assembly();
    rear_drive_unit();
    cockpit_controls();
}

module deck_assembly() {
    color("DimGray") difference() {
        union() {
            translate([0, 0, WHEEL_D/3]) cube([DECK_LEN, DECK_W, 45], center=true);
            translate([DECK_LEN/2 - 20, 0, WHEEL_D/3 + 20]) rotate([0, -RAKE_ANGLE, 0]) cube([60, 40, 100], center=true);
        }
        translate([0, 0, WHEEL_D/3]) cube([DECK_LEN-40, DECK_W-20, 35], center=true);
    }
}

module steering_assembly() {
    translate([DECK_LEN/2 + 10, 0, WHEEL_D/2]) rotate([0, -RAKE_ANGLE, 0]) {
        color("Silver") cylinder(h=STEM_HEIGHT, d=45);
        color("Gray") translate([0, 0, -WHEEL_D/2]) {
            hull() {
                translate([0, 0, WHEEL_D/2]) cube([40, 60, 10], center=true);
                translate([0, 30, 0]) cylinder(d=15, h=20, center=true);
                translate([0, -30, 0]) cylinder(d=15, h=20, center=true);
            }
            rotate([90, 0, 0]) color("Black") torus_wheel(WHEEL_D, 50);
        }
    }
}

module rear_drive_unit() {
    translate([-DECK_LEN/2 + 20, 0, WHEEL_D/2]) {
        rotate([90, 0, 0]) {
            color("Black") torus_wheel(WHEEL_D, 55);
            color("DarkSlateGray") cylinder(h=45, d=WHEEL_D-40, center=true);
            color("Orange") translate([0,0,25]) rotate([0,90,0]) cylinder(h=50, d=6);
        }
    }
}

module cockpit_controls() {
    translate([DECK_LEN/2 + 10 - sin(RAKE_ANGLE)*STEM_HEIGHT, 0, STEM_HEIGHT + 80]) rotate([0, -RAKE_ANGLE, 0]) {
        color("Black") rotate([90, 0, 0]) cylinder(h=500, d=25, center=true);
        translate([30, 0, 10]) rotate([0, -30, 0]) color("DimGray") cube([60, 50, 15], center=true);
        translate([0, -220, 0]) color("Red") rotate([90, 0, 0]) cylinder(h=40, d=35);
    }
}

module torus_wheel(d, w) {
    rotate_extrude() translate([d/2 - w/2, 0, 0]) circle(d=w);
    cylinder(h=w-10, d=d-w, center=true);
}

e_scooter_full();
"""

def test_compilation():
    print("Testing OpenSCAD Compilation...")
    
    # Write SCAD file
    with tempfile.NamedTemporaryFile(suffix='.scad', delete=False, mode='w') as f:
        f.write(SCOOTER_CODE)
        scad_path = f.name
        
    # Output file
    stl_path = scad_path.replace('.scad', '.stl')
    
    print(f"SCAD Path: {scad_path}")
    print(f"STL Path: {stl_path}")
    
    # Command
    cmd = [
        "openscad",
        "-o", stl_path,
        scad_path
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    
    env = os.environ.copy()
    # Force headless?
    # env['DISPLAY'] = '' 
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            env=env
        )
        
        print(f"Return Code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
        
        if os.path.exists(stl_path):
            size = os.path.getsize(stl_path)
            print(f"SUCCESS: STL created. Size: {size} bytes")
            if size < 100:
                print("WARNING: STL seems too small (empty?)")
            os.remove(stl_path)
        else:
            print("FAILURE: STL file not found.")
            
    except Exception as e:
        print(f"EXCEPTION: {e}")
    finally:
        if os.path.exists(scad_path):
            os.remove(scad_path)

if __name__ == "__main__":
    test_compilation()
