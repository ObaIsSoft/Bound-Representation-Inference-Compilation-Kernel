
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from backend.agents.openscad_agent import OpenSCADAgent

def test_bugatti_optimization():
    agent = OpenSCADAgent()
    
    # Bugatti snippet with loops (wheels)
    code = """
    scale_factor = 1/6;
    wheel_diameter_front = 20 * 25.4 * scale_factor;
    tire_width_front = 285 * scale_factor;
    
    module wheels_and_tires() {
        wheel_positions = [
            [-800, 1000, 0],
            [800, 1000, 0]
        ];
        
        for(i = [0:1]) {
            translate(wheel_positions[i]) {
                rotate([0, 90, 0])
                difference() {
                    cylinder(r=wheel_diameter_front/2, h=tire_width_front, center=true);
                    cylinder(r=wheel_diameter_front/2 - 20, h=tire_width_front + 2, center=true);
                }
                
                // Spokes loop (OPTIMIZATION TARGET)
                for(spoke = [0:9]) {
                    rotate([0, 0, spoke * 36])
                        hull() {
                            translate([0, 0, 0])
                                cylinder(r=5, h=tire_width_front * 0.7, center=true);
                            translate([30, 0, 0])
                                cylinder(r=8, h=tire_width_front * 0.7, center=true);
                        }
                }
            }
        }
    }
    
    wheels_and_tires();
    """
    
    print("Starting Optimized Compilation...")
    generator = agent.compile_assembly_progressive(code)
    
    part_count = 0
    errors = 0
    
    for event in generator:
        if event["event"] == "part":
            part_count += 1
            print(f"✓ Part [{event.get('part_index')}] {event['part_name']}")
        elif event["event"] == "part_error":
            print(f"✗ Error: {event['error']}")
            errors += 1
            
    print(f"Total Parts: {part_count}")
    
    # We expect roughly 4 parts: 2 tires (difference) + 2 spoke sets (loops)
    # If optimization works, the 10 spokes per wheel should be 1 part (the loop node)
    # So 2 wheels * (1 tire + 1 spoke loop) = 4 parts.
    # Without optimization: 2 * (1 tire + 10 spokes) = 22 parts.
    
    if part_count <= 6:
        print("SUCCESS: Loop optimization verified (Low part count).")
    else:
        print(f"FAILURE: High part count ({part_count}) indicating loops were unrolled.")

if __name__ == "__main__":
    test_bugatti_optimization()
