// F-22 Raptor 3D Model with Internal Components
// Based on actual specifications and dimensions
// All measurements scaled to 1:1 realistic proportions

// === SCALE FACTOR ===
scale_factor = 1/6;  // 1:6 scale model

// === DIMENSIONS (in mm, converted from specifications and scaled) ===
fuselage_length = 18900 * scale_factor;  // 62 ft = 18.9m, scaled to 3150mm
wingspan = 13560 * scale_factor;         // 44.5 ft = 13.56m, scaled to 2260mm
height = 5080 * scale_factor;            // 16.67 ft = 5.08m, scaled to 847mm
wing_area = 78040 * scale_factor;        // 840 sq ft = 78.04 m²
horizontal_tail_span = 8840 * scale_factor;  // 29 ft = 8.84m, scaled to 1473mm

// Wing parameters
wing_sweep = 42;          // Leading edge sweep angle
wing_root_chord = 6000 * scale_factor;
wing_tip_chord = 1000 * scale_factor;

// Vertical stabilizer parameters
vstab_cant_angle = 27;    // Outward cant angle
vstab_height = 2500 * scale_factor;
vstab_chord_root = 3000 * scale_factor;
vstab_chord_tip = 800 * scale_factor;

// Engine parameters
engine_length = 5200 * scale_factor;
engine_diameter = 1200 * scale_factor;
nozzle_width = 1000 * scale_factor;
nozzle_height = 600 * scale_factor;

// Resolution
$fn = 50;

// === MAIN ASSEMBLY ===
module f22_raptor() {
    color("LightGray", 0.9) {
        fuselage();
    }
    
    color("DarkGray", 0.9) {
        // Wings
        wings();
        
        // Vertical stabilizers
        vertical_stabilizers();
        
        // Horizontal stabilizers
        horizontal_stabilizers();
    }
    
    // Internal components
    color("Silver", 0.3) {
        engines();
    }
    
    color("Red", 0.5) {
        weapons_bays();
    }
    
    color("Orange", 0.7) {
        cockpit_internals();
    }
    
    color("Blue", 0.6) {
        avionics_bay();
    }
    
    color("DarkSlateGray", 0.8) {
        landing_gear_bays();
    }
    
    // External details
    canopy();
    air_intakes();
    engine_nozzles();
}

// === FUSELAGE ===
module fuselage() {
    // Forward fuselage (nose to cockpit)
    hull() {
        // Nose cone
        translate([0, 0, 0])
            sphere(r=400);
        
        // Cockpit area
        translate([0, 5200, 0])
            scale([1, 1, 0.6])
            sphere(r=900);
    }
    
    // Mid fuselage (cockpit to wings)
    hull() {
        translate([0, 5200, 0])
            scale([1, 1, 0.6])
            sphere(r=900);
        
        translate([0, 10400, 0])
            scale([1.2, 1, 0.8])
            sphere(r=1100);
    }
    
    // Rear fuselage (wings to tail)
    hull() {
        translate([0, 10400, 0])
            scale([1.2, 1, 0.8])
            sphere(r=1100);
        
        translate([0, 15800, 0])
            scale([1, 1, 0.7])
            sphere(r=900);
    }
    
    // Tail section
    hull() {
        translate([0, 15800, 0])
            scale([1, 1, 0.7])
            sphere(r=900);
        
        translate([0, 18900, 0])
            scale([0.5, 1, 0.5])
            sphere(r=400);
    }
    
    // Chines (wing-fuselage blend)
    for(side = [-1, 1]) {
        hull() {
            translate([side * 800, 6000, -400])
                rotate([0, 0, 0])
                cylinder(r1=200, r2=100, h=8000);
            
            translate([side * wingspan/2 * 0.3, 10000, -200])
                cylinder(r=150, h=400);
        }
    }
}

// === WINGS ===
module wings() {
    for(side = [-1, 1]) {
        mirror([side < 0 ? 1 : 0, 0, 0])
        translate([0, 10400, -400])
        rotate([0, 0, -wing_sweep])
        difference() {
            // Main wing structure
            hull() {
                // Root
                translate([0, 0, 0])
                    cube([200, wing_root_chord, 400]);
                
                // Mid span
                translate([wingspan/2 * 0.6, 1000, 0])
                    cube([200, 3000, 300]);
                
                // Tip (clipped)
                translate([wingspan/2 - 500, 2000, 0])
                    cube([200, wing_tip_chord, 200]);
            }
            
            // Leading edge sweep cut
            translate([-100, -100, -50])
                cube([wingspan, 100, 500]);
        }
        
        // Wing leading edge extension (LEX)
        mirror([side < 0 ? 1 : 0, 0, 0])
        translate([0, 8000, -300])
        hull() {
            cube([800, 100, 200]);
            translate([1500, 2000, 0])
                cube([200, 100, 150]);
        }
    }
    
    // Wing hardpoints (4 total)
    for(side = [-1, 1]) {
        for(i = [0, 1]) {
            translate([side * (2000 + i * 2000), 12000, -600])
                cylinder(r=80, h=200);
        }
    }
}

// === VERTICAL STABILIZERS ===
module vertical_stabilizers() {
    for(side = [-1, 1]) {
        translate([side * 2200, 15500, 0])
        rotate([0, side * vstab_cant_angle, 0])
        difference() {
            // Main vertical stabilizer
            hull() {
                // Root
                translate([0, 0, 0])
                    cube([200, vstab_chord_root, 100]);
                
                // Tip
                translate([0, vstab_chord_root * 0.4, vstab_height])
                    cube([200, vstab_chord_tip, 100]);
            }
            
            // Rudder cut
            translate([50, vstab_chord_root * 0.6, 0])
                cube([100, vstab_chord_root * 0.4, vstab_height]);
        }
        
        // Moveable rudder
        translate([side * 2200, 15500 + vstab_chord_root * 0.6, 0])
        rotate([0, side * vstab_cant_angle, 0])
        color("SlateGray", 0.9)
        hull() {
            translate([0, 0, 0])
                cube([150, vstab_chord_root * 0.35, 100]);
            
            translate([0, vstab_chord_root * 0.15, vstab_height])
                cube([150, vstab_chord_tip * 0.5, 100]);
        }
    }
}

// === HORIZONTAL STABILIZERS ===
module horizontal_stabilizers() {
    for(side = [-1, 1]) {
        translate([side * horizontal_tail_span/2 * 0.9, 17500, 300])
        rotate([0, 0, -30])
        hull() {
            // Root
            translate([0, 0, 0])
                cube([200, 2000, 100]);
            
            // Tip
            translate([side * 1500, 1000, 0])
                cube([200, 600, 80]);
        }
    }
}

// === ENGINES (Pratt & Whitney F119-PW-100) ===
module engines() {
    for(side = [-1, 1]) {
        translate([side * 1000, 10000, -500])
        rotate([90, 0, 0]) {
            // Engine core
            cylinder(r=engine_diameter/2, h=engine_length);
            
            // Fan section
            translate([0, 0, 0])
                cylinder(r=engine_diameter/2 * 1.1, h=1500);
            
            // Compressor stages
            for(i = [0:10]) {
                translate([0, 0, 1500 + i * 200])
                    cylinder(r=engine_diameter/2 * (1 - i * 0.05), h=180);
            }
            
            // Combustion chamber
            translate([0, 0, 3500])
                cylinder(r=engine_diameter/2 * 0.6, h=800);
            
            // Turbine section
            translate([0, 0, 4300])
                cylinder(r=engine_diameter/2 * 0.7, h=900);
        }
    }
}

// === ENGINE NOZZLES (2D Thrust Vectoring) ===
module engine_nozzles() {
    for(side = [-1, 1]) {
        translate([side * 1000, 18500, -500])
        rotate([90, 0, 0])
        color("DarkGray", 0.9) {
            // Rectangular nozzle
            difference() {
                hull() {
                    cylinder(r=engine_diameter/2, h=600);
                    translate([0, 0, 600])
                        scale([1.4, 0.8, 1])
                        cylinder(r=engine_diameter/2 * 0.9, h=200);
                }
                
                // Interior
                translate([0, 0, -10])
                hull() {
                    cylinder(r=engine_diameter/2 - 100, h=610);
                    translate([0, 0, 600])
                        scale([1.4, 0.8, 1])
                        cylinder(r=engine_diameter/2 * 0.9 - 100, h=210);
                }
            }
            
            // Vectoring flaps
            for(i = [-1, 1]) {
                translate([0, i * nozzle_height/2, 700])
                    cube([nozzle_width, 50, 100], center=true);
            }
        }
    }
}

// === AIR INTAKES ===
module air_intakes() {
    for(side = [-1, 1]) {
        translate([side * 1200, 7000, -800])
        color("DarkSlateGray", 0.9)
        difference() {
            // Intake duct
            hull() {
                // Opening
                translate([0, 0, 0])
                    scale([1, 1, 0.7])
                    sphere(r=600);
                
                // Connection to engine
                translate([side * -200, 3000, 300])
                    rotate([90, 0, 0])
                    cylinder(r=engine_diameter/2, h=100);
            }
            
            // Interior (S-shaped to hide engine face)
            hull() {
                translate([0, 0, 0])
                    scale([1, 1, 0.7])
                    sphere(r=550);
                
                translate([side * -200, 3000, 300])
                    rotate([90, 0, 0])
                    cylinder(r=engine_diameter/2 - 50, h=100);
            }
        }
        
        // Boundary layer splitter
        translate([side * 1200, 6800, -1200])
            cube([side * 50, 300, 400]);
    }
}

// === COCKPIT ===
module canopy() {
    translate([0, 4500, 600])
    color("CyanBlue", 0.3)
    difference() {
        // Canopy bubble
        scale([0.8, 1, 0.6])
            sphere(r=1000);
        
        // Interior cut
        translate([0, 0, -600])
            cube([2000, 3000, 600], center=true);
        
        // Rear cut
        translate([0, 800, 0])
            cube([2000, 1000, 2000], center=true);
    }
}

module cockpit_internals() {
    translate([0, 4500, -200]) {
        // Ejection seat (ACES II variant)
        color("Black", 0.9)
        translate([0, 0, 200])
        union() {
            // Seat base
            cube([500, 600, 100], center=true);
            
            // Seat back
            translate([0, -50, 400])
                cube([500, 100, 800], center=true);
            
            // Headrest
            translate([0, -50, 900])
                cube([400, 150, 200], center=true);
        }
        
        // Control stick (force-sensitive side-stick)
        color("Gray", 0.9)
        translate([300, 200, 300])
            cylinder(r=30, h=400);
        
        // Throttle controls
        color("Gray", 0.9)
        translate([-300, 200, 300])
            cube([80, 300, 200]);
        
        // LCD displays (6 color displays)
        for(i = [0:5]) {
            color("Black", 0.95)
            translate([
                (i % 3 - 1) * 350,
                600,
                400 + floor(i / 3) * 350
            ])
                cube([300, 50, 250], center=true);
        }
        
        // HUD (Head-Up Display)
        color("Green", 0.2)
        translate([0, 800, 700])
            cube([600, 10, 400], center=true);
    }
}

// === WEAPONS BAYS ===
module weapons_bays() {
    // Main ventral weapons bay
    translate([0, 11000, -1200])
    difference() {
        cube([2400, 4000, 1000], center=true);
        
        // Interior cavity
        translate([0, 0, 100])
            cube([2300, 3900, 900], center=true);
    }
    
    // Main bay doors (serrated edges for stealth)
    for(side = [-1, 1]) {
        translate([side * 600, 11000, -1700])
        color("DarkGray", 0.85)
        difference() {
            cube([500, 4000, 50]);
            
            // Serrated edge pattern
            for(i = [0:20]) {
                translate([0, i * 200, 0])
                    rotate([0, 0, 45])
                    cube([100, 100, 60], center=true);
            }
        }
    }
    
    // LAU-142/A AMRAAM launchers (6 positions)
    for(i = [0:5]) {
        translate([
            (i % 2 - 0.5) * 600,
            10000 + floor(i / 2) * 1200,
            -1100
        ])
        color("Gray", 0.9)
        cylinder(r=100, h=400);
    }
    
    // Side weapons bays (for AIM-9 Sidewinders)
    for(side = [-1, 1]) {
        translate([side * 1800, 9000, -600])
        difference() {
            cube([800, 1500, 600], center=true);
            
            translate([0, 0, 50])
                cube([750, 1450, 550], center=true);
        }
        
        // LAU-141/A launcher
        translate([side * 1800, 9000, -500])
        color("Gray", 0.9)
        rotate([0, 0, 90])
            cylinder(r=80, h=300);
    }
    
    // M61A2 Vulcan cannon (20mm, right wing root)
    translate([800, 8500, 200])
    color("DarkGray", 0.95)
    rotate([90, 0, 0]) {
        // Cannon barrel assembly (6 barrels)
        for(i = [0:5]) {
            rotate([0, 0, i * 60])
            translate([120, 0, 0])
                cylinder(r=25, h=2000);
        }
        
        // Cannon housing
        cylinder(r=150, h=500);
        
        // Ammunition feed
        translate([0, 0, -800])
            cylinder(r=100, h=800);
    }
    
    // Ammunition storage (480 rounds)
    translate([600, 7500, 0])
    color("Olive", 0.8)
        cylinder(r=300, h=1200);
}

// === AVIONICS BAY ===
module avionics_bay() {
    translate([0, 3000, -200]) {
        // APG-77 AESA Radar
        translate([0, -1500, 200])
        rotate([90, 0, 0])
            cylinder(r=400, h=300);
        
        // Radar array elements
        translate([0, -1500, 200])
        rotate([90, 0, 0])
        for(x = [-3:3]) {
            for(y = [-3:3]) {
                translate([x * 100, y * 100, 250])
                    cube([80, 80, 50]);
            }
        }
        
        // Electronic warfare systems (ALR-94)
        for(i = [0:7]) {
            translate([
                cos(i * 45) * 700,
                sin(i * 45) * 700,
                0
            ])
            rotate([0, 0, i * 45])
                cube([100, 200, 150], center=true);
        }
        
        // Mission computer bays
        for(i = [0:2]) {
            translate([0, i * 400, -200])
                cube([600, 300, 400], center=true);
        }
    }
}

// === LANDING GEAR BAYS ===
module landing_gear_bays() {
    // Nose gear bay
    translate([0, 3500, -600])
    difference() {
        cube([800, 1500, 800], center=true);
        translate([0, 0, 50])
            cube([750, 1450, 750], center=true);
    }
    
    // Nose landing gear
    translate([0, 3500, -1000])
    color("Silver", 0.95) {
        // Strut
        cylinder(r=60, h=800);
        
        // Wheel
        translate([0, 0, -150])
        rotate([90, 0, 0])
            cylinder(r=200, h=100, center=true);
    }
    
    // Main gear bays (retract sideways)
    for(side = [-1, 1]) {
        translate([side * 1400, 11000, -800])
        difference() {
            cube([1200, 2000, 1000], center=true);
            translate([0, 0, 50])
                cube([1150, 1950, 950], center=true);
        }
        
        // Main landing gear
        translate([side * 1400, 11000, -1200])
        color("Silver", 0.95) {
            // Strut
            cylinder(r=80, h=1000);
            
            // Wheels (dual)
            for(w = [0, 1]) {
                translate([0, w * 250 - 125, -150])
                rotate([90, 0, 0])
                    cylinder(r=300, h=120, center=true);
            }
            
            // Hydraulic actuators
            translate([side * -300, 0, 500])
            rotate([0, -30, 0])
                cylinder(r=50, h=600);
        }
    }
}

// === FUEL SYSTEM ===
module fuel_tanks() {
    // Internal fuel tanks (3 in mid-fuselage)
    fuel_positions = [
        [0, 9000, 0],
        [0, 11000, 0],
        [0, 13000, 0]
    ];
    
    for(pos = fuel_positions) {
        translate(pos)
        color("Yellow", 0.3)
            cylinder(r=800, h=1000, center=true);
    }
}

// === RENDER THE COMPLETE AIRCRAFT ===
f22_raptor();

// Optional: Render fuel tanks (comment out if not needed)
// fuel_tanks();

// Scale indicator
color("Red", 1)
translate([0, -2000, -2000])
    cube([1000, 100, 100]);  // 1 meter reference

echo("F-22 Raptor Specifications (1:6 Scale):");
echo(str("Scale Factor: 1:", 1/scale_factor));
echo(str("Length: ", fuselage_length, " mm (", fuselage_length/1000, " m)"));
echo(str("Wingspan: ", wingspan, " mm (", wingspan/1000, " m)"));
echo(str("Height: ", height, " mm (", height/1000, " m)"));
echo("Full-scale dimensions: 18.9m x 13.56m x 5.08m");
echo("Model includes: Fuselage, Wings, Stabilizers, Engines,");
echo("Weapons Bays, Cockpit, Avionics, Landing Gear");
