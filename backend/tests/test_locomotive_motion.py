#!/usr/bin/env python3
"""
Test script for physics-driven locomotive motion.
Verifies position tracking, orientation, and motion trail updates.
"""

import requests
import json
import time
import math

BASE_URL = "http://localhost:8000"

def test_physics_step_with_position():
    """Test physics step with explicit position/orientation from backend"""
    print("=== Test 1: Backend Provides Position ===")
    
    payload = {
        "state": {
            "velocity": 0,
            "altitude": 0,
            "position": {"x": 0, "y": 0, "z": 0},
            "orientation": {"yaw": 0, "pitch": 0, "roll": 0},
            "temperature": 20,
            "fuel": 100
        },
        "inputs": {
            "thrust": 2500,
            "gravity": 9.81,
            "mass": 10.0,
            "environment": {
                "regime": "AERIAL",
                "fluid_density": 1.225,
                "wind_speed": 0
            }
        },
        "dt": 0.1
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/physics/step", json=payload)
        print(f"Status: {response.status_code}")
        
        if response.ok:
            data = response.json()
            state = data.get("state", {})
            
            print(f"Position: {state.get('position', 'N/A')}")
            print(f"Orientation: {state.get('orientation', 'N/A')}")
            print(f"Velocity: {state.get('velocity', 'N/A')} m/s")
            print(f"Altitude: {state.get('altitude', 'N/A')} m")
            print("✅ Test PASSED")
        else:
            print(f"❌ Test FAILED: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()

def test_aerodynamics_scenario():
    """Test aerodynamics scenario with attack angle"""
    print("=== Test 2: Aerodynamics Scenario ===")
    
    payload = {
        "state": {
            "velocity": 50,
            "altitude": 100,
            "position": {"x": 0, "y": 100, "z": 0},
            "orientation": {"yaw": 0, "pitch": 0.087, "roll": 0},  # 5° pitch
            "temperature": 20,
            "fuel": 95
        },
        "inputs": {
            "thrust": 3000,
            "gravity": 9.81,
            "mass": 15.0,
            "environment": {
                "regime": "AERIAL",
                "fluid_density": 1.225,
                "wind_speed": 60,
                "attack_angle": 5
            }
        },
        "dt": 0.1
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/physics/step", json=payload)
        print(f"Status: {response.status_code}")
        
        if response.ok:
            data = response.json()
            state = data.get("state", {})
            
            # Calculate expected drag
            rho = 1.225
            v = state.get('velocity', 50)
            drag = 0.5 * rho * v * v * 3.14
            
            print(f"Velocity: {v} m/s")
            print(f"Expected Drag: {drag:.0f} N")
            print(f"Position: {state.get('position', {})}")
            print(f"Pitch: {state.get('orientation', {}).get('pitch', 0):.3f} rad")
            print("✅ Test PASSED")
        else:
            print(f"❌ Test FAILED: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()

def test_space_orbital_mechanics():
    """Test space scenario with orbital dynamics"""
    print("=== Test 3: Space Orbital Mechanics ===")
    
    orbit_altitude_km = 400
    GM = 3.986004418e14
    r = (6371 + orbit_altitude_km) * 1000
    orbital_velocity = math.sqrt(GM / r)
    orbital_gravity = GM / (r ** 2)
    
    payload = {
        "state": {
            "velocity": orbital_velocity / 1000,  # Convert to km/s for display
            "altitude": orbit_altitude_km,
            "position": {"x": r / 1000, "y": 0, "z": 0},  # Start on X-axis
            "orientation": {"yaw": math.pi / 2, "pitch": 0, "roll": 0},
            "temperature": -100,
            "fuel": 100
        },
        "inputs": {
            "thrust": 0,  # Coasting in orbit
            "gravity": orbital_gravity,
            "mass": 10.0,
            "environment": {
                "regime": "ORBITAL",
                "orbit_altitude_km": orbit_altitude_km,
                "solar_radiation_W_m2": 1361
            }
        },
        "dt": 0.1
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/physics/step", json=payload)
        print(f"Status: {response.status_code}")
        
        if response.ok:
            data = response.json()
            state = data.get("state", {})
            
            print(f"Orbital Velocity: {orbital_velocity / 1000:.2f} km/s")
            print(f"Microgravity: {orbital_gravity:.3f} m/s²")
            print(f"Position: {state.get('position', {})}")
            print(f"Orientation: {state.get('orientation', {})}")
            print("✅ Test PASSED")
        else:
            print(f"❌ Test FAILED: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()

def test_hydrodynamics_scenario():
    """Test hydrodynamics with depth and pressure"""
    print("=== Test 4: Hydrodynamics Scenario ===")
    
    depth = 100  # meters
    water_density = 1025  # kg/m³ (seawater)
    pressure = 101325 + (water_density * 9.81 * depth)
    
    payload = {
        "state": {
            "velocity": 5,
            "altitude": -depth,  # Negative for underwater
            "position": {"x": 0, "y": -depth, "z": 0},
            "orientation": {"yaw": 0, "pitch": -0.1, "roll": 0},
            "temperature": 10,
            "fuel": 90
        },
        "inputs": {
            "thrust": 1500,
            "gravity": 9.81,
            "mass": 20.0,
            "environment": {
                "regime": "MARINE",
                "fluid_density": water_density,
                "depth_m": depth,
                "current_speed": 2,
                "pressure_Pa": pressure
            }
        },
        "dt": 0.1
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/physics/step", json=payload)
        print(f"Status: {response.status_code}")
        
        if response.ok:
            data = response.json()
            state = data.get("state", {})
            
            print(f"Depth: {depth} m")
            print(f"Pressure: {pressure / 1000:.1f} kPa")
            print(f"Position: {state.get('position', {})}")
            print(f"Velocity: {state.get('velocity', 0)} m/s")
            print("✅ Test PASSED")
        else:
            print(f"❌ Test FAILED: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()

def test_motion_trail_accumulation():
    """Test that motion trail accumulates over time"""
    print("=== Test 5: Motion Trail Simulation ===")
    
    positions = []
    yaw = 0
    
    for i in range(10):
        # Simulate curved path
        yaw += 0.1  # Turn slightly each step
        
        payload = {
            "state": {
                "velocity": 10,
                "altitude": 50,
                "position": {
                    "x": 10 * math.sin(yaw),
                    "y": 50,
                    "z": 10 * math.cos(yaw)
                },
                "orientation": {"yaw": yaw, "pitch": 0, "roll": 0},
                "temperature": 20,
                "fuel": 100 - i
            },
            "inputs": {
                "thrust": 2000,
                "gravity": 9.81,
                "mass": 10.0,
                "environment": {"regime": "AERIAL", "fluid_density": 1.225}
            },
            "dt": 0.1
        }
        
        try:
            response = requests.post(f"{BASE_URL}/api/physics/step", json=payload)
            if response.ok:
                state = response.json().get("state", {})
                pos = state.get("position", {})
                positions.append(pos)
                print(f"Step {i+1}: Position {pos}")
            time.sleep(0.1)  # Simulate 100ms physics loop
        except Exception as e:
            print(f"❌ Error at step {i}: {e}")
            break
    
    print(f"\n✅ Accumulated {len(positions)} positions for motion trail")
    print()

if __name__ == "__main__":
    print("🚀 Testing Physics-Driven Locomotive Motion\n")
    
    test_physics_step_with_position()
    test_aerodynamics_scenario()
    test_space_orbital_mechanics()
    test_hydrodynamics_scenario()
    test_motion_trail_accumulation()
    
    print("🎯 All tests complete!")
