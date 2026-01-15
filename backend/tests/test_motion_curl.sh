# Curl Commands for Testing Physics-Driven Motion
# Copy and paste these into your terminal

# ====================================
# Test 1: Basic Physics Step with Position
# ====================================
curl -X POST http://localhost:8000/api/physics/step \
  -H "Content-Type: application/json" \
  -d '{
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
        "fluid_density": 1.225
      }
    },
    "dt": 0.1
  }' | jq '.'

# ====================================
# Test 2: Aerodynamics with Attack Angle
# ====================================
curl -X POST http://localhost:8000/api/physics/step \
  -H "Content-Type: application/json" \
  -d '{
    "state": {
      "velocity": 50,
      "altitude": 100,
      "position": {"x": 0, "y": 100, "z": 0},
      "orientation": {"yaw": 0, "pitch": 0.087, "roll": 0},
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
  }' | jq '.state | {position, orientation, velocity}'

# ====================================
# Test 3: Space Orbital Mechanics
# ====================================
curl -X POST http://localhost:8000/api/physics/step \
  -H "Content-Type: application/json" \
  -d '{
    "state": {
      "velocity": 7.67,
      "altitude": 400,
      "position": {"x": 6771, "y": 0, "z": 0},
      "orientation": {"yaw": 1.5708, "pitch": 0, "roll": 0},
      "temperature": -100,
      "fuel": 100
    },
    "inputs": {
      "thrust": 0,
      "gravity": 0.089,
      "mass": 10.0,
      "environment": {
        "regime": "ORBITAL",
        "orbit_altitude_km": 400,
        "solar_radiation_W_m2": 1361
      }
    },
    "dt": 0.1
  }' | jq '.state | {position, velocity, orientation}'

# ====================================
# Test 4: Hydrodynamics with Depth
# ====================================
curl -X POST http://localhost:8000/api/physics/step \
  -H "Content-Type: application/json" \
  -d '{
    "state": {
      "velocity": 5,
      "altitude": -100,
      "position": {"x": 0, "y": -100, "z": 0},
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
        "fluid_density": 1025,
        "depth_m": 100,
        "current_speed": 2,
        "pressure_Pa": 1106800
      }
    },
    "dt": 0.1
  }' | jq '.state | {position, velocity, altitude}'

# ====================================
# Test 5: Propulsion Test (Vertical Climb)
# ====================================
curl -X POST http://localhost:8000/api/physics/step \
  -H "Content-Type: application/json" \
  -d '{
    "state": {
      "velocity": 0,
      "altitude": 0,
      "position": {"x": 0, "y": 0, "z": 0},
      "orientation": {"yaw": 0, "pitch": 1.5708, "roll": 0},
      "temperature": 20,
      "fuel": 100,
      "acceleration": 0
    },
    "inputs": {
      "thrust": 5000,
      "gravity": 9.81,
      "mass": 10.0,
      "environment": {
        "regime": "AERIAL",
        "fluid_density": 1.225
      }
    },
    "dt": 0.1
  }' | jq '.state | {position, velocity, altitude, acceleration}'

# ====================================
# Quick Test: Check if Backend is Running
# ====================================
curl http://localhost:8000/health || echo "❌ Backend not running!"

# ====================================
# Loop Test: Simulate Motion Trail (10 steps)
# ====================================
for i in {1..10}; do
  echo "Step $i:"
  curl -X POST http://localhost:8000/api/physics/step \
    -H "Content-Type: application/json" \
    -d "{
      \"state\": {
        \"velocity\": 10,
        \"altitude\": $((i * 5)),
        \"position\": {\"x\": $((i * 2)), \"y\": $((i * 5)), \"z\": $((i * 3))},
        \"orientation\": {\"yaw\": $((i * 0.1)), \"pitch\": 0, \"roll\": 0},
        \"temperature\": 20,
        \"fuel\": $((100 - i))
      },
      \"inputs\": {
        \"thrust\": 2000,
        \"gravity\": 9.81,
        \"mass\": 10.0,
        \"environment\": {\"regime\": \"AERIAL\", \"fluid_density\": 1.225}
      },
      \"dt\": 0.1
    }" | jq '.state.position'
  sleep 0.1
done
