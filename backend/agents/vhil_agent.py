from typing import Dict, Any, List
import logging
from physics import get_physics_kernel

logger = logging.getLogger(__name__)

class VhilAgent:
    """
    VHIL Agent - Virtual Hardware-in-the-Loop Simulation.
    
    Emulates hardware sensors and actuators for:
    - Real-time control loop testing
    - Sensor fusion validation
    - Hardware failure simulation
    - Timing-accurate simulation
    """
    
    def __init__(self):
        self.name = "VhilAgent"
        
        # Initialize Physics Kernel for real dynamics
        self.physics = get_physics_kernel()
        logger.info("VhilAgent: Physics kernel initialized")
        
        # Store gravity constant
        self.g = self.physics.get_constant("g")  # Real gravity from physics
        
        self.sensor_models = {
            "imu": self._emulate_imu,
            "gps": self._emulate_gps,
            "lidar": self._emulate_lidar,
            "camera": self._emulate_camera,
        }
    
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run vHIL simulation step.
        
        Args:
            params: {
                "sensors": List of sensor types to emulate,
                "state": Current system state,
                "dt": Time step (seconds) - optional, uses settings if not provided,
                "noise_level": Optional float (0-1)
            }
        
        Returns:
            {
                "sensor_data": Dict of sensor readings,
                "actuator_commands": Dict of control outputs,
                "timing_ms": float,
                "logs": List of operation logs
            }
        """
        sensors = params.get("sensors", ["imu", "gps"])
        state = params.get("state", {})
        
        # Use settings manager for dt if not explicitly provided
        if "dt" not in params:
            try:
                from config.settings_manager import get_settings_manager
                dt = get_settings_manager().get_vhil_dt()
            except Exception:
                dt = 0.01  # Fallback default
        else:
            dt = params.get("dt", 0.01)
            
        noise_level = params.get("noise_level", 0.1)
        
        logs = [
            f"[VHIL] Emulating {len(sensors)} sensor(s)",
            f"[VHIL] Time step: {dt*1000:.1f} ms",
            f"[VHIL] Noise level: {noise_level:.1%}"
        ]
        
       
        # Emulate sensors
        sensor_data = {}
        for sensor_type in sensors:
            if sensor_type in self.sensor_models:
                sensor_data[sensor_type] = self.sensor_models[sensor_type](state, noise_level)
            else:
                logs.append(f"[VHIL] ⚠ Unknown sensor: {sensor_type}")
        
        # Mock actuator commands (basic control law)
        actuator_commands = self._generate_control_commands(state, sensor_data)
        
        # REAL PHYSICS SIMULATION: Update state using integrated dynamics
        geometry_data = params.get("geometry_data", {})
        if geometry_data and state:
            try:
                from agents.vhil_physics import simulate_aerial_dynamics
                updated_state = simulate_aerial_dynamics(
                    self.physics,
                    state,
                    actuator_commands,
                    geometry_data,
                    dt
                )
                # Update original state reference
                state.update(updated_state)
                logs.append(f"[VHIL] ✓ Physics-based state propagation complete")
                logs.append(f"[VHIL]   Altitude: {updated_state['position']['y']:.1f}m, Velocity: {updated_state['aerodynamics']['velocity_m_s']:.1f}m/s")
            except Exception as e:
                logger.warning(f"Physics simulation failed, using kinematic fallback: {e}")
                logs.append(f"[VHIL] ⚠ Physics simulation error: {str(e)}")
        
        # Timing simulation
        timing_ms = dt * 1000  # Convert to milliseconds
        
        logs.append(f"[VHIL] Generated {len(sensor_data)} sensor reading(s)")
        logs.append(f"[VHIL] Computed {len(actuator_commands)} actuator command(s)")
        
        return {
            "sensor_data": sensor_data,
            "actuator_commands": actuator_commands,
            "updated_state": state,  # Return updated state
            "timing_ms": timing_ms,
            "logs": logs
        }
    
    def _emulate_imu(self, state: Dict, noise: float) -> Dict:
        """Emulate IMU (Inertial Measurement Unit)."""
        import random
        
        # Extract state or use defaults
        accel_x = state.get("acceleration", 0) + random.gauss(0, noise * 0.1)
        accel_y = random.gauss(0, noise * 0.1)
        accel_z = -self.g + random.gauss(0, noise * 0.1)  # Use real gravity constant
        
        gyro_x = random.gauss(0, noise * 0.01)
        gyro_y = random.gauss(0, noise * 0.01)
        gyro_z = random.gauss(0, noise * 0.01)
        
        return {
            "accelerometer": {"x": accel_x, "y": accel_y, "z": accel_z},
            "gyroscope": {"x": gyro_x, "y": gyro_y, "z": gyro_z}
        }
    
    def _emulate_gps(self, state: Dict, noise: float) -> Dict:
        """Emulate GPS sensor."""
        import random
        
        position = state.get("position", {"x": 0, "y": 0, "z": 0})
        
        return {
            "latitude": 37.7749 + (position.get("x", 0) / 111000) + random.gauss(0, noise * 0.00001),
            "longitude": -122.4194 + (position.get("z", 0) / 111000) + random.gauss(0, noise * 0.00001),
            "altitude_m": position.get("y", 0) + random.gauss(0, noise * 5),
            "satellites": 12,
            "hdop": 0.8 + random.random() * noise
        }
    
    def _emulate_lidar(self, state: Dict, noise: float) -> Dict:
        """Emulate LIDAR sensor."""
        import random
        
        # Simplified: return distance to ground
        altitude = state.get("altitude", 10.0)
        
        return {
            "distance_m": altitude + random.gauss(0, noise * 0.1),
            "angle_deg": 0,
            "quality": 255 - int(noise * 50)
        }
    
    def _emulate_camera(self, state: Dict, noise: float) -> Dict:
        """Emulate camera sensor (metadata only)."""
        return {
            "frame_id": int(state.get("time", 0) * 30),  # 30 FPS
            "resolution": "640x480",
            "exposure_ms": 33.3,
            "brightness": 128
        }
    
    def _generate_control_commands(self, state: Dict, sensor_data: Dict) -> Dict:
        """Generate actuator commands based on sensor data."""
        # Simple altitude hold example
        target_alt = 50.0
        current_alt = state.get("altitude", 0)
        error = target_alt - current_alt
        
        throttle = 0.5 + (error * 0.01)  # Proportional control
        throttle = max(0.0, min(1.0, throttle))  # Clamp 0-1
        
        return {
            "throttle": throttle,
            "aileron": 0.0,
            "elevator": 0.0,
            "rudder": 0.0
        }
