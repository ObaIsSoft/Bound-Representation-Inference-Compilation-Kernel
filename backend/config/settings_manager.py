"""
Backend Settings Configuration Manager
Handles runtime configuration for the BRICK OS backend
"""
import json
import os
from typing import Dict, Any
from pydantic import BaseModel

class RuntimeSettings(BaseModel):
    """Backend runtime configuration"""
    simulation_frequency: int = 1000  # Hz
    incremental_compilation: bool = True
    secure_boot: bool = True
    agent_sandboxing: bool = True
    agent_proposals: bool = True
    physics_kernel: str = "EARTH_AERO"
    compiler_optimization: str = "balanced"
    visualization_quality: str = "HIGH"

class SettingsManager:
    """Manages backend settings with file persistence"""
    
    def __init__(self, config_path: str = "data/backend_settings.json"):
        self.config_path = config_path
        self.settings = self._load_settings()
    
    def _load_settings(self) -> RuntimeSettings:
        """Load settings from disk or create defaults"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    return RuntimeSettings(**data)
            except Exception as e:
                print(f"Error loading settings: {e}, using defaults")
                return RuntimeSettings()
        return RuntimeSettings()
    
    def save_settings(self, settings: RuntimeSettings):
        """Persist settings to disk"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(settings.dict(), f, indent=2)
        self.settings = settings
    
    def update_settings(self, updates: Dict[str, Any]) -> RuntimeSettings:
        """Update specific settings fields"""
        current = self.settings.dict()
        current.update(updates)
        new_settings = RuntimeSettings(**current)
        self.save_settings(new_settings)
        return new_settings
    
    def get_settings(self) -> RuntimeSettings:
        """Get current settings"""
        return self.settings
    
    def apply_to_kernel(self):
        """Apply settings to runtime systems (physics kernel, vHIL, etc.)"""
        changes = {}
        
        # 1. Update vHIL time step based on simulation frequency
        dt = 1.0 / self.settings.simulation_frequency  # Convert Hz to seconds
        changes['vhil_dt'] = dt
        changes['simulation_frequency'] = self.settings.simulation_frequency
        
        # 2. Update physics kernel environment parameters
        kernel_config = self._get_physics_kernel_config(self.settings.physics_kernel)
        changes['physics_kernel'] = kernel_config
        
        # 3. Apply compiler optimization flags (for future KCL compilation)
        changes['compiler_optimization'] = self.settings.compiler_optimization
        
        # 4. Update security settings
        changes['secure_boot'] = self.settings.secure_boot
        changes['agent_sandboxing'] = self.settings.agent_sandboxing
        
        print(f"[SETTINGS] Applied Runtime Configuration:")
        print(f"  - Simulation Frequency: {self.settings.simulation_frequency} Hz (dt={dt:.4f}s)")
        print(f"  - Physics Kernel: {self.settings.physics_kernel}")
        print(f"  - Gravity: {kernel_config['gravity']} m/s²")
        print(f"  - Atmospheric Pressure: {kernel_config['pressure']} Pa")
        print(f"  - Compiler Optimization: {self.settings.compiler_optimization}")
        print(f"  - Security: Boot={self.settings.secure_boot}, Sandboxing={self.settings.agent_sandboxing}")
        
        return {"applied": True, "changes": changes}
    
    def _get_physics_kernel_config(self, kernel_name: str) -> Dict[str, Any]:
        """Get physics parameters for specified kernel"""
        kernels = {
            "EARTH_AERO": {
                "gravity": 9.81,
                "pressure": 101325,  # Sea level
                "temperature": 288.15,  # 15°C
                "density": 1.225,  # kg/m³
                "regime": "AERO"
            },
            "EARTH_MARINE": {
                "gravity": 9.81,
                "pressure": 101325,
                "temperature": 293.15,  # 20°C
                "density": 1000,  # Water
                "regime": "MARINE"
            },
            "ORBITAL_LEO": {
                "gravity": 8.8,  # ~400km altitude
                "pressure": 0,  # Vacuum
                "temperature": 273.15,
                "density": 0,
                "regime": "SPACE"
            },
            "ORBITAL_GEO": {
                "gravity": 0.5,  # ~36000km altitude
                "pressure": 0,
                "temperature": 273.15,
                "density": 0,
                "regime": "SPACE"
            }
        }
        return kernels.get(kernel_name, kernels["EARTH_AERO"])
    
    def get_vhil_dt(self) -> float:
        """Get current vHIL time step in seconds"""
        return 1.0 / self.settings.simulation_frequency
    
    def get_kernel_config(self) -> Dict[str, Any]:
        """Get current physics kernel configuration"""
        return self._get_physics_kernel_config(self.settings.physics_kernel)

# Global settings manager
_settings_manager = None

def get_settings_manager() -> SettingsManager:
    """Get or create the global settings manager"""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    return _settings_manager
