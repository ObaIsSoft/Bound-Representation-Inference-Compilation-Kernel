"""
BRICK OS Agent Configuration Management

Centralized configuration for all physics agents with:
- Environment variable overrides
- YAML config file support
- Validation and defaults
- Runtime reconfiguration

Author: BRICK OS Team
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class StructuralConfig:
    """Configuration for Structural Agent"""
    # FEA Solver
    calculix_executable: str = field(default_factory=lambda: os.getenv("CALCULIX_PATH", "ccx"))
    calculix_timeout: int = field(default_factory=lambda: int(os.getenv("CALCULIX_TIMEOUT", "300")))
    
    # Surrogate Model
    surrogate_model_path: Optional[str] = field(default_factory=lambda: os.getenv("SURROGATE_MODEL_PATH"))
    use_surrogate: bool = field(default_factory=lambda: os.getenv("USE_SURROGATE", "true").lower() == "true")
    surrogate_threshold: float = field(default_factory=lambda: float(os.getenv("SURROGATE_THRESHOLD", "0.95")))
    
    # ROM
    rom_energy_threshold: float = field(default_factory=lambda: float(os.getenv("ROM_ENERGY_THRESHOLD", "0.99")))
    rom_max_modes: int = field(default_factory=lambda: int(os.getenv("ROM_MAX_MODES", "50")))
    
    # Buckling
    buckling_modes: int = field(default_factory=lambda: int(os.getenv("BUCKLING_MODES", "10")))
    
    # V&V
    vv20_enabled: bool = field(default_factory=lambda: os.getenv("VV20_ENABLED", "true").lower() == "true")
    mms_tolerance: float = field(default_factory=lambda: float(os.getenv("MMS_TOLERANCE", "1e-6")))


@dataclass
class GeometryConfig:
    """Configuration for Geometry Agent"""
    # CAD Kernel
    cad_kernel: str = field(default_factory=lambda: os.getenv("CAD_KERNEL", "opencascade"))
    
    # Meshing
    gmsh_executable: Optional[str] = field(default_factory=lambda: os.getenv("GMSH_PATH"))
    default_mesh_size: float = field(default_factory=lambda: float(os.getenv("DEFAULT_MESH_SIZE", "0.1")))
    max_mesh_elements: int = field(default_factory=lambda: int(os.getenv("MAX_MESH_ELEMENTS", "100000")))
    
    # File I/O
    step_export_tolerance: float = field(default_factory=lambda: float(os.getenv("STEP_TOLERANCE", "1e-6")))
    
    # Constraint Solver
    constraint_tolerance: float = field(default_factory=lambda: float(os.getenv("CONSTRAINT_TOLERANCE", "1e-6")))
    constraint_max_iterations: int = field(default_factory=lambda: int(os.getenv("CONSTRAINT_MAX_ITER", "100")))
    
    # GD&T
    gdt_standard: str = field(default_factory=lambda: os.getenv("GDT_STANDARD", "ASME_Y14.5"))


@dataclass
class ThermalConfig:
    """Configuration for Thermal Agent"""
    # Solver
    solver_type: str = field(default_factory=lambda: os.getenv("THERMAL_SOLVER", "fipy"))
    fipy_max_iterations: int = field(default_factory=lambda: int(os.getenv("FIPY_MAX_ITER", "1000")))
    fipy_tolerance: float = field(default_factory=lambda: float(os.getenv("FIPY_TOLERANCE", "1e-6")))
    
    # CoolProp
    coolprop_enabled: bool = field(default_factory=lambda: os.getenv("COOLPROP_ENABLED", "true").lower() == "true")
    
    # Thermal-Structural Coupling
    coupling_enabled: bool = field(default_factory=lambda: os.getenv("THERMAL_COUPLING", "true").lower() == "true")
    buckling_risk_threshold: float = field(default_factory=lambda: float(os.getenv("BUCKLING_RISK_THRESHOLD", "2.0")))
    
    # Radiation
    radiation_enabled: bool = field(default_factory=lambda: os.getenv("RADIATION_ENABLED", "true").lower() == "true")
    stefan_boltzmann: float = field(default=5.670374e-8)  # W/m²·K⁴


@dataclass
class MaterialConfig:
    """Configuration for Material Agent"""
    # Data Sources
    database_path: str = field(default_factory=lambda: os.getenv(
        "MATERIAL_DATABASE_PATH", 
        str(Path(__file__).parent.parent.parent / "data" / "materials_database_expanded.json")
    ))
    
    # API
    api_enabled: bool = field(default_factory=lambda: os.getenv("MATERIAL_API_ENABLED", "false").lower() == "true")
    api_endpoint: Optional[str] = field(default_factory=lambda: os.getenv("MATERIAL_API_ENDPOINT"))
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("MATERIAL_API_KEY"))
    api_timeout: int = field(default_factory=lambda: int(os.getenv("MATERIAL_API_TIMEOUT", "30")))
    
    # Cache
    cache_enabled: bool = field(default_factory=lambda: os.getenv("MATERIAL_CACHE", "true").lower() == "true")
    cache_ttl_hours: int = field(default_factory=lambda: int(os.getenv("MATERIAL_CACHE_TTL", "168")))
    
    # Fallback
    fallback_enabled: bool = field(default_factory=lambda: os.getenv("MATERIAL_FALLBACK", "true").lower() == "true")
    
    # Data Quality
    min_provenance_level: str = field(default_factory=lambda: os.getenv("MIN_PROVENANCE", "UNSPECIFIED"))
    warn_on_unspecified: bool = field(default_factory=lambda: os.getenv("WARN_UNSPECIFIED", "true").lower() == "true")


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    format: str = field(default_factory=lambda: os.getenv(
        "LOG_FORMAT", 
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    file: Optional[str] = field(default_factory=lambda: os.getenv("LOG_FILE"))
    max_bytes: int = field(default_factory=lambda: int(os.getenv("LOG_MAX_BYTES", "10485760")))  # 10MB
    backup_count: int = field(default_factory=lambda: int(os.getenv("LOG_BACKUP_COUNT", "5")))


@dataclass
class SystemConfig:
    """System-wide configuration"""
    # Performance
    max_workers: int = field(default_factory=lambda: int(os.getenv("MAX_WORKERS", "4")))
    memory_limit_mb: int = field(default_factory=lambda: int(os.getenv("MEMORY_LIMIT_MB", "4096")))
    
    # Paths
    temp_dir: str = field(default_factory=lambda: os.getenv("TEMP_DIR", "/tmp/brick_os"))
    output_dir: str = field(default_factory=lambda: os.getenv("OUTPUT_DIR", "./output"))
    
    # Features
    debug_mode: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    validate_inputs: bool = field(default_factory=lambda: os.getenv("VALIDATE_INPUTS", "true").lower() == "true")
    
    # Safety
    max_simulation_time: int = field(default_factory=lambda: int(os.getenv("MAX_SIM_TIME", "3600")))
    safe_mode: bool = field(default_factory=lambda: os.getenv("SAFE_MODE", "true").lower() == "true")


@dataclass
class AgentConfiguration:
    """Complete agent configuration"""
    structural: StructuralConfig = field(default_factory=StructuralConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    material: MaterialConfig = field(default_factory=MaterialConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    @classmethod
    def from_file(cls, config_path: str) -> "AgentConfiguration":
        """Load configuration from JSON or YAML file"""
        path = Path(config_path)
        
        if not path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return cls()
        
        try:
            with open(path, 'r') as f:
                if path.suffix in ['.yaml', '.yml']:
                    try:
                        import yaml
                        data = yaml.safe_load(f)
                    except ImportError:
                        logger.error("PyYAML required for YAML config files")
                        return cls()
                else:
                    data = json.load(f)
            
            return cls(
                structural=StructuralConfig(**data.get('structural', {})),
                geometry=GeometryConfig(**data.get('geometry', {})),
                thermal=ThermalConfig(**data.get('thermal', {})),
                material=MaterialConfig(**data.get('material', {})),
                logging=LoggingConfig(**data.get('logging', {})),
                system=SystemConfig(**data.get('system', {}))
            )
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return cls()
    
    def to_file(self, config_path: str):
        """Save configuration to file"""
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'structural': asdict(self.structural),
            'geometry': asdict(self.geometry),
            'thermal': asdict(self.thermal),
            'material': asdict(self.material),
            'logging': asdict(self.logging),
            'system': asdict(self.system)
        }
        
        with open(path, 'w') as f:
            if path.suffix in ['.yaml', '.yml']:
                try:
                    import yaml
                    yaml.dump(data, f, default_flow_style=False)
                except ImportError:
                    json.dump(data, f, indent=2)
            else:
                json.dump(data, f, indent=2)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Check executable paths
        if self.structural.calculix_executable != "ccx":
            if not Path(self.structural.calculix_executable).exists():
                errors.append(f"CalculiX executable not found: {self.structural.calculix_executable}")
        
        if self.geometry.gmsh_executable and not Path(self.geometry.gmsh_executable).exists():
            errors.append(f"Gmsh executable not found: {self.geometry.gmsh_executable}")
        
        # Check material database
        if not Path(self.material.database_path).exists():
            errors.append(f"Material database not found: {self.material.database_path}")
        
        # Check thresholds are reasonable
        if not 0 < self.structural.rom_energy_threshold <= 1:
            errors.append(f"ROM energy threshold must be in (0,1]: {self.structural.rom_energy_threshold}")
        
        if not 0 < self.structural.surrogate_threshold <= 1:
            errors.append(f"Surrogate threshold must be in (0,1]: {self.structural.surrogate_threshold}")
        
        return errors
    
    def apply_logging(self):
        """Apply logging configuration"""
        import logging
        
        level = getattr(logging, self.logging.level.upper(), logging.INFO)
        
        handlers = [logging.StreamHandler()]
        
        if self.logging.file:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                self.logging.file,
                maxBytes=self.logging.max_bytes,
                backupCount=self.logging.backup_count
            )
            handlers.append(file_handler)
        
        logging.basicConfig(
            level=level,
            format=self.logging.format,
            handlers=handlers
        )


# Global configuration instance
_config: Optional[AgentConfiguration] = None


def get_config() -> AgentConfiguration:
    """Get global configuration (singleton)"""
    global _config
    
    if _config is None:
        # Try to load from environment-specified path
        config_path = os.getenv("BRICK_CONFIG_PATH")
        
        if config_path:
            _config = AgentConfiguration.from_file(config_path)
        else:
            # Try default locations
            for path in ["./brick_config.json", "./config/brick_config.json", "~/.brick/config.json"]:
                expanded = Path(path).expanduser()
                if expanded.exists():
                    _config = AgentConfiguration.from_file(str(expanded))
                    break
            else:
                _config = AgentConfiguration()
        
        # Apply logging config
        _config.apply_logging()
        
        # Validate
        errors = _config.validate()
        if errors:
            for error in errors:
                logger.warning(f"Config validation: {error}")
    
    return _config


def reload_config():
    """Reload configuration from disk"""
    global _config
    _config = None
    return get_config()


def create_default_config(path: str = "./brick_config.json"):
    """Create a default configuration file"""
    config = AgentConfiguration()
    config.to_file(path)
    logger.info(f"Default config created at: {path}")


# Example configuration file content
EXAMPLE_CONFIG = """{
  "structural": {
    "calculix_executable": "ccx",
    "calculix_timeout": 300,
    "surrogate_model_path": "./models/fno_surrogate_best.pt",
    "use_surrogate": true,
    "surrogate_threshold": 0.95,
    "rom_energy_threshold": 0.99,
    "rom_max_modes": 50,
    "buckling_modes": 10,
    "vv20_enabled": true,
    "mms_tolerance": 1e-06
  },
  "geometry": {
    "cad_kernel": "opencascade",
    "gmsh_executable": "gmsh",
    "default_mesh_size": 0.1,
    "max_mesh_elements": 100000,
    "step_export_tolerance": 1e-06,
    "constraint_tolerance": 1e-06,
    "constraint_max_iterations": 100,
    "gdt_standard": "ASME_Y14.5"
  },
  "thermal": {
    "solver_type": "fipy",
    "fipy_max_iterations": 1000,
    "fipy_tolerance": 1e-06,
    "coolprop_enabled": true,
    "coupling_enabled": true,
    "buckling_risk_threshold": 2.0,
    "radiation_enabled": true
  },
  "material": {
    "database_path": "./data/materials_database_expanded.json",
    "api_enabled": false,
    "api_timeout": 30,
    "cache_enabled": true,
    "cache_ttl_hours": 168,
    "fallback_enabled": true,
    "min_provenance_level": "UNSPECIFIED",
    "warn_on_unspecified": true
  },
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "max_bytes": 10485760,
    "backup_count": 5
  },
  "system": {
    "max_workers": 4,
    "memory_limit_mb": 4096,
    "temp_dir": "/tmp/brick_os",
    "output_dir": "./output",
    "debug_mode": false,
    "validate_inputs": true,
    "max_simulation_time": 3600,
    "safe_mode": true
  }
}
"""


if __name__ == "__main__":
    # Create example config
    create_default_config()
    
    # Test loading
    config = get_config()
    print("Configuration loaded successfully!")
    print(f"  CalculiX path: {config.structural.calculix_executable}")
    print(f"  Material DB: {config.material.database_path}")
    print(f"  ROM threshold: {config.structural.rom_energy_threshold}")
