from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator
import logging

# Ares Layer: Global Constraints & Unit Enforcement

class AresUnitError(ValueError):
    """Raised when a unit mismatch or invalid unit is detected."""
    pass

class UnitValue(BaseModel):
    value: float
    unit: str
    source: str = "ARES_DEFAULT"
    locked: bool = False

    @field_validator('unit')
    def enforce_si_units(cls, v):
        """
        Comprehensive unit validation across:
        - Mass: kg, g, mg, ton, kiloton, lb, oz
        - Length: m, cm, mm, µm, nm, km, in, ft, yd, mi, mile, feet, inches
        - Volume: L, mL, m³, cm³, gal, qt, pt
        - Pressure/Stress: Pa, kPa, MPa, GPa, bar, psi, atm
        - Force: N, kN, lbf
        - Energy: J, kJ, MJ, Wh, kWh, cal, kcal
        - Power: W, kW, MW, hp
        - Time: s, ms, µs, min, h, day
        - Angular: deg, rad, arcmin, arcsec
        - Temperature: K, °C, °F
        - Electrical: V, kV, mV, A, mA, kA, Ω, kΩ, MΩ
        - Strain: ε, µε, %, ppm
        - Count: count, unit, pcs
        """
        valid_units = [
            # Mass
            'kg', 'g', 'mg', 'ton', 'kiloton', 'lb', 'oz',
            # Length (with aliases)
            'm', 'cm', 'mm', 'µm', 'um', 'nm', 'km', 'in', 'ft', 'yd', 'mi',
            'mile', 'miles', 'feet', 'inches',
            # Volume
            'L', 'mL', 'µL', 'uL', 'm³', 'm3', 'cm³', 'cm3', 'gal', 'qt', 'pt',
            # Pressure/Stress (stress = force/area, same as pressure)
            'Pa', 'kPa', 'MPa', 'GPa', 'bar', 'psi', 'atm',
            # Force (tensile force covered here)
            'N', 'kN', 'lbf',
            # Energy
            'J', 'kJ', 'MJ', 'Wh', 'kWh', 'cal', 'kcal',
            # Power
            'W', 'kW', 'MW', 'hp',
            # Time
            's', 'ms', 'µs', 'us', 'min', 'h', 'day',
            # Angular
            'deg', 'rad', 'arcmin', 'arcsec',
            # Temperature
            'K', '°C', 'C', '°F', 'F',
            # Electrical
            'V', 'kV', 'mV', 'A', 'mA', 'kA', 'Ω', 'ohm', 'kΩ', 'kohm', 'MΩ', 'Mohm',
            # Strain (dimensionless ratio)
            'ε', 'strain', 'µε', 'microstrain', '%', 'ppm',
            # Count
            'count', 'unit', 'pcs',
            # Speed
            'm/s', 'km/h', 'mph', 'knots', 'ft/s',
            # Acceleration
            'm/s²', 'm/s2', 'g', 'ft/s²', 'ft/s2',
            # Torque
            'Nm', 'N-m', 'ft-lb', 'in-lb',
            # Flow Rate (for submersibles/hydraulics)
            'L/min', 'GPM', 'm³/s',
            # Electromagnetism (Extended)
            'T', 'Tesla', 'Gauss', 'H', 'F', 'C', 'Ah',
            # Radiometry / Optics
            'lm', 'lx', 'cd',
            # Fluid Dynamics / Viscosity
            'Pa·s', 'Poise', 'P', 'St', 'Stokes',
            # Frequency / Data
            'Hz', 'kHz', 'MHz', 'GHz', 'bps', 'kbps', 'Mbps', 'Gbps'
        ]
        
        if v not in valid_units:
            raise AresUnitError(f"Unit '{v}' is not recognized by BRICK OS Standards. Supported units: {', '.join(sorted(set(valid_units)))}")
        return v

class AresMiddleware:
    """
    The Gatekeeper for Physical Validity.
    Intercepts Agent outputs and strictly enforces units.
    Provides comprehensive unit conversion utilities.
    Ensures unit consistency throughout the compilation pipeline.
    """
    def __init__(self):
        self.logger = logging.getLogger("AresLayer")
        # Track units used for each parameter to enforce consistency
        self.parameter_unit_lock: Dict[str, str] = {}

    def validate_unit(self, input_data: Dict[str, Any], context_key: str = "unknown"):
        """
        Validates that a dictionary contains valid 'unit' and 'value' keys.
        Example input constraint: {"value": 10, "unit": "kg"}
        """
        # If it's a nested dict like {"mass": {"value": 10, "unit": "kg"}}
        target = input_data
        
        # If the input is the wrapper dict, peel it
        if isinstance(input_data, dict) and context_key in input_data:
             target = input_data[context_key]

        if not isinstance(target, dict):
             # Logic simplifies: if raw value, we assume standard unit or ignore
             return True

        # Check for 'unit'
        if "unit" in target:
             # Create a dummy UnitValue to trigger validator
             try:
                 # Minimal Construction 
                 UnitValue(value=target.get("value", 0), unit=target["unit"])
                 return True
             except Exception as e:
                 raise AresUnitError(f"Ares Validation Failed for '{context_key}': {e}")
        
        # If no unit key, maybe implicit? For now pass.
        return True

    def validate_constraints(self, constraints: Dict[str, Any]) -> Dict[str, UnitValue]:
        validated = {}
        for key, raw_data in constraints.items():
            try:
                # Handle ConstraintNode objects (ISA Schema)
                if hasattr(raw_data, "val"):
                    # Extract PhysicalValue from ConstraintNode
                    pv = raw_data.val
                    uv = UnitValue(value=pv.value, unit=pv.unit, source=pv.source, locked=pv.locked)
                    
                    # Enforce consistency
                    self._enforce_consistency(key, uv)
                    validated[key] = uv
                    
                # Handle Legacy Dicts
                elif isinstance(raw_data, dict):
                    uv = UnitValue(**raw_data)
                    self._enforce_consistency(key, uv)
                    validated[key] = uv
                else:
                    self.logger.warning(f"Constraint '{key}' validation failed: Invalid format {type(raw_data)}")
            except AresUnitError as e:
                self.logger.error(f"Ares Blocked Constraint '{key}': {e}")
                raise e
            except Exception as e:
                self.logger.error(f"Unexpected error in Ares validation for '{key}': {e}")
                raise e
        return validated

    def _enforce_consistency(self, key: str, uv: UnitValue):
        """Helper to check if unit matches previous definitions."""
        if key in self.parameter_unit_lock:
            locked_unit = self.parameter_unit_lock[key]
            if uv.unit != locked_unit:
                self.logger.warning(
                    f"Unit consistency violation: '{key}' was previously defined with unit '{locked_unit}', "
                    f"but agent attempted to use '{uv.unit}'. Enforcing original unit."
                )
                raise AresUnitError(
                    f"Unit mismatch for '{key}': Expected '{locked_unit}', got '{uv.unit}'. "
                    f"Units cannot change mid-compilation."
                )
        else:
            self.parameter_unit_lock[key] = uv.unit
            self.logger.info(f"Locked unit for '{key}': {uv.unit}")

    # --- CONVERSION UTILITIES ---

    def convert_to_kg(self, val: float, unit: str) -> float:
        """Convert mass to kg (SI base)."""
        conversions = {
            'kg': 1.0,
            'g': 0.001,
            'mg': 1e-6,
            'ton': 1000.0,        # metric ton
            'kiloton': 1e6,
            'lb': 0.453592,
            'oz': 0.0283495
        }
        if unit not in conversions:
            raise AresUnitError(f"Cannot convert '{unit}' to kg mass.")
        return val * conversions[unit]

    def convert_to_m(self, val: float, unit: str) -> float:
        """Convert length to m (SI base), including aliases."""
        conversions = {
            'm': 1.0,
            'cm': 0.01,
            'mm': 0.001,
            'µm': 1e-6,
            'um': 1e-6,
            'nm': 1e-9,
            'km': 1000.0,
            'in': 0.0254,
            'inches': 0.0254,
            'ft': 0.3048,
            'feet': 0.3048,
            'yd': 0.9144,
            'mi': 1609.34,
            'mile': 1609.34,
            'miles': 1609.34
        }
        if unit not in conversions:
            raise AresUnitError(f"Cannot convert '{unit}' to m length.")
        return val * conversions[unit]

    def convert_to_mm(self, val: float, unit: str) -> float:
        """Helper for converting valid linear units to mm (KCL default)."""
        return self.convert_to_m(val, unit) * 1000.0

    def convert_to_L(self, val: float, unit: str) -> float:
        """Convert volume to L (liters)."""
        conversions = {
            'L': 1.0,
            'mL': 0.001,
            'µL': 1e-6,
            'uL': 1e-6,
            'm³': 1000.0,
            'm3': 1000.0,
            'cm³': 0.001,
            'cm3': 0.001,
            'gal': 3.78541,  # US gallon
            'qt': 0.946353,
            'pt': 0.473176
        }
        if unit not in conversions:
            raise AresUnitError(f"Cannot convert '{unit}' to L volume.")
        return val * conversions[unit]

    def convert_to_Pa(self, val: float, unit: str) -> float:
        """Convert pressure/stress to Pa (Pascals). Stress units are identical to pressure units."""
        conversions = {
            'Pa': 1.0,
            'kPa': 1000.0,
            'MPa': 1e6,
            'GPa': 1e9,
            'bar': 1e5,
            'psi': 6894.76,
            'atm': 101325.0
        }
        if unit not in conversions:
            raise AresUnitError(f"Cannot convert '{unit}' to Pa pressure/stress.")
        return val * conversions[unit]

    def convert_to_N(self, val: float, unit: str) -> float:
        """Convert force to N (Newtons)."""
        conversions = {
            'N': 1.0,
            'kN': 1000.0,
            'lbf': 4.44822
        }
        if unit not in conversions:
            raise AresUnitError(f"Cannot convert '{unit}' to N force.")
        return val * conversions[unit]

    def convert_to_J(self, val: float, unit: str) -> float:
        """Convert energy to J (Joules)."""
        conversions = {
            'J': 1.0,
            'kJ': 1000.0,
            'MJ': 1e6,
            'Wh': 3600.0,
            'kWh': 3.6e6,
            'cal': 4.184,
            'kcal': 4184.0
        }
        if unit not in conversions:
            raise AresUnitError(f"Cannot convert '{unit}' to J energy.")
        return val * conversions[unit]

    def convert_to_W(self, val: float, unit: str) -> float:
        """Convert power to W (Watts)."""
        conversions = {
            'W': 1.0,
            'kW': 1000.0,
            'MW': 1e6,
            'hp': 745.7  # mechanical horsepower
        }
        if unit not in conversions:
            raise AresUnitError(f"Cannot convert '{unit}' to W power.")
        return val * conversions[unit]

    def convert_to_s(self, val: float, unit: str) -> float:
        """Convert time to s (seconds)."""
        conversions = {
            's': 1.0,
            'ms': 0.001,
            'µs': 1e-6,
            'us': 1e-6,
            'min': 60.0,
            'h': 3600.0,
            'day': 86400.0
        }
        if unit not in conversions:
            raise AresUnitError(f"Cannot convert '{unit}' to s time.")
        return val * conversions[unit]

    def convert_to_rad(self, val: float, unit: str) -> float:
        """Convert angular to rad (radians)."""
        import math
        conversions = {
            'rad': 1.0,
            'deg': math.pi / 180.0,
            'arcmin': math.pi / 10800.0,
            'arcsec': math.pi / 648000.0
        }
        if unit not in conversions:
            raise AresUnitError(f"Cannot convert '{unit}' to rad angle.")
        return val * conversions[unit]

    def convert_to_V(self, val: float, unit: str) -> float:
        """Convert voltage to V (Volts)."""
        conversions = {
            'V': 1.0,
            'kV': 1000.0,
            'mV': 0.001
        }
        if unit not in conversions:
            raise AresUnitError(f"Cannot convert '{unit}' to V voltage.")
        return val * conversions[unit]

    def convert_to_A(self, val: float, unit: str) -> float:
        """Convert current to A (Amperes)."""
        conversions = {
            'A': 1.0,
            'kA': 1000.0,
            'mA': 0.001
        }
        if unit not in conversions:
            raise AresUnitError(f"Cannot convert '{unit}' to A current.")
        return val * conversions[unit]

    def convert_to_ohm(self, val: float, unit: str) -> float:
        """Convert resistance to Ω (Ohms)."""
        conversions = {
            'Ω': 1.0,
            'ohm': 1.0,
            'kΩ': 1000.0,
            'kohm': 1000.0,
            'MΩ': 1e6,
            'Mohm': 1e6
        }
        if unit not in conversions:
            raise AresUnitError(f"Cannot convert '{unit}' to Ω resistance.")
        return val * conversions[unit]

    def convert_strain(self, val: float, unit: str) -> float:
        """
        Convert strain to dimensionless ratio (ε).
        Strain is inherently dimensionless but often expressed in different notations.
        """
        conversions = {
            'ε': 1.0,
            'strain': 1.0,
            'µε': 1e-6,
            'microstrain': 1e-6,
            '%': 0.01,
            'ppm': 1e-6
        }
        if unit not in conversions:
            raise AresUnitError(f"Cannot convert '{unit}' to strain (dimensionless).")
        return val * conversions[unit]

    def convert_to_K(self, val: float, unit: str) -> float:
        """Convert temperature to K (Kelvin)."""
        if unit == 'K':
            return val
        elif unit in ['°C', 'C']:
            return val + 273.15
        elif unit in ['°F', 'F']:
            return (val - 32) * 5/9 + 273.15
        else:
            raise AresUnitError(f"Cannot convert '{unit}' to K temperature.")

    def convert_to_ms(self, val: float, unit: str) -> float:
        """Convert speed to m/s."""
        conversions = {
            'm/s': 1.0,
            'km/h': 0.277778,
            'mph': 0.44704,
            'knots': 0.514444,
            'ft/s': 0.3048
        }
        if unit not in conversions:
             raise AresUnitError(f"Cannot convert '{unit}' to m/s speed.")
        return val * conversions[unit]

    def convert_to_ms2(self, val: float, unit: str) -> float:
        """Convert acceleration to m/s²."""
        conversions = {
            'm/s²': 1.0,
            'm/s2': 1.0,
            'g': 9.80665,
            'ft/s²': 0.3048,
            'ft/s2': 0.3048
        }
        if unit not in conversions:
             raise AresUnitError(f"Cannot convert '{unit}' to m/s² acceleration.")
        return val * conversions[unit]
        
    def convert_to_Nm(self, val: float, unit: str) -> float:
        """Convert torque to N·m."""
        conversions = {
            'Nm': 1.0, 
            'N-m': 1.0,
            'ft-lb': 1.35582,
            'in-lb': 0.112985
        }
        if unit not in conversions:
             raise AresUnitError(f"Cannot convert '{unit}' to Nm torque.")
        return val * conversions[unit]
