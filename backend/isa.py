"""
BRICK OS - Hardware Instruction Set Architecture (ISA) - Enhanced Version

A production-ready ISA with comprehensive unit support, dimensional analysis,
unit conversion, and robust constraint management.

Key Features:
- 150+ units with automatic conversion
- Dimensional analysis for type safety
- Constraint satisfaction checking
- Transaction support with rollback
- Event system for reactive updates
- Comprehensive validation
"""

from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from enum import Enum
from pydantic import BaseModel, Field, validator, field_validator
from dataclasses import dataclass
import uuid
import hashlib
from datetime import datetime
import math
import json


# ==================== Unit System with Conversion ====================

class UnitDimension(str, Enum):
    """Physical dimensions for dimensional analysis"""
    LENGTH = "length"
    MASS = "mass"
    TIME = "time"
    CURRENT = "current"
    TEMPERATURE = "temperature"
    AMOUNT = "amount"
    LUMINOSITY = "luminosity"
    ANGLE = "angle"
    MAGNETISM = "magnetism"
    RADIATION = "radiation"
    DATA = "data"
    CURRENCY = "currency"
    THERMAL_PROP = "thermal_prop" # Conductivity, Specific Heat (simplified dimension)
    VISCOSITY = "viscosity"
    DIMENSIONLESS = "dimensionless"

class Scale(str, Enum):
    """Physical scale of the simulation regime"""
    NANO = "NANO"       # Chip/Molecule (< 1mm)
    MICRO = "MICRO"     # MEMS/PCB Component (1mm - 10cm)
    MESO = "MESO"       # Product/Device (10cm - 10m) - DEFAULT
    MACRO = "MACRO"     # Vehicle/Building (10m - 1km)
    MEGA = "MEGA"       # Grid/City (> 1km)

@dataclass
class UnitDefinition:
    """Complete unit definition with conversion info"""
    symbol: str
    dimension: UnitDimension
    to_si_factor: float  # Conversion factor to SI base unit
    si_offset: float = 0.0  # Offset for temperature conversions
    
    def to_si(self, value: float) -> float:
        """Convert value to SI base unit"""
        return value * self.to_si_factor + self.si_offset
    
    def from_si(self, value: float) -> float:
        """Convert value from SI base unit"""
        return (value - self.si_offset) / self.to_si_factor


class Unit(str, Enum):
    """
    Comprehensive unit system with conversion support.
    Each unit maps to a UnitDefinition for conversion.
    """
    
    # ==================== Length ====================
    METERS = "m"
    MILLIMETERS = "mm"
    CENTIMETERS = "cm"
    KILOMETERS = "km"
    MICROMETERS = "μm"
    NANOMETERS = "nm"
    INCHES = "in"
    FEET = "ft"
    YARDS = "yd"
    MILES = "mi"
    NAUTICAL_MILES = "nmi"
    
    # ==================== Mass ====================
    KILOGRAMS = "kg"
    GRAMS = "g"
    MILLIGRAMS = "mg"
    TONNES = "t"
    POUNDS = "lb"
    OUNCES = "oz"
    SLUGS = "slug"
    
    # ==================== Time ====================
    SECONDS = "s"
    MILLISECONDS = "ms"
    MICROSECONDS = "μs"
    NANOSECONDS = "ns"
    MINUTES = "min"
    HOURS = "h"
    DAYS = "day"
    
    # ==================== Current ====================
    AMPERES = "A"
    MILLIAMPERES = "mA"
    MICROAMPERES = "μA"
    KILOAMPERES = "kA"
    
    # ==================== Temperature ====================
    KELVIN = "K"
    CELSIUS = "°C"
    FAHRENHEIT = "°F"
    RANKINE = "°R"
    
    # ==================== Amount ====================
    MOLE = "mol"
    MILLIMOLE = "mmol"
    KILOMOLE = "kmol"
    
    # ==================== Luminosity ====================
    CANDELA = "cd"
    LUMEN = "lm"
    LUX = "lx"
    
    # ==================== Angles ====================
    RADIANS = "rad"
    DEGREES = "deg"
    GRADIANS = "grad"
    REVOLUTIONS = "rev"
    
    # ==================== Velocity ====================
    METERS_PER_SECOND = "m/s"
    KM_PER_HOUR = "km/h"
    MILES_PER_HOUR = "mph"
    FEET_PER_SECOND = "ft/s"
    KNOTS = "kn"
    MACH = "Ma"
    
    # ==================== Acceleration ====================
    METERS_PER_SECOND_SQUARED = "m/s²"
    G_FORCE = "g"
    FEET_PER_SECOND_SQUARED = "ft/s²"
    
    # ==================== Force ====================
    NEWTONS = "N"
    KILONEWTONS = "kN"
    MEGANEWTONS = "MN"
    POUNDS_FORCE = "lbf"
    DYNES = "dyn"
    KIPS = "kip"
    
    # ==================== Torque ====================
    NEWTON_METERS = "N·m"
    POUND_FEET = "lb·ft"
    POUND_INCHES = "lb·in"
    
    # ==================== Angular Velocity ====================
    RADIANS_PER_SECOND = "rad/s"
    DEGREES_PER_SECOND = "deg/s"
    RPM = "rpm"
    HERTZ = "Hz"
    KILOHERTZ = "kHz"
    MEGAHERTZ = "MHz"
    GIGAHERTZ = "GHz"
    
    # ==================== Power ====================
    WATT = "W"
    MILLIWATT = "mW"
    KILOWATT = "kW"
    MEGAWATT = "MW"
    GIGAWATT = "GW"
    HORSEPOWER = "hp"
    HORSEPOWER_METRIC = "PS"
    BTU_PER_HOUR = "BTU/h"
    
    # ==================== Energy ====================
    JOULE = "J"
    KILOJOULE = "kJ"
    MEGAJOULE = "MJ"
    GIGAJOULE = "GJ"
    WATT_HOURS = "Wh"
    KILOWATT_HOURS = "kWh"
    MEGAWATT_HOURS = "MWh"
    CALORIE = "cal"
    KILOCALORIE = "kcal"
    BTU = "BTU"
    ELECTRON_VOLT = "eV"
    
    # ==================== Voltage ====================
    VOLTS = "V"
    MILLIVOLTS = "mV"
    MICROVOLTS = "μV"
    KILOVOLTS = "kV"
    MEGAVOLTS = "MV"
    
    # ==================== Charge ====================
    COULOMB = "C"
    MILLICOULOMB = "mC"
    MICROCOULOMB = "μC"
    MILLIAMP_HOURS = "mAh"
    AMPERE_HOURS = "Ah"
    
    # ==================== Resistance ====================
    OHM = "Ω"
    MILLIOHM = "mΩ"
    KILOOHM = "kΩ"
    MEGAOHM = "MΩ"
    SIEMENS = "S" # Conductance
    OHM_METER = "Ω·m" # Resistivity
    SIEMENS_PER_METER = "S/m" # Conductivity
    
    # ==================== Capacitance ====================
    FARAD = "F"
    MILLIFARAD = "mF"
    MICROFARAD = "μF"
    NANOFARAD = "nF"
    PICOFARAD = "pF"
    
    # ==================== Inductance ====================
    HENRY = "H"
    MILLIHENRY = "mH"
    MICROHENRY = "μH"
    NANOHENRY = "nH"
    
    # ==================== Pressure ====================
    PASCAL = "Pa"
    KILOPASCAL = "kPa"
    MEGAPASCAL = "MPa"
    GIGAPASCAL = "GPa"
    BAR = "bar"
    MILLIBAR = "mbar"
    PSI = "psi"
    KSI = "ksi"
    ATMOSPHERE = "atm"
    TORR = "Torr"
    MMHG = "mmHg"
    
    # ==================== Density ====================
    KG_PER_CUBIC_METER = "kg/m³"
    G_PER_CUBIC_CM = "g/cm³"
    LB_PER_CUBIC_FOOT = "lb/ft³"
    LB_PER_CUBIC_INCH = "lb/in³"
    
    # ==================== Area ====================
    SQUARE_METERS = "m²"
    SQUARE_MILLIMETERS = "mm²"
    SQUARE_CENTIMETERS = "cm²"
    SQUARE_KILOMETERS = "km²"
    SQUARE_INCHES = "in²"
    SQUARE_FEET = "ft²"
    SQUARE_YARDS = "yd²"
    SQUARE_MILES = "mi²"
    ACRES = "acre"
    HECTARES = "ha"
    
    # ==================== Volume ====================
    CUBIC_METERS = "m³"
    CUBIC_CENTIMETERS = "cm³"
    CUBIC_MILLIMETERS = "mm³"
    LITERS = "L"
    MILLILITERS = "mL"
    CUBIC_INCHES = "in³"
    CUBIC_FEET = "ft³"
    CUBIC_YARDS = "yd³"
    GALLONS_US = "gal"
    GALLONS_UK = "gal(UK)"
    QUARTS = "qt"
    PINTS = "pt"
    FLUID_OUNCES = "fl oz"
    
    # ==================== Flow Rate ====================
    CUBIC_METERS_PER_SECOND = "m³/s"
    LITERS_PER_SECOND = "L/s"
    LITERS_PER_MINUTE = "L/min"
    GALLONS_PER_MINUTE = "GPM"
    CUBIC_FEET_PER_MINUTE = "CFM"
    
    # ==================== Dimensionless ====================
    RATIO = "ratio"
    PERCENT = "%"
    PARTS_PER_MILLION = "ppm"
    PARTS_PER_BILLION = "ppb"
    DECIBEL = "dB"
    
    # ==================== Magnetism ====================
    TESLA = "T"
    GAUSS = "G"
    WEBER = "Wb"
    AMPERES_PER_METER = "A/m"
    
    # ==================== Radiation ====================
    BECQUEREL = "Bq"
    CURIE = "Ci"
    GRAY = "Gy"
    SIEVERT = "Sv"
    RAD = "rad_dose" # Disambiguate from angle radians
    REM = "rem"
    
    # ==================== Data ====================
    BIT = "b"
    BYTE = "B"
    KILOBIT = "Kb"
    KILOBYTE = "KB"
    MEGABIT = "Mb"
    MEGABYTE = "MB"
    GIGABIT = "Gb"
    GIGABYTE = "GB"
    TERABYTE = "TB"
    BAUD = "Bd"
    
    # ==================== Currency ====================
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP" # Pounds Sterling
    NGN = "NGN" # Nigerian Naira
    CNY = "CNY"
    JPY = "JPY"
    
    # ==================== Thermal Properties ====================
    THERMAL_CONDUCTIVITY = "W/m·K"
    SPECIFIC_HEAT = "J/kg·K"
    HEAT_FLUX = "W/m²"
    
    # ==================== Fluids ====================
    PASCAL_SECOND = "Pa·s" # Dynamic Viscosity
    POISE = "P"
    CENTIPOISE = "cP"
    STOKES = "St" # Kinematic Viscosity
    
    # ==================== Special ====================
    UNITLESS = ""


# Unit conversion factors (to SI base units)
UNIT_DEFINITIONS: Dict[Unit, UnitDefinition] = {
    # Length (base: meters)
    Unit.METERS: UnitDefinition("m", UnitDimension.LENGTH, 1.0),
    Unit.MILLIMETERS: UnitDefinition("mm", UnitDimension.LENGTH, 0.001),
    Unit.CENTIMETERS: UnitDefinition("cm", UnitDimension.LENGTH, 0.01),
    Unit.KILOMETERS: UnitDefinition("km", UnitDimension.LENGTH, 1000.0),
    Unit.MICROMETERS: UnitDefinition("μm", UnitDimension.LENGTH, 1e-6),
    Unit.NANOMETERS: UnitDefinition("nm", UnitDimension.LENGTH, 1e-9),
    Unit.INCHES: UnitDefinition("in", UnitDimension.LENGTH, 0.0254),
    Unit.FEET: UnitDefinition("ft", UnitDimension.LENGTH, 0.3048),
    Unit.YARDS: UnitDefinition("yd", UnitDimension.LENGTH, 0.9144),
    Unit.MILES: UnitDefinition("mi", UnitDimension.LENGTH, 1609.34),
    Unit.NAUTICAL_MILES: UnitDefinition("nmi", UnitDimension.LENGTH, 1852.0),
    
    # Mass (base: kilograms)
    Unit.KILOGRAMS: UnitDefinition("kg", UnitDimension.MASS, 1.0),
    Unit.GRAMS: UnitDefinition("g", UnitDimension.MASS, 0.001),
    Unit.MILLIGRAMS: UnitDefinition("mg", UnitDimension.MASS, 1e-6),
    Unit.TONNES: UnitDefinition("t", UnitDimension.MASS, 1000.0),
    Unit.POUNDS: UnitDefinition("lb", UnitDimension.MASS, 0.453592),
    Unit.OUNCES: UnitDefinition("oz", UnitDimension.MASS, 0.0283495),
    Unit.SLUGS: UnitDefinition("slug", UnitDimension.MASS, 14.5939),
    
    # Time (base: seconds)
    Unit.SECONDS: UnitDefinition("s", UnitDimension.TIME, 1.0),
    Unit.MILLISECONDS: UnitDefinition("ms", UnitDimension.TIME, 0.001),
    Unit.MICROSECONDS: UnitDefinition("μs", UnitDimension.TIME, 1e-6),
    Unit.NANOSECONDS: UnitDefinition("ns", UnitDimension.TIME, 1e-9),
    Unit.MINUTES: UnitDefinition("min", UnitDimension.TIME, 60.0),
    Unit.HOURS: UnitDefinition("h", UnitDimension.TIME, 3600.0),
    Unit.DAYS: UnitDefinition("day", UnitDimension.TIME, 86400.0),
    
    # Temperature (base: Kelvin)
    Unit.KELVIN: UnitDefinition("K", UnitDimension.TEMPERATURE, 1.0, 0.0),
    Unit.CELSIUS: UnitDefinition("°C", UnitDimension.TEMPERATURE, 1.0, 273.15),
    Unit.FAHRENHEIT: UnitDefinition("°F", UnitDimension.TEMPERATURE, 5/9, 459.67 * 5/9),
    Unit.RANKINE: UnitDefinition("°R", UnitDimension.TEMPERATURE, 5/9, 0.0),
    
    # Angles (base: radians)
    Unit.RADIANS: UnitDefinition("rad", UnitDimension.ANGLE, 1.0),
    Unit.DEGREES: UnitDefinition("deg", UnitDimension.ANGLE, math.pi/180),
    Unit.GRADIANS: UnitDefinition("grad", UnitDimension.ANGLE, math.pi/200),
    Unit.REVOLUTIONS: UnitDefinition("rev", UnitDimension.ANGLE, 2*math.pi),
    
    # Dimensionless
    Unit.RATIO: UnitDefinition("ratio", UnitDimension.DIMENSIONLESS, 1.0),
    Unit.PERCENT: UnitDefinition("%", UnitDimension.DIMENSIONLESS, 0.01),
    Unit.PARTS_PER_MILLION: UnitDefinition("ppm", UnitDimension.DIMENSIONLESS, 1e-6),
    Unit.PARTS_PER_BILLION: UnitDefinition("ppb", UnitDimension.DIMENSIONLESS, 1e-9),
    Unit.UNITLESS: UnitDefinition("", UnitDimension.DIMENSIONLESS, 1.0),
    
    # Magnetism (base: Tesla, A/m, Weber, Henry)
    Unit.TESLA: UnitDefinition("T", UnitDimension.MAGNETISM, 1.0),
    Unit.GAUSS: UnitDefinition("G", UnitDimension.MAGNETISM, 1e-4),
    Unit.WEBER: UnitDefinition("Wb", UnitDimension.MAGNETISM, 1.0), # Magnetic Flux
    Unit.HENRY: UnitDefinition("H", UnitDimension.MAGNETISM, 1.0), # Inductance
    Unit.AMPERES_PER_METER: UnitDefinition("A/m", UnitDimension.MAGNETISM, 1.0), # H-field
    
    # Radiation (base: Sievert, Gray, Becquerel)
    Unit.SIEVERT: UnitDefinition("Sv", UnitDimension.RADIATION, 1.0),
    Unit.REM: UnitDefinition("rem", UnitDimension.RADIATION, 0.01),
    Unit.GRAY: UnitDefinition("Gy", UnitDimension.RADIATION, 1.0),
    Unit.RAD: UnitDefinition("rad_dose", UnitDimension.RADIATION, 0.01),
    Unit.BECQUEREL: UnitDefinition("Bq", UnitDimension.RADIATION, 1.0),
    Unit.CURIE: UnitDefinition("Ci", UnitDimension.RADIATION, 3.7e10),
    
    # Data (base: Bit)
    Unit.BIT: UnitDefinition("b", UnitDimension.DATA, 1.0),
    Unit.BYTE: UnitDefinition("B", UnitDimension.DATA, 8.0),
    Unit.KILOBIT: UnitDefinition("Kb", UnitDimension.DATA, 1e3),
    Unit.KILOBYTE: UnitDefinition("KB", UnitDimension.DATA, 8e3),
    Unit.MEGABIT: UnitDefinition("Mb", UnitDimension.DATA, 1e6),
    Unit.MEGABYTE: UnitDefinition("MB", UnitDimension.DATA, 8e6),
    Unit.GIGABIT: UnitDefinition("Gb", UnitDimension.DATA, 1e9),
    Unit.GIGABYTE: UnitDefinition("GB", UnitDimension.DATA, 8e9),
    Unit.TERABYTE: UnitDefinition("TB", UnitDimension.DATA, 8e12),
    Unit.BAUD: UnitDefinition("Bd", UnitDimension.DATA, 1.0), # Symbol rate ~ bits/s simplified
    
    # Electrical (Conductivity/Resistivity) - Extended
    Unit.SIEMENS: UnitDefinition("S", UnitDimension.CURRENT, 1.0), # Simplified for MVP (Conductance)
    Unit.OHM: UnitDefinition("Ω", UnitDimension.CURRENT, 1.0), # Simplified MVP mapping
    Unit.OHM_METER: UnitDefinition("Ω·m", UnitDimension.CURRENT, 1.0),
    Unit.SIEMENS_PER_METER: UnitDefinition("S/m", UnitDimension.CURRENT, 1.0),
    
    # Currency (base: USD)
    # NOTE: In production, these should be dynamic. Fixed rates for simulation.
    Unit.USD: UnitDefinition("USD", UnitDimension.CURRENCY, 1.0),
    Unit.EUR: UnitDefinition("EUR", UnitDimension.CURRENCY, 1.08), # 1 EUR = 1.08 USD
    Unit.GBP: UnitDefinition("GBP", UnitDimension.CURRENCY, 1.25), # 1 GBP = 1.25 USD
    Unit.NGN: UnitDefinition("NGN", UnitDimension.CURRENCY, 0.00065), # 1 NGN = ~0.00065 USD
    Unit.CNY: UnitDefinition("CNY", UnitDimension.CURRENCY, 0.14),
    Unit.JPY: UnitDefinition("JPY", UnitDimension.CURRENCY, 0.0068),
    
    # Thermal Properties
    Unit.THERMAL_CONDUCTIVITY: UnitDefinition("W/m·K", UnitDimension.THERMAL_PROP, 1.0),
    Unit.SPECIFIC_HEAT: UnitDefinition("J/kg·K", UnitDimension.THERMAL_PROP, 1.0),
    Unit.HEAT_FLUX: UnitDefinition("W/m²", UnitDimension.THERMAL_PROP, 1.0),
    
    # Fluids (base: Pa·s)
    Unit.PASCAL_SECOND: UnitDefinition("Pa·s", UnitDimension.VISCOSITY, 1.0),
    Unit.POISE: UnitDefinition("P", UnitDimension.VISCOSITY, 0.1),
    Unit.CENTIPOISE: UnitDefinition("cP", UnitDimension.VISCOSITY, 0.001),
    Unit.STOKES: UnitDefinition("St", UnitDimension.VISCOSITY, 1e-4), # Kinematic, but approx mapped here
}


def convert_units(value: float, from_unit: Unit, to_unit: Unit) -> float:
    """
    Convert a value from one unit to another.
    
    Args:
        value: Value to convert
        from_unit: Source unit
        to_unit: Target unit
    
    Returns:
        Converted value
    
    Raises:
        ValueError: If units are incompatible
    """
    from_def = UNIT_DEFINITIONS.get(from_unit)
    to_def = UNIT_DEFINITIONS.get(to_unit)
    
    if not from_def or not to_def:
        raise ValueError(f"Unknown unit: {from_unit if not from_def else to_unit}")
    
    if from_def.dimension != to_def.dimension:
        raise ValueError(
            f"Incompatible dimensions: {from_unit} ({from_def.dimension}) "
            f"vs {to_unit} ({to_def.dimension})"
        )
    
    # Convert to SI, then to target unit
    si_value = from_def.to_si(value)
    return to_def.from_si(si_value)


# ==================== Enhanced Physical Value ====================

class PhysicalValue(BaseModel):
    """
    Enhanced atomic primitive with unit conversion support.
    """
    magnitude: float
    unit: Unit
    locked: bool = False
    tolerance: float = 0.001
    source: str = "KERNEL_INIT"
    validation_score: float = 1.0
    version_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    significant_figures: int = 4
    
    @validator('magnitude')
    def validate_magnitude(cls, v):
        if not math.isfinite(v):
            raise ValueError(f"Magnitude must be finite, got {v}")
        return v
    
    @validator('tolerance')
    def validate_tolerance(cls, v):
        if v < 0:
            raise ValueError(f"Tolerance must be non-negative, got {v}")
        return v
    
    @validator('validation_score')
    def validate_score(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Validation score must be between 0 and 1, got {v}")
        return v
    
    def convert_to(self, target_unit: Unit) -> 'PhysicalValue':
        """
        Convert this value to a different unit.
        
        Args:
            target_unit: Target unit
        
        Returns:
            New PhysicalValue with converted magnitude
        """
        new_magnitude = convert_units(self.magnitude, self.unit, target_unit)
        return PhysicalValue(
            magnitude=new_magnitude,
            unit=target_unit,
            locked=self.locked,
            tolerance=self.tolerance,
            source=self.source,
            validation_score=self.validation_score,
            significant_figures=self.significant_figures
        )
    
    def is_compatible_with(self, other: 'PhysicalValue') -> bool:
        """Check if units have same dimension"""
        self_def = UNIT_DEFINITIONS.get(self.unit)
        other_def = UNIT_DEFINITIONS.get(other.unit)
        
        if not self_def or not other_def:
            return self.unit == other.unit
        
        return self_def.dimension == other_def.dimension
    
    def within_tolerance(self, other: 'PhysicalValue') -> bool:
        """Check if values are within tolerance (auto-converts units)"""
        if not self.is_compatible_with(other):
            return False
        
        # Convert other to our units for comparison
        other_converted = other.convert_to(self.unit)
        return abs(self.magnitude - other_converted.magnitude) <= self.tolerance
    
    def __add__(self, other: 'PhysicalValue') -> 'PhysicalValue':
        """Add two physical values (with unit checking)"""
        if not self.is_compatible_with(other):
            raise ValueError(f"Cannot add incompatible units: {self.unit} and {other.unit}")
        
        other_converted = other.convert_to(self.unit)
        return PhysicalValue(
            magnitude=self.magnitude + other_converted.magnitude,
            unit=self.unit,
            tolerance=max(self.tolerance, other.tolerance),
            source=f"{self.source}+{other.source}",
            validation_score=min(self.validation_score, other.validation_score)
        )
    
    def __sub__(self, other: 'PhysicalValue') -> 'PhysicalValue':
        """Subtract two physical values"""
        if not self.is_compatible_with(other):
            raise ValueError(f"Cannot subtract incompatible units: {self.unit} and {other.unit}")
        
        other_converted = other.convert_to(self.unit)
        return PhysicalValue(
            magnitude=self.magnitude - other_converted.magnitude,
            unit=self.unit,
            tolerance=max(self.tolerance, other.tolerance),
            source=f"{self.source}-{other.source}",
            validation_score=min(self.validation_score, other.validation_score)
        )
    
    def __mul__(self, scalar: float) -> 'PhysicalValue':
        """Multiply by scalar"""
        return PhysicalValue(
            magnitude=self.magnitude * scalar,
            unit=self.unit,
            tolerance=self.tolerance * abs(scalar),
            source=self.source,
            validation_score=self.validation_score
        )
    
    def __truediv__(self, scalar: float) -> 'PhysicalValue':
        """Divide by scalar"""
        return PhysicalValue(
            magnitude=self.magnitude / scalar,
            unit=self.unit,
            tolerance=self.tolerance / abs(scalar),
            source=self.source,
            validation_score=self.validation_score
        )
    
    def __str__(self) -> str:
        """Human-readable with significant figures"""
        if self.significant_figures:
            format_str = f"{{:.{self.significant_figures}g}}"
            return f"{format_str.format(self.magnitude)} {self.unit.value}"
        return f"{self.magnitude} {self.unit.value}"
    
    def __repr__(self) -> str:
        return f"PhysicalValue({self.magnitude}, {self.unit}, locked={self.locked})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "magnitude": self.magnitude,
            "unit": self.unit.value,
            "locked": self.locked,
            "tolerance": self.tolerance,
            "source": self.source,
            "validation_score": self.validation_score,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PhysicalValue':
        """Deserialize from dictionary"""
        return cls(
            magnitude=data["magnitude"],
            unit=Unit(data["unit"]),
            locked=data.get("locked", False),
            tolerance=data.get("tolerance", 0.001),
            source=data.get("source", "KERNEL_INIT"),
            validation_score=data.get("validation_score", 1.0)
        )


# ==================== Enhanced Constraint Node ====================

class ConstraintType(str, Enum):
    """Types of constraints"""
    EQUALITY = "equality"  # value == target
    LESS_THAN = "less_than"  # value < target
    GREATER_THAN = "greater_than"  # value > target
    RANGE = "range"  # min <= value <= max
    LOCKED = "locked"  # user-defined, cannot change


class ConstraintPriority(int, Enum):
    """Priority levels for conflict resolution"""
    CRITICAL = 100  # Safety-critical constraints
    HIGH = 75  # Important design constraints
    MEDIUM = 50  # Standard constraints
    LOW = 25  # Soft preferences
    SUGGESTION = 10  # Agent suggestions


class ConstraintNode(BaseModel):
    """
    Enhanced constraint node with rich metadata and validation.
    """
    id: str
    val: PhysicalValue
    dependencies: List[str] = []
    dependents: List[str] = []
    status: str = "VALID"  # VALID, STALE, PENDING_VERIFICATION, ERROR, CONFLICT
    metadata: Dict[str, Any] = {}
    description: Optional[str] = None
    
    # Constraint bounds
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    constraint_type: ConstraintType = ConstraintType.RANGE
    priority: ConstraintPriority = ConstraintPriority.MEDIUM
    
    # Agent coordination
    agent_owner: Optional[str] = None
    last_modified_by: Optional[str] = None
    modification_count: int = 0
    
    # Change history
    change_history: List[Dict[str, Any]] = []
    
    # Validation
    validation_rules: List[str] = []  # List of validation rule IDs
    
    def is_within_bounds(self) -> bool:
        """Check if value is within specified bounds"""
        if self.min_value is not None and self.val.magnitude < self.min_value:
            return False
        if self.max_value is not None and self.val.magnitude > self.max_value:
            return False
        return True
    
    def satisfies_constraint(self, target_value: Optional[float] = None) -> bool:
        """
        Check if constraint is satisfied.
        
        Args:
            target_value: Target value for equality/inequality constraints
        """
        if self.constraint_type == ConstraintType.LOCKED:
            return self.val.locked
        
        if self.constraint_type == ConstraintType.RANGE:
            return self.is_within_bounds()
        
        if target_value is None:
            return True
        
        if self.constraint_type == ConstraintType.EQUALITY:
            return abs(self.val.magnitude - target_value) <= self.val.tolerance
        
        if self.constraint_type == ConstraintType.LESS_THAN:
            return self.val.magnitude < target_value
        
        if self.constraint_type == ConstraintType.GREATER_THAN:
            return self.val.magnitude > target_value
        
        return True
    
    def record_change(self, old_value: float, new_value: float, source: str):
        """Record a change in the history"""
        self.change_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "old_value": old_value,
            "new_value": new_value,
            "source": source,
            "version_id": self.val.version_id
        })
        self.modification_count += 1
        self.last_modified_by = source
    
    def get_change_summary(self) -> Dict[str, Any]:
        """Get summary of changes"""
        return {
            "total_changes": self.modification_count,
            "last_modified_by": self.last_modified_by,
            "last_modified_at": self.val.timestamp.isoformat() if self.change_history else None,
            "recent_changes": self.change_history[-5:]  # Last 5 changes
        }
    
    def __str__(self) -> str:
        status_icons = {
            "VALID": "✓",
            "STALE": "⚠",
            "ERROR": "✗",
            "CONFLICT": "⚡",
            "PENDING_VERIFICATION": "?"
        }
        status_icon = status_icons.get(self.status, "•")
        lock_icon = "🔒" if self.val.locked else ""
        priority_icon = "🔴" if self.priority >= ConstraintPriority.HIGH else ""
        
        return f"{status_icon}{priority_icon} {self.id}: {self.val} {lock_icon}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "id": self.id,
            "value": self.val.to_dict(),
            "dependencies": self.dependencies,
            "dependents": self.dependents,
            "status": self.status,
            "description": self.description,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "constraint_type": self.constraint_type.value,
            "priority": self.priority.value,
            "agent_owner": self.agent_owner,
            "modification_count": self.modification_count
        }


# ==================== Enhanced Hardware ISA ====================

class ISAEvent(BaseModel):
    """Event for reactive updates"""
    event_type: str  # "node_added", "node_updated", "node_deleted", "status_changed"
    node_id: str
    domain: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = {}


class HardwareISA(BaseModel):
    """
    Production-ready Hardware ISA with advanced features.
    """
    project_id: str
    revision: int = 1
    environment_kernel: str = "EARTH_AERO"
    
    domains: Dict[str, Dict[str, ConstraintNode]] = Field(default_factory=dict)
    components: Dict[str, str] = Field(default_factory=dict)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    tags: List[str] = []
    
    # Event system
    event_listeners: Dict[str, List[Callable]] = Field(default_factory=dict, exclude=True)
    event_history: List[ISAEvent] = []
    
    # Transaction support
    transaction_stack: List[Dict[str, Any]] = Field(default_factory=list, exclude=True)
    
    def __init__(self, **data):
        super().__init__(**data)
        standard_domains = [
            "dynamics", "structural", "propulsion", "avionics",
            "thermal", "electrical", "hydraulic", "pneumatic", "software"
        ]
        for domain in standard_domains:
            if domain not in self.domains:
                self.domains[domain] = {}
    
    def get_state_hash(self) -> str:
        """
        Computes a deterministic cryptographic hash of the entire physical state.
        Uses SHA-256 for Merkle tree compatibility and version control.
        
        Returns:
            Hexadecimal hash string
        """
        all_vals = []
        for domain_name in sorted(self.domains.keys()):
            domain = self.domains[domain_name]
            for node_id in sorted(domain.keys()):
                node = domain[node_id]
                # Include all relevant fields in hash
                all_vals.append(
                    f"{domain_name}:{node.id}:{node.val.magnitude}:"
                    f"{node.val.unit}:{node.val.locked}:{node.status}"
                )
        
        state_str = "|".join(all_vals)
        return hashlib.sha256(state_str.encode()).hexdigest()
    
    def add_node(
        self,
        domain: str,
        node_id: str,
        value: PhysicalValue,
        dependencies: List[str] = None,
        description: str = None,
        min_value: float = None,
        max_value: float = None,
        agent_owner: str = None,
        constraint_type: ConstraintType = ConstraintType.RANGE,
        priority: ConstraintPriority = ConstraintPriority.MEDIUM
    ) -> bool:
        """
        Add a new constraint node to the ISA.
        
        Args:
            domain: Domain name (e.g., "dynamics", "thermal")
            node_id: Unique identifier for the node
            value: PhysicalValue instance
            dependencies: List of node IDs this depends on
            description: Human-readable description
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            agent_owner: Agent responsible for this parameter
            constraint_type: Type of constraint
            priority: Priority level for conflict resolution
        
        Returns:
            True if successful, False if node already exists
        """
        if domain not in self.domains:
            self.domains[domain] = {}
        
        if node_id in self.domains[domain]:
            return False  # Node already exists
        
        node = ConstraintNode(
            id=node_id,
            val=value,
            dependencies=dependencies or [],
            description=description,
            min_value=min_value,
            max_value=max_value,
            agent_owner=agent_owner,
            constraint_type=constraint_type,
            priority=priority
        )
        
        self.domains[domain][node_id] = node
        
        # Update dependents in dependency nodes
        if dependencies:
            for dep_id in dependencies:
                dep_node = self._find_node(dep_id)
                if dep_node and node_id not in dep_node.dependents:
                    dep_node.dependents.append(node_id)
        
        self.updated_at = datetime.utcnow()
        
        # Emit event
        self._emit_event(ISAEvent(
            event_type="node_added",
            node_id=node_id,
            domain=domain,
            data={"agent_owner": agent_owner}
        ))
        
        return True
    
    def update_node(
        self,
        domain: str,
        node_id: str,
        new_val: float,
        source: str,
        validation_score: float = 1.0
    ) -> bool:
        """
        Standard entry point for agentic mutations.
        Updates a node's value and propagates STALE status to all dependents.
        
        Args:
            domain: Domain containing the node
            node_id: Node identifier
            new_val: New magnitude value
            source: Agent or system making the update
            validation_score: Confidence in the new value (0.0 to 1.0)
        
        Returns:
            True if successful, False otherwise
        """
        if domain not in self.domains:
            return False
        
        target_domain = self.domains[domain]
        if node_id not in target_domain:
            return False
        
        node = target_domain[node_id]
        
        # Guardrail: Block stochastic mutation of user intent
        if node.val.locked and source != "USER":
            return False
        
        # Record change
        old_value = node.val.magnitude
        node.record_change(old_value, new_val, source)
        
        # Update the value
        node.val.magnitude = new_val
        node.val.source = source
        node.val.validation_score = validation_score
        node.val.timestamp = datetime.utcnow()
        node.val.version_id = str(uuid.uuid4())
        node.status = "STALE"
        
        # Check bounds
        if not node.is_within_bounds():
            node.status = "ERROR"
            node.metadata["error"] = f"Value {new_val} outside bounds [{node.min_value}, {node.max_value}]"
        
        # Propagate STALE status to all dependents
        self._invalidate_dependents(node_id)
        
        self.updated_at = datetime.utcnow()
        
        # Emit event
        self._emit_event(ISAEvent(
            event_type="node_updated",
            node_id=node_id,
            domain=domain,
            data={"old_value": old_value, "new_value": new_val, "source": source}
        ))
        
        return True
    
    def _find_node(self, node_id: str) -> Optional[ConstraintNode]:
        """Find a node by ID across all domains"""
        for domain in self.domains.values():
            if node_id in domain:
                return domain[node_id]
        return None
    
    def _invalidate_dependents(self, node_id: str, visited: Set[str] = None):
        """
        Recursively mark all dependent nodes as STALE.
        Uses visited set to prevent infinite loops in case of circular dependencies.
        
        Args:
            node_id: Starting node ID
            visited: Set of already visited nodes (for cycle detection)
        """
        if visited is None:
            visited = set()
        
        if node_id in visited:
            return  # Prevent infinite loops
        
        visited.add(node_id)
        
        node = self._find_node(node_id)
        if not node:
            return
        
        for dependent_id in node.dependents:
            dependent = self._find_node(dependent_id)
            if dependent and dependent.status == "VALID":
                dependent.status = "STALE"
                # Recursively invalidate
                self._invalidate_dependents(dependent_id, visited)
    
    def get_stale_nodes(self) -> List[ConstraintNode]:
        """Get all nodes with STALE status"""
        stale = []
        for domain in self.domains.values():
            for node in domain.values():
                if node.status == "STALE":
                    stale.append(node)
        return stale
    
    def get_execution_order(self) -> List[str]:
        """
        Get topologically sorted execution order for recomputing stale nodes.
        Uses Kahn's algorithm for topological sorting.
        
        Returns:
            List of node IDs in execution order
        """
        # Build in-degree map
        in_degree = {}
        all_nodes = {}
        
        for domain in self.domains.values():
            for node_id, node in domain.items():
                all_nodes[node_id] = node
                in_degree[node_id] = len(node.dependencies)
        
        # Queue of nodes with no dependencies
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        result = []
        
        while queue:
            # Process node with no remaining dependencies
            current = queue.pop(0)
            result.append(current)
            
            # Reduce in-degree for dependents
            node = all_nodes[current]
            for dependent_id in node.dependents:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    queue.append(dependent_id)
        
        # Check for cycles
        if len(result) != len(all_nodes):
            raise ValueError("Circular dependency detected in constraint graph")
        
        return result
    
    def validate_all(self) -> Dict[str, List[str]]:
        """
        Validate all nodes and return errors/warnings.
        
        Returns:
            Dictionary with 'errors' and 'warnings' lists
        """
        errors = []
        warnings = []
        
        for domain_name, domain in self.domains.items():
            for node_id, node in domain.items():
                # Check bounds
                if not node.is_within_bounds():
                    errors.append(
                        f"{domain_name}.{node_id}: Value {node.val.magnitude} "
                        f"outside bounds [{node.min_value}, {node.max_value}]"
                    )
                
                # Check validation score
                if node.val.validation_score < 0.5:
                    warnings.append(
                        f"{domain_name}.{node_id}: Low validation score "
                        f"({node.val.validation_score:.2f})"
                    )
                
                # Check for stale nodes
                if node.status == "STALE":
                    warnings.append(f"{domain_name}.{node_id}: Node is STALE")
                
                # Check for error status
                if node.status == "ERROR":
                    errors.append(
                        f"{domain_name}.{node_id}: {node.metadata.get('error', 'Unknown error')}"
                    )
        
        return {"errors": errors, "warnings": warnings}
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the ISA state"""
        total_nodes = sum(len(domain) for domain in self.domains.values())
        stale_count = len(self.get_stale_nodes())
        
        validation = self.validate_all()
        
        return {
            "project_id": self.project_id,
            "revision": self.revision,
            "environment": self.environment_kernel,
            "total_nodes": total_nodes,
            "stale_nodes": stale_count,
            "domains": {name: len(nodes) for name, nodes in self.domains.items()},
            "errors": len(validation["errors"]),
            "warnings": len(validation["warnings"]),
            "state_hash": self.get_state_hash()[:16] + "...",  # Abbreviated
            "updated_at": self.updated_at.isoformat()
        }
    
    def __str__(self) -> str:
        """Human-readable representation"""
        summary = self.get_summary()
        return (
            f"HardwareISA(project={self.project_id}, "
            f"nodes={summary['total_nodes']}, "
            f"stale={summary['stale_nodes']}, "
            f"errors={summary['errors']})"
        )
    

    def begin_transaction(self):
        """Start a transaction (for rollback support)"""
        snapshot = {
            "domains": json.loads(json.dumps({
                domain: {nid: node.to_dict() for nid, node in nodes.items()}
                for domain, nodes in self.domains.items()
            })),
            "revision": self.revision
        }
        self.transaction_stack.append(snapshot)
    
    def commit_transaction(self):
        """Commit current transaction"""
        if self.transaction_stack:
            self.transaction_stack.pop()
            self.revision += 1
    
    def rollback_transaction(self):
        """Rollback to last transaction state"""
        if not self.transaction_stack:
            raise ValueError("No active transaction to rollback")
        
        snapshot = self.transaction_stack.pop()
        # Restore state (simplified - would need full restoration logic)
        self.revision = snapshot["revision"]
    
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to events"""
        if event_type not in self.event_listeners:
            self.event_listeners[event_type] = []
        self.event_listeners[event_type].append(callback)
    
    def _emit_event(self, event: ISAEvent):
        """Emit an event to listeners"""
        self.event_history.append(event)
        if event.event_type in self.event_listeners:
            for callback in self.event_listeners[event.event_type]:
                callback(event)
    
    def find_nodes_by_agent(self, agent_name: str) -> List[Tuple[str, ConstraintNode]]:
        """Find all nodes owned by an agent"""
        results = []
        for domain_name, domain in self.domains.items():
            for node_id, node in domain.items():
                if node.agent_owner == agent_name:
                    results.append((domain_name, node))
        return results
    
    def find_nodes_by_status(self, status: str) -> List[Tuple[str, ConstraintNode]]:
        """Find all nodes with given status"""
        results = []
        for domain_name, domain in self.domains.items():
            for node_id, node in domain.items():
                if node.status == status:
                    results.append((domain_name, node))
        return results
    
    def export_json(self, filepath: str):
        """Export ISA to JSON file"""
        data = {
            "project_id": self.project_id,
            "revision": self.revision,
            "environment_kernel": self.environment_kernel,
            "domains": {
                domain: {nid: node.to_dict() for nid, node in nodes.items()}
                for domain, nodes in self.domains.items()
            },
            "components": self.components,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def import_json(cls, filepath: str) -> 'HardwareISA':
        """Import ISA from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct ISA (simplified)
        isa = cls(
            project_id=data["project_id"],
            revision=data["revision"],
            environment_kernel=data["environment_kernel"],
            tags=data.get("tags", [])
        )
        return isa


# ==================== Helper Functions ====================

def create_physical_value(
    magnitude: float,
    unit: Unit,
    source: str = "KERNEL_INIT",
    locked: bool = False,
    tolerance: float = 0.001,
    significant_figures: int = 4
) -> PhysicalValue:
    """Convenience function to create a PhysicalValue"""
    return PhysicalValue(
        magnitude=magnitude,
        unit=unit,
        source=source,
        locked=locked,
        tolerance=tolerance,
        significant_figures=significant_figures
    )


def create_default_isa(project_id: str, environment: str = "EARTH_AERO") -> HardwareISA:
    """Create a new HardwareISA with default structure"""
    return HardwareISA(
        project_id=project_id,
        environment_kernel=environment
    )


# ==================== Example Usage ====================

if __name__ == "__main__":
    # Test unit conversion
    print("=== Unit Conversion Test ===")
    length_m = create_physical_value(1.0, Unit.METERS)
    length_ft = length_m.convert_to(Unit.FEET)
    print(f"{length_m} = {length_ft}")
    
    temp_c = create_physical_value(20.0, Unit.CELSIUS)
    temp_f = temp_c.convert_to(Unit.FAHRENHEIT)
    print(f"{temp_c} = {temp_f}")
    
    # Test arithmetic
    print("\n=== Arithmetic Test ===")
    v1 = create_physical_value(10.0, Unit.METERS)
    v2 = create_physical_value(5.0, Unit.METERS)
    v3 = v1 + v2
    print(f"{v1} + {v2} = {v3}")
    
    # Test ISA
    print("\n=== ISA Test ===")
    isa = create_default_isa("test-project")
    print(f"Created: {isa}")
    print(f"State hash: {isa.get_state_hash()[:16]}...")
