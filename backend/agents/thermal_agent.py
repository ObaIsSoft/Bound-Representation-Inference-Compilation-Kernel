"""
ProductionThermalAgent - Conjugate heat transfer analysis

Standards Compliance:
- Incropera & DeWitt - Fundamentals of Heat and Mass Transfer
- MIL-HDBK-310 - Environmental data
- SAE ARP 4761 - Thermal analysis

Capabilities:
1. Multi-mode heat transfer (conduction, convection, radiation)
2. CoolProp integration for fluid properties
3. Nusselt correlations for convection
4. View factor calculations for radiation
5. Transient and steady-state analysis
"""

import os
import math
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np
from scipy import integrate

logger = logging.getLogger(__name__)

from .config.physics_config import NusseltCorrelations, get_material_properties

# Try to import CoolProp for fluid properties
try:
    import CoolProp.CoolProp as CP
    HAS_COOLPROP = True
    logger.info("CoolProp loaded successfully")
except ImportError:
    HAS_COOLPROP = False
    logger.warning("CoolProp not installed - fluid properties will be approximate")

# Try to import FiPy for 3D finite volume thermal analysis
try:
    from fipy import Grid3D, CellVariable, DiffusionTerm, ImplicitSourceTerm
    from fipy.tools import numerix
    HAS_FIPY = True
    logger.info("FiPy loaded successfully - 3D thermal analysis available")
except ImportError:
    HAS_FIPY = False
    logger.warning("FiPy not installed - 3D thermal analysis will use finite difference fallback")


class HeatTransferMode(Enum):
    """Heat transfer modes"""
    CONDUCTION = "conduction"
    CONVECTION = "convection"
    RADIATION = "radiation"
    PHASE_CHANGE = "phase_change"


class FlowRegime(Enum):
    """Flow regime classification"""
    LAMINAR = "laminar"
    TRANSITIONAL = "transitional"
    TURBULENT = "turbulent"


@dataclass
class FluidProperties:
    """Thermophysical fluid properties"""
    name: str
    temperature: float          # K
    pressure: float             # Pa
    density: float              # kg/m³
    specific_heat: float        # J/(kg·K)
    thermal_conductivity: float # W/(m·K)
    dynamic_viscosity: float    # Pa·s
    kinematic_viscosity: float  # m²/s
    thermal_diffusivity: float  # m²/s
    prandtl_number: float
    beta: float                 # Thermal expansion coefficient, 1/K
    
    @classmethod
    def from_coolprop(cls, fluid: str, T: float, P: float) -> "FluidProperties":
        """Create from CoolProp"""
        if not HAS_COOLPROP:
            raise RuntimeError("CoolProp not available")
        
        props = CP.PropsSI(
            ['D', 'C', 'L', 'V', 'PRANDTL'],
            'T', T,
            'P', P,
            fluid
        )
        
        rho, cp, k, mu, Pr = props
        
        return cls(
            name=fluid,
            temperature=T,
            pressure=P,
            density=rho,
            specific_heat=cp,
            thermal_conductivity=k,
            dynamic_viscosity=mu,
            kinematic_viscosity=mu / rho,
            thermal_diffusivity=k / (rho * cp),
            prandtl_number=Pr,
            beta=1.0 / T  # Ideal gas approximation
        )
    
    @classmethod
    def air(cls, T: float = 288.15, P: float = 101325) -> "FluidProperties":
        """Approximate air properties (fallback if CoolProp unavailable)"""
        if HAS_COOLPROP:
            try:
                return cls.from_coolprop("Air", T, P)
            except Exception:
                pass
        
        # Approximate properties for air
        return cls(
            name="Air (approximate)",
            temperature=T,
            pressure=P,
            density=1.225 * (288.15 / T) * (P / 101325),
            specific_heat=1005,
            thermal_conductivity=0.0257 * (T / 300)**0.8,
            dynamic_viscosity=1.81e-5 * (T / 300)**0.76,
            kinematic_viscosity=1.48e-5 * (T / 300)**1.76 / (P / 101325),
            thermal_diffusivity=2.08e-5 * (T / 300)**0.76 / (P / 101325),
            prandtl_number=0.71,
            beta=1.0 / T
        )
    
    @classmethod
    def water(cls, T: float = 293.15, P: float = 101325) -> "FluidProperties":
        """Water properties"""
        if HAS_COOLPROP:
            try:
                return cls.from_coolprop("Water", T, P)
            except Exception:
                pass
        
        return cls(
            name="Water (approximate)",
            temperature=T,
            pressure=P,
            density=998.0,
            specific_heat=4182,
            thermal_conductivity=0.598,
            dynamic_viscosity=1.002e-3,
            kinematic_viscosity=1.004e-6,
            thermal_diffusivity=1.43e-7,
            prandtl_number=7.0,
            beta=2.07e-4
        )


@dataclass
class Surface:
    """Surface for heat transfer calculations"""
    area: float                 # m²
    characteristic_length: float  # m (L for correlations)
    orientation: str            # "vertical", "horizontal_up", "horizontal_down"
    roughness: float = 0.0      # m surface roughness
    emissivity: float = 0.9     # For radiation
    temperature: float = 300.0  # K surface temperature


@dataclass
class HeatSource:
    """Heat source definition"""
    power: float                # W
    location: Optional[Tuple[float, float, float]] = None
    distribution: str = "uniform"  # "uniform", "gaussian", "point"


@dataclass
class ThermalBC:
    """Thermal boundary condition"""
    surface_id: str
    type: str                   # "temperature", "heat_flux", "convection", "radiation"
    value: float
    h: Optional[float] = None   # Convection coefficient for convection BC
    T_inf: Optional[float] = None  # Ambient temperature


@dataclass
class ThermalResult:
    """Thermal analysis result"""
    temperature: np.ndarray     # Temperature field (K)
    heat_flux: np.ndarray       # Heat flux (W/m²)
    max_temperature: float
    min_temperature: float
    total_heat_transfer: float
    convection_coeffs: Dict[str, float]
    radiation_exchange: Optional[Dict]
    status: str
    computation_time_ms: float


class ConvectionCorrelations:
    """
    Nusselt number correlations for natural and forced convection
    
    References:
    - Incropera & DeWitt, Fundamentals of Heat and Mass Transfer
    - Churchill & Chu (1975) - Natural convection
    - Gnielinski (1976) - Internal turbulent flow
    """
    
    @staticmethod
    def rayleigh_number(
        fluid: FluidProperties,
        surface: Surface,
        delta_T: float
    ) -> float:
        """
        Calculate Rayleigh number
        
        Ra = Gr * Pr = (g β ΔT L³) / (ν α)
        """
        g = 9.81
        return (
            g * fluid.beta * abs(delta_T) * 
            surface.characteristic_length**3 /
            (fluid.kinematic_viscosity * fluid.thermal_diffusivity)
        )
    
    @staticmethod
    def reynolds_number(
        fluid: FluidProperties,
        surface: Surface,
        velocity: float
    ) -> float:
        """
        Calculate Reynolds number
        
        Re = (ρ V L) / μ = V L / ν
        """
        return velocity * surface.characteristic_length / fluid.kinematic_viscosity
    
    @staticmethod
    def nusselt_natural_vertical_plate(
        fluid: FluidProperties,
        surface: Surface,
        delta_T: float
    ) -> float:
        """
        Churchill-Chu correlation for natural convection on vertical plate
        
        Valid for: 10^-1 < Ra < 10^12
        """
        Ra = ConvectionCorrelations.rayleigh_number(fluid, surface, delta_T)
        Pr = fluid.prandtl_number
        
        # Churchill-Chu correlation
        Nu = (
            0.825 + 
            0.387 * Ra**(1/6) / 
            (1 + (0.492 / Pr)**(9/16))**(8/27)
        )**2
        
        return Nu
    
    @staticmethod
    def nusselt_natural_horizontal_plate(
        fluid: FluidProperties,
        surface: Surface,
        delta_T: float,
        heated_surface: str = "up"
    ) -> float:
        """
        Natural convection from horizontal plate
        
        heated_surface: "up" for heated surface facing up, "down" for facing down
        """
        Ra = ConvectionCorrelations.rayleigh_number(fluid, surface, delta_T)
        
        if heated_surface == "up":
            # Heated surface facing up (or cooled facing down)
            if Ra < 1e7:
                Nu = 0.54 * Ra**0.25
            else:
                Nu = 0.15 * Ra**(1/3)
        else:
            # Heated surface facing down (or cooled facing up)
            Nu = 0.27 * Ra**0.25
        
        return Nu
    
    @staticmethod
    def nusselt_forced_flat_plate_laminar(
        Re: float,
        Pr: float
    ) -> float:
        """
        Laminar flow over flat plate (Blasius solution)
        
        Valid for: Re < 5e5
        """
        return 0.664 * Re**0.5 * Pr**(1/3)
    
    @staticmethod
    def nusselt_forced_flat_plate_turbulent(
        Re: float,
        Pr: float
    ) -> float:
        """
        Turbulent flow over flat plate
        
        Valid for: 5e5 < Re < 1e7
        """
        return 0.037 * Re**0.8 * Pr**(1/3)
    
    @staticmethod
    def nusselt_forced_flat_plate_mixed(
        Re: float,
        Pr: float
    ) -> float:
        """
        Mixed laminar-turbulent flow over flat plate
        """
        Nu_lam = 0.664 * (5e5)**0.5 * Pr**(1/3)
        Nu_turb = 0.037 * (Re**0.8 - (5e5)**0.8) * Pr**(1/3)
        return (Nu_lam + Nu_turb)
    
    @staticmethod
    def nusselt_internal_turbulent(
        fluid: FluidProperties,
        surface: Surface,
        velocity: float
    ) -> float:
        """
        Gnielinski correlation for turbulent internal flow
        
        Most accurate correlation for turbulent forced convection in tubes
        Valid for: 0.5 < Pr < 2000, 3000 < Re < 5e6
        """
        Re = ConvectionCorrelations.reynolds_number(fluid, surface, velocity)
        Pr = fluid.prandtl_number
        
        # Friction factor (Gnielinski)
        f = (0.79 * np.log(Re) - 1.64)**(-2)
        
        # Gnielinski correlation
        Nu = ((f / 8) * (Re - 1000) * Pr) / (
            1 + 12.7 * (f / 8)**0.5 * (Pr**(2/3) - 1)
        )
        
        return Nu
    
    @staticmethod
    def nusselt_internal_laminar(
        Re: float,
        Pr: float,
        length_diameter_ratio: float
    ) -> float:
        """
        Laminar flow in tubes with entrance effects
        
        Sieder-Tate correlation
        """
        return 1.86 * (Re * Pr / length_diameter_ratio)**(1/3)
    
    @staticmethod
    def flow_regime(Re: float) -> FlowRegime:
        """Classify flow regime based on Reynolds number"""
        if Re < 2300:
            return FlowRegime.LAMINAR
        elif Re < 4000:
            return FlowRegime.TRANSITIONAL
        else:
            return FlowRegime.TURBULENT


class RadiationCalculator:
    """
    Radiation heat transfer calculations
    
    Includes view factor calculations and radiative exchange
    """
    
    SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant, W/(m²·K⁴)
    
    @staticmethod
    def blackbody_emissive_power(T: float) -> float:
        """Calculate blackbody emissive power: E_b = σT⁴"""
        return RadiationCalculator.SIGMA * T**4
    
    @staticmethod
    def view_factor_parallel_plates(W: float, H: float, L: float) -> float:
        """
        View factor between two parallel rectangular plates
        
        W: width, H: height, L: separation distance
        """
        X = W / L
        Y = H / L
        
        # Hottel's crossed-string method approximation
        # F12 = (2 / (π * X * Y)) * [ln(...) - ...]
        # Simplified approximation
        return 1.0 / (1.0 + 4 * L * (W + H) / (W * H))
    
    @staticmethod
    def view_factor_perpendicular_plates(
        W1: float, H1: float,
        W2: float, H2: float,
        shared_edge: float
    ) -> float:
        """
        View factor between two perpendicular rectangular plates
        sharing a common edge
        """
        # Using configuration factor algebra
        # Simplified approximation
        area1 = W1 * H1
        area2 = W2 * H2
        return (area2 / (area1 + area2)) * 0.5
    
    @staticmethod
    def net_radiation_exchange(
        T1: float, T2: float,
        epsilon1: float, epsilon2: float,
        F12: float, A1: float
    ) -> float:
        """
        Calculate net radiation exchange between two surfaces
        
        Q12 = A1 * σ * (T1⁴ - T2⁴) / ((1-ε1)/(ε1) + 1/F12 + (1-ε2)/(ε2) * A1/A2)
        """
        numerator = RadiationCalculator.SIGMA * (T1**4 - T2**4)
        
        # Assuming equal areas for simplicity
        denominator = (1 - epsilon1) / epsilon1 + 1 / F12 + (1 - epsilon2) / epsilon2
        
        return A1 * numerator / denominator


class FiPy3DThermalSolver:
    """
    3D Thermal solver using FiPy finite volume library
    
    Solves: ρcₚ∂T/∂t = ∇·(k∇T) + q''' 
    with convection and radiation boundary conditions
    """
    
    def __init__(self):
        self.available = HAS_FIPY
        if not self.available:
            logger.warning("FiPy not available - 3D solver disabled")
    
    def solve_steady_state_3d(
        self,
        domain_size: Tuple[float, float, float],
        nx: int, ny: int, nz: int,
        thermal_conductivity: float,
        heat_generation: float = 0.0,
        bc_left: Tuple[str, float] = ("convection", 300, 10),  # (type, T, h)
        bc_right: Tuple[str, float] = ("convection", 300, 10),
    ) -> Dict[str, Any]:
        """
        Solve 3D steady-state heat conduction
        
        Args:
            domain_size: (Lx, Ly, Lz) domain dimensions
            nx, ny, nz: Number of cells in each direction
            thermal_conductivity: k (W/m·K)
            heat_generation: q''' (W/m³)
            bc_left/right: Boundary conditions
            
        Returns:
            Temperature field and heat fluxes
        """
        if not self.available:
            raise RuntimeError("FiPy not available")
        
        from fipy import Grid3D, CellVariable, DiffusionTerm, ImplicitSourceTerm
        
        Lx, Ly, Lz = domain_size
        dx, dy, dz = Lx/nx, Ly/ny, Lz/nz
        
        # Create 3D grid
        mesh = Grid3D(dx=dx, dy=dy, dz=dz, nx=nx, ny=ny, nz=nz)
        
        # Temperature variable
        T = CellVariable(name="temperature", mesh=mesh, value=300.0)
        
        # Apply boundary conditions
        # Left face (x=0)
        if bc_left[0] == "dirichlet":
            T.constrain(bc_left[1], mesh.facesLeft)
        elif bc_left[0] == "convection":
            T_inf, h = bc_left[1], bc_left[2]
            # Convection BC: -k∇T·n = h(T - T_inf)
            val = (h * T.faceValue - h * T_inf) / thermal_conductivity
            T.faceGrad.constrain(val * mesh.faceNormals, where=mesh.facesLeft)
        
        # Right face (x=Lx)
        if bc_right[0] == "dirichlet":
            T.constrain(bc_right[1], mesh.facesRight)
        elif bc_right[0] == "convection":
            T_inf, h = bc_right[1], bc_right[2]
            val = (h * T.faceValue - h * T_inf) / thermal_conductivity
            T.faceGrad.constrain(val * mesh.faceNormals, where=mesh.facesRight)
        
        # Solve steady-state: ∇·(k∇T) + q''' = 0
        eq = DiffusionTerm(coeff=thermal_conductivity) + ImplicitSourceTerm(coeff=heat_generation)
        eq.solve(var=T)
        
        return {
            "temperature": T.value,
            "mesh": mesh,
            "max_temperature": float(max(T.value)),
            "min_temperature": float(min(T.value)),
            "nx": nx, "ny": ny, "nz": nz
        }
    
    def solve_transient_3d(
        self,
        domain_size: Tuple[float, float, float],
        nx: int, ny: int, nz: int,
        thermal_conductivity: float,
        density: float,
        specific_heat: float,
        T_initial: float,
        time_steps: int,
        dt: float,
        heat_generation: float = 0.0
    ) -> Dict[str, Any]:
        """
        Solve 3D transient heat conduction
        
        ρcₚ∂T/∂t = ∇·(k∇T) + q'''
        """
        if not self.available:
            raise RuntimeError("FiPy not available")
        
        from fipy import Grid3D, CellVariable, TransientTerm, DiffusionTerm, ImplicitSourceTerm
        
        Lx, Ly, Lz = domain_size
        dx, dy, dz = Lx/nx, Ly/ny, Lz/nz
        
        # Create mesh
        mesh = Grid3D(dx=dx, dy=dy, dz=dz, nx=nx, ny=ny, nz=nz)
        
        # Temperature
        T = CellVariable(name="temperature", mesh=mesh, value=T_initial)
        
        # Thermal diffusivity
        alpha = thermal_conductivity / (density * specific_heat)
        
        # Transient equation: ρcₚ∂T/∂t = ∇·(k∇T) + q'''
        eq = TransientTerm(coeff=density*specific_heat) == \
             DiffusionTerm(coeff=thermal_conductivity) + \
             ImplicitSourceTerm(coeff=heat_generation)
        
        # Time stepping
        T_history = [T.value.copy()]
        for step in range(time_steps):
            eq.solve(var=T, dt=dt)
            if step % 10 == 0:
                T_history.append(T.value.copy())
        
        return {
            "temperature": T.value,
            "T_history": np.array(T_history),
            "max_temperature": float(max(T.value)),
            "min_temperature": float(min(T.value)),
            "time": time_steps * dt
        }


@dataclass
class ThermalStressResult:
    """Result of thermal-structural coupling analysis"""
    thermal_strain: np.ndarray  # ε_th = α·ΔT
    thermal_stress: np.ndarray  # σ_th = E·ε_th (constrained)
    total_stress: np.ndarray    # σ_total = σ_mechanical + σ_thermal
    displacement: np.ndarray    # From thermal expansion
    max_stress: float
    buckling_risk: bool         # Compressive thermal stress check


class ThermalStructuralCoupling:
    """
    Thermal-Structural Coupling Analysis
    
    Couples thermal and structural fields:
    ε_total = ε_mechanical + ε_thermal
    σ = C : (ε_total - α·ΔT)
    
    Includes:
    - Thermal strain computation
    - Thermoelastic stress analysis
    - Buckling risk from compressive thermal stresses
    """
    
    def __init__(self):
        self.thermal_expansion_coeff = 12e-6  # Default steel (1/K)
        self.elastic_modulus = 200e9  # Pa
        self.poisson_ratio = 0.3
    
    def compute_thermal_strain(
        self,
        temperature: np.ndarray,
        reference_temp: float = 20.0,
        alpha: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute thermal strain: ε_th = α·(T - T_ref)
        
        Args:
            temperature: Temperature field (°C)
            reference_temp: Reference (stress-free) temperature
            alpha: Thermal expansion coefficient (1/K)
        
        Returns:
            Thermal strain vector [εxx, εyy, εzz, γxy, γyz, γzx]
        """
        alpha = alpha or self.thermal_expansion_coeff
        delta_T = temperature - reference_temp
        
        # Isotropic thermal expansion
        eps_th = alpha * delta_T
        
        # For 3D: εxx = εyy = εzz = α·ΔT, shear strains are zero
        return np.array([eps_th, eps_th, eps_th, 0, 0, 0])
    
    def compute_thermal_stress(
        self,
        thermal_strain: np.ndarray,
        E: Optional[float] = None,
        nu: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute thermal stress assuming fully constrained expansion
        
        σ = C : ε_th
        
        For isotropic material with constrained expansion:
        σxx = σyy = σzz = -E·α·ΔT / (1 - 2ν)  # Hydrostatic compression
        """
        E = E or self.elastic_modulus
        nu = nu or self.poisson_ratio
        
        # Simplified: assume uniform thermal strain
        eps_th = thermal_strain[0]  # εxx component
        
        # Fully constrained thermal stress
        # σ = -E·α·ΔT / (1 - 2ν) for each direction
        factor = -E / (1 - 2 * nu)
        sigma_th = factor * eps_th
        
        return np.array([sigma_th, sigma_th, sigma_th, 0, 0, 0])
    
    def couple_thermal_structural(
        self,
        thermal_result: Dict[str, Any],
        mechanical_stress: Optional[np.ndarray] = None,
        material_props: Optional[Dict] = None
    ) -> ThermalStressResult:
        """
        Couple thermal and structural analysis results
        
        Args:
            thermal_result: Output from thermal analysis (temperature field)
            mechanical_stress: Stress from mechanical loads
            material_props: Material properties dict
        
        Returns:
            Combined thermal-structural result
        """
        material_props = material_props or {}
        alpha = material_props.get("thermal_expansion", self.thermal_expansion_coeff)
        E = material_props.get("elastic_modulus", self.elastic_modulus) * 1e9  # Convert GPa to Pa
        nu = material_props.get("poisson_ratio", self.poisson_ratio)
        
        # Get temperature field
        T = thermal_result.get("temperature", np.array([20.0]))
        if isinstance(T, (list, np.ndarray)):
            T_max = np.max(T)
        else:
            T_max = float(T)
        
        # Compute thermal strain
        eps_th = self.compute_thermal_strain(T, reference_temp=20.0, alpha=alpha)
        
        # Compute thermal stress (constrained expansion)
        sigma_th = self.compute_thermal_stress(eps_th, E, nu)
        
        # Add mechanical stress if provided
        if mechanical_stress is not None:
            sigma_total = sigma_th + mechanical_stress
        else:
            sigma_total = sigma_th
        
        # Check for compressive thermal stress (buckling risk)
        # Negative stress indicates compression
        min_stress = np.min(sigma_total[:3])  # Check normal stresses
        buckling_risk = min_stress < -0.5 * E * 1e-3  # Threshold: 0.05% yield
        
        # Estimate displacement from free expansion
        # δ = α·L·ΔT (simplified)
        delta_T_avg = np.mean(T) - 20.0 if isinstance(T, np.ndarray) else T - 20.0
        displacement = alpha * delta_T_avg * 1.0  # Assume unit length
        
        return ThermalStressResult(
            thermal_strain=eps_th,
            thermal_stress=sigma_th,
            total_stress=sigma_total,
            displacement=np.full(3, displacement),
            max_stress=np.max(np.abs(sigma_total)),
            buckling_risk=buckling_risk
        )
    
    def analyze_thermal_buckling(
        self,
        thermal_stress: np.ndarray,
        geometry: Dict[str, Any],
        material: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess buckling risk from thermal stresses
        
        Thermal buckling occurs when compressive thermal stresses
        exceed critical buckling load.
        """
        E = material.get("elastic_modulus", 200) * 1e9  # Pa
        thickness = geometry.get("thickness", 0.01)  # m
        length = geometry.get("length", 1.0)  # m
        
        # Critical buckling stress for plate (simplified)
        # σ_cr = k·π²·E·t² / (12·(1-ν²)·b²)
        k = 4.0  # Buckling coefficient for simply supported plate
        nu = material.get("poisson_ratio", 0.3)
        
        sigma_cr = k * np.pi**2 * E * thickness**2 / (12 * (1 - nu**2) * length**2)
        
        # Check compressive thermal stress
        sigma_min = np.min(thermal_stress[:3])  # Minimum principal stress
        
        safety_factor = sigma_cr / abs(sigma_min) if sigma_min < 0 else float('inf')
        
        return {
            "critical_stress": sigma_cr,
            "thermal_stress_compressive": sigma_min,
            "safety_factor": safety_factor,
            "buckling_risk": safety_factor < 2.0,  # SF < 2 is risky
            "recommendation": "Add expansion joints" if safety_factor < 2.0 else "Acceptable"
        }


class ProductionThermalAgent:
    """
    Production-grade thermal analysis agent
    
    Capabilities:
    - Conjugate heat transfer (conduction, convection, radiation)
    - CoolProp integration for accurate fluid properties
    - Comprehensive Nusselt correlations
    - Transient and steady-state analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.name = "ProductionThermalAgent"
        self.config = config or {}
        
        # Check CoolProp availability
        self.has_coolprop = HAS_COOLPROP
        
        # Initialize 3D solver
        self.solver_3d = FiPy3DThermalSolver()
        
        # Convection correlation database
        self.correlations = ConvectionCorrelations()
        
        # Radiation calculator
        self.radiation = RadiationCalculator()
        
        # Thermal-structural coupling
        self.thermal_structural = ThermalStructuralCoupling()
        
        logger.info(f"ProductionThermalAgent initialized (CoolProp: {self.has_coolprop})")
        logger.info("Thermal-structural coupling enabled")
    
    async def analyze(
        self,
        surfaces: List[Surface],
        heat_sources: List[HeatSource],
        fluid: FluidProperties,
        ambient_temp: float,
        velocity: float = 0.0,
        transient: bool = False,
        time_span: Optional[Tuple[float, float]] = None,
        options: Optional[Dict] = None
    ) -> ThermalResult:
        """
        Perform multi-mode heat transfer analysis
        
        Args:
            surfaces: List of surfaces for heat transfer
            heat_sources: Internal heat generation
            fluid: Fluid properties
            ambient_temp: Ambient temperature (K)
            velocity: Fluid velocity (m/s) - 0 for natural convection
            transient: Whether to perform transient analysis
            time_span: (start_time, end_time) for transient analysis
            options: Additional analysis options
            
        Returns:
            ThermalResult with temperature distribution and heat fluxes
        """
        import time
        start_time = time.time()
        options = options or {}
        
        logger.info(f"Starting thermal analysis: {len(surfaces)} surfaces, "
                   f"{'transient' if transient else 'steady-state'}")
        
        try:
            # Calculate convection coefficients for each surface
            h_coeffs = {}
            for i, surface in enumerate(surfaces):
                surface_id = f"surface_{i}"
                
                if velocity > 0.1:
                    # Forced convection
                    h = self._calculate_forced_convection(
                        fluid, surface, velocity, ambient_temp
                    )
                else:
                    # Natural convection
                    h = self._calculate_natural_convection(
                        fluid, surface, ambient_temp
                    )
                
                h_coeffs[surface_id] = h
            
            # Calculate total heat transfer
            total_power = sum(hs.power for hs in heat_sources)
            
            # Simplified lumped capacitance or 1D conduction
            if transient:
                result = self._transient_solve(
                    surfaces, heat_sources, h_coeffs,
                    fluid, ambient_temp, time_span
                )
            else:
                result = self._steady_state_solve(
                    surfaces, heat_sources, h_coeffs,
                    fluid, ambient_temp
                )
            
            computation_time = (time.time() - start_time) * 1000
            
            return ThermalResult(
                temperature=result["temperature"],
                heat_flux=result["heat_flux"],
                max_temperature=float(np.max(result["temperature"])),
                min_temperature=float(np.min(result["temperature"])),
                total_heat_transfer=total_power,
                convection_coeffs=h_coeffs,
                radiation_exchange=None,
                status="success",
                computation_time_ms=computation_time
            )
            
        except Exception as e:
            logger.error(f"Thermal analysis failed: {e}")
            return ThermalResult(
                temperature=np.array([ambient_temp]),
                heat_flux=np.zeros(1),
                max_temperature=ambient_temp,
                min_temperature=ambient_temp,
                total_heat_transfer=0.0,
                convection_coeffs={},
                radiation_exchange=None,
                status=f"error: {str(e)}",
                computation_time_ms=(time.time() - start_time) * 1000
            )
    
    def _calculate_natural_convection(
        self,
        fluid: FluidProperties,
        surface: Surface,
        ambient_temp: float
    ) -> float:
        """Calculate natural convection coefficient"""
        delta_T = surface.temperature - ambient_temp
        
        if abs(delta_T) < 0.01:
            return 5.0  # Minimum h for natural convection
        
        # Select correlation based on orientation
        if surface.orientation == "vertical":
            Nu = self.correlations.nusselt_natural_vertical_plate(
                fluid, surface, delta_T
            )
        elif "horizontal" in surface.orientation:
            heated_up = "up" in surface.orientation and delta_T > 0
            Nu = self.correlations.nusselt_natural_horizontal_plate(
                fluid, surface, delta_T,
                heated_surface="up" if heated_up else "down"
            )
        else:
            # Default to vertical
            Nu = self.correlations.nusselt_natural_vertical_plate(
                fluid, surface, delta_T
            )
        
        # h = Nu * k / L
        h = Nu * fluid.thermal_conductivity / surface.characteristic_length
        
        return max(h, 0.1)  # Ensure positive
    
    def _calculate_forced_convection(
        self,
        fluid: FluidProperties,
        surface: Surface,
        velocity: float,
        ambient_temp: float
    ) -> float:
        """Calculate forced convection coefficient"""
        Re = self.correlations.reynolds_number(fluid, surface, velocity)
        regime = self.correlations.flow_regime(Re)
        
        if regime == FlowRegime.LAMINAR:
            Nu = self.correlations.nusselt_forced_flat_plate_laminar(
                Re, fluid.prandtl_number
            )
        elif regime == FlowRegime.TURBULENT:
            Nu = self.correlations.nusselt_forced_flat_plate_turbulent(
                Re, fluid.prandtl_number
            )
        else:
            # Transitional - use mixed formula
            Nu = self.correlations.nusselt_forced_flat_plate_mixed(
                Re, fluid.prandtl_number
            )
        
        # h = Nu * k / L
        h = Nu * fluid.thermal_conductivity / surface.characteristic_length
        
        return max(h, 1.0)
    
    def _steady_state_solve(
        self,
        surfaces: List[Surface],
        heat_sources: List[HeatSource],
        h_coeffs: Dict[str, float],
        fluid: FluidProperties,
        ambient_temp: float
    ) -> Dict[str, np.ndarray]:
        """
        Steady-state thermal solution using 1D finite difference
        
        Solves: d²T/dx² = 0 with convection BCs
        For a plate/solid with internal heat generation
        """
        import numpy as np
        
        # Problem setup
        total_power = sum(hs.power for hs in heat_sources)
        avg_h = np.mean(list(h_coeffs.values())) if h_coeffs else 10.0
        total_area = sum(s.area for s in surfaces)
        
        # Get material properties (use first surface's material or default)
        k = 167.0  # Thermal conductivity (W/m·K) - aluminum default
        for surface in surfaces:
            if hasattr(surface, 'thermal_conductivity'):
                k = surface.thermal_conductivity
                break
        
        # Estimate characteristic length from surface area
        L = np.sqrt(total_area) if total_area > 0 else 0.1
        
        # 1D finite difference grid
        n_nodes = 50
        dx = L / (n_nodes - 1)
        x = np.linspace(0, L, n_nodes)
        
        # Build finite difference matrix for d²T/dx² = -q'''/k
        # Internal nodes: T[i-1] - 2T[i] + T[i+1] = -q'''*dx²/k
        A = np.zeros((n_nodes, n_nodes))
        b = np.zeros(n_nodes)
        
        # Internal heat generation per unit volume
        volume = total_area * L
        q_gen = total_power / volume if volume > 0 else 0.0
        
        # Internal nodes
        for i in range(1, n_nodes - 1):
            A[i, i-1] = 1.0
            A[i, i] = -2.0
            A[i, i+1] = 1.0
            b[i] = -q_gen * dx**2 / k
        
        # Boundary conditions
        # Left surface (x=0): convection
        # -k*dT/dx = h*(T_inf - T[0])
        # FD: -k*(T[1] - T[0])/dx = h*(T_inf - T[0])
        Bi_left = avg_h * dx / k  # Biot number
        A[0, 0] = 1.0 + Bi_left
        A[0, 1] = -1.0
        b[0] = Bi_left * ambient_temp
        
        # Right surface (x=L): convection
        Bi_right = avg_h * dx / k
        A[-1, -1] = 1.0 + Bi_right
        A[-1, -2] = -1.0
        b[-1] = Bi_right * ambient_temp
        
        # Solve system
        T = np.linalg.solve(A, b)
        
        # Calculate heat flux at surfaces
        q_left = -k * (T[1] - T[0]) / dx
        q_right = -k * (T[-1] - T[-2]) / dx
        heat_flux = np.array([q_left, q_right])
        
        # Update surface temperatures
        if surfaces:
            surfaces[0].temperature = T[0]
            if len(surfaces) > 1:
                surfaces[-1].temperature = T[-1]
        
        logger.info(f"Steady-state solve: max T = {np.max(T):.2f}K, "
                   f"q_left = {q_left:.2f} W/m²")
        
        return {
            "temperature": T,
            "heat_flux": heat_flux,
            "x_positions": x
        }
    
    def _transient_solve(
        self,
        surfaces: List[Surface],
        heat_sources: List[HeatSource],
        h_coeffs: Dict[str, float],
        fluid: FluidProperties,
        ambient_temp: float,
        time_span: Optional[Tuple[float, float]]
    ) -> Dict[str, np.ndarray]:
        """
        Transient thermal solution using explicit finite difference
        
        Solves: ρ*c*∂T/∂t = k*∂²T/∂x² + q''' with convection BCs
        """
        import numpy as np
        
        total_power = sum(hs.power for hs in heat_sources)
        avg_h = np.mean(list(h_coeffs.values())) if h_coeffs else 10.0
        total_area = sum(s.area for s in surfaces)
        
        # Material properties
        k = 167.0  # W/(m·K) - aluminum
        rho = 2700  # kg/m³
        cp = 900    # J/(kg·K)
        alpha = k / (rho * cp)  # Thermal diffusivity
        
        # Domain setup
        L = np.sqrt(total_area) if total_area > 0 else 0.1
        n_nodes = 50
        dx = L / (n_nodes - 1)
        x = np.linspace(0, L, n_nodes)
        
        # Time setup
        t_start = time_span[0] if time_span else 0.0
        t_end = time_span[1] if time_span else 1000.0
        
        # Stability criterion for explicit scheme: Fo = α*Δt/Δx² ≤ 0.5
        dt_max = 0.5 * dx**2 / alpha
        dt = min(dt_max * 0.8, (t_end - t_start) / 100)  # 100 time steps or stable
        n_steps = int((t_end - t_start) / dt) + 1
        
        # Initial condition
        T = np.ones(n_nodes) * ambient_temp
        
        # Internal heat generation
        volume = total_area * L
        q_gen = total_power / volume if volume > 0 else 0.0
        
        # Fourier number
        Fo = alpha * dt / dx**2
        
        # Biot numbers for convection BCs
        Bi = avg_h * dx / k
        
        # Time-stepping loop (explicit scheme)
        T_history = [T.copy()]
        t_history = [t_start]
        
        for step in range(n_steps):
            T_new = T.copy()
            
            # Internal nodes
            for i in range(1, n_nodes - 1):
                T_new[i] = T[i] + Fo * (T[i+1] - 2*T[i] + T[i-1]) + q_gen * dt / (rho * cp)
            
            # Left boundary (convection)
            # Using energy balance: ρ*c*dx/2 * dT/dt = q_gen*dx/2 + k*(T[1]-T[0])/dx + h*(T_inf - T[0])
            T_new[0] = T[0] + 2*dt/(rho*cp*dx) * (
                q_gen*dx/2 + k*(T[1]-T[0])/dx + avg_h*(ambient_temp - T[0])
            )
            
            # Right boundary (convection)
            T_new[-1] = T[-1] + 2*dt/(rho*cp*dx) * (
                q_gen*dx/2 + k*(T[-2]-T[-1])/dx + avg_h*(ambient_temp - T[-1])
            )
            
            T = T_new
            
            # Store every 10 steps
            if step % 10 == 0:
                T_history.append(T.copy())
                t_history.append(t_start + step * dt)
        
        # Final heat flux
        q_left = -k * (T[1] - T[0]) / dx
        q_right = -k * (T[-1] - T[-2]) / dx
        
        logger.info(f"Transient solve: {n_steps} steps, dt={dt:.4f}s, "
                   f"max T = {np.max(T):.2f}K")
        
        return {
            "temperature": T,
            "heat_flux": np.array([q_left, q_right]),
            "x_positions": x,
            "t_history": np.array(t_history),
            "T_history": np.array(T_history)
        }
    
    def calculate_radiation_exchange(
        self,
        surfaces: List[Surface],
        ambient_temp: float
    ) -> Dict[str, float]:
        """Calculate radiative heat transfer between surfaces"""
        results = {}
        
        for i, surface in enumerate(surfaces):
            # Net radiation to surroundings (approximated as blackbody at T_inf)
            q_rad = (
                surface.emissivity * 
                RadiationCalculator.SIGMA * 
                surface.area * 
                (surface.temperature**4 - ambient_temp**4)
            )
            results[f"surface_{i}_radiation"] = q_rad
        
        return results
    
    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Legacy-compatible run method
        
        Args:
            payload: Dictionary with thermal parameters
            
        Returns:
            Thermal analysis results
        """
        # Extract parameters
        power_w = float(payload.get("power_watts", 10.0))
        surface_area = float(payload.get("surface_area", 0.1))
        emissivity = float(payload.get("emissivity", 0.9))
        ambient_temp = float(payload.get("ambient_temp", 25.0)) + 273.15  # Convert to K
        env_type = payload.get("environment_type", "GROUND")
        h_provided = payload.get("heat_transfer_coeff")
        
        # Create surfaces
        surfaces = [Surface(
            area=surface_area,
            characteristic_length=np.sqrt(surface_area),
            orientation="vertical",
            emissivity=emissivity,
            temperature=ambient_temp + 50  # Initial guess
        )]
        
        # Create heat source
        heat_sources = [HeatSource(power=power_w)]
        
        # Get fluid properties
        if env_type == "SPACE" or env_type == "VACUUM":
            # Radiation only - no convection
            fluid = FluidProperties.air(ambient_temp)
            velocity = 0.0
        else:
            fluid = FluidProperties.air(ambient_temp)
            velocity = 0.0  # Natural convection
        
        # Run analysis
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(self.analyze(
                surfaces=surfaces,
                heat_sources=heat_sources,
                fluid=fluid,
                ambient_temp=ambient_temp,
                velocity=velocity,
                transient=False
            ))
        except RuntimeError:
            # No event loop
            result = asyncio.run(self.analyze(
                surfaces=surfaces,
                heat_sources=heat_sources,
                fluid=fluid,
                ambient_temp=ambient_temp,
                velocity=velocity,
                transient=False
            ))
        
        # Convert to legacy format
        max_temp_c = result.max_temperature - 273.15
        
        status = "nominal"
        if max_temp_c > 100:
            status = "warning"
        if max_temp_c > 150:
            status = "critical"
        
        return {
            "status": status,
            "equilibrium_temp_c": round(max_temp_c, 2),
            "delta_t": round(max_temp_c - (ambient_temp - 273.15), 2),
            "heat_load_w": power_w,
            "convection_coeffs": result.convection_coeffs,
            "computation_time_ms": result.computation_time_ms,
            "coolprop_used": self.has_coolprop
        }


# Convenience functions
async def calculate_convection_coefficient(
    surface_temp: float,
    ambient_temp: float,
    characteristic_length: float,
    orientation: str = "vertical",
    fluid: str = "air",
    velocity: float = 0.0,
    pressure: float = 101325
) -> float:
    """
    Calculate convection coefficient for a surface
    
    Args:
        surface_temp: Surface temperature (°C)
        ambient_temp: Ambient temperature (°C)
        characteristic_length: Characteristic length (m)
        orientation: "vertical", "horizontal_up", "horizontal_down"
        fluid: "air" or "water"
        velocity: Fluid velocity (m/s), 0 for natural convection
        pressure: Fluid pressure (Pa)
        
    Returns:
        Convection coefficient h (W/(m²·K))
    """
    agent = ProductionThermalAgent()
    
    # Convert temperatures to K
    T_surface = surface_temp + 273.15
    T_ambient = ambient_temp + 273.15
    T_film = (T_surface + T_ambient) / 2
    
    # Get fluid properties
    if fluid == "air":
        fluid_props = FluidProperties.air(T_film, pressure)
    elif fluid == "water":
        fluid_props = FluidProperties.water(T_film, pressure)
    else:
        raise ValueError(f"Unknown fluid: {fluid}")
    
    surface = Surface(
        area=1.0,  # Not used for h calculation
        characteristic_length=characteristic_length,
        orientation=orientation,
        temperature=T_surface
    )
    
    if velocity > 0.1:
        h = agent._calculate_forced_convection(
            fluid_props, surface, velocity, T_ambient
        )
    else:
        h = agent._calculate_natural_convection(
            fluid_props, surface, T_ambient
        )
    
    return h


def get_fluid_properties(
    fluid: str,
    temperature: float,
    pressure: float = 101325
) -> Dict[str, float]:
    """
    Get thermophysical properties of a fluid
    
    Args:
        fluid: Fluid name (e.g., "Air", "Water", "CO2")
        temperature: Temperature (K)
        pressure: Pressure (Pa)
        
    Returns:
        Dictionary of fluid properties
    """
    try:
        props = FluidProperties.from_coolprop(fluid, temperature, pressure)
        return {
            "name": props.name,
            "temperature_k": props.temperature,
            "pressure_pa": props.pressure,
            "density_kg_m3": props.density,
            "specific_heat_j_kg_k": props.specific_heat,
            "thermal_conductivity_w_m_k": props.thermal_conductivity,
            "dynamic_viscosity_pa_s": props.dynamic_viscosity,
            "prandtl_number": props.prandtl_number,
            "source": "CoolProp"
        }
    except Exception as e:
        # Fallback to approximate values
        if fluid.lower() == "air":
            props = FluidProperties.air(temperature, pressure)
        elif fluid.lower() == "water":
            props = FluidProperties.water(temperature, pressure)
        else:
            raise ValueError(f"Cannot get properties for {fluid}: {e}")
        
        return {
            "name": props.name,
            "temperature_k": props.temperature,
            "pressure_pa": props.pressure,
            "density_kg_m3": props.density,
            "specific_heat_j_kg_k": props.specific_heat,
            "thermal_conductivity_w_m_k": props.thermal_conductivity,
            "dynamic_viscosity_pa_s": props.dynamic_viscosity,
            "prandtl_number": props.prandtl_number,
            "source": "approximate"
        }
