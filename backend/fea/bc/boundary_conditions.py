"""
FIX-204: Boundary Condition Handling

Define and manage boundary conditions for FEA:
- Constraints (displacement, rotation)
- Loads (force, pressure, temperature)
- Support for multiple physics (structural, thermal)
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BCType(Enum):
    """Types of boundary conditions"""
    # Displacement constraints
    FIXED = auto()           # All DOFs fixed
    PINNED = auto()          # Displacement fixed, rotation free
    ROLLER = auto()          # One direction free
    SYMMETRY = auto()        # Symmetry plane
    ANTI_SYMMETRY = auto()   # Anti-symmetry plane
    
    # Force loads
    FORCE = auto()           # Point force
    MOMENT = auto()          # Point moment
    PRESSURE = auto()        # Distributed pressure
    GRAVITY = auto()         # Body force (gravity)
    CENTRIFUGAL = auto()     # Rotational load
    
    # Thermal
    TEMPERATURE = auto()     # Fixed temperature
    HEAT_FLUX = auto()       # Heat flux
    CONVECTION = auto()      # Convection BC
    RADIATION = auto()       # Radiation BC
    
    # Contact
    CONTACT = auto()         # Contact surface
    TIE = auto()             # Tied contact


@dataclass
class Constraint:
    """Displacement constraint definition"""
    ux: bool = False  # Fixed in x
    uy: bool = False  # Fixed in y
    uz: bool = False  # Fixed in z
    rotx: bool = False  # Fixed rotation about x
    roty: bool = False  # Fixed rotation about y
    rotz: bool = False  # Fixed rotation about z
    
    @classmethod
    def fixed(cls) -> 'Constraint':
        """Fully fixed constraint"""
        return cls(True, True, True, True, True, True)
    
    @classmethod
    def pinned(cls) -> 'Constraint':
        """Pinned constraint (no translation)"""
        return cls(True, True, True, False, False, False)
    
    @classmethod
    def roller_x(cls) -> 'Constraint':
        """Roller in x direction (free to move in x)"""
        return cls(False, True, True, False, False, False)
    
    @classmethod
    def roller_y(cls) -> 'Constraint':
        """Roller in y direction"""
        return cls(True, False, True, False, False, False)
    
    @classmethod
    def roller_z(cls) -> 'Constraint':
        """Roller in z direction"""
        return cls(True, True, False, False, False, False)
    
    @classmethod
    def symmetry_x(cls) -> 'Constraint':
        """Symmetry about x=0 plane (fix x displacement)"""
        return cls(True, False, False, True, False, False)
    
    @classmethod
    def symmetry_y(cls) -> 'Constraint':
        """Symmetry about y=0 plane"""
        return cls(False, True, False, False, True, False)
    
    @classmethod
    def symmetry_z(cls) -> 'Constraint':
        """Symmetry about z=0 plane"""
        return cls(False, False, True, False, False, True)
    
    def to_calculix(self) -> str:
        """Convert to CalculiX *BOUNDARY format"""
        dofs = []
        if self.ux: dofs.append("1")
        if self.uy: dofs.append("2")
        if self.uz: dofs.append("3")
        if self.rotx: dofs.append("4")
        if self.roty: dofs.append("5")
        if self.rotz: dofs.append("6")
        return ",".join(dofs) if dofs else ""


@dataclass
class Load:
    """Load definition"""
    magnitude: float
    direction: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    components: Optional[Tuple[float, float, float]] = None  # Direct components
    
    def __post_init__(self):
        if self.components is None:
            # Normalize direction
            norm = sum(d**2 for d in self.direction) ** 0.5
            if norm > 0:
                self.components = tuple(
                    self.magnitude * d / norm for d in self.direction
                )
            else:
                self.components = (0.0, 0.0, 0.0)


@dataclass
class BoundaryCondition:
    """Single boundary condition"""
    name: str
    bc_type: BCType
    entity_type: str  # "node", "element", "surface", "volume"
    entity_ids: List[int]
    constraint: Optional[Constraint] = None
    load: Optional[Load] = None
    value: Optional[float] = None  # For temperature, pressure, etc.
    
    # For time-dependent loads
    amplitude: Optional[str] = None
    
    def validate(self) -> bool:
        """Validate boundary condition"""
        if not self.name:
            raise ValueError("BC name is required")
        
        if not self.entity_ids:
            raise ValueError("Entity IDs are required")
        
        if self.bc_type in [BCType.FIXED, BCType.PINNED, BCType.ROLLER, 
                            BCType.SYMMETRY, BCType.ANTI_SYMMETRY]:
            if self.constraint is None:
                raise ValueError(f"Constraint required for {self.bc_type}")
        
        if self.bc_type in [BCType.FORCE, BCType.MOMENT, BCType.PRESSURE]:
            if self.load is None and self.value is None:
                raise ValueError(f"Load or value required for {self.bc_type}")
        
        return True


class BoundaryConditionManager:
    """
    Manage boundary conditions for FEA analysis.
    
    FIX-204: Implements boundary condition handling for CalculiX.
    
    Usage:
        bcm = BoundaryConditionManager()
        
        # Add fixed constraint
        bcm.add_fixed_constraint("fixed_nodes", node_ids=[1, 2, 3])
        
        # Add force load
        bcm.add_force_load("force_load", node_ids=[10], 
                          magnitude=1000, direction=(0, 0, -1))
        
        # Generate CalculiX input
        bcm.write_to_calculix("model.inp")
    """
    
    def __init__(self):
        self._boundary_conditions: Dict[str, BoundaryCondition] = {}
        self._amplitudes: Dict[str, List[Tuple[float, float]]] = {}
    
    def add_boundary_condition(self, bc: BoundaryCondition) -> None:
        """Add a boundary condition"""
        bc.validate()
        self._boundary_conditions[bc.name] = bc
        logger.info(f"Added BC: {bc.name} ({bc.bc_type.name})")
    
    def add_fixed_constraint(
        self,
        name: str,
        node_ids: List[int],
        fix_rotations: bool = True
    ) -> None:
        """Add fully fixed constraint"""
        constraint = Constraint.fixed() if fix_rotations else Constraint.pinned()
        
        bc = BoundaryCondition(
            name=name,
            bc_type=BCType.FIXED,
            entity_type="node",
            entity_ids=node_ids,
            constraint=constraint
        )
        self.add_boundary_condition(bc)
    
    def add_pinned_constraint(self, name: str, node_ids: List[int]) -> None:
        """Add pinned constraint (no translation)"""
        bc = BoundaryCondition(
            name=name,
            bc_type=BCType.PINNED,
            entity_type="node",
            entity_ids=node_ids,
            constraint=Constraint.pinned()
        )
        self.add_boundary_condition(bc)
    
    def add_roller_constraint(
        self,
        name: str,
        node_ids: List[int],
        direction: str = "x"
    ) -> None:
        """Add roller constraint (free in one direction)"""
        constraint_map = {
            "x": Constraint.roller_x(),
            "y": Constraint.roller_y(),
            "z": Constraint.roller_z()
        }
        
        bc = BoundaryCondition(
            name=name,
            bc_type=BCType.ROLLER,
            entity_type="node",
            entity_ids=node_ids,
            constraint=constraint_map.get(direction, Constraint.roller_x())
        )
        self.add_boundary_condition(bc)
    
    def add_symmetry_constraint(
        self,
        name: str,
        surface_ids: List[int],
        plane: str = "x"
    ) -> None:
        """Add symmetry constraint on a plane"""
        constraint_map = {
            "x": Constraint.symmetry_x(),
            "y": Constraint.symmetry_y(),
            "z": Constraint.symmetry_z()
        }
        
        bc = BoundaryCondition(
            name=name,
            bc_type=BCType.SYMMETRY,
            entity_type="surface",
            entity_ids=surface_ids,
            constraint=constraint_map.get(plane, Constraint.symmetry_x())
        )
        self.add_boundary_condition(bc)
    
    def add_force_load(
        self,
        name: str,
        node_ids: List[int],
        magnitude: float,
        direction: Tuple[float, float, float] = (0.0, 0.0, -1.0)
    ) -> None:
        """Add concentrated force load"""
        load = Load(magnitude=magnitude, direction=direction)
        
        bc = BoundaryCondition(
            name=name,
            bc_type=BCType.FORCE,
            entity_type="node",
            entity_ids=node_ids,
            load=load
        )
        self.add_boundary_condition(bc)
    
    def add_pressure_load(
        self,
        name: str,
        surface_ids: List[int],
        pressure: float
    ) -> None:
        """Add pressure load on surface"""
        bc = BoundaryCondition(
            name=name,
            bc_type=BCType.PRESSURE,
            entity_type="surface",
            entity_ids=surface_ids,
            value=pressure
        )
        self.add_boundary_condition(bc)
    
    def add_gravity_load(
        self,
        name: str,
        magnitude: float = 9.81,
        direction: Tuple[float, float, float] = (0.0, 0.0, -1.0)
    ) -> None:
        """Add gravity load (body force)"""
        load = Load(magnitude=magnitude, direction=direction)
        
        bc = BoundaryCondition(
            name=name,
            bc_type=BCType.GRAVITY,
            entity_type="volume",
            entity_ids=[0],  # Applied to all elements
            load=load
        )
        self.add_boundary_condition(bc)
    
    def add_temperature_bc(
        self,
        name: str,
        node_ids: List[int],
        temperature: float
    ) -> None:
        """Add fixed temperature boundary condition"""
        bc = BoundaryCondition(
            name=name,
            bc_type=BCType.TEMPERATURE,
            entity_type="node",
            entity_ids=node_ids,
            value=temperature
        )
        self.add_boundary_condition(bc)
    
    def add_convection_bc(
        self,
        name: str,
        surface_ids: List[int],
        film_coefficient: float,
        ambient_temperature: float
    ) -> None:
        """Add convection boundary condition"""
        bc = BoundaryCondition(
            name=name,
            bc_type=BCType.CONVECTION,
            entity_type="surface",
            entity_ids=surface_ids,
            value=film_coefficient,
            # Store ambient temp in load for now
            load=Load(magnitude=ambient_temperature)
        )
        self.add_boundary_condition(bc)
    
    def define_amplitude(
        self,
        name: str,
        time_value_pairs: List[Tuple[float, float]]
    ) -> None:
        """
        Define time-varying amplitude.
        
        Args:
            name: Amplitude name
            time_value_pairs: List of (time, value) tuples
        """
        self._amplitudes[name] = time_value_pairs
        logger.info(f"Defined amplitude: {name}")
    
    def write_to_calculix(self, filename: Path) -> None:
        """
        Write all boundary conditions to CalculiX .inp file.
        
        Args:
            filename: Output file path
        """
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filename, 'w') as f:
            f.write("** Boundary Conditions\n")
            f.write("** Generated by BRICK OS FEA Module\n\n")
            
            # Write amplitudes
            if self._amplitudes:
                f.write("** Amplitudes\n")
                for name, pairs in self._amplitudes.items():
                    f.write(f"*AMPLITUDE, NAME={name}, TIME=TOTAL TIME\n")
                    for i in range(0, len(pairs), 4):
                        row = pairs[i:i+4]
                        values = []
                        for t, v in row:
                            values.extend([t, v])
                        f.write(", ".join(f"{v:.6e}" for v in values) + "\n")
                f.write("\n")
            
            # Write constraints (boundary conditions)
            constraints = [
                bc for bc in self._boundary_conditions.values()
                if bc.bc_type in [BCType.FIXED, BCType.PINNED, BCType.ROLLER,
                                 BCType.SYMMETRY, BCType.ANTI_SYMMETRY]
            ]
            
            if constraints:
                f.write("** Constraints\n")
                f.write("*BOUNDARY\n")
                for bc in constraints:
                    dofs = bc.constraint.to_calculix()
                    if dofs:
                        for node_id in bc.entity_ids:
                            f.write(f"{node_id}, {dofs}\n")
                f.write("\n")
            
            # Write loads
            f.write("** Loads\n")
            
            # Concentrated forces
            forces = [
                bc for bc in self._boundary_conditions.values()
                if bc.bc_type == BCType.FORCE
            ]
            
            if forces:
                f.write("*CLOAD\n")
                for bc in forces:
                    amp = f", AMPLITUDE={bc.amplitude}" if bc.amplitude else ""
                    if bc.load and bc.load.components:
                        for i, comp in enumerate(bc.load.components, 1):
                            if abs(comp) > 1e-10:
                                for node_id in bc.entity_ids:
                                    f.write(f"{node_id}, {i}, {comp:.6e}\n")
                f.write("\n")
            
            # Pressure loads
            pressures = [
                bc for bc in self._boundary_conditions.values()
                if bc.bc_type == BCType.PRESSURE
            ]
            
            if pressures:
                f.write("*DLOAD\n")
                for bc in pressures:
                    for surf_id in bc.entity_ids:
                        f.write(f"{surf_id}, P, {bc.value:.6e}\n")
                f.write("\n")
            
            # Gravity
            gravities = [
                bc for bc in self._boundary_conditions.values()
                if bc.bc_type == BCType.GRAVITY
            ]
            
            if gravities:
                f.write("*DLOAD\n")
                for bc in gravities:
                    if bc.load and bc.load.components:
                        # GRAV format: element set, GRAV, magnitude, x-comp, y-comp, z-comp
                        # Simplified: apply to all elements
                        f.write(f"EALL, GRAV, {bc.load.magnitude:.6e}, ")
                        f.write(", ".join(f"{c:.6e}" for c in bc.load.components))
                        f.write("\n")
                f.write("\n")
            
            # Temperature BCs
            temperatures = [
                bc for bc in self._boundary_conditions.values()
                if bc.bc_type == BCType.TEMPERATURE
            ]
            
            if temperatures:
                f.write("** Temperature Boundary Conditions\n")
                f.write("*BOUNDARY\n")
                for bc in temperatures:
                    for node_id in bc.entity_ids:
                        f.write(f"{node_id}, 11, 11, {bc.value:.6f}\n")
                f.write("\n")
            
            # Convection
            convections = [
                bc for bc in self._boundary_conditions.values()
                if bc.bc_type == BCType.CONVECTION
            ]
            
            if convections:
                f.write("** Film Conditions\n")
                for bc in convections:
                    f.write(f"*FILM\n")
                    for surf_id in bc.entity_ids:
                        if bc.load:
                            f.write(f"{surf_id}, F, {bc.load.magnitude:.6f}, {bc.value:.6e}\n")
                f.write("\n")
        
        logger.info(f"Wrote boundary conditions to: {filename}")
    
    def write_to_abaqus(self, filename: Path) -> None:
        """
        Write boundary conditions to ABAQUS format.
        Similar to CalculiX but with ABAQUS-specific syntax.
        """
        # For now, use same format as CalculiX (they're compatible)
        self.write_to_calculix(filename)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all boundary conditions"""
        by_type = {}
        for bc in self._boundary_conditions.values():
            type_name = bc.bc_type.name
            if type_name not in by_type:
                by_type[type_name] = []
            by_type[type_name].append(bc.name)
        
        return {
            "total_boundary_conditions": len(self._boundary_conditions),
            "by_type": by_type,
            "amplitudes": list(self._amplitudes.keys())
        }
    
    def clear(self) -> None:
        """Clear all boundary conditions"""
        self._boundary_conditions.clear()
        self._amplitudes.clear()
        logger.info("Cleared all boundary conditions")


# Convenience functions
def create_fixed_constraint(node_ids: List[int]) -> BoundaryCondition:
    """Create a fixed constraint"""
    return BoundaryCondition(
        name="fixed",
        bc_type=BCType.FIXED,
        entity_type="node",
        entity_ids=node_ids,
        constraint=Constraint.fixed()
    )


def create_force_load(
    node_ids: List[int],
    magnitude: float,
    direction: Tuple[float, float, float] = (0, 0, -1)
) -> BoundaryCondition:
    """Create a force load"""
    return BoundaryCondition(
        name="force",
        bc_type=BCType.FORCE,
        entity_type="node",
        entity_ids=node_ids,
        load=Load(magnitude=magnitude, direction=direction)
    )
