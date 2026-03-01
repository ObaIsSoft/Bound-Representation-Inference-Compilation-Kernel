"""
FIX-206: FEA Input File Generators

Generate complete CalculiX input files from geometry, mesh, and boundary conditions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class Material:
    """Material definition for FEA"""
    name: str
    youngs_modulus: float  # MPa
    poisson_ratio: float
    density: float = 7.8e-9  # tonne/mm³ (default: steel)
    thermal_expansion: float = 12e-6  # 1/K
    conductivity: Optional[float] = None  # W/mK
    specific_heat: Optional[float] = None  # J/kgK
    yield_strength: Optional[float] = None  # MPa
    
    def to_calculix(self) -> str:
        """Generate CalculiX *MATERIAL block"""
        lines = [
            f"*MATERIAL, NAME={self.name}",
            f"*ELASTIC",
            f"{self.youngs_modulus:.1f}, {self.poisson_ratio:.4f}",
            f"*DENSITY",
            f"{self.density:.6e}",
        ]
        
        if self.thermal_expansion:
            lines.extend([
                f"*EXPANSION",
                f"{self.thermal_expansion:.6e}",
            ])
        
        if self.conductivity:
            lines.extend([
                f"*CONDUCTIVITY",
                f"{self.conductivity:.2f}",
            ])
        
        if self.specific_heat:
            lines.extend([
                f"*SPECIFIC HEAT",
                f"{self.specific_heat:.2f}",
            ])
        
        return "\n".join(lines)


@dataclass
class Section:
    """Section property definition"""
    name: str
    material: str
    section_type: str = "solid"  # solid, shell, beam
    thickness: Optional[float] = None  # For shells
    
    def to_calculix(self, element_set: str = "EALL") -> str:
        """Generate CalculiX *SOLID SECTION block"""
        if self.section_type == "solid":
            return f"*SOLID SECTION, ELSET={element_set}, MATERIAL={self.material}"
        elif self.section_type == "shell":
            return f"*SHELL SECTION, ELSET={element_set}, MATERIAL={self.name}\n{self.thickness:.6f}"
        else:
            return f"*SOLID SECTION, ELSET={element_set}, MATERIAL={self.material}"


@dataclass
class Step:
    """Analysis step definition"""
    name: str
    step_type: str = "static"  # static, dynamic, buckle, heat
    nlgeom: bool = False  # Nonlinear geometry
    max_increment: int = 100
    initial_increment: float = 0.1
    min_increment: float = 1e-5
    max_increment_size: float = 1.0
    
    def to_calculix_header(self) -> str:
        """Generate CalculiX *STEP header"""
        nlgeom_str = ", NLGEOM" if self.nlgeom else ""
        return f"*STEP{nlgeom_str}, INC={self.max_increment}"
    
    def to_calculix_footer(self) -> str:
        """Generate CalculiX *END STEP"""
        return "*END STEP"


class InputFileGenerator:
    """
    Generate complete CalculiX input files.
    
    FIX-206: Combines mesh, materials, sections, BCs, and steps into single input file.
    
    Usage:
        generator = InputFileGenerator()
        
        # Set mesh
        generator.set_mesh_file("model.msh")
        
        # Add material
        generator.add_material(Material("Steel", 210000, 0.3))
        
        # Add BCs
        generator.set_boundary_conditions(bc_manager)
        
        # Generate input file
        generator.generate("analysis.inp")
    """
    
    def __init__(self):
        self._mesh_file: Optional[Path] = None
        self._materials: Dict[str, Material] = {}
        self._sections: List[Section] = []
        self._steps: List[Step] = []
        self._bc_manager = None
        self._node_sets: Dict[str, List[int]] = {}
        self._element_sets: Dict[str, List[int]] = {}
    
    def set_mesh_file(self, mesh_file: Path) -> None:
        """Set the mesh file to include"""
        self._mesh_file = Path(mesh_file)
    
    def add_material(self, material: Material) -> None:
        """Add a material"""
        self._materials[material.name] = material
    
    def add_section(self, section: Section) -> None:
        """Add a section property"""
        self._sections.append(section)
    
    def add_step(self, step: Step) -> None:
        """Add an analysis step"""
        self._steps.append(step)
    
    def set_boundary_conditions(self, bc_manager) -> None:
        """Set boundary condition manager"""
        from ..bc.boundary_conditions import BoundaryConditionManager
        self._bc_manager = bc_manager
    
    def define_node_set(self, name: str, node_ids: List[int]) -> None:
        """Define a named node set"""
        self._node_sets[name] = node_ids
    
    def define_element_set(self, name: str, element_ids: List[int]) -> None:
        """Define a named element set"""
        self._element_sets[name] = element_ids
    
    def generate(
        self,
        output_file: Path,
        include_mesh: bool = True
    ) -> Path:
        """
        Generate complete CalculiX input file.
        
        Args:
            output_file: Output file path
            include_mesh: Whether to include mesh data inline
            
        Returns:
            Path to generated file
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        lines = []
        
        # Header
        lines.extend([
            "** ============================================",
            "** BRICK OS - FEA Input File",
            "** Generated by InputFileGenerator",
            "** ============================================",
            "",
        ])
        
        # Include or define mesh
        if include_mesh and self._mesh_file:
            lines.extend([
                "** Mesh data",
                f"*INCLUDE, INPUT={self._mesh_file.name}",
                "",
            ])
        
        # Node sets
        if self._node_sets:
            lines.append("** Node sets")
            for name, nodes in self._node_sets.items():
                lines.append(f"*NSET, NSET={name}")
                for i in range(0, len(nodes), 8):
                    line_nodes = nodes[i:i+8]
                    lines.append(", ".join(str(n) for n in line_nodes))
            lines.append("")
        
        # Element sets
        if self._element_sets:
            lines.append("** Element sets")
            for name, elems in self._element_sets.items():
                lines.append(f"*ELSET, ELSET={name}")
                for i in range(0, len(elems), 8):
                    line_elems = elems[i:i+8]
                    lines.append(", ".join(str(e) for e in line_elems))
            lines.append("")
        
        # Materials
        if self._materials:
            lines.append("** Materials")
            for material in self._materials.values():
                lines.append(material.to_calculix())
                lines.append("")
        
        # Sections
        if self._sections:
            lines.append("** Sections")
            for section in self._sections:
                lines.append(section.to_calculix())
                lines.append("")
        
        # Boundary conditions
        if self._bc_manager:
            lines.append("** Boundary conditions")
            bc_file = output_file.parent / f"{output_file.stem}_bc.inp"
            self._bc_manager.write_to_calculix(bc_file)
            lines.append(f"*INCLUDE, INPUT={bc_file.name}")
            lines.append("")
        
        # Steps
        if self._steps:
            for step in self._steps:
                lines.append(step.to_calculix_header())
                lines.append(f"*STATIC")
                lines.append(f"{step.initial_increment:.6f}, 1.0, {step.min_increment:.6e}, {step.max_increment_size:.6f}")
                lines.append("")
                
                # Include BC file again for each step if needed
                if self._bc_manager:
                    bc_file = output_file.parent / f"{output_file.stem}_bc.inp"
                    lines.append(f"*INCLUDE, INPUT={bc_file.name}")
                
                lines.append(step.to_calculix_footer())
                lines.append("")
        else:
            # Default static step
            lines.extend([
                "** Analysis step",
                "*STEP",
                "*STATIC",
                "0.1, 1.0, 1e-5, 1.0",
            ])
            
            if self._bc_manager:
                bc_file = output_file.parent / f"{output_file.stem}_bc.inp"
                lines.append(f"*INCLUDE, INPUT={bc_file.name}")
            
            lines.extend([
                "*END STEP",
                "",
            ])
        
        # Write file
        with open(output_file, 'w') as f:
            f.write("\n".join(lines))
        
        logger.info(f"Generated input file: {output_file}")
        return output_file
    
    def generate_from_template(
        self,
        template_file: Path,
        output_file: Path,
        substitutions: Dict[str, str]
    ) -> Path:
        """
        Generate input file from template with substitutions.
        
        Args:
            template_file: Template file path
            output_file: Output file path
            substitutions: Dictionary of {{key}} -> value substitutions
            
        Returns:
            Path to generated file
        """
        template_file = Path(template_file)
        output_file = Path(output_file)
        
        with open(template_file) as f:
            content = f.read()
        
        # Apply substitutions
        for key, value in substitutions.items():
            content = content.replace(f"{{{{{key}}}}}", value)
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(content)
        
        return output_file


# Predefined templates
TEMPLATES = {
    "static_linear": """
** Static Linear Analysis
*HEADING
{{title}}

** Include mesh
*INCLUDE, INPUT={{mesh_file}}

** Material
*MATERIAL, NAME={{material_name}}
*ELASTIC
{{youngs_modulus}}, {{poisson_ratio}}
*DENSITY
{{density}}

** Section
*SOLID SECTION, ELSET=EALL, MATERIAL={{material_name}}

** Boundary conditions
*BOUNDARY
{{bc_fixed}}, 1, 6

** Step
*STEP
*STATIC
0.1, 1.0, 1e-5, 1.0

** Loads
*CLOAD
{{load_node}}, {{load_dof}}, {{load_magnitude}}

*NODE PRINT, NSET=NALL
U
*EL PRINT, ELSET=EALL
S

*END STEP
""",

    "modal": """
** Modal Analysis
*HEADING
{{title}}

*INCLUDE, INPUT={{mesh_file}}

*MATERIAL, NAME={{material_name}}
*ELASTIC
{{youngs_modulus}}, {{poisson_ratio}}
*DENSITY
{{density}}

*SOLID SECTION, ELSET=EALL, MATERIAL={{material_name}}

*BOUNDARY
{{bc_fixed}}, 1, 6

*STEP
*FREQUENCY
10

*NODE PRINT, NSET=NALL
U

*END STEP
""",

    "thermal": """
** Thermal Analysis
*HEADING
{{title}}

*INCLUDE, INPUT={{mesh_file}}

*MATERIAL, NAME={{material_name}}
*CONDUCTIVITY
{{conductivity}}
*SPECIFIC HEAT
{{specific_heat}}
*DENSITY
{{density}}

*SOLID SECTION, ELSET=EALL, MATERIAL={{material_name}}

** Initial conditions
*INITIAL CONDITIONS, TYPE=TEMPERATURE
NALL, {{initial_temp}}

** Step
*STEP
*HEAT TRANSFER, STEADY STATE
1.0

** Boundary conditions
*BOUNDARY
{{bc_temp_nodes}}, 11, 11, {{fixed_temp}}

** Heat flux
*CFLUX
{{heat_node}}, 11, {{heat_flux}}

*NODE PRINT, NSET=NALL
NT

*END STEP
"""
}


def generate_static_analysis(
    mesh_file: Path,
    output_file: Path,
    material: Material,
    fixed_nodes: List[int],
    load_node: int,
    load_value: float,
    load_direction: Tuple[int, float] = (3, -1.0)  # dof, magnitude sign
) -> Path:
    """
    Quick generation of static analysis input.
    
    Args:
        mesh_file: Path to mesh file
        output_file: Output input file
        material: Material definition
        fixed_nodes: List of fixed node IDs
        load_node: Node ID where load is applied
        load_value: Load magnitude
        load_direction: (dof, direction_sign) - default (3, -1) = -Z
        
    Returns:
        Path to generated file
    """
    generator = InputFileGenerator()
    generator.set_mesh_file(mesh_file)
    generator.add_material(material)
    generator.add_section(Section("section1", material.name))
    generator.define_node_set("FIXED", fixed_nodes)
    
    # Create BC manager
    from ..bc.boundary_conditions import BoundaryConditionManager
    bcm = BoundaryConditionManager()
    bcm.add_fixed_constraint("fixed", fixed_nodes)
    bcm.add_force_load(
        "load",
        [load_node],
        load_value,
        (0, 0, load_direction[1])
    )
    generator.set_boundary_conditions(bcm)
    
    return generator.generate(output_file)
