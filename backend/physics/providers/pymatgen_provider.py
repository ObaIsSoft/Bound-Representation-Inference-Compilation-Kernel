"""
Pymatgen Provider - Materials Science

Wraps pymatgen library for crystal structures, materials analysis, and properties.
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class PymatgenProvider:
    """
    Provider for materials science using pymatgen library.
    Handles crystal structures, material properties, and phase diagrams.
    """
    
    def __init__(self):
        """Initialize the pymatgen library"""
        try:
            import pymatgen
            from pymatgen.core import Structure, Lattice, Element
            from pymatgen.analysis.structure_matcher import StructureMatcher
            
            self.pymatgen = pymatgen
            self.Structure = Structure
            self.Lattice = Lattice
            self.Element = Element
            self.StructureMatcher = StructureMatcher
            
            logger.info(f"PymatgenProvider initialized (version {pymatgen.__version__})")
            
        except ImportError as e:
            logger.error(f"Failed to import pymatgen: {e}")
            raise RuntimeError(f"pymatgen library is required but not available: {e}")
    
    def get_element_properties(self, symbol: str) -> Dict[str, Any]:
        """
        Get properties of a chemical element.
        
        Args:
            symbol: Element symbol (e.g., 'Fe', 'Al', 'Ti')
        
        Returns:
            Dictionary of element properties
        """
        element = self.Element(symbol)
        
        return {
            "atomic_number": element.Z,
            "atomic_mass": element.atomic_mass,
            "atomic_radius": element.atomic_radius,
            "density": element.density_of_solid if hasattr(element, 'density_of_solid') else None,
            "melting_point": element.melting_point if hasattr(element, 'melting_point') else None,
            "boiling_point": element.boiling_point if hasattr(element, 'boiling_point') else None,
            "electronic_structure": element.electronic_structure,
        }
    
    def create_crystal_structure(
        self,
        lattice_params: List[float],
        species: List[str],
        coords: List[List[float]]
    ) -> Any:
        """
        Create a crystal structure.
        
        Args:
            lattice_params: [a, b, c, alpha, beta, gamma] in Angstroms and degrees
            species: List of element symbols
            coords: List of fractional coordinates [[x1,y1,z1], [x2,y2,z2], ...]
        
        Returns:
            pymatgen Structure object
        """
        lattice = self.Lattice.from_parameters(*lattice_params)
        structure = self.Structure(lattice, species, coords)
        
        return structure
    
    def get_material_density(self, structure: Any) -> float:
        """
        Calculate material density from crystal structure.
        
        Args:
            structure: pymatgen Structure object
        
        Returns:
            Density in g/cm³
        """
        return structure.density
    
    def compare_structures(self, structure1: Any, structure2: Any) -> Dict[str, Any]:
        """
        Compare two crystal structures.
        
        Args:
            structure1: First structure
            structure2: Second structure
        
        Returns:
            Comparison results including RMSD
        """
        matcher = self.StructureMatcher()
        
        return {
            "are_identical": matcher.fit(structure1, structure2),
            "rmsd": matcher.get_rms_dist(structure1, structure2) if matcher.fit(structure1, structure2) else None,
        }
