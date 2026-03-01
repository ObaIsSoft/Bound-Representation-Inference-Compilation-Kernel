"""
FIX-207: Result Parsing

Parse CalculiX results from .frd and .dat files.
Extract displacements, stresses, strains, and other field data.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import struct
import logging

logger = logging.getLogger(__name__)


class ResultType(Enum):
    """Types of FEA results"""
    DISPLACEMENT = "displacement"
    STRESS = "stress"
    STRAIN = "strain"
    TEMPERATURE = "temperature"
    FORCE = "force"
    CONTACT = "contact"
    ERROR = "error"


@dataclass
class NodalResult:
    """Results at a single node"""
    node_id: int
    coordinates: Tuple[float, float, float]
    displacement: Optional[Tuple[float, float, float]] = None
    temperature: Optional[float] = None
    
    # Stress components (xx, yy, zz, xy, yz, zx)
    stress: Optional[Tuple[float, ...]] = None
    
    # Strain components
    strain: Optional[Tuple[float, ...]] = None
    
    # Von Mises stress
    von_mises: Optional[float] = None


@dataclass
class ElementResult:
    """Results at a single element"""
    element_id: int
    element_type: int
    stress: Optional[Tuple[float, ...]] = None
    strain: Optional[Tuple[float, ...]] = None
    von_mises: Optional[float] = None


@dataclass
class FEAResults:
    """Complete FEA results container"""
    job_name: str
    result_file: Path
    
    # Model info
    num_nodes: int
    num_elements: int
    
    # Nodal results
    nodal_results: Dict[int, NodalResult] = field(default_factory=dict)
    
    # Element results
    element_results: Dict[int, ElementResult] = field(default_factory=dict)
    
    # Global results
    max_displacement: float = 0.0
    max_stress: float = 0.0
    max_von_mises: float = 0.0
    
    def get_displacement_vector(self) -> np.ndarray:
        """Get all nodal displacements as array"""
        displacements = []
        for nr in self.nodal_results.values():
            if nr.displacement:
                displacements.append(nr.displacement)
            else:
                displacements.append((0.0, 0.0, 0.0))
        return np.array(displacements)
    
    def get_stress_array(self) -> np.ndarray:
        """Get all element stresses as array"""
        stresses = []
        for er in self.element_results.values():
            if er.stress:
                stresses.append(er.stress)
        return np.array(stresses)
    
    def get_von_mises_array(self) -> np.ndarray:
        """Get all Von Mises stresses"""
        vm = []
        for nr in self.nodal_results.values():
            if nr.von_mises is not None:
                vm.append(nr.von_mises)
        return np.array(vm)
    
    def get_max_stress_location(self) -> Optional[Tuple[int, float]]:
        """Get node ID and value of maximum stress"""
        max_stress = 0.0
        max_node = None
        
        for node_id, nr in self.nodal_results.items():
            if nr.von_mises and nr.von_mises > max_stress:
                max_stress = nr.von_mises
                max_node = node_id
        
        return (max_node, max_stress) if max_node else None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get results summary"""
        return {
            "job_name": self.job_name,
            "num_nodes": self.num_nodes,
            "num_elements": self.num_elements,
            "max_displacement": self.max_displacement,
            "max_stress": self.max_stress,
            "max_von_mises": self.max_von_mises,
            "displacement_magnitude": float(np.linalg.norm(self.get_displacement_vector())),
            "num_stress_results": len(self.element_results)
        }


class ResultParser:
    """
    Parse CalculiX result files.
    
    FIX-207: Implements result parsing from .frd and .dat files.
    
    Usage:
        parser = ResultParser()
        results = parser.parse_frd("job.frd")
        
        print(f"Max displacement: {results.max_displacement}")
        print(f"Max Von Mises: {results.max_von_mises}")
    """
    
    def __init__(self):
        self._results: List[FEAResults] = []
    
    def parse_frd(self, frd_file: Path, job_name: Optional[str] = None) -> FEAResults:
        """
        Parse CalculiX .frd result file.
        
        The .frd format is a binary/text hybrid format containing:
        - Node coordinates
        - Element connectivity
        - Nodal results (displacement, temperature)
        - Element results (stress, strain)
        
        Args:
            frd_file: Path to .frd file
            job_name: Optional job name (default: file stem)
            
        Returns:
            FEAResults with all parsed data
        """
        frd_file = Path(frd_file)
        
        if not frd_file.exists():
            raise FileNotFoundError(f"FRD file not found: {frd_file}")
        
        if job_name is None:
            job_name = frd_file.stem
        
        results = FEAResults(
            job_name=job_name,
            result_file=frd_file,
            num_nodes=0,
            num_elements=0
        )
        
        with open(frd_file, 'r') as f:
            lines = f.readlines()
        
        logger.info(f"Parsing FRD file: {frd_file}")
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Header
            if line.startswith('    1C'):
                # Header info
                pass
            
            # Nodes
            elif line.startswith('    2C'):
                # Node block header
                i += 1
                continue
            
            elif line.startswith(' -1'):
                # Node data: -1 node_id x y z
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        node_id = int(parts[1])
                        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                        
                        results.nodal_results[node_id] = NodalResult(
                            node_id=node_id,
                            coordinates=(x, y, z)
                        )
                        results.num_nodes = max(results.num_nodes, node_id)
                    except (ValueError, IndexError):
                        pass
            
            # Elements
            elif line.startswith('    3C'):
                # Element block header
                i += 1
                continue
            
            elif line.startswith(' -2'):
                # Element data: -2 elem_id elem_type ...
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        elem_id = int(parts[1])
                        elem_type = int(parts[2])
                        
                        results.element_results[elem_id] = ElementResult(
                            element_id=elem_id,
                            element_type=elem_type
                        )
                        results.num_elements = max(results.num_elements, elem_id)
                    except (ValueError, IndexError):
                        pass
            
            # Nodal results
            elif line.startswith('    1P'):
                # Result block header
                # Parse result type
                i = self._parse_nodal_results(lines, i, results)
                continue
            
            # Element results
            elif line.startswith('    3P'):
                # Element result block header
                i = self._parse_element_results(lines, i, results)
                continue
            
            i += 1
        
        # Calculate derived quantities
        self._calculate_derived(results)
        
        self._results.append(results)
        
        logger.info(f"Parsed {results.num_nodes} nodes, {results.num_elements} elements")
        logger.info(f"Max displacement: {results.max_displacement:.6e}")
        logger.info(f"Max Von Mises: {results.max_von_mises:.6e}")
        
        return results
    
    def _parse_nodal_results(
        self,
        lines: List[str],
        start_idx: int,
        results: FEAResults
    ) -> int:
        """Parse nodal result block"""
        i = start_idx
        
        # Header line contains result info
        header = lines[i]
        
        # Extract result type from header
        # Format: 1PSTEP, inc, type, numnod, numelm, ...
        parts = header.split()
        
        result_type = None
        if 'DISP' in header.upper():
            result_type = 'displacement'
        elif 'STRESS' in header.upper():
            result_type = 'stress'
        elif 'TEMP' in header.upper():
            result_type = 'temperature'
        
        i += 1
        
        # Read values
        while i < len(lines) and not lines[i].startswith('    1P') and not lines[i].startswith('    3P'):
            line = lines[i]
            
            if line.startswith(' -1'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        node_id = int(parts[1])
                        values = [float(p) for p in parts[2:]]
                        
                        if node_id in results.nodal_results:
                            nr = results.nodal_results[node_id]
                            
                            if result_type == 'displacement' and len(values) >= 3:
                                nr.displacement = (values[0], values[1], values[2])
                            elif result_type == 'temperature' and len(values) >= 1:
                                nr.temperature = values[0]
                            elif result_type == 'stress' and len(values) >= 6:
                                nr.stress = tuple(values[:6])
                                nr.von_mises = self._calculate_von_mises(nr.stress)
                    except (ValueError, IndexError):
                        pass
            
            i += 1
        
        return i - 1  # Return index before next block
    
    def _parse_element_results(
        self,
        lines: List[str],
        start_idx: int,
        results: FEAResults
    ) -> int:
        """Parse element result block"""
        i = start_idx
        
        header = lines[i]
        
        result_type = None
        if 'STRESS' in header.upper():
            result_type = 'stress'
        elif 'STRAIN' in header.upper():
            result_type = 'strain'
        
        i += 1
        
        while i < len(lines) and not lines[i].startswith('    1P') and not lines[i].startswith('    3P'):
            line = lines[i]
            
            if line.startswith(' -1'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        elem_id = int(parts[1])
                        values = [float(p) for p in parts[2:]]
                        
                        if elem_id in results.element_results:
                            er = results.element_results[elem_id]
                            
                            if result_type == 'stress' and len(values) >= 6:
                                er.stress = tuple(values[:6])
                                er.von_mises = self._calculate_von_mises(er.stress)
                            elif result_type == 'strain' and len(values) >= 6:
                                er.strain = tuple(values[:6])
                    except (ValueError, IndexError):
                        pass
            
            i += 1
        
        return i - 1
    
    def _calculate_derived(self, results: FEAResults) -> None:
        """Calculate derived quantities from results"""
        # Max displacement
        max_disp = 0.0
        for nr in results.nodal_results.values():
            if nr.displacement:
                disp_mag = np.linalg.norm(nr.displacement)
                max_disp = max(max_disp, disp_mag)
        results.max_displacement = max_disp
        
        # Max stress
        max_stress = 0.0
        max_vm = 0.0
        
        for nr in results.nodal_results.values():
            if nr.von_mises:
                max_vm = max(max_vm, nr.von_mises)
            if nr.stress:
                max_stress = max(max_stress, max(abs(s) for s in nr.stress))
        
        for er in results.element_results.values():
            if er.von_mises:
                max_vm = max(max_vm, er.von_mises)
            if er.stress:
                max_stress = max(max_stress, max(abs(s) for s in er.stress))
        
        results.max_stress = max_stress
        results.max_von_mises = max_vm
    
    def _calculate_von_mises(self, stress: Tuple[float, ...]) -> float:
        """Calculate Von Mises stress from components"""
        if len(stress) < 6:
            return 0.0
        
        sx, sy, sz, sxy, syz, szx = stress[:6]
        
        # Von Mises formula
        vm = np.sqrt(0.5 * (
            (sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2 +
            6 * (sxy**2 + syz**2 + szx**2)
        ))
        
        return vm
    
    def parse_dat(self, dat_file: Path, job_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse CalculiX .dat file (ASCII results).
        
        The .dat file contains formatted ASCII output of results.
        
        Args:
            dat_file: Path to .dat file
            job_name: Optional job name
            
        Returns:
            Dictionary with parsed data
        """
        dat_file = Path(dat_file)
        
        if not dat_file.exists():
            raise FileNotFoundError(f"DAT file not found: {dat_file}")
        
        if job_name is None:
            job_name = dat_file.stem
        
        results = {
            "job_name": job_name,
            "displacements": [],
            "stresses": [],
            "forces": [],
            "totals": {}
        }
        
        with open(dat_file, 'r') as f:
            content = f.read()
            lines = f.readlines()
        
        # Parse displacement section
        if 'displacement' in content.lower():
            results["displacements"] = self._parse_displacement_section(content)
        
        # Parse stress section
        if 'stress' in content.lower():
            results["stresses"] = self._parse_stress_section(content)
        
        # Parse total force section
        if 'total force' in content.lower():
            results["totals"]["force"] = self._parse_total_force(content)
        
        return results
    
    def _parse_displacement_section(self, content: str) -> List[Dict]:
        """Parse displacement data from content"""
        displacements = []
        
        # Look for displacement table
        lines = content.split('\n')
        in_displacement = False
        
        for line in lines:
            if 'displacement' in line.lower() and 'node' in line.lower():
                in_displacement = True
                continue
            
            if in_displacement:
                if line.strip() and not line.startswith('-'):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            node_id = int(parts[0])
                            disp = [float(p) for p in parts[1:4]]
                            displacements.append({
                                "node": node_id,
                                "displacement": disp,
                                "magnitude": np.linalg.norm(disp)
                            })
                        except (ValueError, IndexError):
                            pass
                elif 'stress' in line.lower() or 'strain' in line.lower():
                    break
        
        return displacements
    
    def _parse_stress_section(self, content: str) -> List[Dict]:
        """Parse stress data from content"""
        stresses = []
        
        lines = content.split('\n')
        in_stress = False
        
        for line in lines:
            if 'stress' in line.lower() and 'element' in line.lower():
                in_stress = True
                continue
            
            if in_stress:
                if line.strip() and not line.startswith('-'):
                    parts = line.split()
                    if len(parts) >= 7:
                        try:
                            elem_id = int(parts[0])
                            stress = [float(p) for p in parts[1:7]]
                            von_mises = self._calculate_von_mises(tuple(stress))
                            stresses.append({
                                "element": elem_id,
                                "stress": stress,
                                "von_mises": von_mises
                            })
                        except (ValueError, IndexError):
                            pass
                elif line.strip() and not line[0].isdigit():
                    break
        
        return stresses
    
    def _parse_total_force(self, content: str) -> Dict:
        """Parse total force from content"""
        forces = {"fx": 0.0, "fy": 0.0, "fz": 0.0}
        
        # Look for total force line
        for line in content.split('\n'):
            if 'total force' in line.lower():
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        forces["fx"] = float(parts[-3])
                        forces["fy"] = float(parts[-2])
                        forces["fz"] = float(parts[-1])
                    except (ValueError, IndexError):
                        pass
        
        return forces
    
    def export_to_vtk(
        self,
        results: FEAResults,
        vtk_file: Path,
        include_stress: bool = True
    ) -> Path:
        """
        Export results to VTK format for visualization.
        
        Args:
            results: FEAResults to export
            vtk_file: Output VTK file path
            include_stress: Include stress data
            
        Returns:
            Path to exported file
        """
        try:
            import meshio
        except ImportError:
            raise RuntimeError("meshio required for VTK export")
        
        vtk_file = Path(vtk_file)
        vtk_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Build meshio mesh
        points = []
        for node_id in sorted(results.nodal_results.keys()):
            nr = results.nodal_results[node_id]
            points.append(nr.coordinates)
        
        points = np.array(points)
        
        # Create point data
        point_data = {}
        
        # Displacements
        disp = []
        for node_id in sorted(results.nodal_results.keys()):
            nr = results.nodal_results[node_id]
            if nr.displacement:
                disp.append(nr.displacement)
            else:
                disp.append((0.0, 0.0, 0.0))
        
        if disp:
            point_data["displacement"] = np.array(disp)
        
        # Von Mises stress at nodes
        vm = []
        for node_id in sorted(results.nodal_results.keys()):
            nr = results.nodal_results[node_id]
            vm.append(nr.von_mises if nr.von_mises else 0.0)
        
        if any(v > 0 for v in vm):
            point_data["von_mises"] = np.array(vm)
        
        # Create simple mesh (no connectivity for now)
        mesh = meshio.Mesh(points, [], point_data=point_data)
        mesh.write(vtk_file)
        
        logger.info(f"Exported results to VTK: {vtk_file}")
        return vtk_file
    
    def get_results_history(self) -> List[FEAResults]:
        """Get history of parsed results"""
        return self._results.copy()


# Convenience function
def parse_results(frd_file: Path) -> FEAResults:
    """
    Quick result parsing.
    
    Args:
        frd_file: Path to .frd file
        
    Returns:
        FEAResults
    """
    parser = ResultParser()
    return parser.parse_frd(frd_file)
