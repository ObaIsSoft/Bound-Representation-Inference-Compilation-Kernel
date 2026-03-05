"""
Geometry-to-Physics Bridge - Connects GeometryAgent to Physics Kernel

This module bridges the gap between CAD geometry generation and physics analysis,
enabling structural, thermal, and fluid analysis on geometry meshes.

Capabilities:
1. Convert geometry meshes to FEA input format
2. Extract cross-section properties for beam theory
3. Generate boundary conditions from geometry features
4. Interface with StructuralAgent for FEA
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Import centralized defaults
try:
    from backend.agents.config.physics_config import (
        STEEL, DEFAULT_MATERIAL, get_material,
        MESH_DEFAULTS, STRUCTURAL_DEFAULTS
    )
    HAS_DEFAULTS = True
except ImportError:
    HAS_DEFAULTS = False
    STEEL = {"density": 7850.0}
    MESH_DEFAULTS = {"tolerance": 0.01}
    STRUCTURAL_DEFAULTS = {"default_load": 1000.0}

logger = logging.getLogger(__name__)


@dataclass
class CrossSectionProperties:
    """Cross-section properties for structural analysis"""
    area: float  # m²
    moment_of_inertia_x: float  # m⁴
    moment_of_inertia_y: float  # m⁴
    polar_moment: float  # m⁴
    section_modulus_x: float  # m³
    section_modulus_y: float  # m³
    centroid: Tuple[float, float]  # (y, z) relative to section origin


@dataclass
class GeometryAnalysisModel:
    """Complete geometry model ready for physics analysis"""
    vertices: np.ndarray  # (N, 3) array of vertex positions
    faces: np.ndarray  # (M, 3) array of triangle indices
    volumes: np.ndarray  # (P,) array of tetrahedron indices (if 3D mesh)
    
    # Extracted properties
    volume: float  # m³
    surface_area: float  # m²
    bounding_box: Tuple[float, float, float, float, float, float]  # (xmin, ymin, zmin, xmax, ymax, zmax)
    centroid: Tuple[float, float, float]
    
    # Cross-section properties (if beam-like)
    cross_section: Optional[CrossSectionProperties] = None
    
    # Mesh quality metrics
    mesh_quality: Optional[Dict[str, float]] = None


class GeometryPhysicsBridge:
    """
    Bridge between GeometryAgent and Physics Kernel
    
    Converts geometric models to physics-ready analysis models.
    """
    
    def __init__(self):
        self.name = "GeometryPhysicsBridge"
    
    def create_analysis_model(
        self,
        mesh_vertices: np.ndarray,
        mesh_faces: np.ndarray,
        mesh_volumes: Optional[np.ndarray] = None
    ) -> GeometryAnalysisModel:
        """
        Create a complete analysis model from mesh data
        
        Args:
            mesh_vertices: (N, 3) array of vertex positions
            mesh_faces: (M, 3) array of triangle indices
            mesh_volumes: Optional (P, 4) array of tetrahedron indices
            
        Returns:
            GeometryAnalysisModel with extracted properties
        """
        vertices = np.asarray(mesh_vertices)
        faces = np.asarray(mesh_faces)
        
        # Calculate basic properties
        volume = self._calculate_volume(vertices, faces)
        surface_area = self._calculate_surface_area(vertices, faces)
        bbox = self._calculate_bounding_box(vertices)
        centroid = self._calculate_centroid(vertices, faces)
        
        # Calculate mesh quality
        mesh_quality = self._calculate_mesh_quality(vertices, faces)
        
        # Try to extract cross-section properties (for beam-like geometries)
        cross_section = self._extract_cross_section(vertices, faces)
        
        return GeometryAnalysisModel(
            vertices=vertices,
            faces=faces,
            volumes=mesh_volumes if mesh_volumes is not None else np.array([]),
            volume=volume,
            surface_area=surface_area,
            bounding_box=bbox,
            centroid=centroid,
            cross_section=cross_section,
            mesh_quality=mesh_quality
        )
    
    def _calculate_volume(self, vertices: np.ndarray, faces: np.ndarray) -> float:
        """Calculate volume using tetrahedral decomposition"""
        if len(faces) == 0:
            return 0.0
        
        # Use divergence theorem: V = (1/3) * sum(dot(v0, cross(v1, v2))) / 6
        total_volume = 0.0
        
        for face in faces:
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            
            # Signed volume of tetrahedron with origin
            vol = np.dot(v0, np.cross(v1, v2)) / 6.0
            total_volume += vol
        
        return abs(total_volume)
    
    def _calculate_surface_area(self, vertices: np.ndarray, faces: np.ndarray) -> float:
        """Calculate total surface area"""
        if len(faces) == 0:
            return 0.0
        
        total_area = 0.0
        
        for face in faces:
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            
            # Cross product gives twice the triangle area
            area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            total_area += area
        
        return total_area
    
    def _calculate_bounding_box(self, vertices: np.ndarray) -> Tuple[float, ...]:
        """Calculate axis-aligned bounding box"""
        if len(vertices) == 0:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        mins = np.min(vertices, axis=0)
        maxs = np.max(vertices, axis=0)
        
        return (mins[0], mins[1], mins[2], maxs[0], maxs[1], maxs[2])
    
    def _calculate_centroid(self, vertices: np.ndarray, faces: np.ndarray) -> Tuple[float, float, float]:
        """Calculate centroid (center of mass assuming uniform density)"""
        if len(faces) == 0:
            return tuple(np.mean(vertices, axis=0)) if len(vertices) > 0 else (0.0, 0.0, 0.0)
        
        # Weighted average of tetrahedron centroids
        total_vol = 0.0
        centroid = np.zeros(3)
        
        for face in faces:
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            
            # Tetrahedron volume with origin
            vol = np.dot(v0, np.cross(v1, v2)) / 6.0
            
            # Tetrahedron centroid
            tet_centroid = (v0 + v1 + v2) / 4.0
            
            centroid += vol * tet_centroid
            total_vol += vol
        
        if abs(total_vol) < 1e-10:
            return tuple(np.mean(vertices, axis=0))
        
        centroid = centroid / total_vol
        return tuple(centroid)
    
    def _calculate_mesh_quality(self, vertices: np.ndarray, faces: np.ndarray) -> Dict[str, float]:
        """Calculate mesh quality metrics"""
        if len(faces) == 0:
            return {}
        
        aspect_ratios = []
        min_angles = []
        
        for face in faces:
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            
            # Edge lengths
            e0 = np.linalg.norm(v1 - v0)
            e1 = np.linalg.norm(v2 - v1)
            e2 = np.linalg.norm(v0 - v2)
            
            # Aspect ratio
            max_edge = max(e0, e1, e2)
            min_edge = min(e0, e1, e2)
            aspect_ratios.append(max_edge / (min_edge + 1e-10))
            
            # Minimum angle using law of cosines
            # cos(A) = (b² + c² - a²) / (2bc)
            if e0 > 0 and e1 > 0 and e2 > 0:
                # Angle at v0
                cos_a0 = (e2**2 + e0**2 - e1**2) / (2 * e2 * e0)
                angle0 = np.arccos(np.clip(cos_a0, -1, 1))
                
                # Angle at v1
                cos_a1 = (e0**2 + e1**2 - e2**2) / (2 * e0 * e1)
                angle1 = np.arccos(np.clip(cos_a1, -1, 1))
                
                # Angle at v2
                angle2 = np.pi - angle0 - angle1
                
                min_angles.append(min(angle0, angle1, angle2) * 180 / np.pi)
        
        return {
            "num_vertices": len(vertices),
            "num_faces": len(faces),
            "min_aspect_ratio": float(min(aspect_ratios)) if aspect_ratios else 0.0,
            "max_aspect_ratio": float(max(aspect_ratios)) if aspect_ratios else 0.0,
            "avg_aspect_ratio": float(np.mean(aspect_ratios)) if aspect_ratios else 0.0,
            "min_angle_deg": float(min(min_angles)) if min_angles else 0.0,
            "avg_angle_deg": float(np.mean(min_angles)) if min_angles else 0.0,
        }
    
    def _extract_cross_section(
        self,
        vertices: np.ndarray,
        faces: np.ndarray
    ) -> Optional[CrossSectionProperties]:
        """
        Extract cross-section properties for beam theory analysis
        
        Assumes geometry is beam-like (elongated in one direction)
        """
        if len(vertices) < 3 or len(faces) < 2:
            return None
        
        # Determine principal axis from bounding box
        bbox = self._calculate_bounding_box(vertices)
        extents = (
            bbox[3] - bbox[0],  # x extent
            bbox[4] - bbox[1],  # y extent
            bbox[5] - bbox[2]   # z extent
        )
        
        # Find longest axis (beam axis)
        beam_axis = np.argmax(extents)
        
        # Cross-section is perpendicular to beam axis
        # Project vertices to cross-section plane
        if beam_axis == 0:  # X-axis is beam axis, cross-section in YZ plane
            coords = vertices[:, [1, 2]]  # (y, z)
        elif beam_axis == 1:  # Y-axis is beam axis, cross-section in XZ plane
            coords = vertices[:, [0, 2]]  # (x, z)
        else:  # Z-axis is beam axis, cross-section in XY plane
            coords = vertices[:, [0, 1]]  # (x, y)
        
        # Calculate cross-section area using 2D polygon area
        # This is an approximation using convex hull
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(coords)
            area = hull.volume  # In 2D, volume = area
            
            # Calculate centroid of cross-section
            centroid_yz = np.mean(coords[hull.vertices], axis=0)
            
            # Calculate moments of inertia about centroid
            # Using parallel axis theorem
            I_x = 0.0  # About Y axis
            I_y = 0.0  # About Z axis (or vice versa depending on orientation)
            
            for vertex in coords:
                dy = vertex[0] - centroid_yz[0]
                dz = vertex[1] - centroid_yz[1]
                I_x += dz**2  # I = ∫ z² dA
                I_y += dy**2  # I = ∫ y² dA
            
            # Average per vertex (approximation)
            I_x *= area / len(coords)
            I_y *= area / len(coords)
            
            # Section modulus
            max_dy = np.max(np.abs(coords[:, 0] - centroid_yz[0]))
            max_dz = np.max(np.abs(coords[:, 1] - centroid_yz[1]))
            
            S_x = I_x / (max_dz + 1e-10)
            S_y = I_y / (max_dy + 1e-10)
            
            return CrossSectionProperties(
                area=area,
                moment_of_inertia_x=I_x,
                moment_of_inertia_y=I_y,
                polar_moment=I_x + I_y,
                section_modulus_x=S_x,
                section_modulus_y=S_y,
                centroid=tuple(centroid_yz)
            )
        except Exception as e:
            logger.warning(f"Could not extract cross-section: {e}")
            return None
    
    def export_to_calculix(
        self,
        model: GeometryAnalysisModel,
        filepath: str,
        element_type: str = "C3D4"
    ) -> bool:
        """
        Export mesh to CalculiX INP format
        
        Args:
            model: Geometry analysis model
            filepath: Output file path (.inp)
            element_type: CalculiX element type (C3D4=tetrahedron, C3D8=hexahedron)
            
        Returns:
            True if successful
        """
        try:
            with open(filepath, 'w') as f:
                f.write("*Heading\n")
                f.write(f"** Geometry exported from GeometryPhysicsBridge\n")
                f.write("*Node\n")
                
                # Write nodes
                for i, vertex in enumerate(model.vertices, start=1):
                    f.write(f"{i}, {vertex[0]:.6e}, {vertex[1]:.6e}, {vertex[2]:.6e}\n")
                
                # Write elements
                f.write(f"*Element, type={element_type}\n")
                
                if element_type == "C3D4" and len(model.volumes) > 0:
                    # Tetrahedral elements
                    for i, tet in enumerate(model.volumes, start=1):
                        f.write(f"{i}, {tet[0]+1}, {tet[1]+1}, {tet[2]+1}, {tet[3]+1}\n")
                else:
                    # Surface elements (for shell analysis)
                    for i, face in enumerate(model.faces, start=1):
                        if len(face) == 3:
                            f.write(f"{i}, {face[0]+1}, {face[1]+1}, {face[2]+1}\n")
                        else:
                            f.write(f"{i}, {face[0]+1}, {face[1]+1}, {face[2]+1}, {face[3]+1}\n")
                
                # Add element set for all elements
                f.write("*Elset, elset=AllElements, generate\n")
                if len(model.volumes) > 0:
                    f.write(f"1, {len(model.volumes)}, 1\n")
                else:
                    f.write(f"1, {len(model.faces)}, 1\n")
                
                logger.info(f"Exported mesh to {filepath}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to export to CalculiX: {e}")
            return False
    
    def generate_boundary_conditions(
        self,
        model: GeometryAnalysisModel,
        support_faces: Optional[List[int]] = None,
        load_faces: Optional[List[int]] = None,
        load_magnitude: float = None  # Use STRUCTURAL_DEFAULTS if None
    ) -> Dict[str, Any]:
        """
        Generate boundary conditions from geometry features
        
        Args:
            model: Geometry analysis model
            support_faces: Face indices to apply displacement BC
            load_faces: Face indices to apply load
            load_magnitude: Magnitude of applied load (N)
            
        Returns:
            Dictionary with boundary condition specifications
        """
        # Apply defaults if not provided
        if load_magnitude is None:
            load_magnitude = STRUCTURAL_DEFAULTS.get("default_load", 1000.0)
        
        bc = {
            "supports": [],
            "loads": []
        }
        
        # Default: fix bottom faces (minimum Z)
        if support_faces is None:
            vertices = model.vertices
            min_z = np.min(vertices[:, 2])
            tolerance = MESH_DEFAULTS.get("tolerance", 0.01) * (np.max(vertices[:, 2]) - min_z)
            
            # Find faces on bottom
            bottom_faces = []
            for i, face in enumerate(model.faces):
                face_z = np.mean(vertices[face, 2])
                if abs(face_z - min_z) < tolerance:
                    bottom_faces.append(i)
            support_faces = bottom_faces
        
        # Add support boundary condition
        for face_idx in support_faces:
            face = model.faces[face_idx]
            for vertex_idx in face:
                bc["supports"].append({
                    "node": int(vertex_idx + 1),  # 1-based indexing for CalculiX
                    "dof": [1, 2, 3],  # Fix all DOFs
                    "value": [0.0, 0.0, 0.0]
                })
        
        # Default: apply load to top faces (maximum Z)
        if load_faces is None:
            vertices = model.vertices
            max_z = np.max(vertices[:, 2])
            tolerance = MESH_DEFAULTS.get("tolerance", 0.01) * (max_z - np.min(vertices[:, 2]))
            
            top_faces = []
            for i, face in enumerate(model.faces):
                face_z = np.mean(vertices[face, 2])
                if abs(face_z - max_z) < tolerance:
                    top_faces.append(i)
            load_faces = top_faces
        
        # Add load boundary condition
        for face_idx in load_faces:
            face = model.faces[face_idx]
            # Distribute load among face vertices
            load_per_node = load_magnitude / len(face)
            for vertex_idx in face:
                bc["loads"].append({
                    "node": int(vertex_idx + 1),
                    "dof": 3,  # Z-direction
                    "value": -load_per_node  # Downward
                })
        
        return bc
    
    def create_beam_model(
        self,
        length: float,
        cross_section: CrossSectionProperties,
        num_segments: int = 10
    ) -> GeometryAnalysisModel:
        """
        Create a simplified beam model for 1D analysis
        
        This creates a line mesh (beam elements) for efficient beam theory analysis
        """
        # Create line mesh along X axis
        x_coords = np.linspace(0, length, num_segments + 1)
        vertices = np.column_stack([x_coords, np.zeros(num_segments + 1), np.zeros(num_segments + 1)])
        
        # Line elements (2 nodes each)
        faces = np.column_stack([
            np.arange(num_segments),
            np.arange(1, num_segments + 1),
            np.zeros(num_segments, dtype=int)  # Padding for triangular format
        ])
        
        return GeometryAnalysisModel(
            vertices=vertices,
            faces=faces,
            volumes=np.array([]),
            volume=cross_section.area * length,
            surface_area=cross_section.area * 2 + 2 * (cross_section.area**0.5) * length,
            bounding_box=(0.0, 0.0, 0.0, length, 0.0, 0.0),
            centroid=(length / 2, 0.0, 0.0),
            cross_section=cross_section,
            mesh_quality={"type": "beam_1d", "num_segments": num_segments}
        )


# Convenience functions
def extract_cross_section_from_mesh(
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    plane_normal: Tuple[float, float, float] = (1.0, 0.0, 0.0)
) -> Optional[CrossSectionProperties]:
    """Extract cross-section properties from a mesh slice"""
    bridge = GeometryPhysicsBridge()
    model = bridge.create_analysis_model(mesh_vertices, mesh_faces)
    return model.cross_section


def calculate_mass_properties(
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    density: float = None  # Uses STEEL density if None
) -> Dict[str, float]:
    """Calculate mass properties from mesh"""
    if density is None:
        density = STEEL.get("density", 7850.0)  # kg/m³
    """Calculate mass properties from mesh"""
    bridge = GeometryPhysicsBridge()
    model = bridge.create_analysis_model(mesh_vertices, mesh_faces)
    
    mass = model.volume * density
    
    return {
        "volume": model.volume,
        "mass": mass,
        "surface_area": model.surface_area,
        "centroid_x": model.centroid[0],
        "centroid_y": model.centroid[1],
        "centroid_z": model.centroid[2],
    }


def prepare_for_fea(
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Prepare geometry for FEA analysis
    
    Returns dictionary with:
    - analysis_model: GeometryAnalysisModel
    - calculix_path: Path to exported INP file (if output_path provided)
    - boundary_conditions: Generated BCs
    - properties: Extracted geometric properties
    """
    bridge = GeometryPhysicsBridge()
    model = bridge.create_analysis_model(mesh_vertices, mesh_faces)
    
    result = {
        "analysis_model": model,
        "properties": {
            "volume": model.volume,
            "surface_area": model.surface_area,
            "bounding_box": model.bounding_box,
            "centroid": model.centroid,
            "mesh_quality": model.mesh_quality,
        }
    }
    
    if model.cross_section:
        result["properties"]["cross_section"] = {
            "area": model.cross_section.area,
            "moment_of_inertia_x": model.cross_section.moment_of_inertia_x,
            "moment_of_inertia_y": model.cross_section.moment_of_inertia_y,
            "polar_moment": model.cross_section.polar_moment,
            "section_modulus_x": model.cross_section.section_modulus_x,
            "section_modulus_y": model.cross_section.section_modulus_y,
        }
    
    # Generate boundary conditions
    result["boundary_conditions"] = bridge.generate_boundary_conditions(model)
    
    # Export to CalculiX if path provided
    if output_path:
        calculix_path = output_path.replace('.json', '.inp')
        if bridge.export_to_calculix(model, calculix_path):
            result["calculix_path"] = calculix_path
    
    return result
