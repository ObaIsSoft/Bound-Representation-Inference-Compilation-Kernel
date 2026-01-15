"""
OpenSCAD Agent
Compiles OpenSCAD code to STL meshes for rendering in BRICK.
Enables Thingiverse integration and community CAD import.
"""

import subprocess
import tempfile
import os
from typing import Dict, Any, Optional
import trimesh
import numpy as np

class OpenSCADAgent:
    """
    Compiles OpenSCAD scripts to renderable geometry.
    Supports both local OpenSCAD CLI and SolidPython fallback.
    """
    
    def __init__(self):
        self.openscad_path = self._find_openscad()
        self.has_openscad_cli = self.openscad_path is not None
        
    def _find_openscad(self) -> Optional[str]:
        """Find OpenSCAD executable on the system."""
        # Try common paths
        possible_paths = [
            'openscad',  # In PATH
            '/usr/local/bin/openscad',
            '/opt/homebrew/bin/openscad',
            '/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD',
            '/Applications/OpenSCAD-2021.01.app/Contents/MacOS/OpenSCAD',
        ]
        
        for path in possible_paths:
            try:
                result = subprocess.run(
                    [path, '--version'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return path
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        return None
    
    def _check_openscad_installed(self) -> bool:
        """Check if OpenSCAD CLI is available."""
        return self.openscad_path is not None
    
    def compile_to_stl(self, scad_code: str, output_path: str = None) -> Dict[str, Any]:
        """
        Compiles OpenSCAD code to STL using the CLI.
        Now includes auto-optimization for high $fn values.
        """
        if not self.has_openscad_cli:
            return {"success": False, "error": "OpenSCAD CLI not available"}
            
        # --- Auto-Optimization: Clamp $fn to reasonable values ---
        # High $fn values (e.g. > 50) cause exponential slowdowns.
        # We replace global $fn definitions with a capped value.
        import re
        
        # Regex to find $fn = <number>;
        optimized_code = scad_code
        fn_match = re.search(r'\$fn\s*=\s*(\d+);', scad_code)
        if fn_match:
            val = int(fn_match.group(1))
            if val > 32:
                print(f"Optimizing: Reducing $fn from {val} to 32 for performance.")
                optimized_code = re.sub(r'\$fn\s*=\s*\d+;', '$fn = 32;', scad_code)
        
        # Also handle local defaults if they appear frequently
        optimized_code = re.sub(r'\$fn\s*=\s*([5-9]\d|\d{3,});', '$fn=32;', optimized_code)
        optimized_code = re.sub(r'\$fn\s*=\s*([5-9]\d|\d{3,})\)', '$fn=32)', optimized_code)

        with tempfile.NamedTemporaryFile(suffix=".scad", delete=False, mode='w') as tmp_scad:
            tmp_scad.write(optimized_code)
            scad_path = tmp_scad.name
        
        if output_path is None:
            stl_file = tempfile.NamedTemporaryFile(suffix='.stl', delete=False)
            stl_path = stl_file.name
            stl_file.close()
        else:
            stl_path = output_path
        
        try:
            # Compile OpenSCAD to STL with proper headless flags
            # --export-format forces binary STL output
            # -o specifies output file
            env = os.environ.copy()
            env['DISPLAY'] = ''  # Ensure headless mode on Linux/Mac
            
            result = subprocess.run(
                [
                    self.openscad_path,
                    '--export-format', 'binstl',  # Binary STL (smaller, faster)
                    '-o', stl_path,
                    scad_path
                ],
                capture_output=True,
                text=True,
                timeout=120,  # Increased to 120s for complex models
                env=env
            )
            
            if result.returncode != 0:
                error_msg = f"OpenSCAD compilation failed (exit code {result.returncode})"
                if result.stderr:
                    error_msg += f"\nSTDERR: {result.stderr}"
                if result.stdout:
                    error_msg += f"\nSTDOUT: {result.stdout}"
                    
                return {
                    "success": False,
                    "error": error_msg,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            
            # Check if STL file was actually created
            if not os.path.exists(stl_path):
                return {
                    "success": False,
                    "error": f"OpenSCAD did not generate output file. STDERR: {result.stderr}"
                }
            
            # Check if file has content
            if os.path.getsize(stl_path) == 0:
                return {
                    "success": False,
                    "error": "OpenSCAD generated empty STL file. Check your code for errors."
                }
            
            # Load STL mesh using trimesh
            try:
                mesh = trimesh.load(stl_path, file_type='stl')
            except Exception as load_err:
                return {
                    "success": False,
                    "error": f"Failed to load STL file: {str(load_err)}. File size: {os.path.getsize(stl_path)} bytes"
                }
            
            # Extract geometry data for Three.js
            vertices = mesh.vertices.tolist()
            faces = mesh.faces.tolist()
            normals = mesh.vertex_normals.tolist()
            
            # Calculate bounding box
            bounds = mesh.bounds
            center = mesh.centroid.tolist()
            
            return {
                "success": True,
                "vertices": vertices,
                "faces": faces,
                "normals": normals,
                "bounds": {
                    "min": bounds[0].tolist(),
                    "max": bounds[1].tolist()
                },
                "center": center,
                "volume": float(mesh.volume),
                "surface_area": float(mesh.area),
                "stl_path": stl_path,
                "vertex_count": len(vertices),
                "face_count": len(faces)
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "OpenSCAD compilation timed out (>120s). Try simplifying the model or reducing $fn value."
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Mesh processing failed: {str(e)}"
            }
        finally:
            # Cleanup temporary SCAD file
            if os.path.exists(scad_path):
                os.unlink(scad_path)
    
    def validate_syntax(self, scad_code: str) -> Dict[str, Any]:
        """
        Quick syntax validation without full compilation.
        """
        # Basic syntax checks
        errors = []
        
        # Check for balanced braces
        if scad_code.count('{') != scad_code.count('}'):
            errors.append("Unbalanced braces")
        
        # Check for balanced parentheses
        if scad_code.count('(') != scad_code.count(')'):
            errors.append("Unbalanced parentheses")
        
        # Check for balanced brackets
        if scad_code.count('[') != scad_code.count(']'):
            errors.append("Unbalanced brackets")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Get OpenSCAD agent capabilities."""
        return {
            "agent": "OpenSCADAgent",
            "openscad_cli_available": self.has_openscad_cli,
            "supported_formats": ["scad", "stl"],
            "features": [
                "OpenSCAD script compilation",
                "STL mesh generation",
                "Thingiverse compatibility",
                "Syntax validation"
            ]
        }
