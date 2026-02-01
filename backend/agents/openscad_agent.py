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
    
    def compile_to_stl(self, scad_code: str, output_path: str = None, timeout: int = 120) -> Dict[str, Any]:
        """
        Compiles OpenSCAD code to STL using the CLI.
        Now includes auto-optimization for high $fn values.
        """
        if not self.has_openscad_cli:
            return {"success": False, "error": "OpenSCAD CLI not available"}
            
        # --- Auto-Optimization: Clamp $fn to reasonable values ---
        import re
        
        optimized_code = scad_code
        
        # Detect if script is heavy (heuristic)
        is_heavy = len(scad_code) > 2000
        target_fn = 24 if is_heavy else 32
        
        # Regex to find $fn = <number>; handling mostly standard formatting
        # We replace any $fn > target with target
        
        def fn_reducer(match):
            val = int(match.group(1))
            if val > target_fn:
                print(f"Optimizing: Reducing $fn from {val} to {target_fn}")
                return f"$fn={target_fn};"
            return match.group(0)
            
        optimized_code = re.sub(r'\$fn\s*=\s*(\d+);', fn_reducer, optimized_code)
        
        # Also handle inline $fn (common in spheres) e.g. sphere(r=10, $fn=100)
        # Replacing all might be aggressive, but safest for performance
        optimized_code = re.sub(r'\$fn\s*=\s*(\d{2,})', f'$fn={target_fn}', optimized_code)

        with tempfile.NamedTemporaryFile(suffix=".scad", delete=False, mode='w') as tmp_scad:
            tmp_scad.write(optimized_code)
            scad_path = tmp_scad.name
        
        if output_path is None:
            stl_file = tempfile.NamedTemporaryFile(suffix='.stl', delete=False)
            stl_path = stl_file.name
            stl_file.close()
            os.unlink(stl_path) # Ensure OpenSCAD creates it
        else:
            stl_path = output_path
        
        try:
            # Compile OpenSCAD to STL with proper headless flags
            # Setup environment for headless mode
            env = os.environ.copy()
            env['DISPLAY'] = ''  # Ensure headless mode on Linux/Mac
            
            # OpenSCAD command - use only universally supported flags
            # Removed --enable=fast-csg and --enable=lazy-union for compatibility
            # These flags are only available in OpenSCAD 2021.01+ and cause errors on older versions
            cmd = [
                self.openscad_path,
                '--export-format', 'binstl',  # Binary STL (supported since 2015)
                '-o', stl_path,
                scad_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,  # Use dynamic timeout
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
                print(f"[OpenSCAD] ERROR: Output file not created at {stl_path}")
                print(f"[OpenSCAD] Command: {' '.join(cmd)}")
                print(f"[OpenSCAD] STDOUT: {result.stdout}")
                print(f"[OpenSCAD] STDERR: {result.stderr}")
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

            # --- Generate SDF Grid (Phase 8/9: True SDF Support) ---
            # Optimize: Adaptive resolution based on complexity
            # 32^3 = 32k points. 64^3 = 262k points (too slow for real-time).
            vertex_count = len(vertices)
            
            # Optimization: Adaptive resolution based on complexity
            if vertex_count > 50000:
                print(f"[OpenSCAD] Mesh very complex ({vertex_count}v). Forcing low-res SDF (16).")
                resolution = 16
            elif vertex_count > 10000:
                print(f"[OpenSCAD] Mesh complex ({vertex_count}v). Reducing SDF resolution to 24.")
                resolution = 24
            else:
                resolution = 32

            sdf_flat = []
            sdf_min = 0
            sdf_max = 0
            min_pt = center
            max_pt = center # Fallback

            if resolution > 0:
                # 1. Create grid points
                # Add 10% padding
                size = bounds[1] - bounds[0]
                max_dim = np.max(size)
                center_pt = (bounds[0] + bounds[1]) / 2
                
                # Make a cubic bound centered on object
                half_size = (max_dim * 1.2) / 2
                min_pt = center_pt - half_size
                max_pt = center_pt + half_size
                
                x = np.linspace(min_pt[0], max_pt[0], resolution)
                y = np.linspace(min_pt[1], max_pt[1], resolution)
                z = np.linspace(min_pt[2], max_pt[2], resolution)
                
                # Create coordinate grid
                # Note: Meshgrid order='ij' matches array indexing (x, y, z)
                grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
                grid_points = np.stack((grid_x, grid_y, grid_z), axis=-1).reshape(-1, 3)
                
                # 2. Compute Signed Distance
                try:
                    print(f"[OpenSCAD] Baking SDF ({resolution}^3) for {vertex_count} vertices...")
                    
                    # Check for basic validity
                    if mesh.is_empty:
                        raise ValueError("Mesh is empty")
                        
                    # For non-watertight meshes, signed_distance can be flaky.
                    # We can try to use 'scan' method if available or just proceed.
                    # Trimesh signed_distance usually handles non-watertight by ray casting, 
                    # but can be wrong.
                    
                    # ensure we are using the robust method
                    sdf_values = trimesh.proximity.signed_distance(mesh, grid_points)
                    
                    # Check if we got valid values
                    if sdf_values is None or len(sdf_values) == 0:
                        raise ValueError("Trimesh returned empty SDF")
                        
                    sdf_flat = sdf_values.astype(float).tolist()
                    sdf_min = float(np.min(sdf_values))
                    sdf_max = float(np.max(sdf_values))
                    
                    # If dynamics range is 0, something is wrong (unless flat plane?)
                    if sdf_min == sdf_max:
                        print(f"[OpenSCAD] Warning: SDF constant value {sdf_min}. Mesh might be invalid/2D.")
                        
                    print(f"[OpenSCAD] SDF Bake Complete. Range: [{sdf_min:.3f}, {sdf_max:.3f}]")
                    
                except Exception as sdf_err:
                    print(f"[OpenSCAD] SDF generation failed: {sdf_err}")
                    # Don't fail the whole compile, but maybe return error metadata
                    sdf_flat = []
            else:
                # Fallback bounds for non-SDF return
                min_pt = bounds[0]
                max_pt = bounds[1]

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
                "face_count": len(faces),
                
                # SDF Data
                "sdf_data": sdf_flat,
                "sdf_resolution": resolution,
                "sdf_bounds": [min_pt.tolist(), max_pt.tolist()],
                "sdf_range": [sdf_min, sdf_max]
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
                "Syntax validation",
                "SDF Baking (Trimesh)",
                "Progressive Assembly Rendering"
            ],
            "progressive_rendering": True
        }
    
    # ===== PROGRESSIVE ASSEMBLY RENDERING =====
    
    def compile_assembly_progressive(self, scad_code: str):
        """
        Compile OpenSCAD assembly progressively using parallel execution.
        Yields parts as they complete.
        
        Args:
            scad_code: OpenSCAD source code
            
        """
        try:
            from .openscad_parser import OpenSCADParser
        except ImportError:
            from agents.openscad_parser import OpenSCADParser
        import concurrent.futures
        import time
        
        # --- INTELLIGENT AUTO-CENTERING (Analytic) ---
        # "Intelligence" means checking the AST for global transforms without compiling.
        
        try:
            from .geometry_estimator import GeometryEstimator
            
            # 1. Parse Initial (Fast)
            parser = OpenSCADParser()
            ast_nodes = parser.parse(scad_code)
            
            # 2. Analytic Bounds Check (Instant)
            estimator = GeometryEstimator()
            bounds, center = estimator.calculate_bounds(ast_nodes, parser.variables)
            
            cx, cy, cz = center["x"], center["y"], center["z"]
            
            # 3. Apply Correction if Needed
            offset_vec = [0, 0, 0]
            if abs(cx) > 100 or abs(cy) > 100 or abs(cz) > 100:
                print(f"[OpenSCAD] Intelligence: Model off-center ({cx:.1f}, {cy:.1f}, {cz:.1f}). Applying correction.")
                offset_vec = [-cx, -cy, -cz]
                scad_code = f"translate([{offset_vec[0]}, {offset_vec[1]}, {offset_vec[2]}]) {{ \n{scad_code}\n }}"
                
                # Re-parse needed? Yes, structure changed (though logically strictly deeper).
                # But progressive compiler re-parses anyway at line 329.
                # So we just update scad_code and let it flow.
                
                # Update metadata for the stream
                sys_bounds = bounds
                sys_center = center
            else:
                print("[OpenSCAD] Model is already valid/centered.")
                
        except Exception as e:
            print(f"[OpenSCAD] Auto-centering analysis failed: {e}. Proceeding raw.")

        # --- END INTELLIGENT AUTO-CENTERING ---
        
        # Parse code into AST (Freshly centered if updated)
        parser = OpenSCADParser()
        try:
            ast_nodes = parser.parse(scad_code)
        except Exception as e:
            yield {
                "success": False,
                "error": f"Parse error: {str(e)}",
                "event": "error"
            }
            return
        
        # Instead of flattening, we traverse recursively to bundle transforms
        # and respect atomic units (Booleans like Hull/Difference).
        
        compilable_nodes = []
        
        def _collect_parts_recursive(node, transform_stack=[]):
            """
            Traverse tree to find renderable parts, propagating transforms.
            """
            # 1. Atomic Units: Booleans (Hull, Difference), Primitives, & LOOPS
            # We treat these as single renderable objects (Mesh) to prevent explosion of parts
            if node.node_type.value in ['boolean', 'primitive', 'loop']:
                # For booleans (hull/diff) and loops, the 'code' includes the children block/statements.
                # Just need to check if we should wrap it in transforms.
                
                # Clone the node to attach the context (wrapper code) 
                # or just yield it with metadata.
                # We need to render: transform_stack + node.code
                
                # Create a synthetic wrapper for compilation
                wrapper_code = "\n".join(transform_stack)
                
                # Store the wrapper in the node (hacky but ephemeral)
                node.temp_wrapper = wrapper_code
                node.temp_idx = len(compilable_nodes)
                compilable_nodes.append(node)
                return

            # 2. Containers: Transforms
            if node.node_type.value in ['transform']:
                # Add this transform to stack
                # Ensure we handle the header correctly (it might be empty for some transforms?)
                header = node.header if hasattr(node, 'header') else ""
                new_stack = transform_stack + [header] if header else transform_stack
                # Recurse
                for child in node.children:
                    _collect_parts_recursive(child, new_stack)
                    
            # 3. Transparent Containers: Modules, Conditionals
            # We must traverse these to find the geometry inside.
            # For modules, the parser has already inlined the body into .children
            # Note: Conditionals (IF) could also be atomic, but traversing them is safer for now
            # unless we can evaluate them statically (which parser tries to do).
            elif node.node_type.value in ['module', 'conditional']:
                for child in node.children:
                    _collect_parts_recursive(child, transform_stack)
        
        # Start traversal from roots
        for root in ast_nodes:
            _collect_parts_recursive(root)
        
        # Filter is no longer needed as we collected specifically
        # compilable_nodes list is already populated

        
        if not compilable_nodes:
            yield {
                "success": False,
                "error": "No compilable geometry found",
                "event": "error"
            }
            return
        
        total_parts = len(compilable_nodes)
        completed = 0
        
        # Generate global variable header
        variable_header_lines = []
        for name, value in parser.variables.items():
            # Convert Python value to SCAD value string
            if isinstance(value, bool):
                val_str = "true" if value else "false"
            elif isinstance(value, (list, tuple)):
                # Simple list handling (no recursion for MVP)
                val_str = str(list(value)).replace("'", '"')
            else:
                val_str = str(value)
            variable_header_lines.append(f"{name} = {val_str};")
        
        variable_header = "\n".join(variable_header_lines)


        # Analyze variables for Physical Intelligence (Scale, Units)
        # Defaults
        scale_factor = 1.0
        
        # Check for common scale variable names
        if 'scale_factor' in parser.variables:
            try:
                scale_factor = float(parser.variables['scale_factor'])
            except: pass
        elif 'scale' in parser.variables:
             try:
                scale_factor = float(parser.variables['scale'])
             except: pass
             
        # Heuristic: If scale is suspiciously small (like 1/6), assume it's a model
        # and we might want to report "Real World" dims.
        
        # Yield initial status with Intelligence Metadata
        yield {
            "event": "start",
            "total_parts": total_parts,
            "message": f"Compiling {total_parts} parts in parallel...",
            "metadata": {
                "scale_factor": scale_factor,
                "is_scaled_model": scale_factor != 1.0,
                "estimated_real_scale": 1.0 / scale_factor if scale_factor != 0 else 1.0
            }
        }
        
        # Compile parts in parallel
        max_workers = min(4, total_parts)  # Limit to 4 concurrent compilations
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all compilation jobs
            future_to_node = {
                executor.submit(self._compile_node, node, idx, variable_header): (node, idx)
                for idx, node in enumerate(compilable_nodes)
            }
            
            # Yield results as they complete
            for future in concurrent.futures.as_completed(future_to_node):
                node, idx = future_to_node[future]
                completed += 1
                
                try:
                    result = future.result()
                    
                    if result.get("success"):
                        yield {
                            "event": "part",
                            "part_id": f"{node.name}_{idx}",
                            "part_name": node.name,
                            "part_index": idx,
                            "depth": node.depth,
                            "vertices": result["vertices"],
                            "faces": result["faces"],
                            "normals": result.get("normals", []),
                            "bounds": result.get("bounds"),
                            "center": result.get("center"),
                            "volume": result.get("volume", 0),
                            "progress": completed / total_parts,
                            "completed": completed,
                            "total": total_parts
                        }
                    else:
                        # Part failed, but continue with others
                        yield {
                            "event": "part_error",
                            "part_id": f"{node.name}_{idx}",
                            "error": result.get("error", "Unknown error"),
                            "progress": completed / total_parts
                        }
                        
                except Exception as e:
                    yield {
                        "event": "part_error",
                        "part_id": f"{node.name}_{idx}",
                        "error": str(e),
                        "progress": completed / total_parts
                    }
        
        # Yield completion
        yield {
            "event": "complete",
            "total_parts": total_parts,
            "completed": completed,
            "message": f"Assembly complete: {completed}/{total_parts} parts rendered"
        }
    
    def _compile_node(self, node, idx: int, variable_header: str = "") -> Dict[str, Any]:
        """
        Compile a single AST node to geometry.
        
        Args:
            node: ASTNode to compile
            idx: Node index for unique identification
            variable_header: Global variables to prepend
            
        Returns:
            Compilation result with vertices, faces, etc.
        """
        # Generate standalone OpenSCAD code for this node
        scad_code = self._generate_scad_for_node(node)
        
        # Prepend transform stack (if we used context-aware traversal)
        if hasattr(node, 'temp_wrapper') and node.temp_wrapper:
             scad_code = f"{node.temp_wrapper}\n{scad_code}"
        
        # Prepend global variables
        if variable_header:
            scad_code = f"{variable_header}\n{scad_code}"
        
        # Compile using existing compile_to_stl method
        result = self.compile_to_stl(scad_code)
        
        return result
    
    def _generate_scad_for_node(self, node) -> str:
        """
        Generate standalone OpenSCAD code for a single node.
        
        Args:
            node: ASTNode to generate code for
            
        Returns:
            OpenSCAD code string
        """
        try:
            from .openscad_parser import NodeType
        except ImportError:
            from agents.openscad_parser import NodeType
        
        # For primitives, just return the code
        if node.node_type == NodeType.PRIMITIVE:
            return node.code
        
        # For modules, we need the module definition + instantiation
        if node.node_type == NodeType.MODULE:
            # node.children contains the parsed module body
            # We need to reconstruct: module definition + module call
            # But the parser already inlined the body into children
            # So we just need to render the children as standalone code
            
            # Generate code from children (the inlined module body)
            children_code = []
            for child in node.children:
                child_code = self._generate_scad_for_node(child)
                if child_code:
                    children_code.append(child_code)
            
            # Return the children code (primitives/transforms from module body)
            return "\n".join(children_code)
        
        # For transforms, include the transform + child
        if node.node_type == NodeType.TRANSFORM:
            return node.code
        
        # For booleans, include all children
        if node.node_type == NodeType.BOOLEAN:
            return node.code
        
        # Default: return the code as-is
        return node.code

