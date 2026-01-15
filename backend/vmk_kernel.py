
import numpy as np
from typing import List, Callable, Dict, Optional, Tuple, Any
from pydantic import BaseModel
import math

class ToolProfile(BaseModel):
    id: str
    radius: float # in microns (μm)
    type: str     # 'BALL', 'FLAT', 'V_BIT'
    angle: Optional[float] = None # For V_BIT (degrees)

class AABB(BaseModel):
    """Axis Aligned Bounding Box for spatial optimization"""
    min_point: List[float]
    max_point: List[float]

    def expand(self, margin: float):
        """Expand the bounding box by margin"""
        self.min_point = [x - margin for x in self.min_point]
        self.max_point = [x + margin for x in self.max_point]

    def overlaps(self, p: np.ndarray, radius: float) -> bool:
        """Check if point p (with radius) overlaps this AABB"""
        # A simplified check: is p within box expanded by radius?
        for i in range(3):
            if p[i] < self.min_point[i] - radius or p[i] > self.max_point[i] + radius:
                return False
        return True

class VMKInstruction(BaseModel):
    tool_id: str
    path: List[List[float]] = [] # Optional for lattice
    aabb: Optional[AABB] = None 
    type: str = "CUT" # CUT, LATTICE
    lattice_type: Optional[str] = None
    period: Optional[float] = None
    thickness: Optional[float] = None

class SymbolicMachiningKernel:
    """
    HWC Kernel: Virtual CNC Engine.
    Achieves 100% precision by treating geometry as a subtractive function.
    Logic: FinalShape = Stock - Sum(ToolPaths)
    
    OPTIMIZED: Uses BVH/AABB for fast rejection.
    """
    def __init__(self, stock_dims: List[float]):
        self.stock_dims = np.array(stock_dims)
        self.history: List[VMKInstruction] = []
        self.tools: Dict[str, ToolProfile] = {}

    def register_tool(self, tool: ToolProfile):
        self.tools[tool.id] = tool

    def execute_gcode(self, instruction: VMKInstruction):
        """Records a toolpath instruction into the symbolic history."""
        # Pre-calculate AABB for optimization
        
        # Handle instructions without paths (e.g. LATTICE field definition)
        if not instruction.path:
            # If no path, AABB covers entire stock? Or defined explicitly?
            # User instruction might want to apply lattice everywhere.
            # Let's default to full stock AABB if not specified.
            if not instruction.aabb:
                instruction.aabb = AABB(
                    min_point=(-self.stock_dims/2).tolist(), 
                    max_point=(self.stock_dims/2).tolist()
                )
            self.history.append(instruction)
            return

        points = np.array(instruction.path)
        min_p = np.min(points, axis=0)
        max_p = np.max(points, axis=0)
        
        tool = self.tools.get(instruction.tool_id)
        margin = tool.radius if tool else 0.0
        
        aabb = AABB(
            min_point=(min_p - margin).tolist(),
            max_point=(max_p + margin).tolist()
        )
        instruction.aabb = aabb
        self.history.append(instruction)

    def get_sdf(self, p: np.ndarray) -> float:
        """
        The Core Resolver: Computes the 'Symbolic Truth' at point p.
        This looks for the Minimum Euclidean Distance to the surface.
        positive = outside, negative = inside.
        """
        # 1. Initial Stock SDF (Box SDF)
        # d_stock = signed distance to stock box
        q = np.abs(p) - self.stock_dims / 2.0
        d = np.linalg.norm(np.maximum(q, 0.0)) + min(max(q[0], max(q[1], q[2])), 0.0)
        
        # 2. Subtractive Iteration
        
        d_cut_global = float('inf') 
        
        for instr in self.history:
            # Check for LATTICE Instruction
            if instr.type == "LATTICE":
                # Lattice is handled as an intersection or subtraction? 
                # User says: "compile custom micro-architectures"
                # Usually lattice replaces material. 
                # But VMK is subtractive. 
                # To make a lattice, we subtract the "Void Phase".
                # If Gyroid SDF > 0 (Solid), SDF < 0 (Void).
                # We want to keep Solid. So we subtract where Void.
                # subtraction: max(d, -d_void).
                # d_void is the SDF of the lattice structure. 
                # If we want to keep the lattice, we are cutting away the empty space.
                
                # Evaluation
                val = self._evaluate_lattice_sdf(p, instr)
                
                # Logic: If val > 0, we are in Lattice Solid. We KEEP this.
                # If val < 0, we are in Lattice Void. We REMOVE this.
                # Start with Solid Block.
                # Cut away Void.
                # Void is defined by region where Lattice SDF < 0? 
                # Actually, standard Gyroid: > t is one domain, < -t is other. 
                # Membrane is near 0.
                # Let's assume generate_unit_cell_sdf returns distance to Surface.
                # Inside Wall (+), Outside Wall (-)? Or vice versa.
                # Agent returns: abs(val) - thickness.
                # If result < 0: Inside Wall (Keep).
                # If result > 0: Outside Wall (Void).
                # So we subtract where result > 0.
                
                # Operation: d = max(d, val) ?? 
                # If val > 0 (Void), max(d, val) -> positive (Void).
                # If val < 0 (Solid), max(d, neg) -> d (Original stock or cut).
                # This performs Intersection? 
                # Yes, Intersection = max(A, B).
                
                # If we intersect Block with Lattice:
                # Result is Lattice shaped block.
                d = max(d, val)
                continue

            # Standard Toolpath Logic
            tool = self.tools[instr.tool_id]
            
            if not instr.aabb.overlaps(p, tool.radius * 2.0): 
                 continue

            path_dist = self._capsule_sweep_sdf(p, instr.path, tool.radius)
            d_cut_global = min(d_cut_global, path_dist)
            
        # Boolean Subtraction for standard tools: max(Stock, -UnionOfCuts)
        if d_cut_global != float('inf'):
             d = max(d, -d_cut_global)
            
        return d

    def _evaluate_lattice_sdf(self, p: np.ndarray, instr: VMKInstruction) -> float:
        """
        Evaluates Periodic Lattice SDF.
        """
        # 1. Coordinate Transform (Modulo)
        period = instr.period if instr.period is not None else 10.0
        thickness = instr.thickness if instr.thickness is not None else 0.5
        l_type = instr.lattice_type if instr.lattice_type is not None else 'GYROID'
        
        # Periodic P: p_mod = p % period
        # But we need continuous phase for smooth derivatives? 
        # sin(2pi * p / period) handles periodicity naturally without modulo discontinuity at boundary.
        # But modulo is needed if we want exact cell repetition logic? 
        # Actually sin(kx) IS periodic. We just scale p.
        
        # Scale to 0..2pi space
        # x_scaled = x * (2pi / period)
        k = 2.0 * np.pi / period
        p_scaled = p * k
        
        x, y, z = p_scaled[0], p_scaled[1], p_scaled[2]
        
        val = 0.0
        if l_type == "GYROID":
            val = np.sin(x)*np.cos(y) + np.sin(y)*np.cos(z) + np.sin(z)*np.cos(x)
        elif l_type == "SCHWARZ_P":
            val = np.cos(x) + np.cos(y) + np.cos(z)
            
        # Distance approx: abs(val) - t
        # If < 0: Within wall thickness (Solid)
        # If > 0: In void
        return abs(val) - thickness
            
    def _capsule_sweep_sdf(self, p: np.ndarray, path: List[List[float]], r: float) -> float:
        """Calculates the distance from point p to a swept tool path (Multi-segment Capsule)."""
        min_dist = float('inf')
        
        # This loop is still O(M) where M is path length.
        # Further optimization: Path segment BVH within the instruction.
        
        path_np = np.array(path)
        if len(path_np) < 2:
            return np.linalg.norm(p - path_np[0]) - r
            
        # Vectorized calculation could replace the loop for Python performance
        # But implementing naive loop for clarity first as requested.
        
        for i in range(len(path) - 1):
            a = path_np[i]
            b = path_np[i+1]
            pa = p - a
            ba = b - a
            
            # Project point onto line segment
            len_sq = np.dot(ba, ba)
            if len_sq < 1e-9:
                h = 0.0 # Point segment
            else:
                h = np.clip(np.dot(pa, ba) / len_sq, 0.0, 1.0)
            
            # Distance vector
            d_vec = pa - ba * h
            dist = np.linalg.norm(d_vec) - r
            
            min_dist = min(min_dist, dist)
            
        return min_dist

    def get_state(self) -> Dict[str, Any]:
        """Return the current kernel state for visualization"""
        return {
            "stock_dims": self.stock_dims.tolist(),
            "tools": {k: v.dict() for k, v in self.tools.items()},
            "history": [
                {
                    "tool_id": instr.tool_id,
                    "path": instr.path,
                    "aabb": instr.aabb.dict() if instr.aabb else None
                }
                for instr in self.history
            ]
        }
    
    def get_sdf_grid(self, dims: Tuple[int, int, int] = (64, 64, 32), padding: float = 2.0) -> np.ndarray:
        """
        Generate a dense SDF grid for the entire stock volume.
        Used for visualization (Raymarching) and Manifold Repair (Marching Cubes).
        WARNING: O(N*M) where N is grid size and M is history length. Can be slow in Python.
        
        Args:
            dims: Grid resolution (x, y, z)
            padding: Extra space around stock (in units) to ensure isosurface assumes closed volume.
        """
        nx, ny, nz = dims
        
        # Ranges with padding
        x_range = np.linspace(-self.stock_dims[0]/2 - padding, self.stock_dims[0]/2 + padding, nx)
        y_range = np.linspace(-self.stock_dims[1]/2 - padding, self.stock_dims[1]/2 + padding, ny)
        z_range = np.linspace(-self.stock_dims[2]/2 - padding, self.stock_dims[2]/2 + padding, nz)
        
        # Initialize with max float
        grid = np.zeros((nx, ny, nz), dtype=np.float32)
        
        # Iterate (Iterative approach for now - slow but safe)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Force Boundary Closure: If on edge of grid, return MAX_DIST
                    if i==0 or i==nx-1 or j==0 or j==ny-1 or k==0 or k==nz-1:
                        grid[i, j, k] = 100.0 # Force Outside
                        continue

                    p = np.array([x_range[i], y_range[j], z_range[k]])
                    grid[i, j, k] = self.get_sdf(p)
                    
        return grid
