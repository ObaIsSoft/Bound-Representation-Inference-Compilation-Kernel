"""
VMK Process Simulation Extension

Extends the SymbolicMachiningKernel with physics-based process simulation:
- G-code parsing (G0, G1, G2, G3)
- Feed rate and spindle speed handling
- Machining time calculation
- Cutting force estimation
- Tool wear modeling
- Surface finish prediction
- Material removal rate calculation

Reference:
- Altintas, Y. (2012). Manufacturing Automation: Metal Cutting Mechanics
- Boothroyd, G. (2006). Fundamentals of Metal Machining
"""

import numpy as np
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

from backend.vmk_kernel import SymbolicMachiningKernel, ToolProfile, VMKInstruction, AABB


class GCodeCommand(Enum):
    """G-code motion commands"""
    RAPID = "G0"      # Rapid positioning
    LINEAR = "G1"     # Linear interpolation
    CW_ARC = "G2"     # Clockwise arc
    CCW_ARC = "G3"    # Counter-clockwise arc
    DWELL = "G4"      # Dwell
    PLANE_XY = "G17"  # XY plane selection
    PLANE_ZX = "G18"  # ZX plane selection
    PLANE_YZ = "G19"  # YZ plane selection
    INCH = "G20"      # Inch units
    METRIC = "G21"    # Metric units
    ABSOLUTE = "G90"  # Absolute positioning
    RELATIVE = "G91"  # Relative positioning


@dataclass
class MachiningParameters:
    """Complete machining parameters for an operation"""
    feed_rate: float          # mm/min
    spindle_speed: float      # RPM
    depth_of_cut: float       # mm (axial)
    width_of_cut: float       # mm (radial)
    tool_radius: float        # mm
    coolant_on: bool = False
    num_flutes: int = 4       # Number of tool flutes (default: 4)
    
    # Derived parameters
    feed_per_tooth: float = field(init=False)
    cutting_speed: float = field(init=False)
    material_removal_rate: float = field(init=False)
    
    def __post_init__(self):
        self.feed_per_tooth = self.feed_rate / (self.spindle_speed * self.num_flutes)
        self.cutting_speed = 2 * np.pi * self.tool_radius * self.spindle_speed / 1000  # m/min
        # MRR = width * depth * feed_rate
        self.material_removal_rate = self.width_of_cut * self.depth_of_cut * self.feed_rate  # mm³/min


@dataclass
class ProcessMetrics:
    """Complete process simulation results"""
    # Time metrics
    total_time_seconds: float
    rapid_time_seconds: float
    cutting_time_seconds: float
    
    # Distance metrics
    total_distance_mm: float
    rapid_distance_mm: float
    cutting_distance_mm: float
    
    # Force metrics
    avg_cutting_force_n: float
    max_cutting_force_n: float
    torque_nm: float
    power_kw: float
    
    # Tool metrics
    tool_wear_percent: float
    estimated_tool_life_minutes: float
    
    # Surface quality
    surface_roughness_ra: float  # μm
    
    # Material
    material_removed_cm3: float
    chip_thickness_mm: float
    
    # Cost (if rates provided)
    machine_cost_usd: Optional[float] = None
    tool_cost_usd: Optional[float] = None


@dataclass
class SimulatedInstruction:
    """VMK instruction with process simulation data"""
    instruction: VMKInstruction
    params: MachiningParameters
    start_time: float          # seconds from start
    duration: float           # seconds
    distance: float           # mm
    is_rapid: bool
    estimated_forces: List[float]  # Force at each point along path


class GCodeParser:
    """
    Parse G-code files into VMK instructions with machining parameters.
    
    Supports:
    - G0: Rapid positioning
    - G1: Linear interpolation
    - G2/G3: Circular arcs (clockwise/counter-clockwise)
    """
    
    # Configurable defaults
    DEFAULT_FEED = 1000.0       # mm/min
    DEFAULT_SPINDLE = 1000.0    # RPM
    DEFAULT_TOOL_RADIUS = 5.0   # mm
    
    def __init__(self, default_feed: float = None, default_spindle: float = None, 
                 default_tool_radius: float = None):
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.current_feed = default_feed or self.DEFAULT_FEED
        self.current_spindle = default_spindle or self.DEFAULT_SPINDLE
        self.current_tool_radius = default_tool_radius or self.DEFAULT_TOOL_RADIUS
        self.absolute_mode = True
        self.metric_units = True
        
    def parse_file(self, filepath: str) -> List[Dict[str, Any]]:
        """Parse G-code file into structured operations"""
        with open(filepath, 'r') as f:
            lines = f.readlines()
        return self.parse_lines(lines)
    
    def parse_lines(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Parse G-code lines into operations"""
        operations = []
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip().upper()
            
            # Skip empty lines and comments
            if not line or line.startswith(';'):
                continue
            if line.startswith('(') and line.endswith(')'):
                continue
                
            # Remove inline comments
            if ';' in line:
                line = line.split(';')[0]
            
            op = self._parse_line(line, line_num)
            if op:
                operations.append(op)
        
        return operations
    
    def _parse_line(self, line: str, line_num: int) -> Optional[Dict[str, Any]]:
        """Parse single G-code line"""
        
        # Extract feed and spindle first (can appear on any line)
        f = self._extract_float(line, 'F')
        s = self._extract_float(line, 'S')
        
        if f:
            self.current_feed = f
        if s:
            self.current_spindle = s
        
        # Check for M commands, T commands, etc. (no G code)
        g_match = re.search(r'G(\d+)', line)
        if not g_match:
            self._parse_misc(line)
            return None
        
        g_code = f"G{int(g_match.group(1))}"
        
        # Extract coordinates
        x = self._extract_float(line, 'X')
        y = self._extract_float(line, 'Y')
        z = self._extract_float(line, 'Z')
        
        # Build target position
        target = self.current_position.copy()
        if x is not None:
            target[0] = x if self.absolute_mode else self.current_position[0] + x
        if y is not None:
            target[1] = y if self.absolute_mode else self.current_position[1] + y
        if z is not None:
            target[2] = z if self.absolute_mode else self.current_position[2] + z
        
        # Create operation
        op = {
            'line': line_num,
            'g_code': g_code,
            'start': self.current_position.copy(),
            'end': target,
            'feed_rate': self.current_feed,
            'spindle_speed': self.current_spindle,
            'is_rapid': g_code in ['G0', 'G00']
        }
        
        # Handle arcs (G2/G3)
        if g_code in ['G2', 'G3', 'G02', 'G03']:
            i = self._extract_float(line, 'I')
            j = self._extract_float(line, 'J')
            r = self._extract_float(line, 'R')
            
            op['is_arc'] = True
            op['clockwise'] = g_code in ['G2', 'G02']
            op['center'] = self._calculate_arc_center(
                self.current_position, target, i, j, r, op['clockwise']
            )
            op['radius'] = r if r else np.linalg.norm(op['center'] - self.current_position[:2])
        
        # Update current position
        self.current_position = target
        
        return op
    
    def _extract_float(self, line: str, char: str) -> Optional[float]:
        """Extract float value following character"""
        pattern = rf'{char}(-?\d+\.?\d*)'
        match = re.search(pattern, line)
        return float(match.group(1)) if match else None
    
    def _parse_misc(self, line: str):
        """Parse non-G commands (M, T, etc.)"""
        # Tool change
        t_match = re.search(r'T(\d+)', line)
        if t_match:
            tool_num = int(t_match.group(1))
            logger.debug(f"Tool change to T{tool_num}")
        
        # Spindle control
        if 'M3' in line or 'M03' in line:
            logger.debug("Spindle on (CW)")
        elif 'M4' in line or 'M04' in line:
            logger.debug("Spindle on (CCW)")
        elif 'M5' in line or 'M05' in line:
            logger.debug("Spindle off")
        
        # Coolant
        if 'M8' in line or 'M08' in line:
            logger.debug("Coolant on")
        elif 'M9' in line or 'M09' in line:
            logger.debug("Coolant off")
    
    def _calculate_arc_center(self, start: np.ndarray, end: np.ndarray, 
                              i: Optional[float], j: Optional[float],
                              r: Optional[float], clockwise: bool) -> np.ndarray:
        """Calculate arc center from I,J or R"""
        if i is not None and j is not None:
            # I,J are relative to start
            return np.array([start[0] + i, start[1] + j])
        
        if r:
            # Calculate from radius
            mid = (start[:2] + end[:2]) / 2
            dist = np.linalg.norm(end[:2] - start[:2])
            h = np.sqrt(max(0, r**2 - (dist/2)**2))
            
            # Perpendicular direction
            dx = end[1] - start[1]
            dy = start[0] - end[0]
            
            if clockwise:
                h = -h
            
            perp = np.array([dx, dy])
            perp = perp / np.linalg.norm(perp) * h
            
            return mid + perp
        
        # Fallback to midpoint
        return (start[:2] + end[:2]) / 2


class MachiningPhysics:
    """
    Physics models for machining process.
    
    Implements mechanistic cutting force models based on:
    - Altintas, Y. (2012). Manufacturing Automation
    """
    
    # Cutting force coefficients (N/mm²) - typical values for steel
    DEFAULT_KTC = 1500  # Tangential cutting coefficient
    DEFAULT_KRC = 600   # Radial cutting coefficient
    DEFAULT_KAC = 300   # Axial cutting coefficient
    
    # Edge force coefficients (N/mm)
    DEFAULT_KTE = 30
    DEFAULT_KRE = 15
    DEFAULT_KAE = 10
    
    # Taylor tool life constants (typical for carbide tools)
    DEFAULT_TAYLOR_N = 0.25
    DEFAULT_TAYLOR_C = 300  # m/min
    
    def __init__(self, material: str = "steel", taylor_n: float = None, taylor_c: float = None):
        """
        Initialize machining physics.
        
        Args:
            material: Workpiece material (steel, aluminum, titanium)
            taylor_n: Taylor tool life exponent (default: 0.25 for carbide)
            taylor_c: Taylor tool life constant in m/min (default: 300)
        """
        self.material = material
        self.coeffs = self._get_material_coefficients(material)
        self.taylor_n = taylor_n or self.DEFAULT_TAYLOR_N
        self.taylor_c = taylor_c or self.DEFAULT_TAYLOR_C
    
    def _get_material_coefficients(self, material: str) -> Dict[str, float]:
        """Get cutting coefficients for material"""
        coefficients = {
            "steel": {
                "Ktc": 1500, "Krc": 600, "Kac": 300,
                "Kte": 30, "Kre": 15, "Kae": 10,
                "specific_cutting_energy": 5.0  # J/mm³
            },
            "aluminum": {
                "Ktc": 800, "Krc": 400, "Kac": 200,
                "Kte": 15, "Kre": 8, "Kae": 5,
                "specific_cutting_energy": 1.5
            },
            "titanium": {
                "Ktc": 2500, "Krc": 1000, "Kac": 500,
                "Kte": 50, "Kre": 25, "Kae": 15,
                "specific_cutting_energy": 8.0
            }
        }
        return coefficients.get(material.lower(), coefficients["steel"])
    
    def calculate_forces(self, params: MachiningParameters) -> Dict[str, float]:
        """
        Calculate cutting forces using mechanistic model.
        
        Ft = Ktc * h * b + Kte * b  (Tangential)
        Fr = Krc * h * b + Kre * b  (Radial)
        Fa = Kac * h * b + Kae * b  (Axial)
        
        Where:
        - h: uncut chip thickness (mm)
        - b: width of cut (mm)
        """
        c = self.coeffs
        
        # Uncut chip thickness from feed per tooth
        h = params.feed_per_tooth  # mm
        b = params.width_of_cut    # mm
        
        # Cutting forces per flute
        Ft = c["Ktc"] * h * b + c["Kte"] * b
        Fr = c["Krc"] * h * b + c["Kre"] * b
        Fa = c["Kac"] * h * b + c["Kae"] * b
        
        # Resultant force
        F_resultant = np.sqrt(Ft**2 + Fr**2 + Fa**2)
        
        # Torque at tool
        torque = Ft * params.tool_radius / 1000  # Nm
        
        # Power
        power = torque * params.spindle_speed * 2 * np.pi / 60 / 1000  # kW
        
        return {
            "Ft": Ft, "Fr": Fr, "Fa": Fa,
            "resultant": F_resultant,
            "torque_nm": torque,
            "power_kw": power
        }
    
    def calculate_tool_wear(self, params: MachiningParameters, 
                           cutting_time_minutes: float) -> float:
        """
        Estimate tool wear using Taylor's tool life equation.
        
        VT^n = C
        Where:
        - V: cutting speed (m/min)
        - T: tool life (min)
        - n, C: constants
        """
        V = params.cutting_speed  # m/min
        
        if V <= 0:
            return 0.0
        
        # Tool life at this cutting speed
        T = (self.taylor_c / V) ** (1/self.taylor_n)  # minutes
        
        # Wear percentage
        wear_percent = (cutting_time_minutes / T) * 100
        
        return min(wear_percent, 100.0)
    
    def calculate_surface_roughness(self, params: MachiningParameters) -> float:
        """
        Estimate theoretical surface roughness.
        
        For turning/milling: Ra ≈ f² / (8 * R)
        Where:
        - f: feed per revolution (mm/rev)
        - R: tool nose radius (mm)
        
        For end milling, approximates with tool radius.
        """
        # Feed per revolution
        f_rev = params.feed_rate / params.spindle_speed  # mm/rev
        
        # Tool nose radius (use tool radius as approximation)
        R = params.tool_radius
        
        if R <= 0:
            return 0.0
        
        # Theoretical peak-to-valley height
        Rt = (f_rev ** 2) / (8 * R)
        
        # Average roughness Ra ≈ Rt / 4 (approximate)
        Ra = Rt / 4
        
        # Convert to μm
        return Ra * 1000


class ProcessSimulator:
    """
    Complete machining process simulator.
    
    Integrates G-code parsing, VMK geometry tracking, and physics simulation.
    """
    
    # Configurable defaults
    DEFAULT_RAPID_FEED = 10000.0  # mm/min for rapid moves
    DEFAULT_DEPTH_OF_CUT = 2.0    # mm
    DEFAULT_WIDTH_OF_CUT = 5.0    # mm
    
    def __init__(self, stock_dims: List[float], material: str = "steel",
                 rapid_feed: float = None, default_doc: float = None,
                 default_woc: float = None):
        """
        Initialize process simulator.
        
        Args:
            stock_dims: [length, width, height] in mm
            material: Workpiece material (steel, aluminum, titanium)
            rapid_feed: Rapid traverse feed rate (mm/min)
            default_doc: Default depth of cut (mm)
            default_woc: Default width of cut (mm)
        """
        self.kernel = SymbolicMachiningKernel(stock_dims=stock_dims)
        self.physics = MachiningPhysics(material=material)
        self.parser = GCodeParser()
        self.material = material
        
        # Configurable parameters
        self.rapid_feed = rapid_feed or self.DEFAULT_RAPID_FEED
        self.default_depth_of_cut = default_doc or self.DEFAULT_DEPTH_OF_CUT
        self.default_width_of_cut = default_woc or self.DEFAULT_WIDTH_OF_CUT
        
        # Simulation state
        self.simulated_instructions: List[SimulatedInstruction] = []
        self.total_time = 0.0
        self.total_distance = 0.0
        self.total_cutting_time = 0.0
        
    def simulate_gcode(self, gcode_file: str, tool_library: Optional[Dict] = None) -> ProcessMetrics:
        """
        Simulate complete G-code file.
        
        Args:
            gcode_file: Path to G-code file
            tool_library: Dict of tool_id -> ToolProfile
            
        Returns:
            Complete process metrics
        """
        # Parse G-code
        operations = self.parser.parse_file(gcode_file)
        logger.info(f"Parsed {len(operations)} operations from {gcode_file}")
        
        # Register tools
        if tool_library:
            for tool_id, tool in tool_library.items():
                self.kernel.register_tool(tool)
        
        # Simulate each operation
        for i, op in enumerate(operations):
            self._simulate_operation(op, i)
        
        # Calculate final metrics
        return self._calculate_metrics()
    
    def simulate_operations(self, operations: List[Dict[str, Any]], 
                           tool: ToolProfile) -> ProcessMetrics:
        """
        Simulate list of operations (for programmatic use).
        
        Args:
            operations: List of operation dicts from GCodeParser
            tool: Tool to use
            
        Returns:
            ProcessMetrics
        """
        self.kernel.register_tool(tool)
        self.current_tool_id = tool.id
        
        for i, op in enumerate(operations):
            self._simulate_operation(op, i)
        
        return self._calculate_metrics()
    
    def _simulate_operation(self, op: Dict[str, Any], index: int):
        """Simulate single operation"""
        # Create VMK instruction
        tool_id = getattr(self, 'current_tool_id', f"tool_{index}")
        
        # Handle arcs by segmenting
        if op.get('is_arc'):
            path = self._segment_arc(op)
        else:
            path = [op['start'].tolist(), op['end'].tolist()]
        
        instruction = VMKInstruction(
            tool_id=tool_id,
            path=path
        )
        
        # Calculate parameters
        is_rapid = op['is_rapid']
        feed_rate = op['feed_rate'] if not is_rapid else self.rapid_feed
        
        # Get tool info if available
        tool = self.kernel.tools.get(tool_id)
        tool_radius = tool.radius if tool else self.current_tool_radius if hasattr(self, 'current_tool_radius') else 5.0
        
        # Get operation-specific parameters or use defaults
        depth_of_cut = op.get('depth_of_cut', self.default_depth_of_cut)
        width_of_cut = op.get('width_of_cut', self.default_width_of_cut)
        
        params = MachiningParameters(
            feed_rate=feed_rate,
            spindle_speed=op['spindle_speed'],
            depth_of_cut=depth_of_cut,
            width_of_cut=width_of_cut,
            tool_radius=tool_radius
        )
        
        # Calculate time and distance
        distance = self._calculate_path_distance(path)
        duration = (distance / feed_rate) * 60 if feed_rate > 0 else 0  # seconds
        
        # Calculate forces (only for cutting, not rapid)
        forces = []
        if not is_rapid:
            force_data = self.physics.calculate_forces(params)
            avg_force = force_data['resultant']
            # Simulate force variation along path
            forces = [avg_force * (0.8 + 0.4 * np.random.random()) 
                     for _ in range(len(path))]
            self.total_cutting_time += duration
        
        # Create simulated instruction
        sim_instr = SimulatedInstruction(
            instruction=instruction,
            params=params,
            start_time=self.total_time,
            duration=duration,
            distance=distance,
            is_rapid=is_rapid,
            estimated_forces=forces
        )
        
        self.simulated_instructions.append(sim_instr)
        
        # Update totals
        self.total_time += duration
        self.total_distance += distance
        
        # Execute in VMK
        self.kernel.execute_gcode(instruction)
    
    def _segment_arc(self, op: Dict[str, Any], segments: int = 20) -> List[List[float]]:
        """Segment arc into linear points"""
        start = op['start']
        end = op['end']
        center = op.get('center', (start[:2] + end[:2]) / 2)
        clockwise = op.get('clockwise', True)
        
        # Calculate angles
        start_angle = np.arctan2(start[1] - center[1], start[0] - center[0])
        end_angle = np.arctan2(end[1] - center[1], end[0] - center[0])
        
        # Adjust for direction
        if clockwise:
            while end_angle > start_angle:
                end_angle -= 2 * np.pi
        else:
            while end_angle < start_angle:
                end_angle += 2 * np.pi
        
        # Generate points
        path = [start.tolist()]
        for i in range(1, segments):
            t = i / segments
            angle = start_angle + t * (end_angle - start_angle)
            x = center[0] + op['radius'] * np.cos(angle)
            y = center[1] + op['radius'] * np.sin(angle)
            z = start[2] + t * (end[2] - start[2])
            path.append([x, y, z])
        
        path.append(end.tolist())
        return path
    
    def _calculate_path_distance(self, path: List[List[float]]) -> float:
        """Calculate total distance of path"""
        if len(path) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(path) - 1):
            p1 = np.array(path[i])
            p2 = np.array(path[i+1])
            total += np.linalg.norm(p2 - p1)
        
        return total
    
    def _calculate_metrics(self) -> ProcessMetrics:
        """Calculate final process metrics"""
        # Time breakdown
        rapid_time = sum(si.duration for si in self.simulated_instructions if si.is_rapid)
        cutting_time = sum(si.duration for si in self.simulated_instructions if not si.is_rapid)
        
        # Distance breakdown
        rapid_dist = sum(si.distance for si in self.simulated_instructions if si.is_rapid)
        cutting_dist = sum(si.distance for si in self.simulated_instructions if not si.is_rapid)
        
        # Forces
        all_forces = []
        for si in self.simulated_instructions:
            all_forces.extend(si.estimated_forces)
        
        avg_force = np.mean(all_forces) if all_forces else 0
        max_force = np.max(all_forces) if all_forces else 0
        
        # Torque and power (average)
        avg_torque = 0
        avg_power = 0
        if self.simulated_instructions:
            cutting_instr = [si for si in self.simulated_instructions if not si.is_rapid]
            if cutting_instr:
                avg_torque = np.mean([
                    si.params.tool_radius * np.mean(si.estimated_forces) / 1000 / 1000
                    for si in cutting_instr
                ])
                avg_power = np.mean([
                    si.params.spindle_speed * 2 * np.pi / 60 * avg_torque / 1000
                    for si in cutting_instr
                ])
        
        # Tool wear
        tool_wear = 0
        if cutting_time > 0 and cutting_instr:
            # Calculate average cutting parameters from actual operations
            avg_feed = np.mean([si.params.feed_rate for si in cutting_instr])
            avg_spindle = np.mean([si.params.spindle_speed for si in cutting_instr])
            avg_doc = np.mean([si.params.depth_of_cut for si in cutting_instr])
            avg_woc = np.mean([si.params.width_of_cut for si in cutting_instr])
            avg_radius = np.mean([si.params.tool_radius for si in cutting_instr])
            
            avg_params = MachiningParameters(
                feed_rate=avg_feed,
                spindle_speed=avg_spindle,
                depth_of_cut=avg_doc,
                width_of_cut=avg_woc,
                tool_radius=avg_radius
            )
            tool_wear = self.physics.calculate_tool_wear(avg_params, cutting_time / 60)
        
        # Surface roughness
        surface_roughness = 0
        if self.simulated_instructions:
            cutting_instr = [si for si in self.simulated_instructions if not si.is_rapid]
            if cutting_instr:
                avg_params = cutting_instr[0].params  # Use first cutting op
                surface_roughness = self.physics.calculate_surface_roughness(avg_params)
        
        # Material removed
        material_removed = 0
        for si in self.simulated_instructions:
            if not si.is_rapid:
                material_removed += si.params.material_removal_rate * (si.duration / 60)
        
        return ProcessMetrics(
            total_time_seconds=self.total_time,
            rapid_time_seconds=rapid_time,
            cutting_time_seconds=cutting_time,
            total_distance_mm=self.total_distance,
            rapid_distance_mm=rapid_dist,
            cutting_distance_mm=cutting_dist,
            avg_cutting_force_n=avg_force,
            max_cutting_force_n=max_force,
            torque_nm=avg_torque,
            power_kw=avg_power,
            tool_wear_percent=tool_wear,
            estimated_tool_life_minutes=(100 / max(tool_wear, 0.01) * cutting_time / 60) if tool_wear > 0 else float('inf'),
            surface_roughness_ra=surface_roughness,
            material_removed_cm3=material_removed / 1000,
            chip_thickness_mm=avg_params.feed_per_tooth if self.simulated_instructions else 0
        )
    
    def get_toolpath_visualization(self) -> Dict[str, Any]:
        """Get toolpath data for visualization"""
        return {
            'instructions': [
                {
                    'path': si.instruction.path,
                    'is_rapid': si.is_rapid,
                    'feed_rate': si.params.feed_rate,
                    'forces': si.estimated_forces
                }
                for si in self.simulated_instructions
            ],
            'total_time': self.total_time,
            'stock_dims': self.kernel.stock_dims.tolist()
        }


def simulate_machining_process(gcode_file: str, 
                               stock_dims: List[float],
                               material: str = "steel",
                               tool_library: Optional[Dict] = None) -> ProcessMetrics:
    """
    Convenience function for quick process simulation.
    
    Usage:
        metrics = simulate_machining_process(
            gcode_file="part.gcode",
            stock_dims=[100, 100, 50],
            material="aluminum"
        )
        print(f"Machining time: {metrics.cutting_time_seconds/60:.1f} minutes")
        print(f"Surface finish: Ra {metrics.surface_roughness_ra:.2f} μm")
    """
    simulator = ProcessSimulator(stock_dims=stock_dims, material=material)
    return simulator.simulate_gcode(gcode_file, tool_library)
