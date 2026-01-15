
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CircuitAdapter:
    """
    Lightweight Circuit Simulator.
    Uses Modified Nodal Analysis (MNA) to solve linear circuits (DC Operating Point).
    Equation: [G] [V] = [I]
    """
    
    def __init__(self):
        self.name = "Python-MNA-Solver"
        
    def run_simulation(self, params: dict) -> dict:
        """
        Solve Circuit - supports both simple calculations and full netlists.
        
        Simple mode: {"type": "RESISTOR", "voltage": 5, "resistance": 10}
        Netlist mode: {"components": [{...}]}
        """
        logger.info("[CIRCUIT] Initializing solver...")
        
        # Simple calculation mode (for basic Ohm's law, power, etc.)
        if "type" in params and "components" not in params:
            return self._solve_simple(params)
        
        # Full netlist mode (MNA)
        components = params.get("components", [])
        if not components:
            return {"status": "error", "message": "Empty netlist"}
            
        # 1. Parse Netlist & Build Graph
        # Identify Max Node ID
        max_node = 0
        nodes = set()
        
        # MNA Matrices
        # Dimension = NumNodes (excluding 0/GND) + NumVoltageSources
        # Assume Node 0 is GND.
        
        # Scan for detailed node count and sources
        v_sources = []
        for c in components:
            for n in c["nodes"]:
                nodes.add(n)
                max_node = max(max_node, n)
            if c["type"] == "V":
                v_sources.append(c)
                
        num_vars = max_node + len(v_sources)
        G = np.zeros((num_vars, num_vars))
        I = np.zeros(num_vars)
        
        # Map source current var indices
        # Node vars: 0..max_node-1 (Indices 0 to max_node-1 maps to Nodes 1..max_node)
        # Source vars: max_node..end
        
        source_idx_map = {}
        for i, src in enumerate(v_sources):
            source_idx_map[id(src)] = max_node + i
            
        # 2. Stamp Matrix
        for c in components:
            n1 = c["nodes"][0]
            n2 = c["nodes"][1]
            
            # Helper to access matrix index (Node 0 is filtered out)
            def idx(n): return n - 1
            
            if c["type"] == "R":
                g = 1.0 / float(c["value"])
                # Stamp Diagonals and Off-diagonals
                if n1 > 0:
                    G[idx(n1), idx(n1)] += g
                    if n2 > 0: G[idx(n1), idx(n2)] -= g
                if n2 > 0:
                    G[idx(n2), idx(n2)] += g
                    if n1 > 0: G[idx(n2), idx(n1)] -= g
                    
            elif c["type"] == "V":
                # Voltage Source Stamps (Augmented lines)
                s_idx = source_idx_map[id(c)]
                val = float(c["value"])
                
                # I vector gets value
                I[s_idx] = val
                
                # G matrix gets +/- 1 for constraint equations
                if n1 > 0:
                    G[idx(n1), s_idx] += 1
                    G[s_idx, idx(n1)] += 1
                if n2 > 0:
                    G[idx(n2), s_idx] -= 1
                    G[s_idx, idx(n2)] -= 1
                    
        # 3. Solve Linear System
        try:
            x = np.linalg.solve(G, I)
        except np.linalg.LinAlgError:
            return {"status": "failed", "error": "Singular matrix (Unsolvable circuit)"}
            
        # 4. Format Results
        # x contains [V1, V2, ... Vn, I_source1...]
        results = {}
        for n in range(1, max_node + 1):
            results[f"V_node_{n}"] = x[n-1]
            
        # Extract currents for sources
        # Current is often negative convention output
        for i, src in enumerate(v_sources):
            current = x[max_node + i]
            results[f"I_source_{src.get('id', i)}"] = current
            
        return {
            "status": "solved",
            "method": "MNA (Modified Nodal Analysis)",
            "voltages": results
        }
    
    def _solve_simple(self, params: dict) -> dict:
        """
        Simple circuit calculations (Ohm's law, power, etc.)
        """
        calc_type = params.get("type", "").upper()
        
        if calc_type == "RESISTOR" or calc_type == "OHM":
            # Ohm's Law: V = IR, P = VI = I²R = V²/R
            voltage = params.get("voltage", None)
            current = params.get("current", None)
            resistance = params.get("resistance", None)
            
            # Calculate missing parameter
            if voltage is not None and resistance is not None:
                current = voltage / resistance
                power = voltage * current
            elif current is not None and resistance is not None:
                voltage = current * resistance
                power = current**2 * resistance
            elif voltage is not None and current is not None:
                resistance = voltage / current
                power = voltage * current
            else:
                return {"status": "error", "message": "Need at least 2 of: voltage, current, resistance"}
            
            return {
                "status": "solved",
                "method": "Ohm's Law",
                "voltage_v": float(voltage),
                "current_a": float(current),
                "resistance_ohm": float(resistance),
                "power_w": float(power)
            }
        
        elif calc_type == "SERIES":
            # Series resistors: R_total = R1 + R2 + ...
            resistors = params.get("resistors", [])
            if not resistors:
                return {"status": "error", "message": "No resistors provided"}
            
            R_total = sum(resistors)
            
            # If voltage provided, calculate current
            voltage = params.get("voltage", None)
            if voltage:
                current = voltage / R_total
                power = voltage * current
            else:
                current = None
                power = None
            
            return {
                "status": "solved",
                "method": "Series Resistors",
                "total_resistance_ohm": float(R_total),
                "current_a": float(current) if current else None,
                "power_w": float(power) if power else None
            }
        
        elif calc_type == "PARALLEL":
            # Parallel resistors: 1/R_total = 1/R1 + 1/R2 + ...
            resistors = params.get("resistors", [])
            if not resistors:
                return {"status": "error", "message": "No resistors provided"}
            
            R_total = 1 / sum(1/r for r in resistors)
            
            # If voltage provided, calculate total current
            voltage = params.get("voltage", None)
            if voltage:
                current = voltage / R_total
                power = voltage * current
            else:
                current = None
                power = None
            
            return {
                "status": "solved",
                "method": "Parallel Resistors",
                "total_resistance_ohm": float(R_total),
                "current_a": float(current) if current else None,
                "power_w": float(power) if power else None
            }
        
        else:
            return {"status": "error", "message": f"Unknown calculation type: {calc_type}"}
