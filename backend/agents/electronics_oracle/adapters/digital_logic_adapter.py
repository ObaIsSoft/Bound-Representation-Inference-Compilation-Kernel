"""Digital Logic Adapter - Boolean algebra, flip-flops, timing"""
import numpy as np
from typing import Dict, Any

class DigitalLogicAdapter:
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        sim_type = params.get("type", "LOGIC_GATE").upper()
        
        if sim_type == "LOGIC_GATE":
            gate = params.get("gate", "AND").upper()
            A = params.get("input_a", 0)
            B = params.get("input_b", 0)
            
            gates = {
                "AND": A and B,
                "OR": A or B,
                "NOT": not A,
                "NAND": not (A and B),
                "NOR": not (A or B),
                "XOR": A ^ B
            }
            output = gates.get(gate, 0)
            return {"status": "solved", "method": f"{gate} Gate", "output": int(output)}
        
        elif sim_type == "PROPAGATION_DELAY":
            t_pd = params.get("gate_delay_ns", 10)
            num_gates = params.get("num_gates", 5)
            total_delay = t_pd * num_gates
            max_freq = 1 / (total_delay * 1e-9) if total_delay > 0 else float('inf')
            return {"status": "solved", "method": "Propagation Delay", "total_delay_ns": float(total_delay), "max_frequency_hz": float(max_freq)}
        
        return {"status": "error", "message": "Unknown digital type"}
