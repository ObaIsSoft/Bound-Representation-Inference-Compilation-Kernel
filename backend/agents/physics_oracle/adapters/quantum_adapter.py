"""
Quantum Mechanics Adapter
Handles quantum computing, tunneling, and superconductivity.
"""

import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class QuantumAdapter:
    """
    Quantum Mechanics Solver
    Domains: Quantum Computing, Tunneling, Superconductivity
    """
    
    # Physical constants
    H_BAR = 1.054571817e-34  # Reduced Planck constant (J·s)
    K_B = 1.380649e-23  # Boltzmann constant (J/K)
    E_CHARGE = 1.602176634e-19  # Elementary charge (C)
    
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point"""
        sim_type = params.get("type", "GATE").upper()
        
        logger.info(f"[QUANTUM] Solving {sim_type}...")
        
        if sim_type == "GATE":
            return self._solve_gate(params)
        elif sim_type == "TUNNEL":
            return self._solve_tunnel(params)
        elif sim_type == "SUPERCONDUCT":
            return self._solve_superconduct(params)
        else:
            return {"status": "error", "message": f"Unknown quantum type: {sim_type}"}
    
    def _solve_gate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Quantum gate operations"""
        gate_type = params.get("gate", "HADAMARD").upper()
        
        # Define common quantum gates
        gates = {
            "HADAMARD": np.array([[1, 1], [1, -1]]) / np.sqrt(2),
            "PAULI_X": np.array([[0, 1], [1, 0]]),
            "PAULI_Y": np.array([[0, -1j], [1j, 0]]),
            "PAULI_Z": np.array([[1, 0], [0, -1]]),
            "CNOT": np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]])
        }
        
        gate = gates.get(gate_type, gates["HADAMARD"])
        
        return {
            "status": "solved",
            "method": "Quantum Gate",
            "gate_type": gate_type,
            "matrix": gate.tolist(),
            "is_unitary": bool(np.allclose(gate @ gate.conj().T, np.eye(len(gate)))),
            "determinant": float(np.abs(np.linalg.det(gate)))
        }
    
    def _solve_tunnel(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Quantum tunneling probability"""
        # Barrier parameters
        barrier_height = params.get("barrier_ev", 1.0) * self.E_CHARGE  # Convert eV to J
        barrier_width = params.get("width_m", 1e-9)  # 1 nm
        particle_energy = params.get("energy_ev", 0.5) * self.E_CHARGE
        mass = params.get("mass_kg", 9.109e-31)  # Electron mass
        
        # Wave vector inside barrier
        k = np.sqrt(2 * mass * (barrier_height - particle_energy)) / self.H_BAR
        
        # Transmission coefficient (WKB approximation)
        T = np.exp(-2 * k * barrier_width)
        
        return {
            "status": "solved",
            "method": "WKB Approximation",
            "transmission_probability": float(T),
            "reflection_probability": float(1 - T),
            "tunneling_likely": T > 0.01
        }
    
    def _solve_superconduct(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Superconductivity calculations"""
        # BCS theory
        T_c = params.get("critical_temp_k", 9.2)  # Niobium
        temperature = params.get("temperature_k", 4.2)  # Liquid helium
        
        # Energy gap: Δ(T) ≈ Δ(0) * √(1 - T/T_c)
        if temperature < T_c:
            delta_0 = 1.76 * self.K_B * T_c  # BCS gap at T=0
            delta_T = delta_0 * np.sqrt(1 - temperature / T_c)
            is_superconducting = True
        else:
            delta_T = 0
            is_superconducting = False
        
        # Critical magnetic field
        H_c0 = params.get("critical_field_t", 0.2)  # Tesla
        H_c = H_c0 * (1 - (temperature / T_c)**2) if temperature < T_c else 0
        
        return {
            "status": "solved",
            "method": "BCS Theory",
            "is_superconducting": is_superconducting,
            "energy_gap_j": float(delta_T),
            "critical_field_t": float(H_c),
            "coherence_length_m": float(self.H_BAR / np.sqrt(2 * 9.109e-31 * delta_T)) if delta_T > 0 else 0
        }
