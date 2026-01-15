"""
Communication Systems Adapter
Handles modulation, channel capacity, coding, and BER analysis.
"""

import numpy as np
from typing import Dict, Any

class CommunicationSystemsAdapter:
    """Communication Systems Solver"""
    
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        sim_type = params.get("type", "SHANNON").upper()
        
        if sim_type == "SHANNON":
            return self._solve_shannon(params)
        elif sim_type == "AM":
            return self._solve_am(params)
        elif sim_type == "FM":
            return self._solve_fm(params)
        elif sim_type == "BER":
            return self._solve_ber(params)
        elif sim_type == "HAMMING":
            return self._solve_hamming(params)
        else:
            return {"status": "error", "message": f"Unknown communication type: {sim_type}"}
    
    def _solve_shannon(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Shannon-Hartley Theorem: C = B log₂(1 + SNR)
        """
        B = params.get("bandwidth_hz", 1e6)
        SNR_linear = params.get("snr_linear", 100)
        
        # If SNR given in dB
        if params.get("snr_db") is not None:
            SNR_db = params["snr_db"]
            SNR_linear = 10 ** (SNR_db / 10)
        
        # Channel capacity
        C = B * np.log2(1 + SNR_linear)
        
        # Spectral efficiency
        spectral_efficiency = C / B
        
        return {
            "status": "solved",
            "method": "Shannon-Hartley Theorem",
            "channel_capacity_bps": float(C),
            "spectral_efficiency_bps_hz": float(spectral_efficiency),
            "snr_linear": float(SNR_linear)
        }
    
    def _solve_am(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Amplitude Modulation: s(t) = Ac[1 + m(t)]cos(ωct)
        """
        Ac = params.get("carrier_amplitude_v", 1.0)
        Am = params.get("message_amplitude_v", 0.5)
        
        # Modulation index
        mu = Am / Ac
        
        # Power calculations
        Pc = Ac**2 / 2  # Carrier power
        Psb = (mu**2 * Ac**2) / 4  # Sideband power (both)
        Pt = Pc + Psb  # Total power
        
        # Efficiency
        efficiency = Psb / Pt
        
        # Bandwidth
        fm = params.get("message_frequency_hz", 1000)
        BW = 2 * fm
        
        return {
            "status": "solved",
            "method": "Amplitude Modulation (AM)",
            "modulation_index": float(mu),
            "total_power_w": float(Pt),
            "efficiency": float(efficiency),
            "bandwidth_hz": float(BW)
        }
    
    def _solve_fm(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Frequency Modulation
        """
        fc = params.get("carrier_frequency_hz", 1e6)
        fm = params.get("message_frequency_hz", 1000)
        delta_f = params.get("frequency_deviation_hz", 75000)
        
        # Modulation index
        beta = delta_f / fm
        
        # Carson's rule for bandwidth
        BW = 2 * (delta_f + fm)
        
        return {
            "status": "solved",
            "method": "Frequency Modulation (FM)",
            "modulation_index": float(beta),
            "bandwidth_hz": float(BW),
            "frequency_deviation_hz": delta_f
        }
    
    def _solve_ber(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Bit Error Rate for BPSK
        BER = Q(√(2Eb/N0))
        """
        Eb_N0_db = params.get("eb_n0_db", 10)
        Eb_N0 = 10 ** (Eb_N0_db / 10)
        
        # Q-function approximation
        x = np.sqrt(2 * Eb_N0)
        Q = 0.5 * (1 - np.tanh(x / np.sqrt(2)))
        
        BER = Q
        
        return {
            "status": "solved",
            "method": "Bit Error Rate (BPSK)",
            "ber": float(BER),
            "eb_n0_db": Eb_N0_db
        }
    
    def _solve_hamming(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hamming code parameters
        """
        k = params.get("data_bits", 4)
        
        # Calculate parity bits: 2^r >= k + r + 1
        r = 0
        while 2**r < k + r + 1:
            r += 1
        
        n = k + r  # Total bits
        
        # Code rate
        code_rate = k / n
        
        # Hamming distance (for single error correction)
        d_min = 3
        
        return {
            "status": "solved",
            "method": "Hamming Code",
            "data_bits": k,
            "parity_bits": r,
            "total_bits": n,
            "code_rate": float(code_rate),
            "min_hamming_distance": d_min
        }
