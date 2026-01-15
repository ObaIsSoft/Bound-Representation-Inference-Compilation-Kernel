"""Signal Processing Adapter - FFT, filters, sampling"""
import numpy as np
from typing import Dict, Any

class SignalProcessingAdapter:
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        sim_type = params.get("type", "NYQUIST").upper()
        
        if sim_type == "NYQUIST":
            fmax = params.get("max_frequency_hz", 1000)
            fs_min = 2 * fmax
            return {"status": "solved", "method": "Nyquist Theorem", "min_sampling_rate_hz": float(fs_min)}
        
        elif sim_type == "SNR":
            signal_power = params.get("signal_power_w", 1.0)
            noise_power = params.get("noise_power_w", 0.01)
            SNR = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
            return {"status": "solved", "method": "Signal-to-Noise Ratio", "snr_db": float(SNR)}
        
        elif sim_type == "QUANTIZATION":
            bits = params.get("bits", 8)
            Vref = params.get("reference_voltage_v", 5.0)
            resolution = Vref / (2**bits)
            return {"status": "solved", "method": "Quantization", "resolution_v": float(resolution), "levels": 2**bits}
        
        return {"status": "error", "message": "Unknown signal processing type"}
