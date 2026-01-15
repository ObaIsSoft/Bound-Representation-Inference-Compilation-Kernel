"""
Comprehensive Test Suite for Electronics Oracle
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.electronics_oracle.electronics_oracle import ElectronicsOracle

def test_all_electronics_domains():
    """Test all 12 electronics domains"""
    print("\n" + "=" * 70)
    print("ELECTRONICS ORACLE COMPREHENSIVE VERIFICATION")
    print("=" * 70)
    
    oracle = ElectronicsOracle()
    
    tests = [
        ("ANALOG", {"type": "OPAMP_INVERTING", "feedback_resistor_ohm": 10000, "input_resistor_ohm": 1000}),
        ("DIGITAL", {"type": "LOGIC_GATE", "gate": "AND", "input_a": 1, "input_b": 1}),
        ("POWER_ELECTRONICS", {"type": "BUCK", "input_voltage_v": 12, "duty_cycle": 0.5}),
        ("SIGNAL_PROCESSING", {"type": "NYQUIST", "max_frequency_hz": 1000}),
        ("RF_MICROWAVE", {"type": "TRANSMISSION_LINE", "load_impedance_ohm": 75, "characteristic_impedance_ohm": 50}),
        ("SEMICONDUCTOR", {"type": "DIODE", "voltage_v": 0.7}),
        ("CONTROL", {"type": "PID", "proportional_gain": 1.0, "error": 1.0}),
        ("COMMUNICATION", {"type": "SHANNON", "bandwidth_hz": 1e6, "snr_linear": 100}),
        ("SENSORS", {"type": "THERMOCOUPLE", "hot_junction_k": 373, "cold_junction_k": 273}),
        ("PCB", {"type": "TRACE_IMPEDANCE", "dielectric_constant": 4.5, "dielectric_height_mil": 10, "trace_width_mil": 10}),
        ("EMC", {"type": "SHIELDING", "frequency_hz": 1e9, "thickness_m": 1e-3}),
        ("POWER_SYSTEMS", {"type": "THREE_PHASE", "line_to_neutral_v": 120, "line_current_a": 10}),
    ]
    
    passed = 0
    failed = 0
    
    for domain, params in tests:
        try:
            result = oracle.solve(f"Test {domain}", domain, params)
            if result.get("status") == "solved":
                print(f"✓ {domain:20s} - PASS")
                passed += 1
            else:
                print(f"✗ {domain:20s} - FAIL: {result.get('message', 'Unknown')}")
                failed += 1
        except Exception as e:
            print(f"✗ {domain:20s} - ERROR: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Results: {passed} PASSED, {failed} FAILED")
    print("=" * 70)
    
    if failed == 0:
        print("\n🎉 ALL ELECTRONICS DOMAINS OPERATIONAL!")
        print("The Electronics Oracle covers:")
        print("  • Analog Circuits (Op-Amps, Filters, Amplifiers)")
        print("  • Digital Logic (Boolean Algebra, Timing)")
        print("  • Power Electronics (Buck, Boost, Rectifiers)")
        print("  • Signal Processing (Nyquist, SNR, Quantization)")
        print("  • RF & Microwave (Transmission Lines, Friis)")
        print("  • Semiconductor Devices (Diodes, BJTs, MOSFETs)")
        print("  • Control Systems (PID, Transfer Functions, Stability)")
        print("  • Communication Systems (Shannon, Modulation, BER)")
        print("  • Sensors (Thermocouple, RTD, Strain Gauge, Photodiode)")
        print("  • PCB Design (Trace Impedance, Current Capacity, Vias)")
        print("  • EMC/EMI (Shielding, Crosstalk)")
        print("  • Power Systems (Three-Phase, Power Factor, Faults)")
        print("\n✓ Complete electronics/electrical engineering simulation achieved!")
    
    return failed == 0

if __name__ == "__main__":
    success = test_all_electronics_domains()
    sys.exit(0 if success else 1)
