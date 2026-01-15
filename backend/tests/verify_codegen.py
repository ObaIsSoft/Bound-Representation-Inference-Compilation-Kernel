import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.codegen_agent import CodegenAgent
from agents.electronics_agent import ElectronicsAgent

def verify_codegen():
    print("--- Verifying Firmware Synthesis (CodegenAgent) ---")
    
    # 1. Simulate Electronics Agent Run
    elec_agent = ElectronicsAgent()
    elec_params = {
        "components": ["motor_2207", "mcu_generic", "imu_bosch"],
        "motor_count": 4
    }
    
    # 2. RUN CODEGEN directly with Mocked Data (simulating Electronics output)
    # We construct the resolved_components list manually to test broad coverage
    mock_resolved = [
        {"id": "m1", "name": "Main Motor", "category": "motor"},
        {"id": "s1", "name": "Aileron Servo", "category": "servo"},
        {"id": "imu", "name": "Bosch IMU", "category": "sensor"}
    ]
    
    codegen = CodegenAgent()
    code_params = {
        "resolved_components": mock_resolved
    }
    code_result = codegen.run(code_params)
    
    # 3. Validation
    source = code_result["firmware_source_code"]
    print("\n--- Generated Generic Firmware ---")
    print(source)
    
    # Check Headers
    assert "#include <Arduino.h>" in source
    assert "#include <Servo.h>" in source # From 'servo' category
    assert "#include <MPU6050.h>" in source # From 'sensor' name matching 'imu'
    
    # Check Setup
    assert "pinMode(PA0, OUTPUT);" in source # Motor
    assert "servo_1.attach(PA1);" in source # Servo (next available generic pin)
    assert "Wire.begin(); imu.initialize();" in source # IMU logic
    
    print("\n✅ PASS: Generic Synthesis Verified")

if __name__ == "__main__":
    verify_codegen()
