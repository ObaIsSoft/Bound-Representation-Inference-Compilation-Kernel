from typing import Dict, Any, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class CodegenAgent:
    """
    Codegen Agent (Generic).
    Synthesis Layer: Converts high-level 'Electronics' components into low-level C++ Firmware.
    Supports Motors, Servos, Sensors, and LEDs via Category Detection.
    """
    def __init__(self):
        self.name = "CodegenAgent"
        
        # Load Hardware Definitions
        from config.hardware_definitions import HARDWARE_DEFS, DEFAULT_TARGET
        self.hardware_defs = HARDWARE_DEFS
        self.default_target = DEFAULT_TARGET
        
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate Firmware.
        Expected params:
        - resolved_components: List[Dict]
        - target_platform: str (optional, e.g. "STM32F405")
        """
        logger.info(f"{self.name} synthesizing generic firmware...")
        
        # Determine Platform
        target = params.get("target_platform", self.default_target)
        if target not in self.hardware_defs:
            logger.warning(f"Target '{target}' not found. Fallback to {self.default_target}")
            target = self.default_target
            
        hw_config = self.hardware_defs[target]
        available_pwm = list(hw_config["pwm"]) # Copy to consume
        available_i2c = list(hw_config["i2c"])
        
        # Inputs
        components = params.get("resolved_components", [])
        
        # Synthesis State
        includes = set(["<Arduino.h>"])
        globals_code = []
        setup_code = []
        loop_code = []
        
        pin_map = {}
        pwm_idx = 0
        
        # 1. Iterate Components
        # 1. Iterate Components
        import sqlite3
        import json
        conn = sqlite3.connect("data/materials.db")
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        # Pre-fetch all library mappings
        cur.execute("SELECT * FROM library_mappings")
        libs_data = cur.fetchall()
        
        for i, comp in enumerate(components):
            cat = comp.get("category", "unknown").lower()
            name = comp.get("name", f"device_{i}")
            safe_name = name.replace(" ", "_").upper() + f"_{i}"
            
            # Match Logic
            matched_lib = None
            for lib in libs_data:
                trigger = lib["category_trigger"]
                if trigger in cat or trigger in name.lower():
                    matched_lib = lib
                    break
            
            if matched_lib:
                # Dynamic Code Injection from DB
                lib_includes = json.loads(matched_lib["includes_json"])
                includes.update(lib_includes)
                
                # Context formatting
                assigned_pin = "0"
                setup_tmpl = matched_lib["setup_template"] or ""
                deps = dict(matched_lib).get("dependencies_json", "")
                needs_pwm = (deps and "pwm" in deps.lower()) or "{pin}" in setup_tmpl
                 
                if needs_pwm or cat == "motor" or cat == "servo":
                     if pwm_idx < len(available_pwm):
                        assigned_pin = str(available_pwm[pwm_idx])
                        pwm_idx += 1
                        pin_map[safe_name] = assigned_pin
                     else:
                        logger.warning(f"No Pins left for {name}")
                
                ctx = {"safe_name": safe_name, "pin": assigned_pin, "led_count": "16"}
                
                if matched_lib["globals_template"]:
                    globals_code.append(matched_lib["globals_template"].format(**ctx))
                    
                if setup_tmpl:
                    setup_code.append(setup_tmpl.format(**ctx))
                    
            elif cat == "motor":
                 # Generic Motor Fallback
                 if pwm_idx < len(available_pwm):
                    pin = available_pwm[pwm_idx]
                    pwm_idx += 1
                    pin_map[safe_name] = pin
                    setup_code.append(f"  // Motor: {name}")
                    setup_code.append(f"  pinMode({pin}, OUTPUT);")
                    setup_code.append(f"  analogWrite({pin}, 0);")
        
        conn.close()
                    
        # 2. Render
        code = self._render_cpp(includes, globals_code, setup_code)
        
        logs = [
            f"Synthesized firmware for {len(components)} components.",
            f"Used {pwm_idx} PWM channels.",
        ]

        return {
            "status": "success",
            "firmware_source_code": code,
            "pinout_map": pin_map,
            "logs": logs
        }
        
    def _render_cpp(self, includes: set, globals_code: List[str], setup_code: List[str]) -> str:
        timestamp = datetime.now().isoformat()
        
        inc_str = "\n".join([f"#include {inc}" for inc in sorted(list(includes))])
        glob_str = "\n".join(globals_code)
        setup_str = "\n".join(setup_code)
        
        code = f"""/**
 * BRICK OS Autogenerated Firmware (Generic)
 * Timestamp: {timestamp}
 * Target: STM32F405 (Generic)
 */

{inc_str}

// --- Globals ---
{glob_str}

void setup() {{
  Serial.begin(115200);
  Serial.println("BRICK OS: Booting...");
  
{setup_str}

  Serial.println("BRICK OS: Setup Complete.");
}}

void loop() {{
  // Main Logic Loop (1000Hz)
  delay(1);
}}
"""
        return code
