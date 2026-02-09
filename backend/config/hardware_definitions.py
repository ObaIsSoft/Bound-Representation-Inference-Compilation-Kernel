"""
Hardware Definitions for BRICK OS.
Centralizes pinout maps and platform constraints.
"""

HARDWARE_DEFS = {
    "STM32F405": {
        "pwm": ["PA0", "PA1", "PA2", "PA3", "PA6", "PA7", "PB0", "PB1", "PC6", "PC7"],
        "i2c": [{"scl": "PB8", "sda": "PB9"}, {"scl": "PB10", "sda": "PB11"}],
        "uart": [{"tx": "PB6", "rx": "PB7"}, {"tx": "PA9", "rx": "PA10"}],
        "spi": [{"sck": "PA5", "miso": "PA6", "mosi": "PA7"}],
        "led": "PC13"
    },
    "ESP32": {
        "pwm": ["13", "12", "14", "27", "26", "25", "33", "32"],
        "i2c": [{"scl": "22", "sda": "21"}],
        "uart": [{"tx": "17", "rx": "16"}],
        "led": "2"
    }
}

DEFAULT_TARGET = "STM32F405"

# Library mappings for code generation (replaces database table)
LIBRARY_MAPPINGS = [
    {
        "category_trigger": "servo",
        "includes": ["<Servo.h>"],
        "globals_template": "Servo {safe_name};",
        "setup_template": "  {safe_name}.attach({pin});",
        "dependencies": ["pwm"]
    },
    {
        "category_trigger": "motor",
        "includes": [],
        "globals_template": "",
        "setup_template": "  pinMode({pin}, OUTPUT);\n  analogWrite({pin}, 0);",
        "dependencies": ["pwm"]
    },
    {
        "category_trigger": "led",
        "includes": [],
        "globals_template": "",
        "setup_template": "  pinMode({pin}, OUTPUT);",
        "dependencies": []
    },
    {
        "category_trigger": "sensor",
        "includes": [],
        "globals_template": "",
        "setup_template": "  // Initialize sensor: {safe_name}",
        "dependencies": []
    },
]
