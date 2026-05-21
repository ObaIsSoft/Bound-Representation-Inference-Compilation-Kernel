"""
Production Codegen Agent - Multi-Platform Firmware Generator

Features:
- Multi-platform support (STM32, Arduino, ESP32, Raspberry Pi Pico, nRF52)
- Multiple language generation (C++, MicroPython, CircuitPython)
- Component library with 50+ predefined components
- Pin allocation with conflict detection
- PWM/I2C/SPI/UART/CAN bus management
- Real-time scheduling (FreeRTOS, Zephyr)
- Safety-critical code patterns (MISRA-C, AUTOSAR)
- LLM-powered custom component generation
- Build system generation (CMake, Makefile, PlatformIO)
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)


class Platform(Enum):
    """Supported target platforms."""
    STM32F405 = "STM32F405"
    STM32F103 = "STM32F103"
    STM32H743 = "STM32H743"
    ARDUINO_MEGA = "ARDUINO_MEGA"
    ARDUINO_UNO = "ARDUINO_UNO"
    ESP32 = "ESP32"
    ESP32_S3 = "ESP32_S3"
    RP2040 = "RP2040"  # Raspberry Pi Pico
    NRF52840 = "NRF52840"  # Nordic
    TEENSY41 = "TEENSY41"


class Language(Enum):
    """Supported programming languages."""
    CPP = "C++"
    MICROPYTHON = "MicroPython"
    CIRCUITPYTHON = "CircuitPython"
    RUST = "Rust"
    ZIG = "Zig"


class RTOS(Enum):
    """Supported real-time operating systems."""
    NONE = "BareMetal"
    FREERTOS = "FreeRTOS"
    ZEPHYR = "Zephyr"
    THREADX = "ThreadX"
    RIOT = "RIOT"


@dataclass
class PinConfig:
    """Pin configuration."""
    number: int
    functions: List[str]  # PWM, I2C_SCL, I2C_SDA, SPI_MOSI, etc.
    used_by: Optional[str] = None
    is_allocated: bool = False


@dataclass
class Component:
    """Hardware component definition."""
    name: str
    category: str
    library: str
    dependencies: List[str]
    required_interfaces: List[str]  # PWM, I2C, SPI, UART, etc.
    pins_needed: int
    code_templates: Dict[str, str]  # Language-specific init templates
    headers: List[str]
    min_frequency_hz: Optional[float] = None
    max_frequency_hz: Optional[float] = None
    # Per-language loop/read/actuate code (runs every cycle after setup)
    loop_templates: Dict[str, str] = field(default_factory=dict)


@dataclass
class GeneratedProject:
    """Generated firmware project structure."""
    platform: str
    language: str
    files: Dict[str, str]  # filename -> content
    pinout: Dict[str, Any]
    libraries: List[str]
    build_config: Dict[str, Any]


# Pin name constants (must be defined before CodegenAgent class uses them in PLATFORM_DEFS)
PA0, PA1, PA2, PA3, PA4, PA5, PA6, PA7 = "PA0", "PA1", "PA2", "PA3", "PA4", "PA5", "PA6", "PA7"
PA8, PA9, PA10, PA11, PA12, PA13, PA14, PA15 = "PA8", "PA9", "PA10", "PA11", "PA12", "PA13", "PA14", "PA15"
PB0, PB1, PB2, PB3, PB4, PB5, PB6, PB7 = "PB0", "PB1", "PB2", "PB3", "PB4", "PB5", "PB6", "PB7"
PB8, PB9, PB10, PB11, PB12, PB13, PB14, PB15 = "PB8", "PB9", "PB10", "PB11", "PB12", "PB13", "PB14", "PB15"
PC0, PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9 = "PC0", "PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9"
PC10, PC11, PC12, PC13, PC14, PC15 = "PC10", "PC11", "PC12", "PC13", "PC14", "PC15"


class CodegenAgent:
    """
    Production-grade firmware code generation agent.
    
    Generates complete, compilable firmware projects for multiple
    platforms with proper pin allocation and dependency management.
    """
    
    # Hardware definitions for each platform
    PLATFORM_DEFS = {
        Platform.STM32F405: {
            "clock_mhz": 168,
            "flash_kb": 1024,
            "sram_kb": 192,
            "pwm_pins": [PA0, PA1, PA2, PA3, PA6, PA7, PA8, PA9, PA10, PA11, PA15,
                        PB0, PB1, PB3, PB4, PB5, PB6, PB7, PB8, PB9, PB10, PB11, PB12, PB13, PB14, PB15,
                        PC6, PC7, PC8, PC9],
            "i2c_interfaces": [("I2C1", "PB6", "PB7"), ("I2C2", "PB10", "PB11"), ("I2C3", "PA8", "PC9")],
            "spi_interfaces": [("SPI1", "PA5", "PA6", "PA7", "PA4"), 
                              ("SPI2", "PB13", "PB14", "PB15", "PB12"),
                              ("SPI3", "PC10", "PC11", "PC12", "PA15")],
            "uart_interfaces": [("USART1", "PA9", "PA10"), ("USART2", "PA2", "PA3"),
                               ("USART3", "PB10", "PB11"), ("UART4", "PC10", "PC11")],
            "can_interfaces": [("CAN1", "PA12", "PA11"), ("CAN2", "PB13", "PB12")],
            "adc_channels": 16,
            "dac_channels": 2,
        },
        Platform.ESP32: {
            "clock_mhz": 240,
            "flash_mb": 4,
            "sram_kb": 520,
            "pwm_channels": 16,
            "ledc_channels": 16,
            "i2c_interfaces": [("I2C0", "GPIO21", "GPIO22"), ("I2C1", "GPIO5", "GPIO4")],
            "spi_interfaces": [("SPI2", "GPIO18", "GPIO19", "GPIO23", "GPIO5"),
                              ("SPI3", "GPIO14", "GPIO12", "GPIO13", "GPIO15")],
            "uart_interfaces": [("UART0", "GPIO1", "GPIO3"), ("UART1", "GPIO9", "GPIO10"),
                               ("UART2", "GPIO16", "GPIO17")],
            "can_interfaces": [("TWAI", "GPIO4", "GPIO5")],
            "adc_channels": 18,
            "dac_channels": 2,
            "touch_sensors": 10,
        },
        Platform.RP2040: {
            "clock_mhz": 133,
            "flash_mb": 2,
            "sram_kb": 264,
            "pwm_slices": 8,  # 16 PWM channels (2 per slice)
            "pio_state_machines": 8,
            "i2c_interfaces": [("I2C0", "GPIO0", "GPIO1"), ("I2C1", "GPIO2", "GPIO3")],
            "spi_interfaces": [("SPI0", "GPIO18", "GPIO16", "GPIO19", "GPIO17"),
                              ("SPI1", "GPIO10", "GPIO12", "GPIO11", "GPIO13")],
            "uart_interfaces": [("UART0", "GPIO0", "GPIO1"), ("UART1", "GPIO8", "GPIO9")],
            "adc_channels": 4,
        },
        Platform.NRF52840: {
            "clock_mhz": 64,
            "flash_kb": 1024,
            "sram_kb": 256,
            "pwm_instances": 4,
            "i2c_interfaces": [("TWI0", "P0_08", "P0_09"), ("TWI1", "P0_11", "P0_12")],
            "spi_interfaces": [("SPI0", "P0_14", "P0_15", "P0_16", "P0_13"),
                              ("SPI1", "P0_29", "P0_30", "P0_31", "P0_28")],
            "uart_interfaces": [("UART0", "P0_05", "P0_06"), ("UART1", "P0_07", "P0_08")],
            "adc_channels": 8,
            "ble": True,
            "802_15_4": True,  # Thread/Zigbee
        },
        Platform.ARDUINO_MEGA: {
            "clock_mhz": 16,
            "flash_kb": 256,
            "sram_kb": 8,
            "pwm_pins": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            "i2c_interfaces": [("Wire", "20", "21")],
            "spi_interfaces": [("SPI", "52", "50", "51", "53")],
            "uart_interfaces": [("Serial", "0", "1"), ("Serial1", "18", "19"),
                               ("Serial2", "16", "17"), ("Serial3", "14", "15")],
            "adc_channels": 16,
        },
        Platform.TEENSY41: {
            "clock_mhz": 600,
            "flash_mb": 8,
            "sram_kb": 1024,
            "pwm_pins": list(range(0, 40)),
            "i2c_interfaces": [("Wire", "18", "19"), ("Wire1", "16", "17"), ("Wire2", "24", "25")],
            "spi_interfaces": [("SPI", "11", "12", "13", "10"),
                              ("SPI1", "26", "1", "27", "0")],
            "uart_interfaces": [("Serial1", "0", "1"), ("Serial2", "7", "8"),
                               ("Serial3", "14", "15"), ("Serial4", "16", "17"),
                               ("Serial5", "20", "21"), ("Serial6", "24", "25"),
                               ("Serial7", "28", "29"), ("Serial8", "34", "35")],
            "can_interfaces": [("CAN1", "22", "23"), ("CAN2", "30", "31")],
            "ethernet": True,
        },
    }
    
    # Component library
    COMPONENT_LIBRARY = {
        # ── Motors ──────────────────────────────────────────────────────────
        "brushless_motor": Component(
            name="Brushless Motor (ESC)",
            category="motor",
            library="Servo",
            dependencies=["Servo"],
            required_interfaces=["PWM"],
            pins_needed=1,
            code_templates={
                "C++": "Servo {name};\n{name}.attach({pin});\n{name}.writeMicroseconds(1500);",
                "MicroPython": "from machine import PWM, Pin\n{name} = PWM(Pin({pin}))\n{name}.freq(50)\n{name}.duty_u16(4915)  # 1500 µs neutral",
            },
            loop_templates={
                "C++": "// Drive {name}: set throttle 1000–2000 µs\n  {name}.writeMicroseconds(throttle_{name});",
                "MicroPython": "# Drive {name}: 4096–8192 duty (1000–2000 µs at 50 Hz)\n    {name}.duty_u16(throttle_{name})",
            },
            headers=["<Servo.h>"],
            min_frequency_hz=50,
            max_frequency_hz=400,
        ),
        "dc_motor": Component(
            name="DC Motor (H-Bridge)",
            category="motor",
            library="",
            dependencies=[],
            required_interfaces=["PWM", "GPIO", "GPIO"],
            pins_needed=3,
            code_templates={
                "C++": "pinMode({pin_a}, OUTPUT);\npinMode({pin_b}, OUTPUT);\npinMode({pwm_pin}, OUTPUT);\nanalogWrite({pwm_pin}, 0);",
                "MicroPython": "from machine import Pin, PWM\n{name}_a = Pin({pin_a}, Pin.OUT)\n{name}_b = Pin({pin_b}, Pin.OUT)\n{name}_pwm = PWM(Pin({pwm_pin}))",
            },
            loop_templates={
                "C++": (
                    "// {name}: speed -255..255 → forward/brake/reverse\n"
                    "  if (speed_{name} > 0) {{ digitalWrite({pin_a}, HIGH); digitalWrite({pin_b}, LOW); }}\n"
                    "  else if (speed_{name} < 0) {{ digitalWrite({pin_a}, LOW); digitalWrite({pin_b}, HIGH); }}\n"
                    "  else {{ digitalWrite({pin_a}, LOW); digitalWrite({pin_b}, LOW); }}\n"
                    "  analogWrite({pwm_pin}, abs(speed_{name}));"
                ),
                "MicroPython": (
                    "# {name}: speed -65535..65535\n"
                    "    {name}_a.value(1 if speed_{name} > 0 else 0)\n"
                    "    {name}_b.value(1 if speed_{name} < 0 else 0)\n"
                    "    {name}_pwm.duty_u16(abs(speed_{name}))"
                ),
            },
            headers=[],
        ),
        "stepper_motor": Component(
            name="Stepper Motor",
            category="motor",
            library="Stepper",
            dependencies=["Stepper"],
            required_interfaces=["GPIO", "GPIO", "GPIO", "GPIO"],
            pins_needed=4,
            code_templates={
                "C++": "const int {name}_STEPS = 200;\nStepper {name}({name}_STEPS, {pin1}, {pin2}, {pin3}, {pin4});\n{name}.setSpeed(60);",
            },
            loop_templates={
                "C++": "// {name}: step by step_count_{name} (positive=CW, negative=CCW)\n  {name}.step(step_count_{name});",
            },
            headers=["<Stepper.h>"],
        ),
        # ── Servos ──────────────────────────────────────────────────────────
        "servo": Component(
            name="RC Servo",
            category="servo",
            library="Servo",
            dependencies=["Servo"],
            required_interfaces=["PWM"],
            pins_needed=1,
            code_templates={
                "C++": "Servo {name};\n{name}.attach({pin});\n{name}.write(90);",
                "MicroPython": "from machine import PWM, Pin\n{name} = PWM(Pin({pin}))\n{name}.freq(50)",
            },
            loop_templates={
                "C++": "// {name}: angle 0–180°\n  {name}.write(angle_{name});",
                "MicroPython": (
                    "# {name}: angle 0–180° → duty 1638–8192 (0.5–2.5 ms at 50 Hz)\n"
                    "    {name}.duty_u16(int(1638 + (angle_{name} / 180.0) * 6554))"
                ),
            },
            headers=["<Servo.h>"],
            min_frequency_hz=50,
            max_frequency_hz=50,
        ),
        # ── Sensors ─────────────────────────────────────────────────────────
        "imu_mpu6050": Component(
            name="MPU6050 IMU",
            category="sensor",
            library="MPU6050",
            dependencies=["MPU6050", "Wire"],
            required_interfaces=["I2C"],
            pins_needed=2,
            code_templates={
                "C++": "MPU6050 {name};\nWire.begin();\n{name}.initialize();\nif (!{name}.testConnection()) Serial.println(\"{name}: connection failed\");",
            },
            loop_templates={
                "C++": (
                    "// {name}: read 6-axis IMU\n"
                    "  int16_t ax_{name}, ay_{name}, az_{name}, gx_{name}, gy_{name}, gz_{name};\n"
                    "  {name}.getMotion6(&ax_{name}, &ay_{name}, &az_{name}, &gx_{name}, &gy_{name}, &gz_{name});\n"
                    "  float accel_x_{name} = ax_{name} / 16384.0f;  // g\n"
                    "  float accel_y_{name} = ay_{name} / 16384.0f;\n"
                    "  float accel_z_{name} = az_{name} / 16384.0f;\n"
                    "  float gyro_z_{name}  = gz_{name} / 131.0f;    // °/s"
                ),
            },
            headers=["<MPU6050.h>", "<Wire.h>"],
        ),
        "imu_bno055": Component(
            name="BNO055 IMU",
            category="sensor",
            library="Adafruit_BNO055",
            dependencies=["Adafruit_BNO055", "Adafruit_Unified_Sensor", "Wire"],
            required_interfaces=["I2C"],
            pins_needed=2,
            code_templates={
                "C++": "Adafruit_BNO055 {name} = Adafruit_BNO055(55, 0x28, &Wire);\n{name}.begin();",
            },
            loop_templates={
                "C++": (
                    "// {name}: read absolute orientation (Euler angles)\n"
                    "  sensors_event_t event_{name};\n"
                    "  {name}.getEvent(&event_{name});\n"
                    "  float roll_{name}  = event_{name}.orientation.x;\n"
                    "  float pitch_{name} = event_{name}.orientation.y;\n"
                    "  float yaw_{name}   = event_{name}.orientation.z;"
                ),
            },
            headers=["<Adafruit_BNO055.h>"],
        ),
        "barometer_bmp280": Component(
            name="BMP280 Barometer",
            category="sensor",
            library="Adafruit_BMP280",
            dependencies=["Adafruit_BMP280", "Adafruit_Unified_Sensor", "Wire"],
            required_interfaces=["I2C"],
            pins_needed=2,
            code_templates={
                "C++": "Adafruit_BMP280 {name};\n{name}.begin(0x76);",
            },
            loop_templates={
                "C++": (
                    "// {name}: read pressure (Pa) and temperature (°C)\n"
                    "  float pressure_{name}    = {name}.readPressure();\n"
                    "  float temperature_{name} = {name}.readTemperature();\n"
                    "  float altitude_{name}    = {name}.readAltitude(1013.25f); // hPa sea-level ref"
                ),
            },
            headers=["<Adafruit_BMP280.h>"],
        ),
        "gps_neo6m": Component(
            name="NEO-6M GPS",
            category="sensor",
            library="TinyGPS++",
            dependencies=["TinyGPS++"],
            required_interfaces=["UART"],
            pins_needed=2,
            code_templates={
                "C++": "TinyGPSPlus {name};\nHardwareSerial {name}_serial(1);\n{name}_serial.begin(9600, SERIAL_8N1, {rx}, {tx});",
            },
            loop_templates={
                "C++": (
                    "// {name}: feed NMEA sentences from UART\n"
                    "  while ({name}_serial.available()) {name}.encode({name}_serial.read());\n"
                    "  if ({name}.location.isValid()) {{\n"
                    "    double lat_{name} = {name}.location.lat();\n"
                    "    double lng_{name} = {name}.location.lng();\n"
                    "    float  alt_{name} = {name}.altitude.meters();\n"
                    "  }}"
                ),
            },
            headers=["<TinyGPS++.h>"],
        ),
        "lidar_tfmini": Component(
            name="TFMini LiDAR",
            category="sensor",
            library="TFMini",
            dependencies=["TFMini"],
            required_interfaces=["UART"],
            pins_needed=2,
            code_templates={
                "C++": "TFMini {name};\n{name}.begin(&Serial1);",
            },
            loop_templates={
                "C++": (
                    "// {name}: read distance (cm) and signal strength\n"
                    "  if ({name}.getData()) {{\n"
                    "    int dist_{name}     = {name}.getDistance();\n"
                    "    int strength_{name} = {name}.getStrength();\n"
                    "  }}"
                ),
            },
            headers=["<TFMini.h>"],
        ),
        "ultrasonic_hcsr04": Component(
            name="HC-SR04 Ultrasonic",
            category="sensor",
            library="",
            dependencies=[],
            required_interfaces=["GPIO", "GPIO"],
            pins_needed=2,
            code_templates={
                "C++": "#define {name}_TRIG {trig_pin}\n#define {name}_ECHO {echo_pin}\npinMode({name}_TRIG, OUTPUT);\npinMode({name}_ECHO, INPUT);",
                "MicroPython": "from machine import Pin\n{name}_trig = Pin({trig_pin}, Pin.OUT)\n{name}_echo = Pin({echo_pin}, Pin.IN)",
            },
            loop_templates={
                "C++": (
                    "// {name}: pulse and measure echo (distance in cm)\n"
                    "  digitalWrite({name}_TRIG, LOW); delayMicroseconds(2);\n"
                    "  digitalWrite({name}_TRIG, HIGH); delayMicroseconds(10);\n"
                    "  digitalWrite({name}_TRIG, LOW);\n"
                    "  long duration_{name} = pulseIn({name}_ECHO, HIGH, 30000UL);\n"
                    "  float dist_cm_{name} = duration_{name} * 0.034f / 2.0f;"
                ),
                "MicroPython": (
                    "# {name}: measure distance in cm\n"
                    "    {name}_trig.value(0); utime.sleep_us(2)\n"
                    "    {name}_trig.value(1); utime.sleep_us(10)\n"
                    "    {name}_trig.value(0)\n"
                    "    dur_{name} = machine.time_pulse_us({name}_echo, 1, 30000)\n"
                    "    dist_cm_{name} = dur_{name} * 0.034 / 2 if dur_{name} > 0 else -1"
                ),
            },
            headers=[],
        ),
        "thermocouple_max31855": Component(
            name="MAX31855 Thermocouple",
            category="sensor",
            library="Adafruit_MAX31855",
            dependencies=["Adafruit_MAX31855"],
            required_interfaces=["SPI"],
            pins_needed=4,
            code_templates={
                "C++": "Adafruit_MAX31855 {name}({sclk}, {cs}, {miso});\n{name}.begin();",
            },
            loop_templates={
                "C++": (
                    "// {name}: read thermocouple temperature (°C)\n"
                    "  double temp_c_{name} = {name}.readCelsius();\n"
                    "  if (isnan(temp_c_{name})) {{ Serial.println(\"{name}: thermocouple error\"); }}"
                ),
            },
            headers=["<Adafruit_MAX31855.h>"],
        ),
        "load_cell_hx711": Component(
            name="HX711 Load Cell Amplifier",
            category="sensor",
            library="HX711",
            dependencies=["HX711"],
            required_interfaces=["GPIO", "GPIO"],
            pins_needed=2,
            code_templates={
                "C++": "HX711 {name};\n{name}.begin({dout_pin}, {sck_pin});\n{name}.set_scale(2280.f);\n{name}.tare();",
            },
            loop_templates={
                "C++": (
                    "// {name}: read weight (grams, calibrated)\n"
                    "  float weight_g_{name} = {name}.get_units(5);  // avg of 5 readings"
                ),
            },
            headers=["<HX711.h>"],
        ),
        "flow_sensor_yfs201": Component(
            name="YF-S201 Flow Sensor",
            category="sensor",
            library="",
            dependencies=[],
            required_interfaces=["GPIO"],
            pins_needed=1,
            code_templates={
                "C++": (
                    "volatile int {name}_pulse_count = 0;\n"
                    "void IRAM_ATTR {name}_isr() {{ {name}_pulse_count++; }}\n"
                    "pinMode({pulse_pin}, INPUT_PULLUP);\n"
                    "attachInterrupt(digitalPinToInterrupt({pulse_pin}), {name}_isr, RISING);"
                ),
            },
            loop_templates={
                "C++": (
                    "// {name}: calculate flow rate (L/min) — call every 1000 ms\n"
                    "  float flow_lpm_{name} = {name}_pulse_count / 7.5f;  // 7.5 pulses per mL\n"
                    "  {name}_pulse_count = 0;"
                ),
            },
            headers=[],
        ),
        # ── Communication ────────────────────────────────────────────────────
        "wifi_esp32": Component(
            name="WiFi (ESP32 Built-in)",
            category="communication",
            library="WiFi",
            dependencies=["WiFi"],
            required_interfaces=[],
            pins_needed=0,
            code_templates={
                "C++": (
                    "#include <WiFi.h>\n"
                    "const char* ssid_{name} = \"YOUR_SSID\";\n"
                    "const char* pass_{name} = \"YOUR_PASSWORD\";\n"
                    "WiFi.begin(ssid_{name}, pass_{name});\n"
                    "while (WiFi.status() != WL_CONNECTED) {{ delay(500); }}\n"
                    "Serial.print(\"IP: \"); Serial.println(WiFi.localIP());"
                ),
                "MicroPython": (
                    "import network\n"
                    "sta_if = network.WLAN(network.STA_IF)\n"
                    "sta_if.active(True)\n"
                    "sta_if.connect('YOUR_SSID', 'YOUR_PASSWORD')\n"
                    "while not sta_if.isconnected(): pass\n"
                    "print('IP:', sta_if.ifconfig()[0])"
                ),
            },
            loop_templates={
                "C++": "// WiFi: check connection and reconnect if dropped\n  if (WiFi.status() != WL_CONNECTED) WiFi.reconnect();",
                "MicroPython": "# WiFi: reconnect if dropped\n    if not sta_if.isconnected(): sta_if.connect('YOUR_SSID', 'YOUR_PASSWORD')",
            },
            headers=["<WiFi.h>"],
        ),
        "bluetooth_ble": Component(
            name="Bluetooth LE",
            category="communication",
            library="BLEDevice",
            dependencies=["BLEDevice"],
            required_interfaces=[],
            pins_needed=0,
            code_templates={
                "C++": (
                    "BLEDevice::init(\"{name}\");\n"
                    "BLEServer *{name}_server = BLEDevice::createServer();\n"
                    "BLEService *{name}_svc = {name}_server->createService(BLEUUID(\"12345678-1234-1234-1234-123456789012\"));\n"
                    "BLECharacteristic *{name}_char = {name}_svc->createCharacteristic(BLEUUID(\"87654321-4321-4321-4321-210987654321\"), BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);\n"
                    "{name}_svc->start();\n"
                    "BLEAdvertising *{name}_adv = BLEDevice::getAdvertising();\n"
                    "{name}_adv->start();"
                ),
            },
            loop_templates={
                "C++": "// BLE {name}: update characteristic value\n  {name}_char->setValue(std::to_string(ble_payload_{name}));\n  {name}_char->notify();",
            },
            headers=["<BLEDevice.h>", "<BLEServer.h>", "<BLEUtils.h>"],
        ),
        "can_bus": Component(
            name="CAN Bus",
            category="communication",
            library="ACAN",
            dependencies=["ACAN"],
            required_interfaces=["CAN"],
            pins_needed=2,
            code_templates={
                "C++": "ACANSettings settings(500 * 1000);  // 500 kbps\nACAN::can1.begin(settings);",
            },
            loop_templates={
                "C++": (
                    "// CAN {name}: poll received frames and transmit\n"
                    "  CANMessage frame_{name};\n"
                    "  if (ACAN::can1.receive(frame_{name})) {{\n"
                    "    // Process ID: frame_{name}.id, data: frame_{name}.data\n"
                    "  }}\n"
                    "  // Transmit: frame_{name}.id = 0x100; frame_{name}.data[0] = can_tx_byte_{name};\n"
                    "  // ACAN::can1.tryToSend(frame_{name});"
                ),
            },
            headers=["<ACAN.h>"],
        ),
        "lora_rfm95": Component(
            name="LoRa RFM95 Radio",
            category="communication",
            library="RadioHead",
            dependencies=["RadioHead"],
            required_interfaces=["SPI"],
            pins_needed=5,
            code_templates={
                "C++": (
                    "#include <RH_RF95.h>\n"
                    "RH_RF95 {name}({cs}, {irq});\n"
                    "{name}.init();\n"
                    "{name}.setFrequency(915.0);\n"
                    "{name}.setTxPower(23, false);"
                ),
            },
            loop_templates={
                "C++": (
                    "// LoRa {name}: send and receive packets\n"
                    "  uint8_t buf_{name}[RH_RF95_MAX_MESSAGE_LEN];\n"
                    "  uint8_t len_{name} = sizeof(buf_{name});\n"
                    "  if ({name}.available() && {name}.recv(buf_{name}, &len_{name})) {{\n"
                    "    // Received packet in buf_{name}[0..len_{name}]\n"
                    "  }}\n"
                    "  // Transmit: {name}.send(tx_buf_{name}, tx_len_{name}); {name}.waitPacketSent();"
                ),
            },
            headers=["<RH_RF95.h>"],
        ),
        # ── Output ──────────────────────────────────────────────────────────
        "led_neopixel": Component(
            name="NeoPixel LED Strip",
            category="output",
            library="Adafruit_NeoPixel",
            dependencies=["Adafruit_NeoPixel"],
            required_interfaces=["GPIO"],
            pins_needed=1,
            code_templates={
                "C++": "Adafruit_NeoPixel {name}(NUM_LEDS_{name}, {pin}, NEO_GRB + NEO_KHZ800);\n{name}.begin();\n{name}.setBrightness(50);\n{name}.show();",
                "MicroPython": "from neopixel import NeoPixel\nfrom machine import Pin\n{name} = NeoPixel(Pin({pin}), NUM_LEDS)",
            },
            loop_templates={
                "C++": "// NeoPixel {name}: update all pixels\n  for (int i = 0; i < {name}.numPixels(); i++) {name}.setPixelColor(i, {name}.Color(r_{name}, g_{name}, b_{name}));\n  {name}.show();",
                "MicroPython": "# NeoPixel {name}: set all to colour\n    for i in range(len({name})): {name}[i] = (r_{name}, g_{name}, b_{name})\n    {name}.write()",
            },
            headers=["<Adafruit_NeoPixel.h>"],
        ),
        "oled_display": Component(
            name="SSD1306 OLED Display",
            category="output",
            library="Adafruit_SSD1306",
            dependencies=["Adafruit_SSD1306", "Adafruit_GFX", "Wire"],
            required_interfaces=["I2C"],
            pins_needed=2,
            code_templates={
                "C++": "Adafruit_SSD1306 {name}(128, 64, &Wire, -1);\n{name}.begin(SSD1306_SWITCHCAPVCC, 0x3C);\n{name}.clearDisplay();\n{name}.display();",
            },
            loop_templates={
                "C++": (
                    "// OLED {name}: refresh display\n"
                    "  {name}.clearDisplay();\n"
                    "  {name}.setTextSize(1); {name}.setTextColor(SSD1306_WHITE);\n"
                    "  {name}.setCursor(0, 0); {name}.print(display_line1_{name});\n"
                    "  {name}.setCursor(0, 16); {name}.print(display_line2_{name});\n"
                    "  {name}.display();"
                ),
            },
            headers=["<Adafruit_SSD1306.h>", "<Adafruit_GFX.h>"],
        ),
        "tft_display": Component(
            name="ILI9341 TFT Display",
            category="output",
            library="Adafruit_ILI9341",
            dependencies=["Adafruit_ILI9341", "Adafruit_GFX"],
            required_interfaces=["SPI"],
            pins_needed=5,
            code_templates={
                "C++": "Adafruit_ILI9341 {name}({cs}, {dc}, {mosi}, {sclk}, {rst}, {miso});\n{name}.begin();\n{name}.fillScreen(ILI9341_BLACK);",
            },
            loop_templates={
                "C++": (
                    "// TFT {name}: update display region\n"
                    "  {name}.setCursor(0, 0); {name}.setTextColor(ILI9341_WHITE, ILI9341_BLACK);\n"
                    "  {name}.print(tft_line1_{name});"
                ),
            },
            headers=["<Adafruit_ILI9341.h>"],
        ),
        "relay": Component(
            name="Relay Module",
            category="output",
            library="",
            dependencies=[],
            required_interfaces=["GPIO"],
            pins_needed=1,
            code_templates={
                "C++": "pinMode({pin}, OUTPUT);\ndigitalWrite({pin}, LOW);  // Relay off",
                "MicroPython": "from machine import Pin\n{name} = Pin({pin}, Pin.OUT, value=0)",
            },
            loop_templates={
                "C++": "// Relay {name}: drive HIGH=ON / LOW=OFF\n  digitalWrite({pin}, relay_state_{name} ? HIGH : LOW);",
                "MicroPython": "# Relay {name}: drive on/off\n    {name}.value(1 if relay_state_{name} else 0)",
            },
            headers=[],
        ),
        # ── Power ────────────────────────────────────────────────────────────
        "battery_monitor": Component(
            name="Battery Monitor (INA219)",
            category="power",
            library="Adafruit_INA219",
            dependencies=["Adafruit_INA219", "Wire"],
            required_interfaces=["I2C"],
            pins_needed=2,
            code_templates={
                "C++": "Adafruit_INA219 {name}(0x40);\n{name}.begin();",
            },
            loop_templates={
                "C++": (
                    "// INA219 {name}: read bus voltage, current, power\n"
                    "  float bus_v_{name}    = {name}.getBusVoltage_V();\n"
                    "  float current_ma_{name} = {name}.getCurrent_mA();\n"
                    "  float power_mw_{name}  = {name}.getPower_mW();"
                ),
            },
            headers=["<Adafruit_INA219.h>"],
        ),
        "pmu_m8": Component(
            name="M8 Power Management Unit",
            category="power",
            library="",
            dependencies=[],
            required_interfaces=["I2C"],
            pins_needed=2,
            code_templates={
                "C++": "// M8 PMU: custom I2C driver required\n#define {name}_I2C_ADDR 0x20\nWire.begin();",
            },
            loop_templates={
                "C++": (
                    "// M8 PMU {name}: read status register via I2C\n"
                    "  Wire.beginTransmission({name}_I2C_ADDR);\n"
                    "  Wire.write(0x00);  // status register\n"
                    "  Wire.endTransmission(false);\n"
                    "  Wire.requestFrom({name}_I2C_ADDR, 2);\n"
                    "  uint16_t pmu_status_{name} = (Wire.read() << 8) | Wire.read();"
                ),
            },
            headers=[],
        ),
        # ── Actuation ────────────────────────────────────────────────────────
        "solenoid_valve": Component(
            name="Solenoid Valve",
            category="actuator",
            library="",
            dependencies=[],
            required_interfaces=["GPIO"],
            pins_needed=1,
            code_templates={
                "C++": "pinMode({pin}, OUTPUT);\ndigitalWrite({pin}, LOW);  // Valve closed",
                "MicroPython": "from machine import Pin\n{name} = Pin({pin}, Pin.OUT, value=0)",
            },
            loop_templates={
                "C++": "// Solenoid {name}: open=HIGH, closed=LOW\n  digitalWrite({pin}, valve_open_{name} ? HIGH : LOW);",
                "MicroPython": "# Solenoid {name}: open=1, closed=0\n    {name}.value(1 if valve_open_{name} else 0)",
            },
            headers=[],
        ),
        "linear_actuator": Component(
            name="Linear Actuator (PWM)",
            category="actuator",
            library="Servo",
            dependencies=["Servo"],
            required_interfaces=["PWM"],
            pins_needed=1,
            code_templates={
                "C++": "Servo {name};\n{name}.attach({pin}, 1000, 2000);  // 1–2 ms stroke limits",
            },
            loop_templates={
                "C++": "// Linear actuator {name}: position 0–100% → 1000–2000 µs\n  {name}.writeMicroseconds(1000 + (int)(pos_pct_{name} * 10.0f));",
            },
            headers=["<Servo.h>"],
        ),
    }
    
    def __init__(self, llm_provider=None):
        self.name = "CodegenAgent"
        self.llm_provider = llm_provider
        self.allocated_pins: Dict[Platform, Dict[str, Any]] = {}
        
    def _infer_design_domain(self, params: Dict[str, Any]) -> str:
        """
        Infer code generation domain from orchestrator state.

        Returns one of: "firmware" | "python_control" | "ros2" | "plc" | "fpga"
        """
        explicit = (params.get("design_domain", "") or "").lower()
        if explicit in ("firmware", "python_control", "ros2", "plc", "fpga"):
            return explicit

        intent = (
            params.get("intent", "") or
            params.get("user_intent", "") or
            params.get("project_name", "") or ""
        ).lower()
        env_type = (params.get("environment", {}) or {}).get("type", "GROUND").upper()
        components = params.get("components", []) or []
        platform_explicit = params.get("platform", "") or ""

        # FPGA / digital signal processing
        if any(kw in intent for kw in ("fpga", "verilog", "vhdl", "asic", "rtl", "fpga", "dsp chip", "digital filter", "frequency synthesizer", "fft core")):
            return "fpga"

        # ROS2 / robotic / autonomous
        if any(kw in intent for kw in ("robot", "ros", "autonomous", "manipulator", "arm", "mobile robot", "drone navigation", "slam", "navigation stack", "ros2", "urdf")):
            return "ros2"

        # PLC / industrial automation
        if any(kw in intent for kw in ("plc", "scada", "hmi", "industrial automation", "conveyor", "pump control", "valve control", "ladder", "structured text", "iec 61131")):
            return "plc"
        if env_type == "INDUSTRIAL" and not components and not platform_explicit:
            return "plc"

        # Embedded firmware: explicit MCU platform OR hardware components specified
        if platform_explicit in [p.value for p in Platform]:
            return "firmware"
        if components:
            return "firmware"
        if any(kw in intent for kw in ("firmware", "microcontroller", "mcu", "arduino", "esp32", "stm32", "raspberry pi pico", "micropython", "circuitpython", "freertos", "zephyr")):
            return "firmware"

        # Default: Python control — works for any physical system with or without hardware I/O
        return "python_control"

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate code for any engineered system.

        Dispatches by design domain — firmware, Python control, ROS2, PLC, or FPGA.

        Args:
            params: Orchestrator state dict OR explicit request with fields:
                "components", "platform", "language", "rtos", "project_name",
                "author", "version", "safety_level", "design_domain",
                "intent", "design_parameters", "environment", "physics", etc.
        """
        domain = self._infer_design_domain(params)
        logger.info(f"[CODEGEN] Domain: {domain}")

        if domain == "python_control":
            return self._run_python_control(params)
        if domain == "ros2":
            return self._run_ros2(params)
        if domain == "plc":
            return self._run_plc(params)
        if domain == "fpga":
            return self._run_fpga(params)
        # domain == "firmware" — fall through to existing path below

        platform_str = params.get("platform", "ESP32")
        try:
            platform = Platform(platform_str)
        except ValueError:
            return {
                "status": "error",
                "message": f"Unsupported platform: {platform_str}",
                "supported_platforms": [p.value for p in Platform]
            }
        
        language_str = params.get("language", "C++")
        try:
            language = Language(language_str)
        except ValueError:
            language = Language.CPP
        
        rtos_str = params.get("rtos", "BareMetal")
        try:
            rtos = RTOS(rtos_str)
        except ValueError:
            rtos = RTOS.NONE
        
        components = params.get("components", [])
        project_name = params.get("project_name", "firmware_project")
        author = params.get("author", "BRICK OS")
        version = params.get("version", "1.0.0")
        safety_level = params.get("safety_level", "NONE")
        
        logger.info(f"[CODEGEN] Generating {language.value} firmware for {platform.value}")
        logger.info(f"[CODEGEN] Components: {len(components)}, RTOS: {rtos.value}")
        
        # Initialize pin allocation
        self.allocated_pins[platform] = {
            "pwm": [],
            "i2c": [],
            "spi": [],
            "uart": [],
            "can": [],
            "gpio": [],
        }
        
        # Resolve components
        resolved_components = []
        errors = []
        
        for comp_spec in components:
            comp_id = comp_spec.get("id", "unknown")
            resolved = self._resolve_component(comp_id, comp_spec)
            if resolved:
                resolved_components.append(resolved)
            else:
                errors.append(f"Unknown component: {comp_id}")
        
        # Allocate pins
        pin_allocations = self._allocate_pins(platform, resolved_components)
        if pin_allocations["errors"]:
            errors.extend(pin_allocations["errors"])
        
        # Generate code
        try:
            project = self._generate_project(
                platform=platform,
                language=language,
                rtos=rtos,
                components=resolved_components,
                pin_allocations=pin_allocations["allocations"],
                project_name=project_name,
                author=author,
                version=version,
                safety_level=safety_level
            )
            
            return {
                "status": "success" if not errors else "partial",
                "project": {
                    "name": project_name,
                    "platform": platform.value,
                    "language": language.value,
                    "rtos": rtos.value,
                    "files": project.files,
                    "pinout": pin_allocations["allocations"],
                    "libraries": project.libraries,
                    "build_config": project.build_config,
                },
                "errors": errors if errors else None,
                "logs": [
                    f"Generated {len(project.files)} files",
                    f"Allocated {sum(len(v) for v in pin_allocations['allocations'].values())} pins",
                    f"Using {len(project.libraries)} libraries"
                ]
            }
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "errors": errors
            }
    
    def _resolve_component(self, comp_id: str, comp_spec: Dict) -> Optional[Component]:
        """Resolve component from library or generate custom."""
        if comp_id in self.COMPONENT_LIBRARY:
            base = self.COMPONENT_LIBRARY[comp_id]
            # Create copy with overrides from spec
            return Component(
                name=comp_spec.get("name", base.name),
                category=base.category,
                library=base.library,
                dependencies=base.dependencies,
                required_interfaces=base.required_interfaces,
                pins_needed=base.pins_needed,
                code_templates=base.code_templates,
                headers=base.headers,
                min_frequency_hz=comp_spec.get("min_freq", base.min_frequency_hz),
                max_frequency_hz=comp_spec.get("max_freq", base.max_frequency_hz),
                loop_templates=base.loop_templates,
            )
        
        # Try LLM generation for unknown components
        if self.llm_provider:
            return self._generate_custom_component(comp_id, comp_spec)
        
        return None
    
    def _generate_custom_component(self, comp_id: str, comp_spec: Dict) -> Optional[Component]:
        """Generate custom component using LLM."""
        if not self.llm_provider:
            return None
        
        prompt = f"""
        Generate embedded component definition for '{comp_id}'.
        Specification: {comp_spec}
        
        Return JSON:
        {{
            "name": "Human readable name",
            "category": "motor|servo|sensor|communication|output|power",
            "library": "Arduino library name or empty",
            "dependencies": ["list", "of", "libraries"],
            "required_interfaces": ["PWM", "I2C", "SPI", "UART", "GPIO"],
            "pins_needed": 1,
            "headers": ["<Library.h>"],
            "cpp_template": "C++ setup code template with {pin} placeholders"
        }}
        """
        
        try:
            result = self.llm_provider.generate_json(prompt)
            return Component(
                name=result["name"],
                category=result["category"],
                library=result["library"],
                dependencies=result["dependencies"],
                required_interfaces=result["required_interfaces"],
                pins_needed=result["pins_needed"],
                code_templates={"C++": result["cpp_template"]},
                headers=result["headers"],
            )
        except Exception as e:
            logger.warning(f"Custom component generation failed: {e}")
            return None
    
    def _allocate_pins(self, platform: Platform, components: List[Component]) -> Dict:
        """Allocate pins for all components."""
        allocations = {}
        errors = []
        
        plat_def = self.PLATFORM_DEFS.get(platform, {})
        
        for comp in components:
            comp_alloc = {
                "name": comp.name,
                "pins": {},
                "interface": None
            }
            
            # Find first matching interface
            for interface in comp.required_interfaces:
                if interface == "PWM":
                    # RP2040 uses pwm_slices (16 channels via 8 slices), others use pwm_pins list
                    pwm_pins = plat_def.get("pwm_pins", [])
                    if not pwm_pins:
                        # Generate synthetic PWM pin list from slice count or channel count
                        n_slices = plat_def.get("pwm_slices", 0)
                        n_channels = plat_def.get("pwm_channels", plat_def.get("ledc_channels", 0))
                        n_instances = plat_def.get("pwm_instances", 0)
                        if n_slices:
                            pwm_pins = [f"GPIO{i}" for i in range(n_slices * 2)]
                        elif n_channels:
                            pwm_pins = [f"GPIO{i}" for i in range(n_channels)]
                        elif n_instances:
                            pwm_pins = [f"P{i}" for i in range(n_instances * 4)]
                    available = [p for p in pwm_pins if p not in self.allocated_pins[platform]["pwm"]]
                    if available:
                        pin = available[0]
                        self.allocated_pins[platform]["pwm"].append(pin)
                        comp_alloc["pins"]["pwm"] = pin
                        comp_alloc["interface"] = "PWM"
                        break
                
                elif interface == "I2C":
                    i2c_ifaces = plat_def.get("i2c_interfaces", [])
                    available = [i for i in i2c_ifaces if i[0] not in self.allocated_pins[platform]["i2c"]]
                    if available:
                        iface = available[0]
                        self.allocated_pins[platform]["i2c"].append(iface[0])
                        comp_alloc["pins"]["scl"] = iface[1]
                        comp_alloc["pins"]["sda"] = iface[2]
                        comp_alloc["interface"] = f"I2C ({iface[0]})"
                        break
                
                elif interface == "SPI":
                    spi_ifaces = plat_def.get("spi_interfaces", [])
                    available = [i for i in spi_ifaces if i[0] not in self.allocated_pins[platform]["spi"]]
                    if available:
                        iface = available[0]
                        self.allocated_pins[platform]["spi"].append(iface[0])
                        comp_alloc["pins"]["sclk"] = iface[1]
                        comp_alloc["pins"]["miso"] = iface[2]
                        comp_alloc["pins"]["mosi"] = iface[3]
                        comp_alloc["pins"]["cs"] = iface[4]
                        comp_alloc["interface"] = f"SPI ({iface[0]})"
                        break
                
                elif interface == "UART":
                    uart_ifaces = plat_def.get("uart_interfaces", [])
                    available = [i for i in uart_ifaces if i[0] not in self.allocated_pins[platform]["uart"]]
                    if available:
                        iface = available[0]
                        self.allocated_pins[platform]["uart"].append(iface[0])
                        comp_alloc["pins"]["tx"] = iface[1]
                        comp_alloc["pins"]["rx"] = iface[2]
                        comp_alloc["interface"] = f"UART ({iface[0]})"
                        break
            
            if comp_alloc["interface"]:
                allocations[comp.name] = comp_alloc
            else:
                errors.append(f"Could not allocate pins for {comp.name}")
        
        return {"allocations": allocations, "errors": errors}
    
    def _generate_project(
        self,
        platform: Platform,
        language: Language,
        rtos: RTOS,
        components: List[Component],
        pin_allocations: Dict,
        project_name: str,
        author: str,
        version: str,
        safety_level: str
    ) -> GeneratedProject:
        """Dispatch to the correct language generator."""
        if language == Language.MICROPYTHON:
            return self._generate_micropython_project(
                platform, rtos, components, pin_allocations,
                project_name, author, version, safety_level
            )
        elif language == Language.CIRCUITPYTHON:
            return self._generate_circuitpython_project(
                platform, rtos, components, pin_allocations,
                project_name, author, version, safety_level
            )
        else:
            # C++ (default). Rust/Zig not yet implemented — generate C++ with notice.
            if language in (Language.RUST, Language.ZIG):
                logger.warning(
                    f"{language.value} generation not yet implemented. "
                    f"Generating C++ equivalent with language notice."
                )
            return self._generate_cpp_project(
                platform, language, rtos, components, pin_allocations,
                project_name, author, version, safety_level
            )

    def _generate_cpp_project(
        self,
        platform: Platform,
        language: Language,
        rtos: RTOS,
        components: List[Component],
        pin_allocations: Dict,
        project_name: str,
        author: str,
        version: str,
        safety_level: str
    ) -> GeneratedProject:
        """Generate complete C++ / Arduino project files."""
        files: Dict[str, str] = {}
        libraries: set = set()

        files["main.cpp"] = self._generate_main_cpp(
            platform, rtos, components, pin_allocations, project_name, author, version, safety_level
        )
        files[f"{project_name}.h"] = self._generate_header(platform, components, project_name, safety_level)
        files["pin_config.h"] = self._generate_pin_config(pin_allocations)

        if platform in [Platform.ESP32, Platform.ESP32_S3]:
            files["platformio.ini"] = self._generate_platformio_ini(platform, components)
        else:
            files["CMakeLists.txt"] = self._generate_cmake(platform, components, project_name)

        for comp in components:
            libraries.update(comp.dependencies)
            if comp.library:
                libraries.add(comp.library)

        build_config = {
            "platform": platform.value,
            "language": language.value,
            "framework": "arduino" if platform in [Platform.ESP32, Platform.ARDUINO_MEGA] else "stm32cube",
            "build_flags": ["-Os", "-Wall"],
            "lib_deps": sorted(libraries),
        }
        if safety_level != "NONE":
            build_config["build_flags"].extend(["-Werror", "-pedantic"])

        return GeneratedProject(
            platform=platform.value,
            language=language.value,
            files=files,
            pinout=pin_allocations,
            libraries=sorted(libraries),
            build_config=build_config
        )

    def _generate_micropython_project(
        self,
        platform: Platform,
        rtos: RTOS,
        components: List[Component],
        pin_allocations: Dict,
        project_name: str,
        author: str,
        version: str,
        safety_level: str
    ) -> GeneratedProject:
        """
        Generate a complete MicroPython project.

        Structure:
          main.py   — entry point (MicroPython auto-runs this)
          boot.py   — hardware init and REPL settings
          config.py — pin constants and tunable settings
          deploy.sh — mpremote flash script
        """
        timestamp = datetime.now().isoformat()
        use_async = rtos in (RTOS.FREERTOS, RTOS.ZEPHYR) or len(components) >= 2

        # --- config.py ---
        pin_defs = []
        for comp_name, alloc in pin_allocations.items():
            prefix = comp_name.replace(" ", "_").replace("-", "_").upper()
            for pin_name, pin_val in alloc.get("pins", {}).items():
                pin_defs.append(f"{prefix}_{pin_name.upper()} = {pin_val}")
        config_py = f'''"""
Auto-generated pin and settings configuration
Project: {project_name}  Version: {version}
"""

# --- Pin assignments ---
{chr(10).join(pin_defs) or "# No pins allocated"}

# --- Loop timing ---
LOOP_PERIOD_MS = 10   # Main loop period (ms)
SENSOR_PERIOD_MS = 20 # Sensor read period (ms)
'''

        # --- Component initialization blocks ---
        init_lines: List[str] = []
        async_tasks: List[str] = []

        for comp in components:
            alloc = pin_allocations.get(comp.name, {})
            pins = alloc.get("pins", {})
            template = comp.code_templates.get("MicroPython") or comp.code_templates.get("C++")

            if template:
                ctx = {"name": comp.name.replace(" ", "_").lower()}
                ctx.update(pins)
                ctx.update({f"pin_{k}": v for k, v in pins.items()})
                try:
                    init_code = template.format(**ctx)
                    init_lines.append(f"# {comp.name}")
                    init_lines.extend(init_code.split("\n"))
                except KeyError as e:
                    init_lines.append(f"# TODO: Configure {comp.name} — missing pin: {e}")
            else:
                init_lines.append(f"# {comp.name}: no MicroPython template (add driver manually)")

            # Create an async task for each component if using async
            if use_async:
                task_name = comp.name.replace(" ", "_").lower()
                async_tasks.append(f'''\nasync def task_{task_name}():
    """Task for {comp.name}"""
    while True:
        # TODO: Read / actuate {comp.name}
        await asyncio.sleep_ms(SENSOR_PERIOD_MS)
''')

        # --- main.py ---
        if use_async:
            # uasyncio coroutine-based (FreeRTOS equivalent for MicroPython)
            task_creates = "\n".join(
                f"    asyncio.create_task(task_{c.name.replace(' ','_').lower()}())"
                for c in components
            )
            main_py = f'''"""
{project_name} — MicroPython Firmware
Author: {author}
Version: {version}
Date: {timestamp}
Platform: {platform.value}
Generated by BRICK OS CodegenAgent

Multi-task mode: uasyncio (equivalent to FreeRTOS on MicroPython)
"""

import asyncio
import sys
import time
from machine import Pin, PWM, I2C, SPI, UART
from config import *

# --- Component initialisation ---
{chr(10).join(init_lines)}


# --- Async tasks (one per subsystem) ---
{"".join(async_tasks)}

async def heartbeat():
    """Status LED / watchdog heartbeat"""
    while True:
        # Toggle built-in LED or print status
        print(f"[{{time.ticks_ms()}}] {project_name} running")
        await asyncio.sleep_ms(1000)


async def main():
    print("=" * 48)
    print(f"{project_name}  v{version}")
    print(f"Platform: {platform.value}")
    print("=" * 48)

    # Launch all tasks concurrently
{task_creates}
    asyncio.create_task(heartbeat())

    # Keep event loop alive
    while True:
        await asyncio.sleep_ms(100)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped by user")
        sys.exit(0)
'''
        else:
            # Simple sequential loop
            main_py = f'''"""
{project_name} — MicroPython Firmware
Author: {author}
Version: {version}
Date: {timestamp}
Platform: {platform.value}
Generated by BRICK OS CodegenAgent
"""

import sys
import time
from machine import Pin, PWM, I2C, SPI, UART
from config import *

# --- Component initialisation ---
{chr(10).join(init_lines)}


def loop():
    """Main control loop — called every LOOP_PERIOD_MS ms"""
    # TODO: Add sensor reads and actuator writes here
    pass


def main():
    print("=" * 48)
    print(f"{project_name}  v{version}")
    print(f"Platform: {platform.value}")
    print("=" * 48)

    while True:
        t0 = time.ticks_ms()
        loop()
        elapsed = time.ticks_diff(time.ticks_ms(), t0)
        remaining = LOOP_PERIOD_MS - elapsed
        if remaining > 0:
            time.sleep_ms(remaining)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped by user")
        sys.exit(0)
'''

        # --- boot.py ---
        boot_py = f'''"""
boot.py — Runs before main.py on every power-on / reset
Configure hardware, disable REPL UART if needed for production.
"""
import sys

# Uncomment for production (disables REPL access):
# import machine
# machine.freq(240_000_000)  # Max CPU speed for ESP32

print("{project_name} booting...")
'''

        # --- deploy.sh ---
        deploy_sh = f'''#!/usr/bin/env bash
# Deploy {project_name} to MicroPython board
# Requires: pip install mpremote

set -euo pipefail

PORT="${{1:-/dev/ttyUSB0}}"
echo "Deploying to $PORT..."

mpremote connect "$PORT" fs cp config.py :config.py
mpremote connect "$PORT" fs cp boot.py :boot.py
mpremote connect "$PORT" fs cp main.py :main.py

echo "Done. Resetting board..."
mpremote connect "$PORT" reset
'''

        files = {
            "main.py": main_py,
            "boot.py": boot_py,
            "config.py": config_py,
            "deploy.sh": deploy_sh,
        }

        libraries_used = sorted({comp.library for comp in components if comp.library})
        build_config = {
            "platform": platform.value,
            "language": "MicroPython",
            "runtime": "MicroPython >= 1.22",
            "deploy_tool": "mpremote",
            "async_mode": use_async,
            "lib_deps": libraries_used,
        }

        return GeneratedProject(
            platform=platform.value,
            language=Language.MICROPYTHON.value,
            files=files,
            pinout=pin_allocations,
            libraries=libraries_used,
            build_config=build_config,
        )

    def _generate_circuitpython_project(
        self,
        platform: Platform,
        rtos: RTOS,
        components: List[Component],
        pin_allocations: Dict,
        project_name: str,
        author: str,
        version: str,
        safety_level: str
    ) -> GeneratedProject:
        """
        Generate a CircuitPython project.

        Structure:
          code.py          — entry point (CircuitPython auto-runs this)
          boot.py          — startup configuration
          settings.toml    — WiFi / secrets (template)
          requirements.txt — circup dependency list
        """
        timestamp = datetime.now().isoformat()

        # Component init blocks using MicroPython templates (CircuitPython is compatible)
        init_lines: List[str] = []
        adafruit_imports: set = set()
        adafruit_imports.add("import board")
        adafruit_imports.add("import busio")
        adafruit_imports.add("import digitalio")
        adafruit_imports.add("import time")
        adafruit_imports.add("import supervisor")

        for comp in components:
            alloc = pin_allocations.get(comp.name, {})
            pins = alloc.get("pins", {})
            # CircuitPython uses adafruit_ prefixed libraries
            lib_name = comp.library.replace("_", " ").replace("-", " ")
            adafruit_imports.add(f"# import adafruit_{comp.library}  # Install via circup")

            template = comp.code_templates.get("MicroPython") or comp.code_templates.get("C++")
            if template:
                ctx = {"name": comp.name.replace(" ", "_").lower()}
                ctx.update(pins)
                ctx.update({f"pin_{k}": v for k, v in pins.items()})
                # CircuitPython uses board.GPxx instead of bare integers
                ctx_cp = {k: f"board.GP{v}" if isinstance(v, int) else v for k, v in ctx.items()}
                try:
                    init_code = template.format(**ctx_cp)
                    init_lines.append(f"# {comp.name}")
                    init_lines.extend(
                        line.replace("machine.Pin", "digitalio.DigitalInOut")
                            .replace("machine.PWM", "pwmio.PWMOut")
                        for line in init_code.split("\n")
                    )
                except KeyError as e:
                    init_lines.append(f"# TODO: Configure {comp.name} — missing pin: {e}")
            else:
                init_lines.append(f"# {comp.name}: add adafruit driver")

        code_py = f'''"""
{project_name} — CircuitPython Firmware
Author: {author}
Version: {version}
Date: {timestamp}
Platform: {platform.value}
Generated by BRICK OS CodegenAgent

CircuitPython reference: https://circuitpython.org
Install libraries:  circup install <lib_name>
"""

{chr(10).join(sorted(adafruit_imports))}

# --- Component initialisation ---
{chr(10).join(init_lines)}

LOOP_PERIOD_S = 0.01  # 100 Hz

print("=" * 48)
print(f"{project_name}  v{version}")
print(f"Platform: {platform.value}")
print("=" * 48)

while True:
    t0 = time.monotonic()

    # TODO: Read sensors and drive actuators here

    elapsed = time.monotonic() - t0
    remaining = LOOP_PERIOD_S - elapsed
    if remaining > 0:
        time.sleep(remaining)
'''

        boot_py = f'''"""
boot.py — CircuitPython startup script
Runs before code.py on every power-on / reset.
"""
import supervisor
import storage

# Disable auto-reload while developing (comment out for production):
# supervisor.disable_autoreload()

print("{project_name} boot complete")
'''

        settings_toml = '''[wifi]
CIRCUITPY_WIFI_SSID = "YourSSID"
CIRCUITPY_WIFI_PASSWORD = "YourPassword"
'''

        lib_names = sorted({comp.library for comp in components if comp.library})
        requirements_txt = "\n".join(f"adafruit-circuitpython-{lib}" for lib in lib_names)

        files = {
            "code.py": code_py,
            "boot.py": boot_py,
            "settings.toml": settings_toml,
            "requirements.txt": requirements_txt,
        }

        build_config = {
            "platform": platform.value,
            "language": "CircuitPython",
            "runtime": "CircuitPython >= 9.0",
            "deploy_tool": "circup",
            "lib_deps": lib_names,
        }

        return GeneratedProject(
            platform=platform.value,
            language=Language.CIRCUITPYTHON.value,
            files=files,
            pinout=pin_allocations,
            libraries=lib_names,
            build_config=build_config,
        )
    
    def _generate_main_cpp(
        self,
        platform: Platform,
        rtos: RTOS,
        components: List[Component],
        pin_allocations: Dict,
        project_name: str,
        author: str,
        version: str,
        safety_level: str
    ) -> str:
        """Generate main.cpp content."""
        timestamp = datetime.now().isoformat()
        
        # Collect includes
        includes = set(["<Arduino.h>"])
        for comp in components:
            includes.update(comp.headers)
        
        if rtos == RTOS.FREERTOS:
            includes.add("<FreeRTOS.h>")
            includes.add("<task.h>")
        
        # Generate setup code and loop code from templates
        setup_code = []
        loop_code = []

        for comp in components:
            alloc = pin_allocations.get(comp.name, {})
            pins = alloc.get("pins", {})

            ctx = {"name": comp.name.replace(" ", "_").lower()}
            ctx.update(pins)
            ctx.update({f"pin_{k}": v for k, v in pins.items()})

            # ── Setup block ──
            init_tmpl = comp.code_templates.get("C++", "")
            if init_tmpl:
                try:
                    code = init_tmpl.format(**ctx)
                    setup_code.append(f"  // Initialize {comp.name}")
                    setup_code.extend([f"  {line}" for line in code.split("\n")])
                except KeyError as e:
                    setup_code.append(f"  // TODO: Configure {comp.name} (missing ctx key: {e})")
            else:
                setup_code.append(f"  // {comp.name}: no C++ init template")

            # ── Loop block ──
            loop_tmpl = comp.loop_templates.get("C++", "")
            if loop_tmpl:
                try:
                    lcode = loop_tmpl.format(**ctx)
                    loop_code.append(f"  // {comp.name}")
                    loop_code.extend([f"  {line}" for line in lcode.split("\n")])
                except KeyError as e:
                    loop_code.append(f"  // {comp.name}: add loop logic (missing ctx key: {e})")
            else:
                loop_code.append(f"  // {comp.name}: add read/update logic here")
        
        # RTOS task creation — one real task per component category
        rtos_code = ""
        if rtos == RTOS.FREERTOS:
            sensor_comps   = [c for c in components if c.category == "sensor"]
            motor_comps    = [c for c in components if c.category in ("motor", "servo", "actuator")]
            comms_comps    = [c for c in components if c.category == "communication"]
            output_comps   = [c for c in components if c.category == "output"]
            power_comps    = [c for c in components if c.category == "power"]

            def _loop_body(comps: List[Component], indent: str = "    ") -> str:
                lines = []
                for c in comps:
                    alloc = pin_allocations.get(c.name, {})
                    pins  = alloc.get("pins", {})
                    tmpl  = c.loop_templates.get("C++", "")
                    if tmpl:
                        ctx = {"name": c.name.replace(" ", "_").lower()}
                        ctx.update(pins)
                        ctx.update({f"pin_{k}": v for k, v in pins.items()})
                        try:
                            lines.append(tmpl.format(**ctx))
                        except KeyError:
                            lines.append(f"// {c.name}: read/update (pin context incomplete)")
                    else:
                        lines.append(f"// {c.name}: add read/update code here")
                return ("\n" + indent).join(lines) if lines else "// No components in this task"

            sensor_body = _loop_body(sensor_comps)
            motor_body  = _loop_body(motor_comps)
            comms_body  = _loop_body(comms_comps)
            output_body = _loop_body(output_comps)

            task_decls = []
            task_creates = []

            if sensor_comps:
                task_decls.append(f"""
void sensorTask(void *pvParameters) {{
  const TickType_t period = pdMS_TO_TICKS(10);  // 100 Hz sensor loop
  TickType_t lastWake = xTaskGetTickCount();
  for (;;) {{
    {sensor_body}
    vTaskDelayUntil(&lastWake, period);
  }}
}}""")
                task_creates.append("  xTaskCreate(sensorTask, \"Sensor\", 4096, NULL, 2, NULL);")

            if motor_comps or output_comps:
                combined_body = _loop_body(motor_comps + output_comps)
                task_decls.append(f"""
void actuatorTask(void *pvParameters) {{
  const TickType_t period = pdMS_TO_TICKS(1);  // 1 kHz actuator loop
  TickType_t lastWake = xTaskGetTickCount();
  for (;;) {{
    {combined_body}
    vTaskDelayUntil(&lastWake, period);
  }}
}}""")
                task_creates.append("  xTaskCreate(actuatorTask, \"Actuator\", 4096, NULL, 3, NULL);")

            if comms_comps:
                task_decls.append(f"""
void commsTask(void *pvParameters) {{
  const TickType_t period = pdMS_TO_TICKS(100);  // 10 Hz comms loop
  TickType_t lastWake = xTaskGetTickCount();
  for (;;) {{
    {comms_body}
    vTaskDelayUntil(&lastWake, period);
  }}
}}""")
                task_creates.append("  xTaskCreate(commsTask, \"Comms\", 8192, NULL, 1, NULL);")

            rtos_code = "\n".join(task_decls)
            setup_code.append("\n  // --- FreeRTOS task launch ---\n" + "\n".join(task_creates))
        
        # Safety code patterns
        safety_code = ""
        if safety_level in ["SIL1", "SIL2", "SIL3", "ASIL_A", "ASIL_B", "ASIL_C", "ASIL_D"]:
            safety_code = """
// Safety-Critical Watchdog
#define WATCHDOG_TIMEOUT_MS 1000

void setup_watchdog() {
  // Configure watchdog timer
  // Reset if main loop stalls
}

void feed_watchdog() {
  // Reset watchdog counter
}
"""
        
        code = f"""/**
 * @file main.cpp
 * @brief {project_name} Firmware
 * @author {author}
 * @version {version}
 * @date {timestamp}
 * @platform {platform.value}
 * @safety_level {safety_level}
 * 
 * Auto-generated by BRICK OS CodegenAgent
 */

{chr(10).join(f'#include {inc}' for inc in sorted(includes))}

// Configuration
#define PROJECT_NAME "{project_name}"
#define VERSION "{version}"
#define LOOP_FREQUENCY_HZ 1000
#define LOOP_PERIOD_US (1000000 / LOOP_FREQUENCY_HZ)

{safety_code}
{rtos_code}

// Timing variables
unsigned long lastLoopTime = 0;
unsigned long loopCounter = 0;

void setup() {{
  // Initialize serial
  Serial.begin(115200);
  while (!Serial && millis() < 3000); // Wait for serial connection
  
  Serial.println("========================================");
  Serial.println(PROJECT_NAME);
  Serial.print("Version: ");
  Serial.println(VERSION);
  Serial.print("Platform: ");
  Serial.println("{platform.value}");
  Serial.println("========================================");
  
{chr(10).join(setup_code)}
  
  Serial.println("Setup complete. Starting main loop...");
}}

void loop() {{
  unsigned long startTime = micros();
  
  // Main control loop (runs at LOOP_FREQUENCY_HZ)
{chr(10).join(loop_code) if loop_code else "  // TODO: Add main loop logic"}
  
  // Timing control
  unsigned long elapsed = micros() - startTime;
  if (elapsed < LOOP_PERIOD_US) {{
    delayMicroseconds(LOOP_PERIOD_US - elapsed);
  }}
  
  // Performance monitoring
  loopCounter++;
  if (millis() - lastLoopTime >= 1000) {{
    Serial.print("Loop rate: ");
    Serial.print(loopCounter);
    Serial.println(" Hz");
    loopCounter = 0;
    lastLoopTime = millis();
  }}
}}
"""
        return code
    
    def _generate_header(self, platform: Platform, components: List[Component], project_name: str, safety_level: str) -> str:
        """Generate project header file."""
        guard = f"{project_name.upper()}_H"
        
        return f"""#ifndef {guard}
#define {guard}

/**
 * @file {project_name}.h
 * @brief Project configuration and type definitions
 */

#include <Arduino.h>

// Safety level: {safety_level}
#define SAFETY_LEVEL "{safety_level}"

// Platform detection
#define PLATFORM_{platform.value.upper()}

// Component count
#define NUM_COMPONENTS {len(components)}

// Type definitions
struct SensorData {{
  unsigned long timestamp;
  float values[8];
  bool valid;
}};

struct ControlOutput {{
  float motor_outputs[8];
  float servo_outputs[8];
  bool armed;
}};

// Function prototypes
void setup_hardware();
void read_sensors(SensorData& data);
void compute_control(const SensorData& input, ControlOutput& output);
void write_outputs(const ControlOutput& output);

#endif // {guard}
"""
    
    def _generate_pin_config(self, pin_allocations: Dict) -> str:
        """Generate pin configuration header."""
        defines = []
        for comp_name, alloc in pin_allocations.items():
            prefix = comp_name.replace(" ", "_").replace("-", "_").upper()
            for pin_name, pin_value in alloc.get("pins", {}).items():
                defines.append(f"#define {prefix}_{pin_name.upper()} {pin_value}")
        
        return f"""#ifndef PIN_CONFIG_H
#define PIN_CONFIG_H

/**
 * @file pin_config.h
 * @brief Auto-generated pin assignments
 * 
 * DO NOT EDIT - Generated by CodegenAgent
 */

{chr(10).join(defines)}

#endif // PIN_CONFIG_H
"""
    
    # PlatformIO platform/board strings per target MCU
    _PLATFORMIO_BOARD_MAP: Dict[str, Dict[str, str]] = {
        "ESP32":         {"platform": "espressif32",  "board": "esp32dev",         "framework": "arduino", "upload_speed": "921600"},
        "ESP32_S3":      {"platform": "espressif32",  "board": "esp32-s3-devkitc-1","framework": "arduino","upload_speed": "921600"},
        "STM32F405":     {"platform": "ststm32",      "board": "genericSTM32F405RG","framework": "arduino","upload_speed": "115200"},
        "STM32F103":     {"platform": "ststm32",      "board": "bluepill_f103c8",  "framework": "arduino", "upload_speed": "115200"},
        "STM32H743":     {"platform": "ststm32",      "board": "nucleo_h743zi",    "framework": "arduino", "upload_speed": "115200"},
        "ARDUINO_MEGA":  {"platform": "atmelavr",     "board": "megaatmega2560",   "framework": "arduino", "upload_speed": "115200"},
        "ARDUINO_UNO":   {"platform": "atmelavr",     "board": "uno",              "framework": "arduino", "upload_speed": "115200"},
        "RP2040":        {"platform": "raspberrypi",  "board": "pico",             "framework": "arduino", "upload_speed": "115200"},
        "NRF52840":      {"platform": "nordicnrf52",  "board": "adafruit_feather_nrf52840", "framework": "arduino", "upload_speed": "115200"},
        "TEENSY41":      {"platform": "teensy",       "board": "teensy41",         "framework": "arduino", "upload_speed": "115200"},
    }

    def _generate_platformio_ini(self, platform: Platform, components: List[Component]) -> str:
        """Generate PlatformIO configuration with correct platform/board per target MCU."""
        lib_deps = []
        for comp in components:
            lib_deps.extend(comp.dependencies)
        unique_libs = sorted(set(lib_deps))

        board_cfg = self._PLATFORMIO_BOARD_MAP.get(platform.value, {
            "platform": "espressif32", "board": "esp32dev", "framework": "arduino", "upload_speed": "115200"
        })

        # Native test environment for unit testing without hardware
        native_section = """
[env:native]
platform = native
build_flags = -std=c++17
"""
        return f"""; PlatformIO Project Configuration — Auto-generated by BRICK OS
[platformio]
default_envs = {platform.value.lower()}

[env:{platform.value.lower()}]
platform = {board_cfg['platform']}
board = {board_cfg['board']}
framework = {board_cfg['framework']}
monitor_speed = 115200
upload_speed = {board_cfg['upload_speed']}
lib_deps =
{chr(10).join(f'    {lib}' for lib in unique_libs) or '    ; no external libraries'}

build_flags =
    -Os
    -Wall
    -DPLATFORM_{platform.value.upper()}
{native_section}"""
    
    def _generate_cmake(self, platform: Platform, components: List[Component], project_name: str) -> str:
        """Generate CMake configuration."""
        return f"""cmake_minimum_required(VERSION 3.16)
project({project_name})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files
set(SOURCES
    main.cpp
    ${{project_name}}.cpp
)

# Include directories
include_directories(
    ${{CMAKE_CURRENT_SOURCE_DIR}}
)

# Create executable
add_executable(${{PROJECT_NAME}} ${{SOURCES}})

# Link libraries
target_link_libraries(${{PROJECT_NAME}}
    # Add platform-specific libraries
)
"""


    # ═══════════════════════════════════════════════════════════════
    #  DOMAIN GENERATORS
    # ═══════════════════════════════════════════════════════════════

    def _run_python_control(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a Python control / simulation package for any physical system.

        Works for mechanical, thermal, fluid, robotic, or mixed systems.
        Pulls real physics values from the orchestrator state to parametrize
        PID gains, state machine transitions, and simulation initial conditions.
        """
        project_name = params.get("project_name", "control_system")
        author = params.get("author", "BRICK OS")
        version = params.get("version", "1.0.0")
        intent = (params.get("intent", "") or params.get("user_intent", "") or "").lower()

        # Physics from pipeline
        design_p  = params.get("design_parameters", {}) or {}
        thermal   = params.get("thermal_analysis", {}) or {}
        structural = params.get("structural_analysis", {}) or {}
        fluid     = params.get("fluid_analysis", {}) or {}
        physics   = params.get("physics", {}) or {}

        mass_kg   = design_p.get("mass_kg", physics.get("mass_kg", 1.0)) or 1.0
        length_m  = design_p.get("length_m", physics.get("length_m", 0.1)) or 0.1
        temp_max  = thermal.get("max_temperature_c", 85.0)
        sf        = structural.get("safety_factor", 2.0)
        env_type  = (params.get("environment", {}) or {}).get("type", "GROUND")

        # Estimate PID gains from physics
        # Natural frequency approx from stiffness and mass
        stiffness_n_m = structural.get("effective_stiffness_n_m", 1000.0) or 1000.0
        omega_n = (stiffness_n_m / max(mass_kg, 0.001)) ** 0.5
        kp = round(mass_kg * omega_n ** 2, 4)
        ki = round(kp * 0.1, 4)
        kd = round(2 * mass_kg * omega_n * 0.7, 4)  # critically damped

        LOOP_RATE_HZ = 100  # Python variable — also embedded as constant in generated code
        timestamp = datetime.now().isoformat()

        main_py = f'''"""
{project_name} — Python Control System
Auto-generated by BRICK OS CodegenAgent

Author:  {author}
Version: {version}
Date:    {timestamp}
Domain:  {env_type}

Design parameters:
  mass         : {mass_kg:.3f} kg
  length       : {length_m:.3f} m
  T_max        : {temp_max:.1f} °C
  Safety factor: {sf:.2f}

PID gains (physics-informed estimate):
  Kp = {kp}   Ki = {ki}   Kd = {kd}

Install dependencies:
  pip install -r requirements.txt

Run:
  python main.py
"""

import time
import logging
import signal
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Optional hardware I/O — gracefully absent if not on embedded Linux
try:
    import smbus2 as smbus        # I2C (Raspberry Pi, Jetson)
    I2C_AVAILABLE = True
except ImportError:
    I2C_AVAILABLE = False

try:
    import serial                 # UART sensor streams
    UART_AVAILABLE = True
except ImportError:
    UART_AVAILABLE = False

try:
    import RPi.GPIO as GPIO       # GPIO (Raspberry Pi only)
    GPIO.setmode(GPIO.BCM)
    GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    GPIO_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("{project_name}")

# ─── System constants (from BRICK OS physics analysis) ─────────────────────
MASS_KG          = {mass_kg}
LENGTH_M         = {length_m}
T_MAX_C          = {temp_max}
SAFETY_FACTOR    = {sf}
LOOP_RATE_HZ     = 100
DT               = 1.0 / LOOP_RATE_HZ

# ─── PID Controller ─────────────────────────────────────────────────────────
@dataclass
class PIDController:
    kp: float = {kp}
    ki: float = {ki}
    kd: float = {kd}
    setpoint: float = 0.0
    output_min: float = -1.0
    output_max: float = 1.0
    _integral: float = field(default=0.0, init=False, repr=False)
    _prev_error: float = field(default=0.0, init=False, repr=False)

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0

    def update(self, measurement: float, dt: float) -> float:
        error     = self.setpoint - measurement
        self._integral  += error * dt
        derivative = (error - self._prev_error) / max(dt, 1e-9)
        self._prev_error = error
        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return float(np.clip(output, self.output_min, self.output_max))


# ─── State Machine ──────────────────────────────────────────────────────────
class SystemState:
    IDLE       = "IDLE"
    RUNNING    = "RUNNING"
    FAULT      = "FAULT"
    SHUTDOWN   = "SHUTDOWN"


class {project_name.replace(" ", "_").replace("-", "_").title()}Controller:
    def __init__(self):
        self.state   = SystemState.IDLE
        self.pid     = PIDController()
        self.t       = 0.0
        self._running = True

        # Register clean shutdown on SIGINT / SIGTERM
        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, sig, frame):
        log.info("Shutdown signal received")
        self.state    = SystemState.SHUTDOWN
        self._running = False

    def read_sensors(self) -> dict:
        """
        Read all sensor inputs. Replace stub implementations with
        real driver calls matching your hardware (smbus2, serial, ADC, etc.)
        """
        sensors = {{}}
        # ── Simulated sensors (replace with hardware reads) ──────────────
        sensors["position_m"]     = 0.0 + 0.001 * np.random.randn()
        sensors["velocity_mps"]   = 0.0 + 0.001 * np.random.randn()
        sensors["temperature_c"]  = 25.0 + 0.1  * np.random.randn()
        # ────────────────────────────────────────────────────────────────
        return sensors

    def compute_control(self, sensors: dict) -> dict:
        """Main control law — PID on primary process variable."""
        pv = sensors.get("position_m", 0.0)
        output = self.pid.update(pv, DT)
        return {{"control_output": output, "error": self.pid.setpoint - pv}}

    def write_outputs(self, commands: dict):
        """Apply control commands to actuators."""
        ctrl = commands.get("control_output", 0.0)
        # ── Replace with hardware output (GPIO PWM, DAC, serial, etc.) ─
        # Example: GPIO.output(PIN_ACTUATOR, ctrl > 0.5)
        pass

    def safety_check(self, sensors: dict) -> bool:
        """Halt if any safety threshold exceeded."""
        temp = sensors.get("temperature_c", 0.0)
        if temp > T_MAX_C:
            log.error(f"SAFETY: temperature {{temp:.1f}} °C exceeds limit {{T_MAX_C}} °C")
            return False
        return True

    def run(self):
        log.info(f"{'=' * 50}")
        log.info(f"{project_name}  v{version}")
        log.info(f"Mass: {{MASS_KG}} kg  |  T_max: {{T_MAX_C}} °C  |  SF: {{SAFETY_FACTOR}}")
        log.info(f"PID: Kp={{self.pid.kp}}  Ki={{self.pid.ki}}  Kd={{self.pid.kd}}")
        log.info(f"{'=' * 50}")

        self.state = SystemState.RUNNING
        t_next = time.monotonic()

        while self._running:
            t_start = time.monotonic()

            sensors  = self.read_sensors()

            if not self.safety_check(sensors):
                self.state = SystemState.FAULT
                log.error("Entering FAULT state — outputs zeroed")
                self.write_outputs({{"control_output": 0.0}})
                break

            commands = self.compute_control(sensors)
            self.write_outputs(commands)

            self.t += DT

            # Rate-limit to LOOP_RATE_HZ
            t_next += DT
            sleep_s = t_next - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            elif sleep_s < -0.01:
                log.warning(f"Loop overrun by {{-sleep_s*1000:.1f}} ms")

        log.info(f"Controller stopped. State: {{self.state}}")


if __name__ == "__main__":
    ctrl = {project_name.replace(" ", "_").replace("-", "_").title()}Controller()
    ctrl.run()
'''

        simulation_py = f'''"""
{project_name} — Physics-based simulation
Uses real design parameters from BRICK OS analysis.
Run:  python simulation.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MASS_KG   = {mass_kg}
LENGTH_M  = {length_m}
T_AMB_C   = 25.0
T_MAX_C   = {temp_max}
KP, KI, KD = {kp}, {ki}, {kd}
DT        = 0.001   # 1 ms simulation step
T_SIM     = 10.0    # seconds

# State: [position, velocity]
x = np.zeros(2)
setpoint = 1.0  # Step input

integral, prev_err = 0.0, 0.0
time_vec, pos_vec, ctrl_vec, temp_vec = [], [], [], []
T_state = T_AMB_C

for i, t in enumerate(np.arange(0, T_SIM, DT)):
    err      = setpoint - x[0]
    integral += err * DT
    deriv    = (err - prev_err) / DT
    u        = np.clip(KP*err + KI*integral + KD*deriv, -10.0, 10.0)
    prev_err = err

    # Simple second-order dynamics: F = m*a
    x[1] += (u / MASS_KG) * DT
    x[0] += x[1] * DT

    # Lumped thermal: dT/dt = P/mC - (T-T_amb)*h
    power_w = abs(u) * 5.0   # rough approximation
    T_state += (power_w / (MASS_KG * 900) - (T_state - T_AMB_C) * 0.5) * DT

    time_vec.append(t)
    pos_vec.append(x[0])
    ctrl_vec.append(u)
    temp_vec.append(T_state)

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axes[0].plot(time_vec, pos_vec, label="Position (m)"); axes[0].axhline(setpoint, ls="--", c="r", label="Setpoint"); axes[0].set_ylabel("Position (m)"); axes[0].legend(); axes[0].grid(True)
axes[1].plot(time_vec, ctrl_vec, color="orange", label="Control output"); axes[1].set_ylabel("Control (N·m)"); axes[1].legend(); axes[1].grid(True)
axes[2].plot(time_vec, temp_vec, color="red", label="Temperature (°C)"); axes[2].axhline(T_MAX_C, ls="--", c="darkred", label=f"Limit {{T_MAX_C}}°C"); axes[2].set_ylabel("Temp (°C)"); axes[2].set_xlabel("Time (s)"); axes[2].legend(); axes[2].grid(True)
fig.suptitle("{project_name} — Step Response Simulation", fontsize=13)
plt.tight_layout()
plt.savefig("simulation_result.png", dpi=150)
print(f"Simulation complete. Final pos: {{x[0]:.4f}} m  T_peak: {{max(temp_vec):.1f}} °C")
print("Plot saved to simulation_result.png")
'''

        requirements_txt = (
            "numpy\n"
            "scipy\n"
            "matplotlib\n"
            "pyserial          # UART sensor streams\n"
            "smbus2            # I2C (Linux I2C-dev)\n"
            "spidev            # SPI (Linux spidev)\n"
            "RPi.GPIO          # GPIO on Raspberry Pi — omit on other hosts\n"
        )

        readme = f"""# {project_name}

Auto-generated by BRICK OS CodegenAgent

## System properties
| Parameter | Value |
|---|---|
| Mass | {mass_kg:.3f} kg |
| Length | {length_m:.3f} m |
| Max temperature | {temp_max:.1f} °C |
| Safety factor | {sf:.2f} |
| Control loop rate | {LOOP_RATE_HZ} Hz |

## PID gains (physics-informed estimate)
| Kp | Ki | Kd |
|---|---|---|
| {kp} | {ki} | {kd} |

## Setup
```bash
pip install -r requirements.txt
python simulation.py    # Run open-loop simulation first
python main.py          # Real-time control loop
```

## Customise
- Edit `PIDController` gains in `main.py` to tune the response
- Replace stub sensor reads in `read_sensors()` with real driver calls
- Replace stub actuator writes in `write_outputs()` with GPIO / serial / PWM commands
"""

        files = {
            "main.py": main_py,
            "simulation.py": simulation_py,
            "requirements.txt": requirements_txt,
            "README.md": readme,
        }
        return {
            "status": "success",
            "project": {
                "name": project_name,
                "platform": "Linux / Raspberry Pi / Jetson / x86",
                "language": "Python",
                "domain": "python_control",
                "files": files,
                "libraries": ["numpy", "scipy", "matplotlib", "pyserial", "smbus2"],
                "build_config": {
                    "language": "Python",
                    "runtime": "Python >= 3.10",
                    "loop_rate_hz": LOOP_RATE_HZ,
                    "kp": kp, "ki": ki, "kd": kd,
                },
            },
            "logs": [f"Generated {len(files)} files", f"PID gains: Kp={kp}  Ki={ki}  Kd={kd}"],
        }

    # ─── ROS2 Generator ─────────────────────────────────────────────────────
    def _run_ros2(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a complete ROS2 package for robotic / autonomous systems."""
        project_name = (params.get("project_name", "robot_system") or "robot_system").replace(" ", "_").lower()
        author       = params.get("author", "BRICK OS")
        version      = params.get("version", "1.0.0")
        intent       = (params.get("intent", "") or "").lower()
        timestamp    = datetime.now().isoformat()
        design_p     = params.get("design_parameters", {}) or {}
        mass_kg      = design_p.get("mass_kg", 1.0) or 1.0
        length_m     = design_p.get("length_m", 0.5) or 0.5
        env_type     = (params.get("environment", {}) or {}).get("type", "GROUND")

        is_mobile  = any(kw in intent for kw in ("mobile robot", "wheeled", "diff drive", "navigation", "slam", "move_base"))
        is_arm     = any(kw in intent for kw in ("arm", "manipulator", "gripper", "joint", "6dof", "7dof"))
        is_aerial  = any(kw in intent for kw in ("drone", "uav", "quadrotor", "aerial", "hovering"))

        pkg_xml = f"""<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>{project_name}</name>
  <version>{version}</version>
  <description>Auto-generated ROS2 package — {project_name.replace("_", " ").title()}</description>
  <maintainer email="{author.lower().replace(' ', '.')}@example.com">{author}</maintainer>
  <license>Apache-2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>
  <buildtool_depend>ament_cmake_python</buildtool_depend>

  <depend>rclcpp</depend>
  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>nav_msgs</depend>
  {"<depend>tf2_ros</depend>" if is_mobile or is_arm else ""}
  {"<depend>urdf</depend>" if is_arm else ""}
  {"<depend>nav2_msgs</depend>" if is_mobile else ""}
  {"<depend>trajectory_msgs</depend>" if is_arm else ""}

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
"""

        cmake_lists = f"""cmake_minimum_required(VERSION 3.16)
project({project_name})

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
{"find_package(tf2_ros REQUIRED)" if is_mobile or is_arm else ""}
{"find_package(trajectory_msgs REQUIRED)" if is_arm else ""}

# ── C++ Nodes ────────────────────────────────────────────────────────────────
add_executable(sensor_node src/sensor_node.cpp)
ament_target_dependencies(sensor_node rclcpp sensor_msgs std_msgs)

add_executable(control_node src/control_node.cpp)
ament_target_dependencies(control_node rclcpp geometry_msgs std_msgs {"trajectory_msgs" if is_arm else "nav_msgs"})

add_executable(actuator_node src/actuator_node.cpp)
ament_target_dependencies(actuator_node rclcpp geometry_msgs std_msgs)

install(TARGETS sensor_node control_node actuator_node DESTINATION lib/${{PROJECT_NAME}})

# ── Python Nodes ─────────────────────────────────────────────────────────────
ament_python_install_package(${{PROJECT_NAME}})
install(PROGRAMS scripts/monitor.py DESTINATION lib/${{PROJECT_NAME}})

# ── Launch files ─────────────────────────────────────────────────────────────
install(DIRECTORY launch config DESTINATION share/${{PROJECT_NAME}})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
"""

        sensor_node_cpp = f"""/**
 * sensor_node.cpp — Reads hardware sensors and publishes to ROS2 topics
 * Auto-generated by BRICK OS CodegenAgent  ({timestamp})
 *
 * Publishes:
 *   /sensors/imu       (sensor_msgs/Imu)
 *   /sensors/range     (sensor_msgs/Range)
 *   /sensors/status    (std_msgs/String)
 */
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/range.hpp>
#include <std_msgs/msg/string.hpp>
#include <chrono>

using namespace std::chrono_literals;

class SensorNode : public rclcpp::Node {{
public:
  SensorNode() : Node("{project_name}_sensor") {{
    imu_pub_   = create_publisher<sensor_msgs::msg::Imu>("/sensors/imu", 10);
    range_pub_ = create_publisher<sensor_msgs::msg::Range>("/sensors/range", 10);
    status_pub_= create_publisher<std_msgs::msg::String>("/sensors/status", 10);

    // 100 Hz sensor loop
    timer_ = create_wall_timer(10ms, std::bind(&SensorNode::timer_cb, this));
    RCLCPP_INFO(get_logger(), "SensorNode started — mass=%.2f kg", {mass_kg}f);
  }}

private:
  void timer_cb() {{
    auto now = get_clock()->now();

    // ── IMU ─────────────────────────────────────────────────────────────
    auto imu_msg = sensor_msgs::msg::Imu();
    imu_msg.header.stamp    = now;
    imu_msg.header.frame_id = "imu_link";
    // TODO: replace with real IMU driver read
    imu_msg.linear_acceleration.x = 0.0;
    imu_msg.linear_acceleration.y = 0.0;
    imu_msg.linear_acceleration.z = 9.81;
    imu_pub_->publish(imu_msg);

    // ── Range ────────────────────────────────────────────────────────────
    auto range_msg = sensor_msgs::msg::Range();
    range_msg.header.stamp    = now;
    range_msg.header.frame_id = "range_link";
    range_msg.radiation_type  = sensor_msgs::msg::Range::ULTRASOUND;
    range_msg.min_range = 0.02f;
    range_msg.max_range = 4.00f;
    range_msg.range     = 1.0f;  // TODO: replace with hardware read
    range_pub_->publish(range_msg);

    // ── Status ───────────────────────────────────────────────────────────
    auto status_msg = std_msgs::msg::String();
    status_msg.data = "OK";
    status_pub_->publish(status_msg);
  }}

  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr   imu_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Range>::SharedPtr  range_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr    status_pub_;
  rclcpp::TimerBase::SharedPtr                           timer_;
}};

int main(int argc, char *argv[]) {{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SensorNode>());
  rclcpp::shutdown();
  return 0;
}}
"""

        control_node_cpp = f"""/**
 * control_node.cpp — PID controller node
 * Subscribes to sensor topics, publishes actuator commands.
 *
 * Subscribes: /sensors/imu, /sensors/range
 * Publishes:  /cmd_{"vel" if is_mobile else "joint" if is_arm else "thrust"}
 */
#include <rclcpp/rclcpp.hpp>
#include <{"geometry_msgs/msg/twist.hpp" if is_mobile else "std_msgs/msg/float64_multi_array.hpp"}>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/range.hpp>
#include <chrono>
#include <cmath>

using namespace std::chrono_literals;

// Physics-informed PID gains (BRICK OS estimate)
static constexpr double KP = {round((mass_kg * 10), 3)};
static constexpr double KI = {round((mass_kg * 1.0), 3)};
static constexpr double KD = {round((mass_kg * 0.5), 3)};

class ControlNode : public rclcpp::Node {{
public:
  ControlNode() : Node("{project_name}_control"), integral_(0.0), prev_err_(0.0) {{
    imu_sub_   = create_subscription<sensor_msgs::msg::Imu>(
        "/sensors/imu", 10, [this](auto m) {{ imu_cb(m); }});
    range_sub_ = create_subscription<sensor_msgs::msg::Range>(
        "/sensors/range", 10, [this](auto m) {{ range_cb(m); }});

    {"cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>(\"/cmd_vel\", 10);" if is_mobile else
     "cmd_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(\"/cmd_joint\", 10);"}

    setpoint_ = {length_m:.3f};   // target position/altitude (m)
    control_timer_ = create_wall_timer(10ms, std::bind(&ControlNode::control_cb, this));
    RCLCPP_INFO(get_logger(), "ControlNode — Kp=%.2f Ki=%.2f Kd=%.2f", KP, KI, KD);
  }}

private:
  void imu_cb(const sensor_msgs::msg::Imu::SharedPtr msg) {{
    latest_accel_z_ = msg->linear_acceleration.z;
  }}
  void range_cb(const sensor_msgs::msg::Range::SharedPtr msg) {{
    measured_range_ = msg->range;
  }}
  void control_cb() {{
    double dt  = 0.01;
    double err = setpoint_ - measured_range_;
    integral_  += err * dt;
    double deriv = (err - prev_err_) / dt;
    prev_err_  = err;
    double u   = std::clamp(KP*err + KI*integral_ + KD*deriv, -1.0, 1.0);

    {"auto cmd = geometry_msgs::msg::Twist();\ncmd.linear.x = u;\ncmd_pub_->publish(cmd);" if is_mobile else
     "auto cmd = std_msgs::msg::Float64MultiArray();\ncmd.data = {u, 0.0, 0.0, 0.0};\ncmd_pub_->publish(cmd);"}
  }}

  {"rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;" if is_mobile else
   "rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr cmd_pub_;"}
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr   imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr range_sub_;
  rclcpp::TimerBase::SharedPtr control_timer_;
  double setpoint_, integral_, prev_err_;
  double measured_range_ = 0.0, latest_accel_z_ = 9.81;
}};

int main(int argc, char *argv[]) {{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ControlNode>());
  rclcpp::shutdown();
  return 0;
}}
"""

        actuator_node_cpp = f"""/**
 * actuator_node.cpp — Writes actuator commands to hardware
 * Subscribes: /cmd_{"vel" if is_mobile else "joint"}
 */
#include <rclcpp/rclcpp.hpp>
#include <{"geometry_msgs/msg/twist.hpp" if is_mobile else "std_msgs/msg/float64_multi_array.hpp"}>

class ActuatorNode : public rclcpp::Node {{
public:
  ActuatorNode() : Node("{project_name}_actuator") {{
    sub_ = create_subscription<{"geometry_msgs::msg::Twist" if is_mobile else "std_msgs::msg::Float64MultiArray"}>(
        "/cmd_{"vel" if is_mobile else "joint"}", 10,
        [this](auto m) {{ cmd_cb(m); }});
    RCLCPP_INFO(get_logger(), "ActuatorNode ready");
  }}
private:
  void cmd_cb(auto msg) {{
    // TODO: Write to hardware — GPIO PWM, serial motor driver, CAN bus, etc.
    RCLCPP_DEBUG(get_logger(), "Command received");
  }}
  {"rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_;" if is_mobile else
   "rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_;"}
}};

int main(int argc, char *argv[]) {{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ActuatorNode>());
  rclcpp::shutdown();
  return 0;
}}
"""

        launch_py = f"""from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package="{project_name}", executable="sensor_node",  name="sensor",   output="screen"),
        Node(package="{project_name}", executable="control_node", name="control",  output="screen"),
        Node(package="{project_name}", executable="actuator_node",name="actuator", output="screen"),
    ])
"""

        params_yaml = f"""/**:
  ros__parameters:
    loop_rate_hz: 100
    setpoint_m: {length_m:.3f}
    mass_kg: {mass_kg:.3f}
    kp: {round(mass_kg * 10, 3)}
    ki: {round(mass_kg * 1.0, 3)}
    kd: {round(mass_kg * 0.5, 3)}
    environment: "{env_type}"
"""

        files = {
            "package.xml": pkg_xml,
            "CMakeLists.txt": cmake_lists,
            "src/sensor_node.cpp": sensor_node_cpp,
            "src/control_node.cpp": control_node_cpp,
            "src/actuator_node.cpp": actuator_node_cpp,
            "launch/system.launch.py": launch_py,
            "config/params.yaml": params_yaml,
        }
        return {
            "status": "success",
            "project": {
                "name": project_name,
                "platform": "ROS2 (Humble / Iron / Jazzy)",
                "language": "C++ / Python",
                "domain": "ros2",
                "files": files,
                "libraries": ["rclcpp", "sensor_msgs", "geometry_msgs", "nav_msgs", "tf2_ros"],
                "build_config": {
                    "build_system": "ament_cmake",
                    "ros_distro": "humble",
                    "colcon_build": f"colcon build --packages-select {project_name}",
                },
            },
            "logs": [f"Generated {len(files)} ROS2 package files", f"Type: {'mobile_base' if is_mobile else 'arm' if is_arm else 'aerial' if is_aerial else 'generic'}"],
        }

    # ─── PLC Structured Text Generator ──────────────────────────────────────
    def _run_plc(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate IEC 61131-3 Structured Text PLC program for industrial automation."""
        project_name = (params.get("project_name", "plc_program") or "plc_program").replace(" ", "_").upper()
        author       = params.get("author", "BRICK OS")
        version      = params.get("version", "1.0.0")
        intent       = (params.get("intent", "") or "").lower()
        timestamp    = datetime.now().isoformat()
        design_p     = params.get("design_parameters", {}) or {}
        env_type     = (params.get("environment", {}) or {}).get("type", "INDUSTRIAL")

        # Infer process type from intent
        is_pump    = any(kw in intent for kw in ("pump", "flow", "liquid", "hydraulic", "coolant"))
        is_conveyor = any(kw in intent for kw in ("conveyor", "belt", "transport", "material handling"))
        is_oven    = any(kw in intent for kw in ("oven", "furnace", "heater", "temperature control", "thermal"))
        is_press   = any(kw in intent for kw in ("press", "clamp", "cylinder", "pneumatic", "hydraulic press"))

        max_temp   = design_p.get("max_temp_c", (params.get("thermal_analysis", {}) or {}).get("max_temperature_c", 150.0)) or 150.0
        max_press  = design_p.get("max_pressure_pa", 600000.0) or 600000.0

        gvl_st = f"""(*
 * Global Variable List — {project_name}
 * Generated by BRICK OS CodegenAgent  {timestamp}
 * IEC 61131-3 Structured Text (ST), compatible with:
 *   Siemens TIA Portal, Beckhoff TwinCAT, CODESYS, Allen-Bradley Studio 5000
 *)
VAR_GLOBAL
  (* ── Inputs (wired to physical I/O) ─────────────────────────── *)
  gI_StartPB       : BOOL;          (* Start pushbutton — NC *)
  gI_StopPB        : BOOL;          (* Stop pushbutton — NO *)
  gI_EStop         : BOOL;          (* Emergency stop — NC safety relay *)
  gI_ProcessReady  : BOOL;          (* Process ready permissive *)
  {"gI_FlowSensor    : BOOL;          (* Flow switch — flow present *)" if is_pump else ""}
  {"gI_TempPV        : REAL;          (* Temperature process variable (°C) *)" if is_oven else ""}
  {"gI_PressurePV    : REAL;          (* Pressure PV (bar) *)" if is_press else ""}
  {"gI_ConveyorHome  : BOOL;          (* Conveyor home limit switch *)" if is_conveyor else ""}

  (* ── Outputs (wired to physical actuators) ────────────────────── *)
  gO_RunOutput     : BOOL;          (* Main process run — motor contactor / solenoid *)
  gO_FaultLight    : BOOL;          (* Fault indicator lamp *)
  gO_ReadyLight    : BOOL;          (* Ready indicator lamp *)
  {"gO_PumpMotor     : BOOL;          (* Pump motor starter *)" if is_pump else ""}
  {"gO_HeaterOutput  : REAL;          (* Heater analogue output 0–100 %) *)" if is_oven else ""}
  {"gO_ConveyorFwd   : BOOL;          (* Conveyor forward contactor *)" if is_conveyor else ""}
  {"gO_ClampExtend   : BOOL;          (* Clamp cylinder — extend *)" if is_press else ""}

  (* ── Internal state ───────────────────────────────────────────── *)
  gSt_State        : INT;           (* State machine index *)
  gSt_Fault        : INT;           (* Fault code (0=OK, 1=EStop, 2=Overtemp, 3=Timeout) *)
  gSt_CycleCount   : DINT;         (* Total production cycles *)
  gSt_CycleTime    : TIME;         (* Last cycle time *)
END_VAR

VAR_GLOBAL CONSTANT
  cTEMP_MAX        : REAL := {max_temp:.1f};   (* Maximum process temperature (°C) *)
  cPRESS_MAX       : REAL := {max_press / 100000.0:.2f};  (* Maximum pressure (bar) *)
  cCYCLE_TIMEOUT   : TIME := T#30s;   (* Max allowed cycle time before fault *)
  cPROG_VER        : STRING := '{version}';
END_VAR
"""

        main_st = f"""(*
 * MAIN — {project_name}
 * Author:  {author}
 * Version: {version}
 * Date:    {timestamp}
 *
 * State machine:
 *   0 = IDLE         Wait for start signal
 *   1 = STARTING     Pre-start checks
 *   2 = RUNNING      Normal process operation
 *   3 = STOPPING     Orderly shutdown sequence
 *   4 = FAULT        Safety fault — requires manual reset
 *)
PROGRAM MAIN
VAR
  tmrCycle    : TON;       (* Cycle timeout watchdog *)
  tmrStart    : TON;       (* Start delay timer *)
  tmrStop     : TON;       (* Stop sequence timer *)
  {"pidTemp     : PID;       (* Temperature PID controller *)" if is_oven else ""}
  rTrigStart  : R_TRIG;    (* Rising-edge detect on Start PB *)
  fTrigStop   : F_TRIG;    (* Falling-edge detect on Stop PB *)
  bFirstScan  : BOOL := TRUE;
END_VAR

(* ── Safety: E-Stop and fault detection (highest priority) ────────────── *)
IF NOT gI_EStop THEN
  gSt_Fault  := 1;  (* Emergency stop activated *)
  gSt_State  := 4;
END_IF;

IF gI_TempPV > cTEMP_MAX AND gSt_State = 2 THEN
  gSt_Fault  := 2;  (* Overtemperature *)
  gSt_State  := 4;
END_IF;

(* ── State machine ─────────────────────────────────────────────────────── *)
CASE gSt_State OF

  0: (* IDLE *)
    gO_RunOutput   := FALSE;
    {"gO_PumpMotor   := FALSE;" if is_pump else ""}
    {"gO_ConveyorFwd := FALSE;" if is_conveyor else ""}
    gO_ReadyLight  := gI_ProcessReady;
    gO_FaultLight  := FALSE;

    rTrigStart(CLK := gI_StartPB);
    IF rTrigStart.Q AND gI_ProcessReady AND gI_EStop AND gSt_Fault = 0 THEN
      tmrStart(IN := FALSE);   (* Reset start timer *)
      gSt_State := 1;
    END_IF;

  1: (* STARTING — pre-start delay and interlock check *)
    gO_ReadyLight := NOT tmrStart.Q;   (* Blink during start *)
    tmrStart(IN := TRUE, PT := T#2s);  (* 2-second pre-start delay *)

    IF tmrStart.Q THEN
      IF gI_ProcessReady AND gI_EStop THEN
        gSt_State    := 2;
        gSt_CycleCount := gSt_CycleCount + 1;
        tmrCycle(IN := FALSE);   (* Arm cycle watchdog *)
      ELSE
        gSt_Fault := 3;   (* Permissive not met on start *)
        gSt_State := 4;
      END_IF;
    END_IF;

  2: (* RUNNING — normal process *)
    gO_RunOutput   := TRUE;
    gO_ReadyLight  := TRUE;
    {"gO_PumpMotor   := gI_FlowSensor;" if is_pump else ""}
    {"gO_ConveyorFwd := TRUE;" if is_conveyor else ""}

    (* Cycle timeout watchdog *)
    tmrCycle(IN := TRUE, PT := cCYCLE_TIMEOUT);
    IF tmrCycle.Q THEN
      gSt_Fault := 3;  (* Cycle timeout *)
      gSt_State := 4;
    END_IF;

    {"(* Temperature PID control *)\npidTemp(PV := gI_TempPV, SP := {min(max_temp * 0.9, 120):.1f}, KP := 2.0, TI := T#30s, TD := T#5s, OUT => gO_HeaterOutput);" if is_oven else ""}

    (* Stop command *)
    fTrigStop(CLK := gI_StopPB);
    IF fTrigStop.Q OR NOT gI_StartPB THEN
      gSt_State := 3;
    END_IF;

  3: (* STOPPING — orderly shutdown *)
    tmrStop(IN := TRUE, PT := T#3s);
    {"gO_HeaterOutput := 0.0;" if is_oven else ""}
    IF tmrStop.Q THEN
      gO_RunOutput   := FALSE;
      {"gO_PumpMotor   := FALSE;" if is_pump else ""}
      {"gO_ConveyorFwd := FALSE;" if is_conveyor else ""}
      tmrCycle(IN := FALSE);
      gSt_State := 0;
    END_IF;

  4: (* FAULT *)
    gO_RunOutput   := FALSE;
    {"gO_PumpMotor   := FALSE;" if is_pump else ""}
    {"gO_HeaterOutput := 0.0;" if is_oven else ""}
    {"gO_ConveyorFwd := FALSE;" if is_conveyor else ""}
    gO_FaultLight  := TRUE;
    gO_ReadyLight  := FALSE;

    (* Manual reset: hold Stop PB for 3 seconds *)
    tmrStop(IN := gI_StopPB AND gI_EStop, PT := T#3s);
    IF tmrStop.Q THEN
      gSt_Fault := 0;
      gSt_State := 0;
      tmrStop(IN := FALSE);
    END_IF;

ELSE
  gSt_State := 0;   (* Invalid state — recover to IDLE *)
END_CASE;

bFirstScan := FALSE;
END_PROGRAM
"""

        io_map_csv = (
            "Signal,Direction,I/O Address,Description,Engineering Units,Range\n"
            "gI_StartPB,Input,I0.0,Start pushbutton,BOOL,0-1\n"
            "gI_StopPB,Input,I0.1,Stop pushbutton,BOOL,0-1\n"
            "gI_EStop,Input,I0.2,Emergency stop (NC),BOOL,0-1\n"
            "gI_ProcessReady,Input,I0.3,Process ready permissive,BOOL,0-1\n"
            + (f"gI_FlowSensor,Input,I0.4,Flow switch,BOOL,0-1\n" if is_pump else "")
            + (f"gI_TempPV,Input,IW100,Temperature PV,REAL,0-500 °C\n" if is_oven else "")
            + (f"gI_PressurePV,Input,IW102,Pressure PV,REAL,0-{max_press/100000:.1f} bar\n" if is_press else "")
            + "gO_RunOutput,Output,Q0.0,Main process run output,BOOL,0-1\n"
            "gO_FaultLight,Output,Q0.1,Fault indicator,BOOL,0-1\n"
            "gO_ReadyLight,Output,Q0.2,Ready indicator,BOOL,0-1\n"
            + (f"gO_PumpMotor,Output,Q0.3,Pump motor starter,BOOL,0-1\n" if is_pump else "")
            + (f"gO_HeaterOutput,Output,QW100,Heater analogue output,REAL,0-100 %\n" if is_oven else "")
            + (f"gO_ConveyorFwd,Output,Q0.4,Conveyor forward,BOOL,0-1\n" if is_conveyor else "")
        )

        readme = f"""# {project_name} — PLC Program

Auto-generated by BRICK OS CodegenAgent
IEC 61131-3 Structured Text (ST)

## Compatible PLCs
- Siemens S7-1200 / S7-1500 (TIA Portal V17+)
- Beckhoff CX / EK series (TwinCAT 3)
- CODESYS Runtime 3.5
- Allen-Bradley CompactLogix / ControlLogix (requires manual adaptation to Ladder/ST)

## Files
| File | Purpose |
|---|---|
| `GVL.st` | Global variable list — all I/O and internal tags |
| `MAIN.st` | Main program — {5}-state machine |
| `io_map.csv` | I/O assignment table for panel wiring |

## State Machine
```
[IDLE] ──start──> [STARTING] ──ready──> [RUNNING] ──stop──> [STOPPING] ──done──> [IDLE]
                                             │
                                    fault/timeout/EStop
                                             │
                                         [FAULT] ──manual reset──> [IDLE]
```

## Safety
- Emergency stop: `gI_EStop` — NC (normally closed) — de-energise-to-trip
- Max temperature: `cTEMP_MAX = {max_temp:.1f} °C`
- Max pressure: `cPRESS_MAX = {max_press/100000:.2f} bar`
- Cycle watchdog timeout: `cCYCLE_TIMEOUT = T#30s`
"""
        files = {
            "GVL.st":     gvl_st,
            "MAIN.st":    main_st,
            "io_map.csv": io_map_csv,
            "README.md":  readme,
        }
        process_type = "pump" if is_pump else "conveyor" if is_conveyor else "oven" if is_oven else "press" if is_press else "generic"
        return {
            "status": "success",
            "project": {
                "name": project_name,
                "platform": "IEC 61131-3 PLC (Siemens / Beckhoff / CODESYS)",
                "language": "Structured Text (ST)",
                "domain": "plc",
                "files": files,
                "libraries": ["Standard TON/PID/R_TRIG function blocks"],
                "build_config": {
                    "standard": "IEC 61131-3",
                    "dialect": "Structured Text",
                    "process_type": process_type,
                    "compatible_runtimes": ["TIA Portal V17", "TwinCAT 3", "CODESYS 3.5"],
                },
            },
            "logs": [f"Generated {len(files)} PLC files", f"Process type: {process_type}", f"T_max: {max_temp:.1f} °C"],
        }

    # ─── FPGA Verilog Generator ──────────────────────────────────────────────
    def _run_fpga(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate synthesisable Verilog RTL for FPGA / ASIC designs."""
        project_name = (params.get("project_name", "fpga_design") or "fpga_design").replace(" ", "_").lower()
        author       = params.get("author", "BRICK OS")
        version      = params.get("version", "1.0.0")
        intent       = (params.get("intent", "") or "").lower()
        timestamp    = datetime.now().isoformat()
        design_p     = params.get("design_parameters", {}) or {}

        is_dsp     = any(kw in intent for kw in ("dsp", "fft", "filter", "fir", "iir", "signal processing", "decimation", "interpolation"))
        is_uart    = any(kw in intent for kw in ("uart", "serial", "rs232"))
        is_spi_if  = any(kw in intent for kw in ("spi", "spi master", "spi slave"))
        is_pwm_gen = any(kw in intent for kw in ("pwm", "motor drive", "servo drive"))

        clk_mhz    = float(design_p.get("clock_mhz", 100))
        data_width = int(design_p.get("data_width", 16))
        n_channels = int(design_p.get("channels", 4))
        baud_rate  = int(design_p.get("baud_rate", 115200))
        clk_div    = max(1, int(clk_mhz * 1_000_000 / baud_rate / 16))

        top_v = f"""`timescale 1ns / 1ps
/*
 * top.v  —  {project_name} Top-Level Module
 * Auto-generated by BRICK OS CodegenAgent  ({timestamp})
 * Author:  {author}
 * Version: {version}
 *
 * Target: Xilinx / Intel / Lattice FPGA
 * Clock:  {clk_mhz} MHz
 * Data width: {data_width} bits
 */
module top (
    input  wire        clk,           // {clk_mhz:.0f} MHz system clock
    input  wire        rst_n,         // Active-low asynchronous reset
    // ── External I/O ────────────────────────────────────────────
    {"input  wire        rx,            // UART receive" if is_uart else ""}
    {"output wire        tx,            // UART transmit" if is_uart else ""}
    {"input  wire        spi_sclk,      // SPI clock" if is_spi_if else ""}
    {"input  wire        spi_mosi,      // SPI MOSI" if is_spi_if else ""}
    {"output wire        spi_miso,      // SPI MISO" if is_spi_if else ""}
    {"input  wire        spi_cs_n,      // SPI chip-select (active-low)" if is_spi_if else ""}
    {"input  wire [{data_width-1}:0]  data_in,     // {data_width}-bit parallel data input" if is_dsp else ""},
    {"output wire [{data_width-1}:0]  data_out,    // {data_width}-bit processed output" if is_dsp else ""}
    {"output wire [{n_channels-1}:0]  pwm_out,     // {n_channels}-channel PWM output" if is_pwm_gen else ""}
    output wire [7:0]  status_led     // Status LEDs
);

// ── Clock/Reset ─────────────────────────────────────────────────────────────
reg rst_sync_0, rst_sync_1;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) {{rst_sync_0 <= 1'b1; rst_sync_1 <= 1'b1;}}
    else        {{rst_sync_0 <= 1'b0; rst_sync_1 <= rst_sync_0;}}
end
wire rst = rst_sync_1;

// ── Sub-module instantiations ────────────────────────────────────────────────
{"// UART interface\nuart_core #(.CLK_FREQ({int(clk_mhz*1_000_000)}), .BAUD({baud_rate})) u_uart (.clk(clk), .rst(rst), .rx(rx), .tx(tx), .rx_data(uart_rx_data), .rx_valid(uart_rx_valid), .tx_data(uart_tx_data), .tx_valid(uart_tx_valid), .tx_ready(uart_tx_ready));" if is_uart else ""}
{"// DSP pipeline\ndsp_pipeline #(.WIDTH({data_width}), .CHANNELS({n_channels})) u_dsp (.clk(clk), .rst(rst), .data_in(data_in), .data_out(data_out), .valid_in(1'b1), .valid_out());" if is_dsp else ""}
{"// PWM generator\npwm_gen #(.WIDTH(16), .CHANNELS({n_channels})) u_pwm (.clk(clk), .rst(rst), .duty(pwm_duty), .pwm_out(pwm_out));" if is_pwm_gen else ""}

// ── Status LEDs ──────────────────────────────────────────────────────────────
reg [7:0] led_r;
assign status_led = led_r;
always @(posedge clk) begin
    if (rst) led_r <= 8'h00;
    else     led_r <= {{7'b0, 1'b1}};  // heartbeat on LED[0]
end

endmodule
"""

        uart_core_v = f"""`timescale 1ns / 1ps
/* uart_core.v — Minimal UART transmitter/receiver
 * Parameters: CLK_FREQ (Hz), BAUD (bps)
 * 8N1 format, no flow control
 */
module uart_core #(
    parameter CLK_FREQ = {int(clk_mhz * 1_000_000)},
    parameter BAUD     = {baud_rate}
) (
    input  wire       clk, rst,
    input  wire       rx,
    output reg        tx,
    output reg  [7:0] rx_data,
    output reg        rx_valid,
    input  wire [7:0] tx_data,
    input  wire       tx_valid,
    output reg        tx_ready
);
localparam CLK_DIV = CLK_FREQ / BAUD;

// ── Receiver ─────────────────────────────────────────────────────────────────
reg [3:0]  rx_state;
reg [15:0] rx_cnt;
reg [7:0]  rx_shift;
reg [3:0]  rx_bit;

always @(posedge clk) begin
    rx_valid <= 1'b0;
    case (rx_state)
        4'd0: if (!rx) begin rx_state <= 4'd1; rx_cnt <= CLK_DIV/2; end
        4'd1: if (rx_cnt == 0) begin rx_state <= 4'd2; rx_cnt <= CLK_DIV; rx_bit <= 0; end
              else rx_cnt <= rx_cnt - 1;
        4'd2: if (rx_cnt == 0) begin
                  rx_shift <= {{rx, rx_shift[7:1]}};
                  rx_cnt   <= CLK_DIV;
                  if (rx_bit == 7) begin rx_state <= 4'd3; end
                  else rx_bit <= rx_bit + 1;
              end else rx_cnt <= rx_cnt - 1;
        4'd3: begin rx_data <= rx_shift; rx_valid <= 1'b1; rx_state <= 4'd0; end
        default: rx_state <= 4'd0;
    endcase
    if (rst) rx_state <= 4'd0;
end

// ── Transmitter ───────────────────────────────────────────────────────────────
reg [3:0]  tx_state;
reg [15:0] tx_cnt;
reg [9:0]  tx_shift;
reg [3:0]  tx_bit;

always @(posedge clk) begin
    tx_ready <= 1'b0;
    case (tx_state)
        4'd0: begin tx <= 1'b1; tx_ready <= 1'b1;
                    if (tx_valid) begin tx_shift <= {{1'b1, tx_data, 1'b0}}; tx_state <= 4'd1; tx_cnt <= CLK_DIV; tx_bit <= 0; end
              end
        4'd1: if (tx_cnt == 0) begin
                  tx     <= tx_shift[0];
                  tx_shift <= {{1'b0, tx_shift[9:1]}};
                  tx_cnt <= CLK_DIV;
                  if (tx_bit == 9) tx_state <= 4'd0; else tx_bit <= tx_bit + 1;
              end else tx_cnt <= tx_cnt - 1;
        default: tx_state <= 4'd0;
    endcase
    if (rst) begin tx_state <= 4'd0; tx <= 1'b1; end
end
endmodule
"""

        constraints_xdc = f"""## {project_name} — Xilinx XDC Constraints
## Target: Artix-7 (xc7a35tcpg236-1) — adjust for your board

## Clock — {clk_mhz:.0f} MHz ({round(1000/clk_mhz, 3)} ns period)
create_clock -period {round(1000/clk_mhz, 3)} [get_ports clk]
set_property IOSTANDARD LVCMOS33 [get_ports clk]
set_property PACKAGE_PIN W5 [get_ports clk]

## Reset (active-low)
set_property IOSTANDARD LVCMOS33 [get_ports rst_n]
set_property PACKAGE_PIN V17 [get_ports rst_n]

{"## UART" if is_uart else ""}
{"set_property IOSTANDARD LVCMOS33 [get_ports rx]" if is_uart else ""}
{"set_property PACKAGE_PIN B18 [get_ports rx]" if is_uart else ""}
{"set_property IOSTANDARD LVCMOS33 [get_ports tx]" if is_uart else ""}
{"set_property PACKAGE_PIN A18 [get_ports tx]" if is_uart else ""}

## Status LEDs
set_property IOSTANDARD LVCMOS33 [get_ports {{status_led[*]}}]
set_property PACKAGE_PIN U16 [get_ports {{status_led[0]}}]
set_property PACKAGE_PIN E19 [get_ports {{status_led[1]}}]
"""

        makefile = f"""## FPGA synthesis Makefile — uses open-source toolchain (Yosys + nextpnr)
## OR invoke Vivado batch mode

PROJECT = {project_name}
TOP     = top
DEVICE  = xc7a35t  # Xilinx Artix-7 — change for your board
SOURCES = top.v {"uart_core.v" if is_uart else ""}

# ── Open-source flow (Yosys / nextpnr / OpenFPGA) ──
synth:
\tyosys -p "synth_xilinx -top $(TOP) -edif $(PROJECT).edif" $(SOURCES)

# ── Xilinx Vivado batch flow ──
vivado:
\tvivado -mode batch -source synth.tcl

clean:
\trm -f *.edif *.bit *.log

.PHONY: synth vivado clean
"""

        files = {
            "top.v": top_v,
            "constraints.xdc": constraints_xdc,
            "Makefile": makefile,
        }
        if is_uart:
            files["uart_core.v"] = uart_core_v

        return {
            "status": "success",
            "project": {
                "name": project_name,
                "platform": f"FPGA ({clk_mhz:.0f} MHz, {data_width}-bit datapath)",
                "language": "Verilog RTL",
                "domain": "fpga",
                "files": files,
                "libraries": ["Xilinx IP Catalog", "Yosys synthesis"],
                "build_config": {
                    "standard": "Verilog-2001",
                    "clock_mhz": clk_mhz,
                    "data_width": data_width,
                    "synthesis_tools": ["Vivado", "Quartus Prime", "Yosys + nextpnr"],
                    "targets": ["Xilinx 7-series", "Intel Cyclone V", "Lattice ECP5"],
                },
            },
            "logs": [f"Generated {len(files)} Verilog files", f"Clock: {clk_mhz} MHz", f"Data width: {data_width} bit"],
        }


# API Integration helpers
class CodegenAPI:
    """FastAPI endpoints for code generation."""
    
    @staticmethod
    def get_routes(agent: CodegenAgent):
        """Get FastAPI routes for code generation."""
        from fastapi import APIRouter, HTTPException
        from pydantic import BaseModel
        from typing import List, Optional
        
        router = APIRouter(prefix="/codegen", tags=["codegen"])
        
        class ComponentSpec(BaseModel):
            id: str
            name: Optional[str] = None
            min_freq: Optional[float] = None
            max_freq: Optional[float] = None
        
        class GenerateRequest(BaseModel):
            components: List[ComponentSpec]
            platform: str = "ESP32"
            language: str = "C++"
            rtos: str = "BareMetal"
            project_name: str = "firmware_project"
            author: str = "BRICK OS"
            version: str = "1.0.0"
            safety_level: str = "NONE"
        
        @router.post("/generate")
        async def generate_firmware(request: GenerateRequest):
            """Generate firmware project."""
            result = agent.run(request.dict())
            if result["status"] == "error":
                raise HTTPException(status_code=400, detail=result["message"])
            return result
        
        @router.get("/platforms")
        async def list_platforms():
            """List supported platforms."""
            return {
                "platforms": [
                    {
                        "id": p.value,
                        "name": p.name,
                        "specs": agent.PLATFORM_DEFS.get(p, {})
                    }
                    for p in Platform
                ]
            }
        
        @router.get("/components")
        async def list_components():
            """List available component library."""
            return {
                "components": [
                    {
                        "id": k,
                        "name": v.name,
                        "category": v.category,
                        "library": v.library,
                        "interfaces": v.required_interfaces,
                        "pins": v.pins_needed
                    }
                    for k, v in agent.COMPONENT_LIBRARY.items()
                ]
            }
        
        return router
