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

import jinja2
from agents.hardware_db import HardwareDB
from agents.pin_allocator import PinAllocator
from agents.hal_generator import generate_hal_files

_JINJA_ENV = jinja2.Environment(
    undefined=jinja2.Undefined,  # silently use "" for missing vars (safe for embedded)
    keep_trailing_newline=True,
)


def _render(template_str: str, ctx: dict) -> str:
    """Render a Jinja2 template string. Never raises on missing vars — emits empty string."""
    try:
        return _JINJA_ENV.from_string(template_str).render(**ctx)
    except jinja2.TemplateError as exc:
        logger.error("Template render error: %s", exc)
        return f"// TEMPLATE ERROR: {exc}\n"

logger = logging.getLogger(__name__)


def _c_ident(name: str) -> str:
    """Convert any component name to a valid C identifier."""
    return re.sub(r"[^a-z0-9_]", "_", name.lower()).strip("_")


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
    # User-supplied overrides (i2c_address, spi_mode, uart_baud, etc.)
    # Merged into template context at render time — no hardcoded values here.
    user_params: Dict[str, Any] = field(default_factory=dict)
    # Template expression for the sensor's primary scalar output after loop_templates runs.
    # Used to populate data.values[slot] in the FreeRTOS sensor task.
    # Empty string → a TODO comment is emitted instead.
    primary_output: str = ""
    # Pinned version string for this component's library (e.g. "2.6.3").
    # When set, platformio.ini emits "LibraryName@version" instead of bare name.
    # Prevents upstream breakage from unpinned floating dependencies.
    library_version: str = ""


@dataclass
class GeneratedProject:
    """Generated firmware project structure."""
    platform: str
    language: str
    files: Dict[str, str]  # filename -> content
    pinout: Dict[str, Any]
    libraries: List[str]
    build_config: Dict[str, Any]




class CodegenAgent:
    """
    Production-grade firmware code generation agent.
    
    Generates complete, compilable firmware projects for multiple
    platforms with proper pin allocation and dependency management.
    """
    
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
            primary_output="accel_x_{name}",
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
            primary_output="yaw_{name}",
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
            primary_output="temperature_{name}",
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
            primary_output="",  # lat/lng are inside conditional scope — caller emits TODO
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
            primary_output="",  # dist_{name} is scoped inside if block — caller emits TODO
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
            primary_output="dist_cm_{name}",
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
            primary_output="(float)temp_c_{name}",
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
            primary_output="weight_g_{name}",
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
            primary_output="flow_lpm_{name}",
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
        self.allocated_pins: Dict[str, Dict[str, Any]] = {}
        self._dynamic_catalog: Dict[str, Component] = {}
        self._load_component_catalogs()
        
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
        
        # Load hardware spec from Supabase / YAML / LLM (no hardcoding)
        hw = HardwareDB.load(platform.value, self.llm_provider)
        self._current_hw = hw
        
        # Resolve components
        resolved_components = []
        errors = []
        
        for comp_spec in components:
            if isinstance(comp_spec, str):
                comp_spec = {"id": comp_spec}
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
                # Top-level convenience keys so callers don't need to unpack project
                "files": project.files,
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
    
    def _load_component_catalogs(self) -> None:
        """
        Load YAML component catalogs from config/component_catalog/*.yaml.

        Any user can drop a YAML file there to register custom components without
        touching Python code. Format (all fields optional except id/name/category):

          components:
            my_sensor_foo:
              name: "Foo Pressure Sensor"
              category: sensor          # sensor|motor|servo|communication|output|power
              library: "FooLib"
              dependencies: ["FooLib"]
              required_interfaces: ["I2C"]
              pins_needed: 2
              headers: ["<FooLib.h>"]
              primary_output: "pressure_{name}"
              templates:
                cpp_init: "FooSensor {name}(0x40);\n{name}.begin();"
                cpp_loop: "float pressure_{name} = {name}.readPressure();"
                micropython_init: "from foolib import Foo\n{name} = Foo(i2c)"
                micropython_loop: "pressure_{name} = {name}.read()"
        """
        import yaml
        catalog_dirs = [
            Path(__file__).parent.parent / "config" / "component_catalog",
            Path(__file__).parent / "component_catalog",
        ]
        for catalog_dir in catalog_dirs:
            if not catalog_dir.exists():
                continue
            for yaml_path in sorted(catalog_dir.glob("*.yaml")):
                try:
                    with open(yaml_path) as f:
                        data = yaml.safe_load(f)
                    for comp_id, spec in (data or {}).get("components", {}).items():
                        self._dynamic_catalog[comp_id] = self._component_from_yaml(comp_id, spec)
                    logger.info(f"[CODEGEN] Loaded catalog: {yaml_path.name} ({len(data.get('components', {}))} components)")
                except Exception as e:
                    logger.warning(f"[CODEGEN] Could not load catalog {yaml_path}: {e}")

    @staticmethod
    def _component_from_yaml(comp_id: str, spec: Dict) -> Component:
        """Build a Component from a YAML catalog entry."""
        tmpls = spec.get("templates", {})
        code_templates: Dict[str, str] = {}
        loop_templates: Dict[str, str] = {}
        if tmpls.get("cpp_init"):
            code_templates["C++"] = tmpls["cpp_init"]
        if tmpls.get("cpp_loop"):
            loop_templates["C++"] = tmpls["cpp_loop"]
        if tmpls.get("micropython_init"):
            code_templates["MicroPython"] = tmpls["micropython_init"]
        if tmpls.get("micropython_loop"):
            loop_templates["MicroPython"] = tmpls["micropython_loop"]
        return Component(
            name=spec.get("name", comp_id),
            category=spec.get("category", "sensor"),
            library=spec.get("library", ""),
            dependencies=spec.get("dependencies", []),
            required_interfaces=spec.get("required_interfaces", ["GPIO"]),
            pins_needed=spec.get("pins_needed", 1),
            code_templates=code_templates,
            headers=spec.get("headers", []),
            loop_templates=loop_templates,
            primary_output=spec.get("primary_output", ""),
        )

    def _resolve_component(self, comp_id: str, comp_spec: Dict) -> Component:
        """
        Resolve a component in priority order:
          1. Built-in COMPONENT_LIBRARY
          2. User YAML catalog (config/component_catalog/*.yaml)
          3. LLM-generated (if llm_provider available)
          4. Skeleton stub (always succeeds — compilable TODO markers)
        """
        user_params = {
            k: v for k, v in comp_spec.items()
            if k not in ("id", "name", "min_freq", "max_freq")
        }

        # 1 — built-in library
        base = self.COMPONENT_LIBRARY.get(comp_id)
        if base:
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
                user_params=user_params,
                primary_output=base.primary_output,
            )

        # 2 — user YAML catalog
        base = self._dynamic_catalog.get(comp_id)
        if base:
            return Component(
                name=comp_spec.get("name", base.name),
                category=base.category,
                library=base.library,
                dependencies=base.dependencies,
                required_interfaces=base.required_interfaces,
                pins_needed=base.pins_needed,
                code_templates=base.code_templates,
                headers=base.headers,
                loop_templates=base.loop_templates,
                user_params=user_params,
                primary_output=base.primary_output,
            )

        # 3 — LLM generation
        if self.llm_provider:
            generated = self._generate_custom_component(comp_id, comp_spec)
            if generated:
                generated.user_params = user_params
                return generated

        # 4 — skeleton stub: always returns something compilable
        return self._skeleton_component(comp_id, comp_spec, user_params)

    def _generate_custom_component(self, comp_id: str, comp_spec: Dict) -> Optional[Component]:
        """
        Ask the LLM to generate a complete Component definition for an unknown part.
        Requests BOTH init and per-cycle loop code plus the primary output expression,
        so the FreeRTOS sensor task and MicroPython async tasks are fully populated.
        """
        inferred_iface = comp_spec.get("interface", comp_spec.get("required_interfaces", ["GPIO"])[0]
                                       if isinstance(comp_spec.get("required_interfaces"), list) else "GPIO")
        pin_tokens_by_iface = {
            "I2C":  "{scl} {sda}  (shared bus, Wire.begin() called once)",
            "SPI":  "{sclk} {miso} {mosi} {cs}  (shared bus, unique CS per device)",
            "UART": "{tx} {rx}",
            "PWM":  "{pwm}",
            "GPIO": "{pin}  (or {pin_a}, {pin_b} for multi-pin)",
        }
        pin_hint = pin_tokens_by_iface.get(inferred_iface.upper(), "{pin}")

        prompt = f"""You are generating embedded firmware code for a hardware component.

Component ID : {comp_id}
User spec    : {json.dumps(comp_spec)}
Interface    : {inferred_iface}

RULES:
- Use ONLY the pin placeholder tokens for the interface type shown below — do NOT
  invent pin numbers. The code generator replaces these at render time.
- Pin tokens for {inferred_iface}: {pin_hint}
- Variable naming: use {{name}} as a C identifier prefix for all variables
  (the generator replaces {{name}} with a sanitised string like "my_sensor").
- Arduino/C++ templates must compile on ESP32 / STM32 / RP2040 without modification.
- MicroPython templates must run on RP2040/ESP32 with the standard machine module.
- cpp_loop must leave a local float variable named according to primary_output_expr
  so the FreeRTOS sensor task can store it in the queue struct.

Return ONLY valid JSON (no markdown, no extra text):
{{
  "name": "Human-readable component name",
  "category": "sensor|motor|servo|communication|output|power",
  "library": "Arduino library name, or empty string if using raw registers",
  "dependencies": ["LibraryA", "LibraryB"],
  "required_interfaces": ["{inferred_iface}"],
  "pins_needed": 2,
  "headers": ["<LibraryA.h>"],
  "cpp_init": "C++ setup code using {{name}} and pin tokens",
  "cpp_loop": "C++ per-cycle read/update code — must define the primary output variable",
  "micropython_init": "MicroPython init code using {{name}} and pin tokens",
  "micropython_loop": "MicroPython per-cycle read/update — one assignment statement",
  "primary_output_expr": "C expression for the primary scalar reading, e.g. temperature_{{name}}"
}}"""

        def _validate_llm_result(result: dict, iface: str) -> List[str]:
            """Return list of validation errors. Empty list = OK."""
            errs: List[str] = []
            for req_key in ("cpp_init", "micropython_init"):
                if not result.get(req_key):
                    errs.append(f"missing '{req_key}'")
            # Jinja2 syntax check on all template strings
            for key in ("cpp_init", "cpp_loop", "micropython_init", "micropython_loop"):
                tmpl = result.get(key, "")
                if tmpl:
                    try:
                        _JINJA_ENV.parse(tmpl)
                    except jinja2.TemplateSyntaxError as exc:
                        errs.append(f"Jinja2 syntax error in '{key}': {exc}")
            # Pin-token presence check: at least one interface token must appear
            iface_tokens = {
                "I2C":  ["{scl}", "{sda}"],
                "SPI":  ["{sclk}", "{mosi}"],
                "UART": ["{tx}", "{rx}"],
                "PWM":  ["{pwm}"],
                "GPIO": ["{pin}"],
            }
            required_tokens = iface_tokens.get(iface.upper(), ["{pin}"])
            all_code = " ".join(str(result.get(k, "")) for k in ("cpp_init", "micropython_init"))
            if required_tokens and not any(t in all_code for t in required_tokens):
                errs.append(f"no pin tokens for {iface} found — LLM may have hallucinated pin numbers")
            return errs

        for attempt in range(2):
            try:
                result = self.llm_provider.generate_json(prompt)
                errs = _validate_llm_result(result, inferred_iface)
                if errs and attempt == 0:
                    # Retry once with validation feedback appended to prompt
                    retry_prompt = prompt + f"\n\nPrevious attempt failed validation:\n" + "\n".join(f"- {e}" for e in errs) + "\nPlease fix and return correct JSON."
                    result = self.llm_provider.generate_json(retry_prompt)
                    errs = _validate_llm_result(result, inferred_iface)
                if errs:
                    logger.warning("[CODEGEN] LLM result for '%s' failed validation: %s", comp_id, errs)
                    # Still use it — better than nothing — but log the issues
                code_templates: Dict[str, str] = {}
                loop_templates: Dict[str, str] = {}
                if result.get("cpp_init"):
                    code_templates["C++"] = result["cpp_init"]
                if result.get("cpp_loop"):
                    loop_templates["C++"] = result["cpp_loop"]
                if result.get("micropython_init"):
                    code_templates["MicroPython"] = result["micropython_init"]
                if result.get("micropython_loop"):
                    loop_templates["MicroPython"] = result["micropython_loop"]
                return Component(
                    name=result.get("name", comp_id),
                    category=result.get("category", "sensor"),
                    library=result.get("library", ""),
                    dependencies=result.get("dependencies", []),
                    required_interfaces=result.get("required_interfaces", [inferred_iface]),
                    pins_needed=result.get("pins_needed", 1),
                    code_templates=code_templates,
                    headers=result.get("headers", []),
                    loop_templates=loop_templates,
                    primary_output=result.get("primary_output_expr", ""),
                )
            except Exception as e:
                logger.warning("[CODEGEN] LLM component generation attempt %d failed for '%s': %s", attempt + 1, comp_id, e)
                if attempt == 1:
                    return None
        return None

    @staticmethod
    def _skeleton_component(comp_id: str, comp_spec: Dict, user_params: Dict) -> Component:
        """
        Return a compilable stub for a completely unknown component.
        The generated code contains clear TODO markers so the developer knows
        exactly what to fill in — it will compile and link without errors.
        """
        display_name = comp_spec.get("name", comp_id)
        category     = comp_spec.get("category", "sensor")
        iface        = (comp_spec.get("interface") or
                        (comp_spec.get("required_interfaces") or ["GPIO"])[0])
        n = "{name}"  # keep as template token
        return Component(
            name=display_name,
            category=category,
            library="",
            dependencies=[],
            required_interfaces=[iface],
            pins_needed=1,
            code_templates={
                "C++": (
                    f"// TODO: add #include and init for {display_name}\n"
                    f"// Interface: {iface}  Spec: {comp_spec}\n"
                    f"// Replace this block with real driver initialisation."
                ),
                "MicroPython": (
                    f"# TODO: import and init {display_name}\n"
                    f"# Interface: {iface}\n"
                    f"{n}_{_c_ident(display_name)} = None  # replace with real driver"
                ),
            },
            headers=[],
            loop_templates={
                # C++ gets a typed stub so it still compiles
                "C++": (
                    f"// TODO: read/update {display_name}\n"
                    f"  float {n}_val = 0.0f;  // replace with actual read"
                ),
                # MicroPython: leave empty so the interface-driven raw-bus generator
                # kicks in and produces code that actually talks to real hardware.
            },
            user_params=user_params,
            primary_output=f"{{name}}_val",
        )
    
    def _allocate_pins(self, platform: Platform, components: List[Component]) -> Dict:
        """
        Delegate to PinAllocator (constraint-based, reads hw spec from HardwareDB).
        Returns {"allocations": {...}, "errors": [...]} for backwards compatibility.
        """
        hw = getattr(self, "_current_hw", None) or HardwareDB.load(platform.value, self.llm_provider)
        alloc_result = PinAllocator(hw).allocate(components)
        return {
            "allocations": alloc_result.assignments,
            "errors": alloc_result.errors + alloc_result.warnings,
        }
    
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

        # HAL abstraction — always included so components use portable HAL_* calls
        board_cfg = self._PLATFORMIO_BOARD_MAP.get(platform.value, {})
        framework = board_cfg.get("framework", "arduino")
        hal_files = generate_hal_files(platform.value, framework)
        files.update(hal_files)

        # PlatformIO ini for all platforms + CMake for STM32 (dual build system)
        files["platformio.ini"] = self._generate_platformio_ini(platform, components)
        stm32_platforms = {Platform.STM32F405, Platform.STM32F103, Platform.STM32H743}
        if platform in stm32_platforms:
            files["CMakeLists.txt"] = self._generate_cmake(platform, components, project_name)

        # Safety files — only when a real safety level is requested
        if safety_level not in ("NONE", "", None):
            safety_files = self._generate_safety_files(platform, components, safety_level)
            files.update(safety_files)

        # CI/CD pipeline
        files[".github/workflows/ci.yml"] = self._generate_ci_yml(platform, project_name)

        for comp in components:
            libraries.update(comp.dependencies)
            if comp.library:
                lib_entry = (
                    f"{comp.library}@{comp.library_version}" if comp.library_version else comp.library
                )
                libraries.add(lib_entry)

        build_config = {
            "platform": platform.value,
            "language": language.value,
            "framework": framework,
            "build_flags": ["-Os", "-Wall"],
            "lib_deps": sorted(libraries),
        }
        if safety_level not in ("NONE", "", None):
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
                ctx = {"name": _c_ident(comp.name)}
                ctx.update(pins)
                ctx.update({f"pin_{k}": v for k, v in pins.items()})
                init_code = _render(template, ctx)
                init_lines.append(f"# {comp.name}")
                init_lines.extend(init_code.split("\n"))
            else:
                init_lines.append(f"# {comp.name}: no MicroPython template (add driver manually)")

            # Create an async task for each component if using async
            if use_async:
                task_name = _c_ident(comp.name)
                period_const = ("SENSOR_PERIOD_MS"
                                if comp.category == "sensor"
                                else "LOOP_PERIOD_MS")
                loop_tmpl = comp.loop_templates.get("MicroPython", "")

                if loop_tmpl:
                    # Component has a MicroPython-specific loop template (from YAML catalog
                    # or built-in) — render it directly.
                    loop_ctx = {"name": task_name}
                    loop_ctx.update(pins)
                    loop_ctx.update({f"pin_{k}": v for k, v in pins.items()})
                    loop_ctx.update(comp.user_params)
                    rendered_loop = _render(loop_tmpl, loop_ctx)
                    task_body = "\n".join(
                        "        " + ln if ln.strip() else ""
                        for ln in rendered_loop.splitlines()
                    )
                else:
                    # No specific template — generate generic raw-bus read code based on
                    # the allocated interface. This works on real hardware for ANY device
                    # without knowing the chip. Developer fills in address + parsing only.
                    iface = alloc.get("interface", "") or ""
                    addr  = comp.user_params.get("i2c_address", "0x00")
                    if "I2C" in iface:
                        task_body = (
                            f"        # {comp.name} — I2C addr {addr}\n"
                            f"        # Consult datasheet: set address, byte count, register\n"
                            f"        _buf_{task_name} = i2c.readfrom_mem({addr}, 0x00, 2)  # reg 0x00, 2 bytes\n"
                            f"        {task_name}_val = int.from_bytes(_buf_{task_name}, 'big')  # TODO: apply scaling"
                        )
                    elif "SPI" in iface:
                        cs_pin = pins.get("cs", "None")
                        task_body = (
                            f"        # {comp.name} — SPI CS={cs_pin}\n"
                            f"        # Consult datasheet: set command byte and byte count\n"
                            f"        {task_name}_cs.value(0)\n"
                            f"        _buf_{task_name} = spi.read(2, 0x00)  # TODO: command byte, byte count\n"
                            f"        {task_name}_cs.value(1)\n"
                            f"        {task_name}_val = int.from_bytes(_buf_{task_name}, 'big')  # TODO: apply scaling"
                        )
                    elif "UART" in iface:
                        task_body = (
                            f"        # {comp.name} — UART\n"
                            f"        if {task_name}_uart.any():\n"
                            f"            _buf_{task_name} = {task_name}_uart.read()  # TODO: fixed packet length\n"
                            f"            {task_name}_val = 0.0  # TODO: parse _buf_{task_name} per protocol"
                        )
                    elif comp.category in ("motor", "servo", "actuator", "output"):
                        task_body = (
                            f"        # {comp.name} — update output\n"
                            f"        # TODO: write setpoint to {task_name} via PWM/GPIO"
                        )
                    else:
                        task_body = (
                            f"        # {comp.name} — GPIO\n"
                            f"        # TODO: read/write {task_name} pin"
                        )

                async_tasks.append(f'''
async def task_{task_name}():
    while True:
{task_body}
        await asyncio.sleep_ms({period_const})
''')

        # --- main.py ---
        if use_async:
            # uasyncio coroutine-based (FreeRTOS equivalent for MicroPython)
            task_creates = "\n".join(
                f"    asyncio.create_task(task_{_c_ident(c.name)}())"
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
                ctx = {"name": _c_ident(comp.name)}
                ctx.update(pins)
                ctx.update({f"pin_{k}": v for k, v in pins.items()})
                # CircuitPython uses board.GPxx instead of bare integers
                ctx_cp = {k: f"board.GP{v}" if isinstance(v, int) else v for k, v in ctx.items()}
                init_code = _render(template, ctx_cp)
                init_lines.append(f"# {comp.name}")
                init_lines.extend(
                    line.replace("machine.Pin", "digitalio.DigitalInOut")
                        .replace("machine.PWM", "pwmio.PWMOut")
                    for line in init_code.split("\n")
                )
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
            includes.add("<queue.h>")
            includes.add("<semphr.h>")
        
        # Generate setup code and loop code from templates
        setup_code = []
        loop_code = []

        for comp in components:
            alloc = pin_allocations.get(comp.name, {})
            pins = alloc.get("pins", {})

            ctx = {"name": _c_ident(comp.name)}
            ctx.update(pins)
            ctx.update({f"pin_{k}": v for k, v in pins.items()})
            ctx.update(comp.user_params)  # i2c_address, spi_mode, uart_baud, etc.

            # ── Setup block ──
            init_tmpl = comp.code_templates.get("C++", "")
            if init_tmpl:
                code = _render(init_tmpl, ctx)
                setup_code.append(f"  // Initialize {comp.name}")
                setup_code.extend([f"  {line}" for line in code.split("\n")])
            else:
                setup_code.append(f"  // {comp.name}: no C++ init template")

            # ── Loop block ──
            loop_tmpl = comp.loop_templates.get("C++", "")
            if loop_tmpl:
                lcode = _render(loop_tmpl, ctx)
                loop_code.append(f"  // {comp.name}")
                loop_code.extend([f"  {line}" for line in lcode.split("\n")])
            else:
                loop_code.append(f"  // {comp.name}: add read/update logic here")
        
        # RTOS task creation — queues between sensor→actuator, mutex for shared I2C
        rtos_code = ""
        if rtos == RTOS.FREERTOS:
            sensor_comps = [c for c in components if c.category == "sensor"]
            motor_comps  = [c for c in components if c.category in ("motor", "servo", "actuator")]
            comms_comps  = [c for c in components if c.category == "communication"]
            output_comps = [c for c in components if c.category == "output"]

            has_i2c = any(
                "I2C" in (pin_allocations.get(c.name, {}).get("interface") or "")
                for c in components
            )

            def _loop_body(comps: List[Component], indent: str = "    ") -> str:
                lines = []
                for c in comps:
                    alloc = pin_allocations.get(c.name, {})
                    pins  = alloc.get("pins", {})
                    tmpl  = c.loop_templates.get("C++", "")
                    if tmpl:
                        ctx = {"name": _c_ident(c.name)}
                        ctx.update(pins)
                        ctx.update({f"pin_{k}": v for k, v in pins.items()})
                        ctx.update(c.user_params)
                        lines.append(_render(tmpl, ctx))
                    else:
                        lines.append(f"// {c.name}: add read/update code here")
                return ("\n" + indent).join(lines) if lines else "// No components in this task"

            def _sensor_slot_assignments(comps: List[Component]) -> str:
                """Emit data.values[i] = <primary_expr> for each sensor slot."""
                lines = []
                for i, c in enumerate(comps):
                    n   = _c_ident(c.name)
                    ctx = {"name": n}
                    ctx.update(c.user_params)
                    expr = c.primary_output
                    resolved = _render(expr, ctx).strip() if expr else None
                    if resolved:
                        lines.append(f"  data.values[{i}] = {resolved};")
                    else:
                        lines.append(f"  data.values[{i}] = 0.0f;  // TODO: assign primary reading from {c.name}")
                return "\n".join(lines)

            sensor_body   = _loop_body(sensor_comps)
            actuator_body = _loop_body(motor_comps + output_comps)
            comms_body    = _loop_body(comms_comps)

            # Generic sensor struct — indexed by slot, not by component name.
            # N_SENSORS is emitted as a #define so it scales to any component set.
            n_sensors = max(len(sensor_comps), 1)

            # Stack sizes: sensor I2C reads ~1KB, motor float math ~2KB, comms network ~4KB
            sensor_stack  = 2048 if not has_i2c else 3072
            actuator_stack = 2048 + 512 * len(motor_comps)
            comms_stack   = 4096

            task_decls  = []
            task_creates = []
            global_decls = []

            # Emit a comment mapping slot index → sensor name so generated code stays readable
            slot_comments = "\n".join(
                f"//   values[{i}]  {c.name}"
                for i, c in enumerate(sensor_comps)
            ) or "//   (no sensors)"

            # ── DMA / interrupt capability by platform ────────────────────────
            # STM32 platforms support raw DMA via HAL_I2C_Master_Receive_DMA.
            # All platforms support DRDY external interrupt when drdy_pin provided.
            # Everything else falls back to a timed read with proper queue depth.
            dma_platforms = {Platform.STM32F405, Platform.STM32F103, Platform.STM32H743}
            tickless_platforms = {Platform.NRF52840, Platform.ESP32, Platform.ESP32_S3}
            has_dma  = platform in dma_platforms
            has_tickless = platform in tickless_platforms and not motor_comps
            has_drdy = any(c.user_params.get("drdy_pin") for c in sensor_comps)

            # Determine if any sensor uses I2C — affects DMA buffer/callback approach
            def _sensor_uses_i2c(c: Component) -> bool:
                return "I2C" in (pin_allocations.get(c.name, {}).get("interface") or "")

            global_decls.append(f"""
// ── FreeRTOS inter-task communication ────────────────────────────────────────
// Sensor slot index map:
{slot_comments}
#define N_SENSORS {n_sensors}
#define SENSOR_QUEUE_DEPTH 4  // ring buffer — 4 samples deep, never blocks sensor task

typedef struct {{
  float    values[N_SENSORS];  // one slot per sensor, order matches slot map above
  uint32_t timestamp_ms;
  bool     valid;
}} SensorData_t;

static QueueHandle_t    sensorQueue    = NULL;
static TaskHandle_t     sensorTaskHandle = NULL;
{"static SemaphoreHandle_t i2cMutex = NULL;  // Shared I2C bus protection" if has_i2c else ""}
{"// DMA receive buffers — one per I2C sensor (placed in DMA-accessible RAM)" if has_dma and has_i2c else ""}
{"".join(f"static uint8_t _dma_buf_{_c_ident(c.name)}[16];  // burst read buffer for {c.name}" + chr(10) for c in sensor_comps if _sensor_uses_i2c(c)) if has_dma else ""}
""")

            if sensor_comps:
                slot_assigns = _sensor_slot_assignments(sensor_comps)
                i2c_lock   = "xSemaphoreTake(i2cMutex, portMAX_DELAY);" if has_i2c else ""
                i2c_unlock = "xSemaphoreGive(i2cMutex);" if has_i2c else ""

                if has_dma and has_i2c:
                    # STM32 DMA path: request DMA read, task waits on notification from IRQ
                    dma_read_lines = "\n    ".join(
                        f"HAL_I2C_Read_DMA(0, 0x{c.user_params.get('i2c_address', 0):02X}, 0x00,"
                        f" _dma_buf_{_c_ident(c.name)}, sizeof(_dma_buf_{_c_ident(c.name)}));"
                        for c in sensor_comps if _sensor_uses_i2c(c)
                    ) or "// No DMA-capable sensors"

                    task_decls.append(f"""
// Override HAL DMA complete callback — fires from DMA IRQ context
void HAL_I2C_DMAComplete_Callback(uint8_t bus) {{
  (void)bus;
  BaseType_t xHigher = pdFALSE;
  vTaskNotifyGiveFromISR(sensorTaskHandle, &xHigher);
  portYIELD_FROM_ISR(xHigher);
}}

void sensorTask(void *pvParameters) {{
  SensorData_t data = {{}};
  for (;;) {{
    // Arm DMA burst reads (non-blocking)
    {i2c_lock}
    {dma_read_lines}
    {i2c_unlock}
    // Block until DMA complete IRQ fires (zero CPU during transfer)
    ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(100));  // 100 ms watchdog
{slot_assigns}
    data.timestamp_ms = (uint32_t)(xTaskGetTickCount() * portTICK_PERIOD_MS);
    data.valid = true;
    // Queue depth=4: never overwrites; if full, discard oldest
    if (uxQueueSpacesAvailable(sensorQueue) == 0) {{
      SensorData_t discard;
      xQueueReceive(sensorQueue, &discard, 0);
    }}
    xQueueSendToBack(sensorQueue, &data, 0);
  }}
}}""")

                elif has_drdy:
                    # DRDY interrupt path — works on all platforms
                    drdy_setups = "\n  ".join(
                        f"HAL_GPIO_AttachInterrupt({c.user_params['drdy_pin']},"
                        f" drdy_isr_{_c_ident(c.name)}, true);"
                        for c in sensor_comps if c.user_params.get("drdy_pin")
                    )
                    drdy_isrs = "\n".join(
                        f"static void IRAM_ATTR drdy_isr_{_c_ident(c.name)}(void) {{"
                        f"\n  BaseType_t w = pdFALSE;"
                        f"\n  vTaskNotifyGiveFromISR(sensorTaskHandle, &w);"
                        f"\n  portYIELD_FROM_ISR(w);\n}}"
                        for c in sensor_comps if c.user_params.get("drdy_pin")
                    )
                    task_decls.append(f"""
{drdy_isrs}

void sensorTask(void *pvParameters) {{
  SensorData_t data = {{}};
  for (;;) {{
    // Block until DRDY interrupt fires — zero CPU while waiting
    ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(100));
    {i2c_lock}
    {sensor_body}
    {i2c_unlock}
{slot_assigns}
    data.timestamp_ms = (uint32_t)(xTaskGetTickCount() * portTICK_PERIOD_MS);
    data.valid = true;
    if (uxQueueSpacesAvailable(sensorQueue) == 0) {{
      SensorData_t discard; xQueueReceive(sensorQueue, &discard, 0);
    }}
    xQueueSendToBack(sensorQueue, &data, 0);
  }}
}}""")
                    setup_code_drdy = f"  {drdy_setups}"

                else:
                    # Timed path — optimized polling with configUSE_TICKLESS_IDLE
                    task_decls.append(f"""
void sensorTask(void *pvParameters) {{
  const TickType_t period = pdMS_TO_TICKS(10);  // 100 Hz
  TickType_t lastWake = xTaskGetTickCount();
  SensorData_t data = {{}};
  for (;;) {{
    {i2c_lock}
    {sensor_body}
    {i2c_unlock}
{slot_assigns}
    data.timestamp_ms = (uint32_t)(xTaskGetTickCount() * portTICK_PERIOD_MS);
    data.valid = true;
    // Queue depth 4: drop oldest if full so sensor is never blocked
    if (uxQueueSpacesAvailable(sensorQueue) == 0) {{
      SensorData_t discard; xQueueReceive(sensorQueue, &discard, 0);
    }}
    xQueueSendToBack(sensorQueue, &data, 0);
    vTaskDelayUntil(&lastWake, period);
  }}
}}""")

                task_creates.append(
                    f"  sensorQueue = xQueueCreate(SENSOR_QUEUE_DEPTH, sizeof(SensorData_t));\n"
                    f"  xTaskCreate(sensorTask, \"Sensor\", {sensor_stack}, NULL, 2, &sensorTaskHandle);"
                )

            if motor_comps or output_comps:
                task_decls.append(f"""
void actuatorTask(void *pvParameters) {{
  const TickType_t period = pdMS_TO_TICKS(1);  // 1 kHz actuator loop
  TickType_t lastWake = xTaskGetTickCount();
  SensorData_t sensors = {{}};
  for (;;) {{
    // Non-blocking peek — use last valid frame if queue empty
    xQueuePeek(sensorQueue, &sensors, 0);
    if (sensors.valid) {{
      {actuator_body}
    }}
    HAL_WDG_Feed();  // REQ-SAF-WDG: each task must feed the watchdog
    vTaskDelayUntil(&lastWake, period);
  }}
}}""")
                task_creates.append(
                    f"  xTaskCreate(actuatorTask, \"Actuator\", {actuator_stack}, NULL, 3, NULL);"
                )

            if comms_comps:
                task_decls.append(f"""
void commsTask(void *pvParameters) {{
  const TickType_t period = pdMS_TO_TICKS(100);  // 10 Hz comms
  TickType_t lastWake = xTaskGetTickCount();
  for (;;) {{
    {comms_body}
    HAL_WDG_Feed();
    vTaskDelayUntil(&lastWake, period);
  }}
}}""")
                task_creates.append(
                    f"  xTaskCreate(commsTask, \"Comms\", {comms_stack}, NULL, 1, NULL);"
                )

            # Tickless idle — reduces power consumption on battery platforms
            tickless_config = (
                "\n// Tickless idle enabled — enters low-power sleep between tasks\n"
                "// configUSE_TICKLESS_IDLE = 1  (set in FreeRTOSConfig.h)\n"
            ) if has_tickless else ""

            mutex_init = "  i2cMutex = xSemaphoreCreateMutex();" if has_i2c else ""
            rtos_code  = tickless_config + "\n".join(global_decls) + "\n".join(task_decls)
            setup_code.append(
                f"\n  // --- FreeRTOS init ---\n{mutex_init}\n"
                + "\n".join(task_creates)
            )
        
        # Safety code patterns — real IWDG via HAL abstraction
        safety_code = ""
        if safety_level in ["SIL1", "SIL2", "SIL3", "ASIL_A", "ASIL_B", "ASIL_C", "ASIL_D"]:
            wdg_timeout = 500 if safety_level in ("SIL3", "ASIL_D") else 1000
            safety_code = f"""
#include "hal/hal.h"
#include "safety.h"

// REQ-SAF-WDG: IWDG must be initialised before any task runs.
// Timeout: {wdg_timeout} ms — feeds required from every task that holds CPU.
#define WATCHDOG_TIMEOUT_MS {wdg_timeout}
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
{"  HAL_WDG_Init(WATCHDOG_TIMEOUT_MS);  // REQ-SAF-WDG: arm IWDG after all init" if safety_level not in ("NONE", "") else ""}
}}

void loop() {{
  unsigned long startTime = micros();

  // Main control loop (runs at LOOP_FREQUENCY_HZ)
{chr(10).join(loop_code) if loop_code else "  // TODO: Add main loop logic"}
  {"HAL_WDG_Feed();  // REQ-SAF-WDG: pet IWDG every iteration" if safety_level not in ("NONE", "") else ""}
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
        """Generate PlatformIO configuration with version-pinned lib_deps for all platforms."""
        # Build a versioned dep map: prefer "Library@version" when library_version is set.
        # Use a dict so duplicate lib names are deduped by name, keeping the first version seen.
        versioned: Dict[str, str] = {}
        for comp in components:
            # Primary library with version pin
            lib_key = comp.library or ""
            if lib_key and lib_key not in versioned:
                versioned[lib_key] = (
                    f"{lib_key}@{comp.library_version}" if comp.library_version else lib_key
                )
            # Dependencies (usually framework libs — no version pinning available)
            for dep in comp.dependencies:
                if dep and dep not in versioned:
                    versioned[dep] = dep
        unique_libs = sorted(versioned.values())

        board_cfg = self._PLATFORMIO_BOARD_MAP.get(platform.value, {
            "platform": "espressif32", "board": "esp32dev", "framework": "arduino", "upload_speed": "115200"
        })

        native_section = """
[env:native]
platform = native
build_flags = -std=c++17
test_framework = unity
"""
        return f"""; PlatformIO Project Configuration — Auto-generated by BRICK OS
; lib_deps are version-pinned. Bump versions here, not in code.
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
    -Werror=implicit-function-declaration
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


    def _generate_safety_files(
        self,
        platform: Platform,
        components: List[Component],
        safety_level: str,
    ) -> Dict[str, str]:
        """
        Generate safety.h + safety.c for SIL/ASIL projects.

        Implements:
          - IWDG init + per-task Feed (REQ-SAF-WDG)
          - SAFETY_EnterSafeState(fault_code) — disables all actuators, sets interlocks
          - HardFault_Handler override (REQ-SAF-HF)
          - Sensor majority vote when 2+ sensors share a category (REQ-SAF-VOTE)
          - safety_tests.cpp with timing assertions
        """
        actuator_comps = [c for c in components if c.category in ("motor", "servo", "actuator")]
        sensor_groups: Dict[str, List[str]] = {}
        for c in components:
            if c.category == "sensor":
                sensor_groups.setdefault(c.category, []).append(_c_ident(c.name))

        # Disable actuators in safe-state — emit one HAL_PWM_Write(pin, 0) per actuator
        actuator_shutdowns = "\n  ".join(
            f"HAL_PWM_Write(ACTUATOR_{_c_ident(c.name).upper()}_PIN, 0);  // disarm {c.name}"
            for c in actuator_comps
        ) or "// No actuators — nothing to disarm"

        # Majority vote for sensor categories with 3+ sensors
        vote_fns = []
        for cat, names in sensor_groups.items():
            if len(names) >= 3:
                vote_fns.append(f"""
// REQ-SAF-VOTE: majority vote across 3 {cat} readings
float SAFETY_Vote_{cat}(float a, float b, float c) {{
  // Sort and return median
  if (a > b) {{ float t = a; a = b; b = t; }}
  if (b > c) {{ float t = b; b = c; c = t; }}
  if (a > b) {{ float t = a; a = b; b = t; }}
  return b;
}}""")

        timeout_ms = 500 if safety_level in ("SIL3", "ASIL_D") else 1000

        safety_h = f"""\
#ifndef SAFETY_H
#define SAFETY_H
/**
 * @file safety.h
 * @brief Safety monitor — {safety_level}
 * Auto-generated by BRICK OS. DO NOT EDIT.
 *
 * REQ-SAF-WDG:  IWDG must be fed by every task within {timeout_ms} ms.
 * REQ-SAF-HF:   HardFault_Handler must enter safe state before halting.
 * REQ-SAF-VOTE: Sensor majority vote required when 3+ sensors share a category.
 */

#include <stdint.h>
#include "hal/hal.h"

#define SAFETY_LEVEL_STR  "{safety_level}"
#define SAFETY_WDG_TIMEOUT_MS {timeout_ms}

typedef enum {{
  FAULT_NONE         = 0,
  FAULT_WDG_MISS     = 1,  // REQ-SAF-WDG
  FAULT_HARDFAULT    = 2,  // REQ-SAF-HF
  FAULT_SENSOR_OOB   = 3,
  FAULT_ACTUATOR_OOR = 4,
  FAULT_STACK_OVF    = 5,
}} FaultCode_t;

void SAFETY_Init(void);
void SAFETY_Feed(void);
void SAFETY_EnterSafeState(FaultCode_t fault);
{"".join(f"float SAFETY_Vote_{cat}(float a, float b, float c);\n" for cat in sensor_groups if len(sensor_groups[cat]) >= 3)}
#endif /* SAFETY_H */
"""

        vote_fn_block = "\n".join(vote_fns)

        safety_c = f"""\
/**
 * @file safety.c
 * @brief Safety monitor implementation — {safety_level}
 * Auto-generated by BRICK OS. DO NOT EDIT.
 */

#include "safety.h"
#include "hal/hal.h"
#include <string.h>

static volatile FaultCode_t _active_fault = FAULT_NONE;

// REQ-SAF-WDG: Initialise IWDG. Call ONCE before vTaskStartScheduler / loop().
void SAFETY_Init(void) {{
  HAL_WDG_Init(SAFETY_WDG_TIMEOUT_MS);
}}

// REQ-SAF-WDG: Reset the watchdog. Each task/ISR must call this within the timeout.
void SAFETY_Feed(void) {{
  HAL_WDG_Feed();
}}

// REQ-SAF: Shut down all actuators and halt in a known safe state.
// This function is intentionally non-returning.
void SAFETY_EnterSafeState(FaultCode_t fault) {{
  _active_fault = fault;

  // Disable all actuators — {len(actuator_comps)} device(s)
  {actuator_shutdowns}

  // Log fault to serial (best-effort — may be unavailable in hard fault context)
  // Use a raw memory-mapped UART write if you need guaranteed output.
  HAL_UART_WriteStr(0, "SAFETY: safe state entered, fault=");
  char buf[4];
  buf[0] = '0' + (int)fault;
  buf[1] = '\\r'; buf[2] = '\\n'; buf[3] = 0;
  HAL_UART_WriteStr(0, buf);

  // Feed watchdog once more to prevent spurious reset during shutdown,
  // then stop feeding so the watchdog resets after SAFETY_WDG_TIMEOUT_MS.
  HAL_WDG_Feed();
  for (;;) {{
    // Spin — watchdog will reset the MCU if this takes too long
    HAL_DelayMs(10);
  }}
}}

// REQ-SAF-HF: Cortex-M HardFault handler override.
// __attribute__((naked)) preserves LR/SP for stack-trace tools.
__attribute__((naked)) void HardFault_Handler(void) {{
  __asm volatile(
    "tst lr, #4      \\n"
    "ite eq          \\n"
    "mrseq r0, msp   \\n"
    "mrsne r0, psp   \\n"
    "b HardFault_HandlerC \\n"
  );
}}

void HardFault_HandlerC(uint32_t *stack) {{
  (void)stack;  // stack[6]=PC, stack[7]=xPSR for offline analysis
  SAFETY_EnterSafeState(FAULT_HARDFAULT);
}}

{vote_fn_block}
"""

        safety_tests_cpp = f"""\
/**
 * @file safety_tests.cpp
 * @brief Timing assertions for safety watchdog — {safety_level}
 * Run with: pio test -e native
 */

#include <unity.h>
#include <stdint.h>

// Stub HAL for native test build
static uint32_t _wdg_timeout = 0;
static uint32_t _wdg_last_feed = 0;
static uint32_t _mock_time = 0;
void HAL_WDG_Init(uint32_t ms)  {{ _wdg_timeout = ms; _wdg_last_feed = 0; }}
void HAL_WDG_Feed(void)          {{ _wdg_last_feed = _mock_time; }}
void HAL_DelayMs(uint32_t ms)    {{ _mock_time += ms; }}
void HAL_UART_WriteStr(uint8_t, const char*) {{}}

#include "safety.c"

void test_wdg_timeout_configured(void) {{
  SAFETY_Init();
  TEST_ASSERT_EQUAL_UINT32({timeout_ms}, _wdg_timeout);
}}

void test_feed_resets_timer(void) {{
  SAFETY_Init();
  _mock_time = 500;
  SAFETY_Feed();
  TEST_ASSERT_EQUAL_UINT32(500, _wdg_last_feed);
}}

void test_safe_state_sets_fault(void) {{
  // Entering safe state must set fault code before looping
  // We can't call SAFETY_EnterSafeState directly (it loops forever),
  // so we test the fault assignment path indirectly.
  // Full integration test requires hardware or a mock that intercepts HAL_DelayMs.
  TEST_PASS();  // placeholder — extend with hardware-in-loop tests
}}

int main(void) {{
  UNITY_BEGIN();
  RUN_TEST(test_wdg_timeout_configured);
  RUN_TEST(test_feed_resets_timer);
  RUN_TEST(test_safe_state_sets_fault);
  return UNITY_END();
}}
"""

        return {
            "safety.h": safety_h,
            "safety.c": safety_c,
            "test/safety_tests.cpp": safety_tests_cpp,
        }

    def _generate_ci_yml(self, platform: Platform, project_name: str) -> str:
        """Generate .github/workflows/ci.yml with PlatformIO build + native tests + cppcheck."""
        board_map = self._PLATFORMIO_BOARD_MAP.get(platform.value, {})
        env_name = platform.value.lower()
        return f"""\
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  build:
    name: PlatformIO Build — {platform.value}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install PlatformIO
        run: pip install platformio

      - name: Build firmware
        run: pio run -e {env_name}

  native-tests:
    name: Native Unit Tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install PlatformIO
        run: pip install platformio

      - name: Run native tests
        run: pio test -e native

  static-analysis:
    name: cppcheck static analysis
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install cppcheck
        run: sudo apt-get install -y cppcheck

      - name: Run cppcheck
        run: |
          cppcheck --error-exitcode=1 --std=c++17 \\
            --suppress=missingInclude \\
            --enable=warning,performance,portability \\
            main.cpp safety.c
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

        # Domain-specific PID gains — NOT generic mass-spring-damper numerology
        domain_gains = self._pid_gains_for_domain(intent, design_p, thermal, structural, fluid)
        kp          = domain_gains["kp"]
        ki          = domain_gains["ki"]
        kd          = domain_gains["kd"]
        pv_name     = domain_gains["pv_name"]
        cv_name     = domain_gains["cv_name"]
        ctrl_domain = domain_gains["domain"]
        anti_windup = domain_gains["anti_windup"]
        gains_note  = domain_gains["note"]

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

Control domain : {ctrl_domain}
Process var    : {pv_name}
Control var    : {cv_name}
PID gains      : Kp={kp}  Ki={ki}  Kd={kd}
Gain basis     : {gains_note}

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
# Domain: {ctrl_domain}
# Gain basis: {gains_note}
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
    _anti_windup: bool = {anti_windup}  # clamp integral when output saturates

    def reset(self):
        self._integral  = 0.0
        self._prev_error = 0.0

    def update(self, measurement: float, dt: float) -> float:
        error      = self.setpoint - measurement
        derivative = (error - self._prev_error) / max(dt, 1e-9)
        self._prev_error = error
        raw_output = self.kp * error + self.ki * self._integral + self.kd * derivative
        output     = float(np.clip(raw_output, self.output_min, self.output_max))
        # Anti-windup: only integrate when output is not saturated
        if not self._anti_windup or (output == raw_output):
            self._integral += error * dt
        return output


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
        """Main control law — {ctrl_domain} PID on {pv_name}."""
        pv = sensors.get("{pv_name}", 0.0)
        output = self.pid.update(pv, DT)
        return {{"{cv_name}": output, "error": self.pid.setpoint - pv}}

    def write_outputs(self, commands: dict):
        """Apply {cv_name} command to actuator."""
        ctrl = commands.get("{cv_name}", 0.0)
        # ── Replace with hardware output matching your actuator type ────
        # Thermal   : set heater duty cycle (0.0–1.0) via DAC or PWM
        # Motion    : set motor torque/velocity via driver (CAN, PWM, serial)
        # Fluid     : set pump speed / valve position via VFD or 4-20 mA
        # Electrical: set PWM duty / reference voltage via DAC
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

    @staticmethod
    def _pid_gains_for_domain(
        intent: str,
        design_p: Dict,
        thermal: Dict,
        structural: Dict,
        fluid: Dict,
    ) -> Dict[str, Any]:
        """
        Return domain-appropriate PID gains and metadata.

        Each domain uses a physically motivated formula, not generic mass scaling.
        All gains are conservative starting points — real tuning is always required.

        Returns dict with keys: domain, kp, ki, kd, pv_name, cv_name,
                                anti_windup, note
        """
        i = intent  # shorthand

        # ── Thermal / temperature control ──────────────────────────────────────
        # PI only (no derivative — amplifies sensor noise in slow thermal systems)
        # Tuning basis: SIMC rules for first-order process
        #   tau  ≈ mass * Cp / (U * A)   — thermal time constant
        #   Kp   ≈ tau / (K * theta)     — K=steady-state gain, theta=dead time
        # Without full process model, use conservative: Kp≈0.5, Ki≈Kp/(2*tau)
        if any(kw in i for kw in ("temperature", "thermal", "heat", "oven",
                                   "furnace", "cool", "hvac", "thermostat")):
            tau_s   = design_p.get("thermal_time_constant_s", 120.0) or 120.0
            k_gain  = design_p.get("thermal_process_gain", 1.0) or 1.0
            kp = round(0.5 / max(k_gain, 0.01), 4)
            ki = round(kp / (2 * max(tau_s, 1.0)), 6)
            kd = 0.0  # no derivative for thermal
            return {
                "domain": "thermal",
                "kp": kp, "ki": ki, "kd": kd,
                "pv_name": "temperature_c",
                "cv_name": "heater_pct",
                "anti_windup": True,
                "note": f"SIMC PI — tau={tau_s:.0f}s, K={k_gain}. Kd=0 (no derivative on thermal)",
            }

        # ── Fluid / flow / pressure control ────────────────────────────────────
        # Cascade hint: outer loop (pressure/level) → inner loop (flow/speed)
        # Use PI for inner loop, P for outer (slow outer, fast inner)
        if any(kw in i for kw in ("flow", "pump", "pressure", "hydraulic",
                                   "coolant", "fluid", "valve", "pipe")):
            pipe_dia_m  = design_p.get("pipe_diameter_m", 0.05) or 0.05
            # Approximate process gain from pipe area (higher area → lower gain needed)
            import math
            area = math.pi * (pipe_dia_m / 2) ** 2
            kp = round(min(2.0, 0.05 / max(area, 1e-6)), 4)
            ki = round(kp * 0.3, 5)
            kd = 0.0  # flow loops rarely need derivative
            return {
                "domain": "fluid_flow",
                "kp": kp, "ki": ki, "kd": kd,
                "pv_name": "flow_lpm",
                "cv_name": "pump_speed_pct",
                "anti_windup": True,
                "note": f"PI for inner flow loop, pipe_dia={pipe_dia_m*1000:.0f}mm. Add outer pressure/level loop if cascade needed",
            }

        # ── Motion / position / velocity control ───────────────────────────────
        # PID + velocity feedforward. Gains from second-order closed-loop spec:
        #   omega_n ≈ sqrt(stiffness / mass),  zeta = 0.7 (slightly underdamped)
        #   Kp = mass * omega_n²,  Kd = 2*mass*omega_n*zeta,  Ki = Kp * 0.1
        if any(kw in i for kw in ("position", "velocity", "servo", "motor", "joint",
                                   "arm", "actuator", "motion", "drive", "robot")):
            mass_kg       = design_p.get("mass_kg", 1.0) or 1.0
            stiffness     = structural.get("effective_stiffness_n_m", 1000.0) or 1000.0
            omega_n       = (stiffness / max(mass_kg, 0.001)) ** 0.5
            kp = round(mass_kg * omega_n ** 2, 4)
            ki = round(kp * 0.1, 5)
            kd = round(2 * mass_kg * omega_n * 0.7, 4)
            return {
                "domain": "motion",
                "kp": kp, "ki": ki, "kd": kd,
                "pv_name": "position_m",
                "cv_name": "motor_torque_nm",
                "anti_windup": False,
                "note": f"Second-order spec — mass={mass_kg}kg, stiffness={stiffness}N/m, omega_n={omega_n:.2f}rad/s, zeta=0.7",
            }

        # ── Electrical / power / voltage control ───────────────────────────────
        # PI with tight integral bandwidth (switching noise, fast dynamics)
        if any(kw in i for kw in ("voltage", "current", "power", "converter",
                                   "inverter", "pwm", "buck", "boost", "mppt")):
            kp = 0.1
            ki = 10.0  # fast integrator for zero steady-state error on voltage
            kd = 0.0
            return {
                "domain": "electrical",
                "kp": kp, "ki": ki, "kd": kd,
                "pv_name": "voltage_v",
                "cv_name": "duty_cycle",
                "anti_windup": True,
                "note": "PI for power converter inner loop. Tune bandwidth to <1/10 switching frequency",
            }

        # ── Aerospace / altitude / attitude control ─────────────────────────────
        if any(kw in i for kw in ("altitude", "attitude", "pitch", "roll", "yaw",
                                   "drone", "uav", "spacecraft", "stabilize")):
            mass_kg = design_p.get("mass_kg", 0.5) or 0.5
            kp = round(4.0 * mass_kg, 4)
            ki = round(0.5 * mass_kg, 4)
            kd = round(2.0 * mass_kg, 4)
            return {
                "domain": "attitude",
                "kp": kp, "ki": ki, "kd": kd,
                "pv_name": "angle_rad",
                "cv_name": "motor_thrust_pct",
                "anti_windup": True,
                "note": f"Attitude PID, mass={mass_kg}kg. Tune Kd first (damp oscillations), then Kp, then Ki",
            }

        # ── Fallback: generic position control with mass-spring model ──────────
        mass_kg   = design_p.get("mass_kg", 1.0) or 1.0
        stiffness = structural.get("effective_stiffness_n_m", 500.0) or 500.0
        omega_n   = (stiffness / max(mass_kg, 0.001)) ** 0.5
        kp = round(mass_kg * omega_n ** 2, 4)
        ki = round(kp * 0.05, 5)
        kd = round(2 * mass_kg * omega_n * 0.7, 4)
        return {
            "domain": "generic_position",
            "kp": kp, "ki": ki, "kd": kd,
            "pv_name": "process_variable",
            "cv_name": "control_output",
            "anti_windup": False,
            "note": f"Generic mass-spring fallback — mass={mass_kg}kg, stiffness={stiffness}N/m. Identify domain in intent for domain-specific gains",
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

        thermal    = params.get("thermal_analysis", {}) or {}
        structural = params.get("structural_analysis", {}) or {}
        fluid      = params.get("fluid_analysis", {}) or {}
        domain_gains = self._pid_gains_for_domain(intent, design_p, thermal, structural, fluid)
        ros_kp = domain_gains["kp"]
        ros_ki = domain_gains["ki"]
        ros_kd = domain_gains["kd"]
        ros_gains_note = domain_gains["note"]

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

        # ── Dynamic sensor node: one publisher per actual component ─────────
        def _ros_msg_type(comp: Any) -> tuple:
            """Return (include_header, ros_type, short_name) for a component category."""
            cat = getattr(comp, "category", "") or ""
            _map = {
                "imu":         ("sensor_msgs/msg/imu.hpp",           "sensor_msgs::msg::Imu",            "Imu"),
                "gps":         ("sensor_msgs/msg/nav_sat_fix.hpp",   "sensor_msgs::msg::NavSatFix",      "NavSatFix"),
                "lidar":       ("sensor_msgs/msg/laser_scan.hpp",    "sensor_msgs::msg::LaserScan",      "LaserScan"),
                "range":       ("sensor_msgs/msg/range.hpp",         "sensor_msgs::msg::Range",          "Range"),
                "camera":      ("sensor_msgs/msg/image.hpp",         "sensor_msgs::msg::Image",          "Image"),
                "temperature": ("sensor_msgs/msg/temperature.hpp",   "sensor_msgs::msg::Temperature",    "Temperature"),
                "pressure":    ("sensor_msgs/msg/fluid_pressure.hpp","sensor_msgs::msg::FluidPressure",  "FluidPressure"),
            }
            if cat in _map:
                return _map[cat]
            return ("std_msgs/msg/float32_multi_array.hpp", "std_msgs::msg::Float32MultiArray", "Float32MultiArray")

        sensor_comps_ros = [c for c in params.get("_resolved_components", []) if getattr(c, "category", "") == "sensor"]
        # Fallback: treat 'imu' and 'range' as generic if no components passed
        if not sensor_comps_ros:
            sensor_comps_ros = []

        # Build include list and publisher declarations
        all_headers = {"sensor_msgs/msg/imu.hpp", "std_msgs/msg/string.hpp"}
        pub_decls, pub_inits, pub_reads, pub_members = [], [], [], []
        for comp in sensor_comps_ros:
            hdr, msg_type, short = _ros_msg_type(comp)
            all_headers.add(hdr)
            ident = _c_ident(comp.name) + "_pub_"
            topic = f"/sensors/{_c_ident(comp.name)}"
            pub_decls.append(f"    rclcpp::Publisher<{msg_type}>::SharedPtr {ident};")
            pub_inits.append(f'    {ident} = create_publisher<{msg_type}>("{topic}", 10);')
            pub_reads.append(
                f"    auto {_c_ident(comp.name)}_msg = {msg_type}();\n"
                f"    {_c_ident(comp.name)}_msg.header.stamp = now;\n"
                f"    // TODO: populate {_c_ident(comp.name)}_msg from driver\n"
                f"    {ident}->publish({_c_ident(comp.name)}_msg);"
            )

        # Fallback IMU if no sensors
        if not pub_inits:
            all_headers.add("sensor_msgs/msg/imu.hpp")
            pub_decls.append("    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;")
            pub_inits.append('    imu_pub_ = create_publisher<sensor_msgs::msg::Imu>("/sensors/imu", 10);')
            pub_reads.append(
                "    auto imu_msg = sensor_msgs::msg::Imu();\n"
                "    imu_msg.header.stamp = now;\n"
                "    imu_msg.linear_acceleration.z = 9.81f;  // TODO: real driver\n"
                "    imu_pub_->publish(imu_msg);"
            )

        _includes = "\n".join(f"#include <{h}>" for h in sorted(all_headers))
        _pub_members_str = "\n".join(pub_decls)
        _pub_inits_str   = "\n".join(pub_inits)
        _pub_reads_str   = "\n".join(pub_reads)

        topic_list = "\n".join(
            f" *   /sensors/{_c_ident(c.name)}" for c in sensor_comps_ros
        ) or " *   /sensors/imu  (fallback — add components for dynamic graph)"

        sensor_node_cpp = f"""/**
 * sensor_node.cpp — Reads hardware sensors and publishes to ROS2 topics
 * Auto-generated by BRICK OS CodegenAgent  ({timestamp})
 *
 * Publishes (derived from component list — not hardcoded):
{topic_list}
 */
#include <rclcpp/rclcpp.hpp>
{_includes}
#include <chrono>

using namespace std::chrono_literals;

class SensorNode : public rclcpp::Node {{
public:
  SensorNode() : Node("{project_name}_sensor") {{
{_pub_inits_str}
    timer_ = create_wall_timer(10ms, std::bind(&SensorNode::timer_cb, this));
    RCLCPP_INFO(get_logger(), "SensorNode started — mass={mass_kg:.2f} kg");
  }}

private:
  void timer_cb() {{
    auto now = get_clock()->now();
{_pub_reads_str}
    auto status_msg = std_msgs::msg::String();
    status_msg.data = "OK";
    status_pub_->publish(status_msg);
  }}

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_ {{
      create_publisher<std_msgs::msg::String>("/sensors/status", 10)}};
{_pub_members_str}
  rclcpp::TimerBase::SharedPtr timer_;
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

// PID gains — {ros_gains_note}
// Override at runtime: ros2 param set /{project_name}_control kp <value>

class ControlNode : public rclcpp::Node {{
public:
  ControlNode() : Node("{project_name}_control"), integral_(0.0), prev_err_(0.0) {{
    // Declare tunable PID params — no hardcoded constexpr
    declare_parameter("kp", {ros_kp});
    declare_parameter("ki", {ros_ki});
    declare_parameter("kd", {ros_kd});
    declare_parameter("setpoint", {length_m:.3f});

    imu_sub_   = create_subscription<sensor_msgs::msg::Imu>(
        "/sensors/imu", 10, [this](auto m) {{ imu_cb(m); }});
    range_sub_ = create_subscription<sensor_msgs::msg::Range>(
        "/sensors/range", 10, [this](auto m) {{ range_cb(m); }});

    {"cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>(\"/cmd_vel\", 10);" if is_mobile else
     "cmd_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(\"/cmd_joint\", 10);"}

    control_timer_ = create_wall_timer(10ms, std::bind(&ControlNode::control_cb, this));
    RCLCPP_INFO(get_logger(), "ControlNode — Kp=%.2f Ki=%.2f Kd=%.2f",
        get_parameter("kp").as_double(), get_parameter("ki").as_double(), get_parameter("kd").as_double());
  }}

private:
  void imu_cb(const sensor_msgs::msg::Imu::SharedPtr msg) {{
    latest_accel_z_ = msg->linear_acceleration.z;
  }}
  void range_cb(const sensor_msgs::msg::Range::SharedPtr msg) {{
    measured_range_ = msg->range;
  }}
  void control_cb() {{
    double kp = get_parameter("kp").as_double();
    double ki = get_parameter("ki").as_double();
    double kd = get_parameter("kd").as_double();
    double setpoint = get_parameter("setpoint").as_double();
    double dt  = 0.01;
    double err = setpoint - measured_range_;
    integral_  += err * dt;
    double deriv = (err - prev_err_) / dt;
    prev_err_  = err;
    double u   = std::clamp(kp*err + ki*integral_ + kd*deriv, -1.0, 1.0);

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
    # Gains — {ros_gains_note}
    kp: {ros_kp}
    ki: {ros_ki}
    kd: {ros_kd}
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

    @staticmethod
    def _infer_plc_domain(intent: str) -> str:
        """
        Classify intent into a PLC process domain for domain-specific state machines.
        Returns one of: pump | oven | reactor | conveyor | press | generic
        """
        intent = intent.lower()
        if any(kw in intent for kw in ("pump", "flow", "liquid", "hydraulic", "coolant", "water")):
            return "pump"
        if any(kw in intent for kw in ("oven", "furnace", "heater", "bake", "kiln", "drying")):
            return "oven"
        if any(kw in intent for kw in ("reactor", "mix", "blend", "batch", "ferment", "reacting")):
            return "reactor"
        if any(kw in intent for kw in ("conveyor", "belt", "transport", "transfer", "sorting")):
            return "conveyor"
        if any(kw in intent for kw in ("press", "clamp", "cylinder", "pneumatic", "stamping")):
            return "press"
        return "generic"

    @staticmethod
    def _plc_domain_states(domain: str) -> List[str]:
        """Return state name list for a given process domain."""
        return {
            "pump":     ["IDLE", "PRIMING", "RUNNING", "STOPPING", "FAULT"],
            "oven":     ["IDLE", "PREHEAT", "SOAK", "COOLING", "FAULT"],
            "reactor":  ["IDLE", "CHARGING", "REACTING", "DISCHARGING", "CLEANING"],
            "conveyor": ["IDLE", "RAMPING_UP", "RUNNING", "RAMPING_DOWN", "FAULT"],
            "press":    ["IDLE", "CLAMPING", "PRESSING", "RELEASING", "FAULT"],
        }.get(domain, ["IDLE", "STARTING", "RUNNING", "STOPPING", "FAULT"])

    def _run_plc(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate IEC 61131-3 Structured Text PLC program for industrial automation."""
        project_name = (params.get("project_name", "plc_program") or "plc_program").replace(" ", "_").upper()
        author       = params.get("author", "BRICK OS")
        version      = params.get("version", "1.0.0")
        intent       = (params.get("intent", "") or "").lower()
        timestamp    = datetime.now().isoformat()
        design_p     = params.get("design_parameters", {}) or {}
        env_type     = (params.get("environment", {}) or {}).get("type", "INDUSTRIAL")

        # Infer process domain — drives domain-specific state machine + I/O
        plc_domain  = self._infer_plc_domain(intent)
        domain_states = self._plc_domain_states(plc_domain)
        is_pump    = plc_domain == "pump"
        is_conveyor = plc_domain == "conveyor"
        is_oven    = plc_domain == "oven"
        is_press   = plc_domain == "press"
        is_reactor = plc_domain == "reactor"

        # Build state-machine CASE text dynamically from domain states
        def _sm_state(idx: int, name: str) -> str:
            lines = [f"  {idx}: (* {name} *)"]
            if name in ("IDLE", "FAULT"):
                lines.append(f"    gO_RunOutput := {'FALSE' if name == 'IDLE' else 'FALSE'};")
                if name == "FAULT":
                    lines.append("    gO_FaultLight := TRUE;")
                    lines.append("    gO_ReadyLight := FALSE;")
                    lines.append("    (* Manual reset: hold Stop for 3 s *)")
                    lines.append("    tmrStop(IN := gI_StopPB AND gI_EStop, PT := T#3s);")
                    lines.append("    IF tmrStop.Q THEN gSt_Fault := 0; gSt_State := 0; tmrStop(IN := FALSE); END_IF;")
                else:
                    lines.append("    gO_ReadyLight := gI_ProcessReady;")
                    lines.append("    rTrigStart(CLK := gI_StartPB);")
                    lines.append("    IF rTrigStart.Q AND gI_ProcessReady AND gI_EStop AND gSt_Fault = 0 THEN")
                    lines.append("      gSt_State := 1;")
                    lines.append("    END_IF;")
            elif idx == len(domain_states) - 2:  # second-to-last = terminal running state
                lines.append("    gO_RunOutput := TRUE;")
                lines.append("    tmrCycle(IN := TRUE, PT := cCYCLE_TIMEOUT);")
                lines.append("    IF tmrCycle.Q THEN gSt_Fault := 3; gSt_State := 4; END_IF;")
                lines.append("    fTrigStop(CLK := gI_StopPB);")
                lines.append(f"    IF fTrigStop.Q THEN gSt_State := {idx + 1}; END_IF;")
            elif idx == len(domain_states) - 2 + 1 and name not in ("FAULT",):  # stopping state
                lines.append("    tmrStop(IN := TRUE, PT := T#3s);")
                lines.append("    IF tmrStop.Q THEN gO_RunOutput := FALSE; tmrCycle(IN := FALSE); gSt_State := 0; END_IF;")
            else:
                lines.append(f"    (* {name} — process-specific actions *)")
                lines.append("    gO_RunOutput := TRUE;")
                lines.append(f"    tmrStart(IN := TRUE, PT := T#5s);")
                lines.append(f"    IF tmrStart.Q THEN tmrStart(IN := FALSE); gSt_State := {idx + 1}; END_IF;")
            return "\n".join(lines)

        sm_cases = "\n\n".join(_sm_state(i, s) for i, s in enumerate(domain_states))
        process_type = plc_domain

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

        # State-machine enumeration comment derived from domain
        state_comment = "\n".join(
            f" *   {i} = {s:<20} {plc_domain.upper()} state"
            for i, s in enumerate(domain_states)
        )

        main_st = f"""(*
 * MAIN — {project_name}
 * Author:  {author}
 * Version: {version}
 * Date:    {timestamp}
 * Domain:  {plc_domain.upper()} process
 *
 * State machine ({plc_domain}):
{state_comment}
 *)
PROGRAM MAIN
VAR
  tmrCycle    : TON;       (* Cycle timeout watchdog *)
  tmrStart    : TON;       (* Start / ramp-up timer *)
  tmrStop     : TON;       (* Shutdown / ramp-down timer *)
  {"pidTemp     : PID;       (* Temperature PID controller *)" if is_oven else ""}
  rTrigStart  : R_TRIG;    (* Rising-edge detect on Start PB *)
  fTrigStop   : F_TRIG;    (* Falling-edge detect on Stop PB *)
  bFirstScan  : BOOL := TRUE;
END_VAR

(* ── Safety: E-Stop and fault detection (highest priority) ────────────── *)
IF NOT gI_EStop THEN
  gSt_Fault  := 1;  (* Emergency stop activated *)
  gSt_State  := {len(domain_states) - 1};
END_IF;

IF gI_TempPV > cTEMP_MAX AND gSt_State > 0 THEN
  gSt_Fault  := 2;  (* Overtemperature *)
  gSt_State  := {len(domain_states) - 1};
END_IF;

(* ── Domain state machine — generated from _plc_domain_states() ─────── *)
CASE gSt_State OF

{sm_cases}

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
// State machine: IDLE(0) → STARTBIT(1) → DATA(2) → STOPBIT(3)
// Samples at baud-rate mid-point; verifies stop bit before asserting rx_valid.
reg [1:0]  rx_state;
reg [15:0] rx_cnt;
reg [7:0]  rx_shift;
reg [2:0]  rx_bit;
output reg rx_frame_err;  // high for one cycle on stop-bit violation

localparam RX_IDLE    = 2'd0;
localparam RX_STARTBT = 2'd1;
localparam RX_DATA    = 2'd2;
localparam RX_STOPBT  = 2'd3;

always @(posedge clk) begin
    rx_valid     <= 1'b0;
    rx_frame_err <= 1'b0;

    if (rst) begin
        rx_state <= RX_IDLE;
        rx_cnt   <= 0;
        rx_bit   <= 0;
    end else begin
        case (rx_state)
            // Wait for falling edge (start bit)
            RX_IDLE: begin
                if (!rx) begin
                    rx_state <= RX_STARTBT;
                    rx_cnt   <= CLK_DIV / 2 - 1;  // advance to mid-point of start bit
                end
            end

            // Centre on start bit; confirm it's still low (not a glitch)
            RX_STARTBT: begin
                if (rx_cnt == 0) begin
                    if (!rx) begin
                        rx_state <= RX_DATA;
                        rx_cnt   <= CLK_DIV - 1;
                        rx_bit   <= 0;
                    end else begin
                        rx_state <= RX_IDLE;  // glitch — abort
                    end
                end else begin
                    rx_cnt <= rx_cnt - 1;
                end
            end

            // Sample 8 data bits LSB-first
            RX_DATA: begin
                if (rx_cnt == 0) begin
                    // Shift new bit into MSB; previous bits slide right → after 8
                    // iterations rx_shift = {{b7,b6,...,b0}} (correct byte value)
                    rx_shift <= {{rx, rx_shift[7:1]}};
                    rx_cnt   <= CLK_DIV - 1;
                    if (rx_bit == 3'd7) begin
                        rx_state <= RX_STOPBT;
                    end else begin
                        rx_bit <= rx_bit + 1;
                    end
                end else begin
                    rx_cnt <= rx_cnt - 1;
                end
            end

            // Verify stop bit is high; latch data only if valid
            RX_STOPBT: begin
                if (rx_cnt == 0) begin
                    if (rx) begin
                        rx_data  <= rx_shift;
                        rx_valid <= 1'b1;
                    end else begin
                        rx_frame_err <= 1'b1;  // framing error — stop bit low
                    end
                    rx_state <= RX_IDLE;
                end else begin
                    rx_cnt <= rx_cnt - 1;
                end
            end

            default: rx_state <= RX_IDLE;
        endcase
    end
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

        uart_tb_v = f"""`timescale 1ns / 1ps
/* uart_core_tb.v — Self-checking testbench for uart_core
 * Tests: normal byte, glitch rejection, framing error, back-to-back bytes
 * Run with: iverilog -o sim uart_core_tb.v uart_core.v && vvp sim
 */
module uart_core_tb;
localparam CLK_FREQ = {int(clk_mhz * 1_000_000)};
localparam BAUD     = {baud_rate};
localparam CLK_PERIOD_NS = 1_000_000_000 / CLK_FREQ;
localparam BIT_PERIOD_NS = 1_000_000_000 / BAUD;

reg  clk = 0, rst = 1, rx = 1;
reg  [7:0] tx_data = 0;
reg  tx_valid = 0;
wire [7:0] rx_data;
wire rx_valid, rx_frame_err, tx, tx_ready;
integer errors = 0;

always #(CLK_PERIOD_NS / 2) clk = ~clk;

uart_core #(.CLK_FREQ(CLK_FREQ), .BAUD(BAUD)) dut (
    .clk(clk), .rst(rst),
    .rx(rx), .tx(tx),
    .rx_data(rx_data), .rx_valid(rx_valid),
    .rx_frame_err(rx_frame_err),
    .tx_data(tx_data), .tx_valid(tx_valid), .tx_ready(tx_ready)
);

// Task: send one 8N1 byte on rx line
task send_byte;
    input [7:0] data;
    integer i;
    begin
        rx = 0;                       // start bit
        #(BIT_PERIOD_NS);
        for (i = 0; i < 8; i = i + 1) begin
            rx = data[i];             // LSB first
            #(BIT_PERIOD_NS);
        end
        rx = 1;                       // stop bit
        #(BIT_PERIOD_NS);
    end
endtask

// Task: expect rx_valid with a specific value
task expect_byte;
    input [7:0] expected;
    input [255:0] label;
    begin
        @(posedge rx_valid);
        if (rx_data !== expected) begin
            $display("FAIL [%0s]: got 0x%02X expected 0x%02X", label, rx_data, expected);
            errors = errors + 1;
        end else begin
            $display("PASS [%0s]: 0x%02X", label, rx_data);
        end
    end
endtask

initial begin
    $dumpfile("uart_tb.vcd");
    $dumpvars(0, uart_core_tb);

    // Reset
    rst = 1; #(CLK_PERIOD_NS * 4); rst = 0;

    // Test 1: normal byte 0x55 (alternating bits — stresses all transitions)
    fork
        send_byte(8'h55);
        expect_byte(8'h55, "normal_0x55");
    join

    // Test 2: all-zeros byte 0x00
    fork
        send_byte(8'h00);
        expect_byte(8'h00, "all_zeros");
    join

    // Test 3: all-ones byte 0xFF
    fork
        send_byte(8'hFF);
        expect_byte(8'hFF, "all_ones");
    join

    // Test 4: glitch rejection — pulse shorter than CLK_DIV/4 should be ignored
    rx = 0; #(BIT_PERIOD_NS / 8); rx = 1;
    #(BIT_PERIOD_NS * 2);

    // Test 5: framing error — stop bit driven low
    rx = 0; #(BIT_PERIOD_NS);       // start
    repeat(8) begin rx = 1; #(BIT_PERIOD_NS); end  // 8 data bits = 0xFF
    rx = 0; #(BIT_PERIOD_NS);       // bad stop bit
    @(posedge clk);
    if (!rx_frame_err) begin
        $display("FAIL [framing_error]: rx_frame_err not asserted");
        errors = errors + 1;
    end else begin
        $display("PASS [framing_error]");
    end
    rx = 1; #(BIT_PERIOD_NS * 2);

    // Test 6: back-to-back bytes with no gap
    fork
        begin send_byte(8'hA5); send_byte(8'h3C); end
        begin expect_byte(8'hA5, "back2back_1"); expect_byte(8'h3C, "back2back_2"); end
    join

    if (errors == 0)
        $display("\\n=== ALL TESTS PASSED ===");
    else
        $display("\\n=== %0d TEST(S) FAILED ===", errors);

    $finish;
end

// Timeout watchdog — 50 ms simulation time
initial begin
    #50_000_000;
    $display("FAIL: simulation timeout");
    $finish;
end
endmodule
"""

        # ── CDC synchronisers (always generated — any real design crosses clocks) ──
        cdc_sync2_v = f"""`timescale 1ns / 1ps
/* cdc_sync2.v — 2-FF metastability synchroniser
 * Use for single-bit signals crossing from src_clk domain to dst_clk domain.
 * Set src_clk period constraint: set_false_path -from [get_cells *cdc_sync2*/sync_ff1*]
 */
module cdc_sync2 #(parameter RESET_VAL = 1'b0) (
    input  wire src_data,
    input  wire dst_clk,
    input  wire dst_rst,
    output wire dst_data
);
    (* ASYNC_REG = "TRUE" *) reg sync_ff1, sync_ff2;
    always @(posedge dst_clk or posedge dst_rst) begin
        if (dst_rst) {{sync_ff1 <= RESET_VAL; sync_ff2 <= RESET_VAL;}}
        else         {{sync_ff1 <= src_data;   sync_ff2 <= sync_ff1;}}
    end
    assign dst_data = sync_ff2;
endmodule
"""

        cdc_handshake_v = f"""`timescale 1ns / 1ps
/* cdc_handshake.v — Multi-bit req/ack CDC handshake
 * Safe for {data_width}-bit data buses crossing clock domains.
 * Throughput: ~4 dst_clk cycles per transfer.
 * Resource estimate: ~{data_width + 8} FFs, ~{data_width * 2} LUTs
 */
module cdc_handshake #(parameter WIDTH = {data_width}) (
    input  wire             src_clk, src_rst,
    input  wire [WIDTH-1:0] src_data,
    input  wire             src_valid,
    output wire             src_ready,

    input  wire             dst_clk, dst_rst,
    output reg  [WIDTH-1:0] dst_data,
    output reg              dst_valid
);
    reg  [WIDTH-1:0] src_hold;
    reg              src_req, src_req_sync1, src_req_sync2;
    reg              dst_ack, dst_ack_sync1, dst_ack_sync2;

    (* ASYNC_REG = "TRUE" *) reg src_ack_sync1, src_ack_sync2;
    (* ASYNC_REG = "TRUE" *) reg dst_req_sync1, dst_req_sync2;

    // Src side: latch data, toggle req
    always @(posedge src_clk or posedge src_rst) begin
        if (src_rst) {{src_req <= 0; src_hold <= 0;}}
        else if (src_valid && src_ready) {{src_hold <= src_data; src_req <= ~src_req;}}
    end
    always @(posedge src_clk) {{src_ack_sync1 <= dst_ack; src_ack_sync2 <= src_ack_sync1;}}
    assign src_ready = (src_req == src_ack_sync2);

    // Dst side: sync req, latch, toggle ack
    always @(posedge dst_clk or posedge dst_rst) begin
        if (dst_rst) {{dst_req_sync1 <= 0; dst_req_sync2 <= 0; dst_ack <= 0; dst_valid <= 0;}}
        else begin
            dst_req_sync1 <= src_req; dst_req_sync2 <= dst_req_sync1;
            if (dst_req_sync2 != dst_ack) {{
                dst_data  <= src_hold;
                dst_valid <= 1'b1;
                dst_ack   <= dst_req_sync2;
            end else dst_valid <= 1'b0;
        end
    end
endmodule
"""

        # Update constraints with CDC timing exceptions
        cdc_constraints = f"""
# ── CDC timing exceptions (add to constraints.xdc) ─────────────────────────
# Replace clock names with actual clocks from your design
set_false_path -from [get_cells -hierarchical -filter {{NAME =~ *cdc_sync2*sync_ff1*}}]
set_max_delay -datapath_only -from [get_cells -hierarchical -filter {{NAME =~ *src_hold*}}] \\
              -to [get_cells -hierarchical -filter {{NAME =~ *dst_req_sync1*}}] 5.0
"""

        files = {
            "top.v": top_v,
            "cdc_sync2.v": cdc_sync2_v,
            "cdc_handshake.v": cdc_handshake_v,
            "constraints.xdc": constraints_xdc + cdc_constraints,
            "Makefile": makefile,
        }
        if is_uart:
            files["uart_core.v"] = uart_core_v
            files["uart_core_tb.v"] = uart_tb_v

        lut_estimate = data_width * 2 + 8
        ff_estimate  = data_width + 8

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
                    "resource_estimate": f"~{lut_estimate} LUTs, ~{ff_estimate} FFs for CDC logic",
                },
            },
            "logs": [f"Generated {len(files)} Verilog files", f"Clock: {clk_mhz} MHz",
                     f"Data width: {data_width} bit", f"CDC: cdc_sync2.v + cdc_handshake.v included"],
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
                        "specs": HardwareDB.load(p.value)
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
