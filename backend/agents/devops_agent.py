"""
Production DevOps Diagnostics Agent

Features:
- Multi-cloud health monitoring (AWS, Azure, GCP)
- Docker/Kubernetes diagnostics
- CI/CD pipeline validation
- Security vulnerability scanning
- Performance profiling
- Log aggregation and analysis
- Alert management
- Infrastructure as Code validation
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json
import logging
import os
import re
import subprocess
import shutil
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    DIGITAL_OCEAN = "digital_ocean"
    LINODE = "linode"


class ContainerRuntime(Enum):
    """Supported container runtimes."""
    DOCKER = "docker"
    CONTAINERD = "containerd"
    PODMAN = "podman"
    CRIO = "crio"


class Severity(Enum):
    """Issue severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class HealthCheck:
    """Health check result."""
    name: str
    status: str  # healthy, degraded, unhealthy
    message: str
    severity: Severity
    timestamp: datetime
    metrics: Dict[str, Any]
    recommendations: List[str]


@dataclass
class SecurityIssue:
    """Security vulnerability."""
    id: str
    title: str
    description: str
    severity: Severity
    cve_id: Optional[str]
    fix_available: bool
    fix_command: Optional[str]


class DevOpsAgent:
    """
    Production-grade DevOps diagnostics agent.
    
    Provides comprehensive system health monitoring, security auditing,
    and infrastructure validation across multiple platforms.
    """
    
    # Dockerfile security audit rules
    AUDIT_RULES = [
        {"check": "USER", "message": "Security: Container runs as root (missing USER instruction)", "type": "missing"},
        {"check": "latest", "message": "Avoid: Using 'latest' tag for base images (use specific version)", "type": "avoid"},
        {"check": "ADD", "message": "Caution: ADD can extract archives unexpectedly, prefer COPY", "type": "caution"},
        {"check": "apt-get update", "message": "Security: apt-get update without upgrade may miss security patches", "type": "warning"},
        {"check": "curl | sh", "message": "Security: Piping curl to shell is dangerous", "type": "critical"},
        {"check": "wget | sh", "message": "Security: Piping wget to shell is dangerous", "type": "critical"},
        {"check": "sudo", "message": "Caution: sudo in container indicates potential privilege issues", "type": "caution"},
        {"check": "EXPOSE 22", "message": "Security: SSH port exposed, consider using docker exec instead", "type": "warning"},
        {"check": "EXPOSE 23", "message": "Security: Telnet port exposed (insecure protocol)", "type": "critical"},
        {"check": "EXPOSE 21", "message": "Security: FTP port exposed (insecure protocol)", "type": "warning"},
        {"check": "HEALTHCHECK", "message": "Missing: No HEALTHCHECK defined for container", "type": "missing"},
        {"check": "LABEL maintainer", "message": "Best Practice: Missing maintainer label", "type": "info"},
        {"check": ".dockerignore", "message": "Best Practice: Missing .dockerignore file", "type": "info"},
        {"check": "--no-cache", "message": "Optimization: Consider using --no-cache for production builds", "type": "optimization"},
    ]
    
    # Firmware CI/CD Templates — PlatformIO-based (C++/C/Rust/Zig firmware projects)
    PLATFORMIO_GITHUB_ACTIONS = """name: Firmware CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Cache PlatformIO
      uses: actions/cache@v3
      with:
        path: ~/.platformio
        key: ${{ runner.os }}-pio-${{ hashFiles('platformio.ini') }}

    - name: Install PlatformIO
      run: pip install platformio

    - name: Build firmware
      run: pio run

    - name: Run native unit tests
      run: pio test -e native

    - name: Upload firmware artifact
      uses: actions/upload-artifact@v3
      with:
        name: firmware-${{ github.sha }}
        path: .pio/build/*/firmware.*
        retention-days: 30

  static-analysis:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Install cppcheck
      run: sudo apt-get install -y cppcheck

    - name: Run cppcheck
      run: cppcheck --error-exitcode=1 --enable=warning,performance,portability src/

    - name: Check for unresolved TODOs
      run: |
        count=$(grep -r "TODO\\|FIXME\\|HACK" src/ | wc -l || true)
        echo "Found $count TODO/FIXME comments"
"""

    PLATFORMIO_GITLAB_CI = """stages:
  - build
  - test
  - analysis

variables:
  PIO_CACHE_DIR: "$CI_PROJECT_DIR/.pio"

cache:
  paths:
    - .pio/
    - ~/.platformio/

build:firmware:
  stage: build
  image: python:3.11-slim
  before_script:
    - pip install platformio
  script:
    - pio run
  artifacts:
    paths:
      - .pio/build/*/firmware.*
    expire_in: 1 week

test:native:
  stage: test
  image: python:3.11-slim
  before_script:
    - pip install platformio
  script:
    - pio test -e native

analysis:cppcheck:
  stage: analysis
  image: ubuntu:22.04
  before_script:
    - apt-get update -qq && apt-get install -y cppcheck
  script:
    - cppcheck --error-exitcode=1 --enable=warning,performance,portability src/
"""

    MICROPYTHON_GITHUB_ACTIONS = """name: MicroPython CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  syntax-check:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Syntax check all .py files
      run: find . -name "*.py" -not -path "./.git/*" -exec python -m py_compile {} \\;

    - name: Package for mpremote deployment
      run: |
        mkdir -p dist
        cp *.py dist/ 2>/dev/null || true
        echo "Flash: mpremote connect <port> fs cp dist/*.py :" > dist/DEPLOY.txt

    - name: Upload flash package
      uses: actions/upload-artifact@v3
      with:
        name: micropython-${{ github.sha }}
        path: dist/
        retention-days: 14
"""

    CIRCUITPYTHON_GITHUB_ACTIONS = """name: CircuitPython CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Syntax check
      run: find . -name "*.py" -not -path "./.git/*" -exec python -m py_compile {} \\;

    - name: Install circup (library manager)
      run: pip install circup

    - name: Package deployment
      run: |
        mkdir -p dist
        cp code.py dist/
        [ -f boot.py ] && cp boot.py dist/ || true
        [ -f requirements.txt ] && cp requirements.txt dist/ || true
        echo "Deploy: copy dist/*.py to CIRCUITPY USB drive" > dist/DEPLOY.txt

    - name: Upload deploy package
      uses: actions/upload-artifact@v3
      with:
        name: circuitpython-${{ github.sha }}
        path: dist/
        retention-days: 14
"""

    # Per-platform flash instructions keyed by normalized platform name
    FLASH_INSTRUCTIONS: Dict[str, Dict[str, str]] = {
        "ESP32": {
            "tool": "esptool.py",
            "install": "pip install esptool",
            "erase": "esptool.py --chip esp32 --port {port} erase_flash",
            "flash": "esptool.py --chip esp32 --port {port} --baud 460800 write_flash -z 0x1000 firmware.bin",
            "verify": "esptool.py --chip esp32 --port {port} verify_flash 0x1000 firmware.bin",
            "port_linux": "/dev/ttyUSB0",
            "port_mac": "/dev/cu.usbserial-*",
            "notes": "Hold BOOT button during flashing if device does not auto-enter bootloader mode.",
        },
        "ESP8266": {
            "tool": "esptool.py",
            "install": "pip install esptool",
            "erase": "esptool.py --chip esp8266 --port {port} erase_flash",
            "flash": "esptool.py --chip esp8266 --port {port} --baud 460800 write_flash 0x00000 firmware.bin",
            "port_linux": "/dev/ttyUSB0",
            "port_mac": "/dev/cu.usbserial-*",
            "notes": "GPIO0 must be LOW (GND) during boot to enter flash mode.",
        },
        "STM32": {
            "tool": "OpenOCD or dfu-util",
            "install": "sudo apt install openocd  # macOS: brew install openocd",
            "flash_openocd": "openocd -f interface/stlink.cfg -f target/{target}.cfg -c 'program firmware.bin verify reset exit 0x08000000'",
            "flash_dfu": "dfu-util -d 0483:df11 -a 0 -s 0x08000000:leave -D firmware.bin",
            "notes": "Use ST-Link V2 for best results. DFU mode: BOOT0=HIGH (3.3V) then RESET.",
        },
        "RP2040": {
            "tool": "UF2 drag-and-drop (no installer needed)",
            "install": "No installation required for UF2. picotool: brew install picotool",
            "flash": "Hold BOOTSEL while connecting USB → BOOTFS drive appears → drag firmware.uf2",
            "flash_picotool": "picotool load firmware.uf2 && picotool reboot",
            "notes": "PlatformIO build produces .uf2 automatically. elf2uf2 included in toolchain.",
        },
        "Arduino": {
            "tool": "avrdude or arduino-cli",
            "install": "Included with Arduino IDE, or: sudo apt install avrdude",
            "flash_avrdude": "avrdude -p atmega328p -c arduino -P {port} -b 115200 -U flash:w:firmware.hex:i",
            "flash_cli": "arduino-cli upload -p {port} --fqbn arduino:avr:uno .",
            "notes": "Adjust FQBN (--fqbn) for your specific board variant (Mega, Leonardo, etc.).",
        },
        "MicroPython": {
            "tool": "mpremote + esptool.py (for firmware flash)",
            "install": "pip install mpremote esptool",
            "flash_firmware_esp32": "esptool.py --chip esp32 --port {port} write_flash -z 0x1000 micropython-esp32.bin",
            "deploy_files": "mpremote connect {port} fs cp main.py boot.py :",
            "run": "mpremote connect {port} run main.py",
            "repl": "mpremote connect {port}",
            "notes": "Flash MicroPython .bin firmware first, then copy .py application files.",
        },
        "CircuitPython": {
            "tool": "USB mass storage (no tool needed) + circup",
            "install": "pip install circup",
            "deploy": "Copy code.py (and boot.py if present) to the CIRCUITPY USB drive that appears when device is connected",
            "install_libs": "circup install -r requirements.txt",
            "notes": "Device auto-runs code.py on startup. Serial REPL available at /dev/ttyACM0.",
        },
    }
    
    def __init__(self, llm_provider=None):
        self.name = "DevOpsAgent"
        self.llm_provider = llm_provider
        self.health_history: List[HealthCheck] = []
        self.alert_thresholds = {
            "cpu_percent": 80,
            "memory_percent": 85,
            "disk_percent": 90,
            "response_time_ms": 1000,
        }
        
    def generate_deployment_plan(self, code: Any, geometry: Any) -> Dict[str, Any]:
        """
        Generate firmware flash plan + CI pipeline + manufacturing handoff for a hardware project.

        Called by devops_node after firmware codegen. Firmware projects never use Docker.

        Args:
            code: Generated firmware/code artifact — dict with {files, language, platform, environment_type}
            geometry: Geometry tree from the pipeline (list of nodes or dict)

        Returns:
            {plan, ci_cd_config, flash_instructions, manufacturing_handoff, compliance_checklist}
        """
        language = "cpp"
        platform_name = "ESP32"
        environment_type = "GROUND"
        files: Dict[str, Any] = {}

        if isinstance(code, dict):
            files = code.get("files", {}) or {}
            lang_raw = (code.get("language", "") or "").lower()
            platform_raw = (code.get("platform", "") or "").upper()
            environment_type = (code.get("environment_type", "GROUND") or "GROUND").upper()

            if "micropython" in lang_raw or "main.py" in files:
                language = "micropython"
                platform_name = platform_raw or "ESP32"
            elif "circuitpython" in lang_raw or "code.py" in files:
                language = "circuitpython"
                platform_name = platform_raw or "RP2040"
            elif "rust" in lang_raw:
                language = "rust"
                platform_name = platform_raw or "STM32"
            elif "zig" in lang_raw:
                language = "zig"
                platform_name = platform_raw or "STM32"
            else:
                language = "cpp"
                platform_name = platform_raw or "ESP32"

        platform_key = self._normalize_platform(platform_name)
        ci_result = self._generate_firmware_ci(language, platform_key)
        flash = self._get_flash_instructions(language, platform_key)
        mfg_handoff = self._generate_manufacturing_handoff(geometry, files)
        compliance = self._generate_compliance_checklist(environment_type)

        n_components = len(geometry) if isinstance(geometry, list) else 0

        plan = {
            "platform": platform_key,
            "language": language,
            "environment_type": environment_type,
            "geometry_nodes": n_components,
            "ci_platform": "github",
            "ci_filename": ci_result.get("filename", ".github/workflows/firmware-ci.yml"),
            "docker_enabled": False,
            "steps": ci_result.get("steps", []),
        }

        return {
            "plan": plan,
            "ci_cd_config": {
                "content": ci_result.get("config", ""),
                "filename": ci_result.get("filename", ""),
                "platform": "github",
            },
            "flash_instructions": flash,
            "manufacturing_handoff": mfg_handoff,
            "compliance_checklist": compliance,
        }

    def _normalize_platform(self, raw: str) -> str:
        """Normalize any platform string to a FLASH_INSTRUCTIONS key."""
        r = raw.upper()
        if "ESP32" in r or "ESP-IDF" in r:
            return "ESP32"
        if "ESP8266" in r:
            return "ESP8266"
        if "STM32" in r or "NUCLEO" in r or "DISCO" in r:
            return "STM32"
        if "RP2040" in r or "PICO" in r or "RASPBERRY" in r:
            return "RP2040"
        if "ARDUINO" in r or "AVR" in r or "NANO" in r or "UNO" in r or "MEGA" in r:
            return "Arduino"
        if "MICROPYTHON" in r:
            return "MicroPython"
        if "CIRCUITPYTHON" in r:
            return "CircuitPython"
        return raw if raw else "ESP32"

    def _generate_firmware_ci(self, language: str, platform: str) -> Dict[str, Any]:
        """Select the correct firmware CI template for a language/platform combination."""
        lang = language.lower()
        if lang == "micropython":
            return {
                "config": self.MICROPYTHON_GITHUB_ACTIONS,
                "filename": ".github/workflows/micropython-ci.yml",
                "steps": ["syntax_check", "package_deploy", "upload_artifact"],
                "status": "generated",
            }
        if lang == "circuitpython":
            return {
                "config": self.CIRCUITPYTHON_GITHUB_ACTIONS,
                "filename": ".github/workflows/circuitpython-ci.yml",
                "steps": ["syntax_check", "circup_check", "package_deploy", "upload_artifact"],
                "status": "generated",
            }
        # C++, C, Rust, Zig → PlatformIO
        return {
            "config": self.PLATFORMIO_GITHUB_ACTIONS,
            "filename": ".github/workflows/firmware-ci.yml",
            "steps": ["platformio_build", "native_unit_tests", "cppcheck", "upload_artifact"],
            "status": "generated",
        }

    def _get_flash_instructions(self, language: str, platform: str) -> Dict[str, str]:
        """Return flash instructions for the given language/platform."""
        if language == "micropython":
            return self.FLASH_INSTRUCTIONS.get("MicroPython", {})
        if language == "circuitpython":
            return self.FLASH_INSTRUCTIONS.get("CircuitPython", {})
        return self.FLASH_INSTRUCTIONS.get(platform, self.FLASH_INSTRUCTIONS.get("ESP32", {}))

    def _generate_manufacturing_handoff(self, geometry: Any, files: Dict[str, Any]) -> Dict[str, Any]:
        """Generate manufacturing handoff checklist from available output files."""
        file_keys = [k.lower() for k in (files or {}).keys()]
        has_step = any("step" in f or ".stp" in f for f in file_keys)
        has_stl = any(f.endswith(".stl") for f in file_keys)
        has_gcode = any("gcode" in f or f.endswith(".nc") or f.endswith(".ngc") for f in file_keys)
        has_bom = any("bom" in f or "bill_of_material" in f for f in file_keys)
        has_asm_drawing = any("assembly" in f or "drawing" in f or "asm" in f for f in file_keys)

        checklist = [
            {
                "item": "STEP file (.stp / .step)",
                "ready": has_step,
                "required_for": "CNC machining, injection molding, contract manufacturing",
            },
            {
                "item": "STL file",
                "ready": has_stl,
                "required_for": "3D printing (FDM, SLA, SLS)",
            },
            {
                "item": "GCode / NC program",
                "ready": has_gcode,
                "required_for": "CNC machining — direct machine upload",
            },
            {
                "item": "Bill of Materials CSV",
                "ready": has_bom,
                "required_for": "JLCPCB, Digi-Key, Mouser procurement",
            },
            {
                "item": "Assembly drawing (PDF)",
                "ready": has_asm_drawing,
                "required_for": "Contract manufacturer assembly instructions",
            },
            {
                "item": "Tolerance callouts",
                "ready": False,
                "required_for": "Precision machining — must be added to drawings manually",
            },
        ]

        n_ready = sum(1 for c in checklist if c["ready"])
        return {
            "checklist": checklist,
            "ready_count": n_ready,
            "total_count": len(checklist),
            "jlcpcb_ready": has_bom,
            "cnc_ready": has_step and has_gcode,
            "print_ready": has_stl,
        }

    def _generate_compliance_checklist(self, environment_type: str) -> Dict[str, Any]:
        """Generate compliance standards checklist based on deployment environment."""
        env = (environment_type or "GROUND").upper()

        items = [
            {"standard": "RoHS 3 (EU 2015/863)", "description": "Restriction of hazardous substances in electrical equipment", "applies": True},
            {"standard": "WEEE (EU 2012/19/EU)", "description": "Waste electrical and electronic equipment directive", "applies": True},
        ]

        if env in ("GROUND", "INDOOR", "INDUSTRIAL", "OUTDOOR"):
            items += [
                {"standard": "CE Marking", "description": "Conformité Européenne — EU market entry", "applies": True},
                {"standard": "FCC Part 15", "description": "US electromagnetic emissions for unintentional radiators", "applies": True},
                {"standard": "IEC 61000 (EMC)", "description": "Electromagnetic compatibility — emissions and immunity", "applies": True},
            ]

        if env == "INDUSTRIAL":
            items += [
                {"standard": "IP54 minimum", "description": "Dust + splash protection per IEC 60529", "applies": True},
                {"standard": "IEC 60068", "description": "Environmental testing — temperature, humidity, vibration", "applies": True},
                {"standard": "UL 61010-1", "description": "Safety for measurement, control and laboratory equipment", "applies": True},
                {"standard": "ATEX / IECEx", "description": "Explosive atmosphere rating (if applicable)", "applies": False},
            ]

        if env in ("AEROSPACE", "AVIONICS"):
            items += [
                {"standard": "DO-178C", "description": "Software considerations in airborne systems certification", "applies": True},
                {"standard": "RTCA DO-160G", "description": "Environmental conditions and test procedures for airborne equipment", "applies": True},
                {"standard": "MIL-STD-461G", "description": "EMC requirements for military / aerospace equipment", "applies": True},
                {"standard": "MIL-STD-810H", "description": "Environmental engineering — temperature, shock, vibration, humidity", "applies": True},
            ]

        if env == "SPACE":
            items += [
                {"standard": "ECSS-E-ST-20C", "description": "Space product assurance — electrical and electronic", "applies": True},
                {"standard": "NASA-STD-8739.8", "description": "Workmanship standard for crimping, interconnecting cables", "applies": True},
                {"standard": "NASA SP-R-0022A", "description": "Outgassing requirements for materials in vacuum", "applies": True},
                {"standard": "MIL-STD-1540E", "description": "Product verification requirements for launch, upper stage, and space vehicles", "applies": True},
                {"standard": "Total Ionizing Dose (TID) rating", "description": "Radiation hardness for electronics in orbit", "applies": True},
            ]

        if env == "UNDERWATER":
            items += [
                {"standard": "IP68", "description": "Dust-tight + continuous immersion protection per IEC 60529", "applies": True},
                {"standard": "NEMA 6P", "description": "Waterproof enclosure standard — submersible", "applies": True},
                {"standard": "DNV GL", "description": "Marine / subsea equipment classification (offshore)", "applies": False},
            ]

        if env == "MEDICAL":
            items += [
                {"standard": "IEC 60601-1", "description": "Medical electrical equipment — general safety requirements", "applies": True},
                {"standard": "ISO 14971", "description": "Risk management for medical devices", "applies": True},
                {"standard": "FDA 21 CFR Part 11", "description": "Electronic records and signatures for FDA submissions", "applies": True},
                {"standard": "IEC 62304", "description": "Medical device software — software life cycle processes", "applies": True},
            ]

        return {
            "environment": environment_type,
            "standards": items,
            "total": len(items),
            "mandatory_count": sum(1 for i in items if i["applies"]),
            "note": "Compliance testing must be performed by an accredited third-party laboratory before market entry.",
        }

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute DevOps diagnostics.
        
        Args:
            params: {
                "action": str,  # health_check, audit, generate_pipeline, analyze_logs, 
                               # monitor_resources, security_scan, iac_validate
                ... action-specific parameters
            }
        """
        action = params.get("action", "health_check")
        
        actions = {
            "health_check": self._action_health_check,
            "audit_dockerfile": self._action_audit_dockerfile,
            "audit_compose": self._action_audit_compose,
            "generate_pipeline": self._action_generate_pipeline,
            "analyze_logs": self._action_analyze_logs,
            "monitor_resources": self._action_monitor_resources,
            "security_scan": self._action_security_scan,
            "iac_validate": self._action_iac_validate,
            "docker_diagnostics": self._action_docker_diagnostics,
            "k8s_diagnostics": self._action_k8s_diagnostics,
            "network_check": self._action_network_check,
            "dependency_check": self._action_dependency_check,
        }
        
        if action not in actions:
            return {
                "status": "error",
                "message": f"Unknown action: {action}",
                "available_actions": list(actions.keys())
            }
        
        return actions[action](params)
    
    def _action_health_check(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive system health check."""
        checks = []
        
        # System resources
        disk_check = self._check_disk_space()
        checks.append(disk_check)
        
        memory_check = self._check_memory()
        checks.append(memory_check)
        
        cpu_check = self._check_cpu()
        checks.append(cpu_check)
        
        # Docker health
        docker_check = self._check_docker()
        checks.append(docker_check)
        
        # Network connectivity
        network_check = self._check_network()
        checks.append(network_check)
        
        # Service health
        services = params.get("services", [])
        for service in services:
            service_check = self._check_service(service)
            checks.append(service_check)
        
        # Determine overall status
        critical_count = sum(1 for c in checks if c.severity == Severity.CRITICAL)
        unhealthy_count = sum(1 for c in checks if c.status == "unhealthy")
        
        if critical_count > 0 or unhealthy_count > 2:
            overall_status = "critical"
        elif unhealthy_count > 0:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        # Store in history
        self.health_history.extend(checks)
        
        return {
            "status": "success",
            "overall_status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "message": c.message,
                    "severity": c.severity.value,
                    "metrics": c.metrics,
                    "recommendations": c.recommendations
                }
                for c in checks
            ],
            "summary": {
                "total": len(checks),
                "healthy": sum(1 for c in checks if c.status == "healthy"),
                "degraded": sum(1 for c in checks if c.status == "degraded"),
                "unhealthy": sum(1 for c in checks if c.status == "unhealthy"),
                "critical": critical_count
            }
        }
    
    def _check_disk_space(self) -> HealthCheck:
        """Check disk space usage."""
        try:
            total, used, free = shutil.disk_usage("/")
            percent_used = (used / total) * 100
            
            if percent_used > self.alert_thresholds["disk_percent"]:
                status = "unhealthy"
                severity = Severity.HIGH
                message = f"Disk usage critical: {percent_used:.1f}%"
            elif percent_used > 80:
                status = "degraded"
                severity = Severity.MEDIUM
                message = f"Disk usage high: {percent_used:.1f}%"
            else:
                status = "healthy"
                severity = Severity.INFO
                message = f"Disk usage normal: {percent_used:.1f}%"
            
            return HealthCheck(
                name="disk_space",
                status=status,
                message=message,
                severity=severity,
                timestamp=datetime.now(),
                metrics={
                    "total_gb": round(total / (1024**3), 2),
                    "used_gb": round(used / (1024**3), 2),
                    "free_gb": round(free / (1024**3), 2),
                    "percent_used": round(percent_used, 1)
                },
                recommendations=[
                    "Clean up old logs" if percent_used > 80 else None,
                    "Archive old data" if percent_used > 90 else None
                ]
            )
        except Exception as e:
            return HealthCheck(
                name="disk_space",
                status="error",
                message=f"Failed to check disk: {str(e)}",
                severity=Severity.HIGH,
                timestamp=datetime.now(),
                metrics={},
                recommendations=[]
            )
    
    def _check_memory(self) -> HealthCheck:
        """Check memory usage."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            
            if mem.percent > self.alert_thresholds["memory_percent"]:
                status = "unhealthy"
                severity = Severity.HIGH
                message = f"Memory usage critical: {mem.percent}%"
            elif mem.percent > 75:
                status = "degraded"
                severity = Severity.MEDIUM
                message = f"Memory usage high: {mem.percent}%"
            else:
                status = "healthy"
                severity = Severity.INFO
                message = f"Memory usage normal: {mem.percent}%"
            
            return HealthCheck(
                name="memory",
                status=status,
                message=message,
                severity=severity,
                timestamp=datetime.now(),
                metrics={
                    "total_gb": round(mem.total / (1024**3), 2),
                    "available_gb": round(mem.available / (1024**3), 2),
                    "percent_used": mem.percent
                },
                recommendations=[
                    "Check for memory leaks" if mem.percent > 85 else None,
                    "Restart services" if mem.percent > 90 else None
                ]
            )
        except ImportError:
            return HealthCheck(
                name="memory",
                status="unknown",
                message="psutil not available for memory check",
                severity=Severity.INFO,
                timestamp=datetime.now(),
                metrics={},
                recommendations=["Install psutil for memory monitoring"]
            )
    
    def _check_cpu(self) -> HealthCheck:
        """Check CPU usage."""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            if cpu_percent > self.alert_thresholds["cpu_percent"]:
                status = "degraded"
                severity = Severity.MEDIUM
                message = f"CPU usage high: {cpu_percent:.1f}%"
            else:
                status = "healthy"
                severity = Severity.INFO
                message = f"CPU usage normal: {cpu_percent:.1f}%"
            
            return HealthCheck(
                name="cpu",
                status=status,
                message=message,
                severity=severity,
                timestamp=datetime.now(),
                metrics={
                    "usage_percent": cpu_percent,
                    "cores": cpu_count,
                    "load_avg": os.getloadavg() if hasattr(os, 'getloadavg') else None
                },
                recommendations=[
                    "Investigate high CPU processes" if cpu_percent > 80 else None
                ]
            )
        except ImportError:
            return HealthCheck(
                name="cpu",
                status="unknown",
                message="psutil not available for CPU check",
                severity=Severity.INFO,
                timestamp=datetime.now(),
                metrics={},
                recommendations=["Install psutil for CPU monitoring"]
            )
    
    def _check_docker(self) -> HealthCheck:
        """Check Docker engine status."""
        if not shutil.which("docker"):
            return HealthCheck(
                name="docker",
                status="unknown",
                message="Docker not installed",
                severity=Severity.INFO,
                timestamp=datetime.now(),
                metrics={},
                recommendations=[]
            )
        
        try:
            result = subprocess.run(
                ["docker", "info", "--format", "{{json .}}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                info = json.loads(result.stdout)
                containers_running = info.get("ContainersRunning", 0)
                containers_total = info.get("Containers", 0)
                images = info.get("Images", 0)
                
                return HealthCheck(
                    name="docker",
                    status="healthy",
                    message=f"Docker running ({containers_running} containers)",
                    severity=Severity.INFO,
                    timestamp=datetime.now(),
                    metrics={
                        "containers_running": containers_running,
                        "containers_total": containers_total,
                        "images": images,
                        "version": info.get("ServerVersion", "unknown")
                    },
                    recommendations=[]
                )
            else:
                return HealthCheck(
                    name="docker",
                    status="unhealthy",
                    message="Docker daemon not responding",
                    severity=Severity.HIGH,
                    timestamp=datetime.now(),
                    metrics={},
                    recommendations=["Check Docker daemon status", "Run 'dockerd' or restart Docker service"]
                )
        except Exception as e:
            return HealthCheck(
                name="docker",
                status="error",
                message=f"Docker check failed: {str(e)}",
                severity=Severity.HIGH,
                timestamp=datetime.now(),
                metrics={},
                recommendations=[]
            )
    
    def _check_network(self) -> HealthCheck:
        """Check network connectivity."""
        try:
            # Try to ping common hosts
            hosts = ["8.8.8.8", "1.1.1.1"]
            reachable = 0
            latencies = []
            
            for host in hosts:
                try:
                    result = subprocess.run(
                        ["ping", "-c", "1", "-W", "2", host],
                        capture_output=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        reachable += 1
                        # Extract latency
                        match = re.search(r'time=([0-9.]+) ms', result.stdout.decode())
                        if match:
                            latencies.append(float(match.group(1)))
                except:
                    pass
            
            avg_latency = sum(latencies) / len(latencies) if latencies else None
            
            if reachable == len(hosts):
                status = "healthy"
                message = f"Network connectivity OK ({avg_latency:.1f}ms avg)"
                severity = Severity.INFO
            elif reachable > 0:
                status = "degraded"
                message = f"Partial connectivity ({reachable}/{len(hosts)} hosts)"
                severity = Severity.MEDIUM
            else:
                status = "unhealthy"
                message = "No network connectivity"
                severity = Severity.CRITICAL
            
            return HealthCheck(
                name="network",
                status=status,
                message=message,
                severity=severity,
                timestamp=datetime.now(),
                metrics={
                    "hosts_tested": len(hosts),
                    "hosts_reachable": reachable,
                    "avg_latency_ms": avg_latency
                },
                recommendations=["Check network configuration"] if reachable == 0 else []
            )
        except Exception as e:
            return HealthCheck(
                name="network",
                status="unknown",
                message=f"Network check failed: {str(e)}",
                severity=Severity.INFO,
                timestamp=datetime.now(),
                metrics={},
                recommendations=[]
            )
    
    def _check_service(self, service: Dict[str, Any]) -> HealthCheck:
        """Check individual service health."""
        name = service.get("name", "unknown")
        url = service.get("url")
        port = service.get("port")
        
        try:
            if url:
                import urllib.request
                import socket
                
                start = datetime.now()
                urllib.request.urlopen(url, timeout=5)
                elapsed = (datetime.now() - start).total_seconds() * 1000
                
                return HealthCheck(
                    name=f"service_{name}",
                    status="healthy",
                    message=f"{name} responding ({elapsed:.0f}ms)",
                    severity=Severity.INFO,
                    timestamp=datetime.now(),
                    metrics={"response_time_ms": elapsed},
                    recommendations=[]
                )
            elif port:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex(("localhost", port))
                sock.close()
                
                if result == 0:
                    return HealthCheck(
                        name=f"service_{name}",
                        status="healthy",
                        message=f"{name} port {port} open",
                        severity=Severity.INFO,
                        timestamp=datetime.now(),
                        metrics={"port": port},
                        recommendations=[]
                    )
                else:
                    return HealthCheck(
                        name=f"service_{name}",
                        status="unhealthy",
                        message=f"{name} port {port} closed",
                        severity=Severity.HIGH,
                        timestamp=datetime.now(),
                        metrics={"port": port},
                        recommendations=[f"Check if {name} is running"]
                    )
        except Exception as e:
            return HealthCheck(
                name=f"service_{name}",
                status="error",
                message=f"{name} check failed: {str(e)}",
                severity=Severity.MEDIUM,
                timestamp=datetime.now(),
                metrics={},
                recommendations=[]
            )
    
    def _action_audit_dockerfile(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Audit Dockerfile for security issues."""
        path = params.get("path", "Dockerfile")
        strict = params.get("strict", False)
        
        if not os.path.exists(path):
            return {
                "status": "error",
                "message": f"Dockerfile not found: {path}"
            }
        
        try:
            with open(path, 'r') as f:
                content = f.read()
                lines = content.split('\n')
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to read Dockerfile: {str(e)}"
            }
        
        issues = []
        score = 100
        
        for rule in self.AUDIT_RULES:
            check = rule["check"]
            message = rule["message"]
            rule_type = rule.get("type", "warning")
            
            if rule_type == "missing":
                if check not in content:
                    issues.append({
                        "rule": check,
                        "message": message,
                        "severity": "high" if "Security" in message else "info",
                        "line": None
                    })
                    score -= 10 if "Security" in message else 2
            elif rule_type in ["avoid", "critical", "warning", "caution"]:
                if check in content:
                    # Find line number
                    for i, line in enumerate(lines, 1):
                        if check in line:
                            issues.append({
                                "rule": check,
                                "message": message,
                                "severity": "critical" if rule_type == "critical" else 
                                          "high" if rule_type == "avoid" else "medium",
                                "line": i
                            })
                            score -= 20 if rule_type == "critical" else 10
                            break
        
        # Additional checks
        if "FROM scratch" not in content and not any("distroless" in l for l in lines):
            issues.append({
                "rule": "base_image",
                "message": "Consider using 'FROM scratch' or distroless base image for smaller attack surface",
                "severity": "low",
                "line": None
            })
            score -= 2
        
        # Check for multi-stage builds
        if content.count("FROM ") < 2:
            issues.append({
                "rule": "multi_stage",
                "message": "Consider using multi-stage builds to reduce final image size",
                "severity": "info",
                "line": None
            })
            score -= 1
        
        score = max(0, score)
        
        return {
            "status": "audited",
            "score": score,
            "grade": "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "D" if score >= 60 else "F",
            "issues": issues,
            "critical_count": sum(1 for i in issues if i["severity"] == "critical"),
            "high_count": sum(1 for i in issues if i["severity"] == "high"),
            "medium_count": sum(1 for i in issues if i["severity"] == "medium"),
            "low_count": sum(1 for i in issues if i["severity"] == "low"),
            "passed": score >= 80 and not any(i["severity"] == "critical" for i in issues)
        }
    
    def _action_audit_compose(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Audit docker-compose.yml."""
        path = params.get("path", "docker-compose.yml")
        
        if not os.path.exists(path):
            return {"status": "error", "message": f"Compose file not found: {path}"}
        
        try:
            import yaml
            with open(path, 'r') as f:
                compose = yaml.safe_load(f)
        except Exception as e:
            return {"status": "error", "message": f"Failed to parse compose file: {str(e)}"}
        
        issues = []
        services = compose.get("services", {})
        
        for svc_name, svc_config in services.items():
            # Check for restart policy
            if not svc_config.get("restart"):
                issues.append({
                    "service": svc_name,
                    "issue": "Missing restart policy",
                    "severity": "medium",
                    "fix": f"Add 'restart: unless-stopped' to {svc_name}"
                })
            
            # Check for resource limits
            if not svc_config.get("deploy", {}).get("resources", {}).get("limits"):
                issues.append({
                    "service": svc_name,
                    "issue": "No resource limits defined",
                    "severity": "low",
                    "fix": f"Add memory and CPU limits to {svc_name}"
                })
            
            # Check for health checks
            if not svc_config.get("healthcheck"):
                issues.append({
                    "service": svc_name,
                    "issue": "No healthcheck defined",
                    "severity": "medium",
                    "fix": f"Add healthcheck to {svc_name}"
                })
            
            # Check for secrets in environment
            env = svc_config.get("environment", [])
            for e in env if isinstance(env, list) else [f"{k}={v}" for k, v in env.items()]:
                if any(s in str(e).lower() for s in ["password", "secret", "key", "token"]):
                    if "=" in str(e) and not str(e).split("=", 1)[1].startswith("${"):
                        issues.append({
                            "service": svc_name,
                            "issue": f"Potential hardcoded secret in environment: {e}",
                            "severity": "critical",
                            "fix": "Use Docker secrets or environment file"
                        })
        
        return {
            "status": "audited",
            "services_checked": len(services),
            "issues": issues,
            "passed": len(issues) == 0
        }
    
    def _action_generate_pipeline(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate firmware CI/CD pipeline configuration, dispatched by language."""
        ci_platform = params.get("platform", "github")  # github or gitlab
        language = (params.get("language", "cpp") or "cpp").lower()

        # Dispatch to the correct firmware CI template by language
        if language in ("micropython",):
            if ci_platform == "gitlab":
                # MicroPython GitLab — syntax check + package job
                template = self.MICROPYTHON_GITHUB_ACTIONS.replace(
                    "name: MicroPython CI", "# MicroPython GitLab CI"
                ).replace(
                    "uses: actions/checkout@v3", "- git clone"
                )
            else:
                template = self.MICROPYTHON_GITHUB_ACTIONS
            filename = ".github/workflows/micropython-ci.yml" if ci_platform == "github" else ".gitlab-ci.yml"
            steps = ["syntax_check", "package_deploy", "upload_artifact"]

        elif language in ("circuitpython",):
            template = self.CIRCUITPYTHON_GITHUB_ACTIONS
            filename = ".github/workflows/circuitpython-ci.yml" if ci_platform == "github" else ".gitlab-ci.yml"
            steps = ["syntax_check", "circup_check", "package_deploy", "upload_artifact"]

        else:
            # C++, C, Rust, Zig — all go through PlatformIO
            if ci_platform == "gitlab":
                template = self.PLATFORMIO_GITLAB_CI
                filename = ".gitlab-ci.yml"
            else:
                template = self.PLATFORMIO_GITHUB_ACTIONS
                filename = ".github/workflows/firmware-ci.yml"
            steps = ["platformio_build", "native_unit_tests", "cppcheck", "upload_artifact"]

        return {
            "status": "success",
            "platform": ci_platform,
            "language": language,
            "config": template,
            "filename": filename,
            "steps": steps,
        }
    
    def _generate_custom_pipeline(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate custom pipeline using LLM."""
        if not self.llm_provider:
            return {"status": "error", "message": "No LLM available for custom pipeline generation"}
        
        prompt = f"""
        Generate a CI/CD pipeline configuration for:
        - Platform: {params.get('platform', 'github')}
        - Language: {params.get('language', 'python')}
        - Test framework: {params.get('test_framework', 'pytest')}
        - Docker: {params.get('enable_docker', True)}
        - Security scanning: {params.get('enable_security', True)}
        - Deployment: {params.get('enable_deploy', False)} to {params.get('deploy_target', 'none')}
        
        Return only the YAML configuration, no explanation.
        """
        
        try:
            config = self.llm_provider.generate(prompt)
            return {
                "status": "success",
                "platform": params.get("platform"),
                "config": config,
                "source": "llm"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"LLM generation failed: {str(e)}"
            }
    
    def _action_analyze_logs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze log files for errors and patterns."""
        path = params.get("path")
        service = params.get("service", "application")
        
        if not path or not os.path.exists(path):
            return {"status": "error", "message": f"Log file not found: {path}"}
        
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            return {"status": "error", "message": f"Failed to read logs: {str(e)}"}
        
        # Analyze patterns
        errors = []
        warnings = []
        patterns = {
            "error": re.compile(r'(ERROR|FATAL|CRITICAL|Exception|Traceback)', re.IGNORECASE),
            "warning": re.compile(r'(WARNING|WARN)', re.IGNORECASE),
            "info": re.compile(r'(INFO|DEBUG)', re.IGNORECASE),
        }
        
        for i, line in enumerate(lines, 1):
            if patterns["error"].search(line):
                errors.append({"line": i, "content": line.strip()[:200]})
            elif patterns["warning"].search(line):
                warnings.append({"line": i, "content": line.strip()[:200]})
        
        # Identify top errors
        error_types = {}
        for e in errors:
            # Extract error type
            match = re.search(r'(\w+Error|\w+Exception)', e["content"])
            if match:
                err_type = match.group(1)
                error_types[err_type] = error_types.get(err_type, 0) + 1
        
        return {
            "status": "success",
            "service": service,
            "total_lines": len(lines),
            "error_count": len(errors),
            "warning_count": len(warnings),
            "error_rate": round(len(errors) / len(lines) * 100, 2) if lines else 0,
            "top_errors": sorted(error_types.items(), key=lambda x: -x[1])[:10],
            "recent_errors": errors[-5:] if errors else [],
            "summary": f"Found {len(errors)} errors and {len(warnings)} warnings in {len(lines)} lines"
        }
    
    def _action_monitor_resources(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor system resources over time. Default 5s to avoid blocking the pipeline."""
        duration_seconds = params.get("duration", 5)
        interval_seconds = params.get("interval", 1)
        
        try:
            import psutil
            
            samples = []
            start_time = datetime.now()
            
            while (datetime.now() - start_time).total_seconds() < duration_seconds:
                sample = {
                    "timestamp": datetime.now().isoformat(),
                    "cpu_percent": psutil.cpu_percent(interval=1),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_percent": psutil.disk_usage('/').percent,
                }
                
                # Add network I/O if available
                net_io = psutil.net_io_counters()
                sample["net_sent_mb"] = round(net_io.bytes_sent / (1024**2), 2)
                sample["net_recv_mb"] = round(net_io.bytes_recv / (1024**2), 2)
                
                samples.append(sample)
                
                if (datetime.now() - start_time).total_seconds() < duration_seconds:
                    import time
                    time.sleep(interval_seconds)
            
            # Calculate statistics
            cpu_samples = [s["cpu_percent"] for s in samples]
            mem_samples = [s["memory_percent"] for s in samples]
            
            return {
                "status": "success",
                "duration": duration_seconds,
                "samples": len(samples),
                "statistics": {
                    "cpu_avg": round(sum(cpu_samples) / len(cpu_samples), 1),
                    "cpu_max": round(max(cpu_samples), 1),
                    "cpu_min": round(min(cpu_samples), 1),
                    "memory_avg": round(sum(mem_samples) / len(mem_samples), 1),
                    "memory_max": round(max(mem_samples), 1),
                    "memory_min": round(min(mem_samples), 1),
                },
                "data": samples,
                "alerts": [
                    {"type": "cpu_high", "value": max(cpu_samples), "threshold": self.alert_thresholds["cpu_percent"]}
                    if max(cpu_samples) > self.alert_thresholds["cpu_percent"] else None,
                    {"type": "memory_high", "value": max(mem_samples), "threshold": self.alert_thresholds["memory_percent"]}
                    if max(mem_samples) > self.alert_thresholds["memory_percent"] else None,
                ]
            }
            
        except ImportError:
            return {
                "status": "error",
                "message": "psutil required for resource monitoring",
                "install_command": "pip install psutil"
            }
    
    def _action_security_scan(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform security scan on project."""
        scan_type = params.get("type", "dependencies")  # dependencies, code, container
        path = params.get("path", ".")
        
        vulnerabilities = []
        
        if scan_type == "dependencies":
            # Check Python dependencies
            if os.path.exists("requirements.txt"):
                if not shutil.which("safety"):
                    vulnerabilities.append({
                        "type": "tool_missing",
                        "package": "safety",
                        "error": "safety not installed — dependency vulnerability scan skipped",
                        "fix": "pip install safety",
                        "severity": "INFO",
                    })
                else:
                    try:
                        result = subprocess.run(
                            ["safety", "check", "-r", "requirements.txt", "--json"],
                            capture_output=True,
                            text=True,
                            timeout=60
                        )
                        if result.stdout:
                            safety_data = json.loads(result.stdout)
                            for vuln in safety_data.get("vulnerabilities", []):
                                vulnerabilities.append({
                                    "package": vuln.get("package_name"),
                                    "installed_version": vuln.get("vulnerable_spec"),
                                    "vulnerable_spec": vuln.get("vulnerable_spec"),
                                    "CVE": vuln.get("cve"),
                                    "severity": vuln.get("severity", "unknown"),
                                    "advisory": vuln.get("advisory"),
                                })
                    except Exception as e:
                        vulnerabilities.append({
                            "type": "scan_error",
                            "package": "safety",
                            "error": f"safety scan failed: {str(e)}",
                            "severity": "INFO",
                        })
            
            # Bandit code scan
            if os.path.exists("src") or os.path.exists("backend"):
                src_dir = "src" if os.path.exists("src") else "backend"
                if not shutil.which("bandit"):
                    vulnerabilities.append({
                        "type": "tool_missing",
                        "package": "bandit",
                        "error": "bandit not installed — static code analysis skipped",
                        "fix": "pip install bandit",
                        "severity": "INFO",
                    })
                else:
                    try:
                        result = subprocess.run(
                            ["bandit", "-r", src_dir, "-f", "json"],
                            capture_output=True,
                            text=True,
                            timeout=60
                        )
                        if result.stdout:
                            bandit_data = json.loads(result.stdout)
                            for issue in bandit_data.get("results", []):
                                vulnerabilities.append({
                                    "type": "code",
                                    "test_id": issue.get("test_id"),
                                    "test_name": issue.get("test_name"),
                                    "severity": issue.get("issue_severity"),
                                    "confidence": issue.get("issue_confidence"),
                                    "filename": issue.get("filename"),
                                    "line": issue.get("line_number"),
                                    "code": issue.get("code"),
                                })
                    except Exception as e:
                        vulnerabilities.append({
                            "type": "scan_error",
                            "package": "bandit",
                            "error": f"bandit scan failed: {str(e)}",
                            "severity": "INFO",
                        })
        
        elif scan_type == "container":
            image = params.get("image")
            if image:
                if not shutil.which("trivy"):
                    vulnerabilities.append({
                        "type": "tool_missing",
                        "package": "trivy",
                        "error": "trivy not installed — container scan skipped",
                        "fix": "Install trivy: https://aquasecurity.github.io/trivy/latest/getting-started/installation/",
                        "severity": "INFO",
                    })
                else:
                    try:
                        proc = subprocess.run(
                            ["trivy", "image", "--format", "json", image],
                            capture_output=True,
                            text=True,
                            timeout=120
                        )
                        if proc.stdout:
                            trivy_data = json.loads(proc.stdout)
                            for scan_result in trivy_data.get("Results", []):
                                for vuln in (scan_result.get("Vulnerabilities") or []):
                                    vulnerabilities.append({
                                        "type": "container",
                                        "vulnerability_id": vuln.get("VulnerabilityID"),
                                        "pkg_name": vuln.get("PkgName"),
                                        "installed_version": vuln.get("InstalledVersion"),
                                        "fixed_version": vuln.get("FixedVersion"),
                                        "severity": vuln.get("Severity"),
                                        "title": vuln.get("Title"),
                                    })
                    except Exception as e:
                        vulnerabilities.append({
                            "type": "scan_error",
                            "package": "trivy",
                            "error": f"trivy scan failed: {str(e)}",
                            "severity": "INFO",
                        })
        
        critical = sum(1 for v in vulnerabilities if v.get("severity") == "CRITICAL")
        high = sum(1 for v in vulnerabilities if v.get("severity") == "HIGH")
        
        return {
            "status": "scanned",
            "scan_type": scan_type,
            "vulnerabilities_found": len(vulnerabilities),
            "critical": critical,
            "high": high,
            "medium": sum(1 for v in vulnerabilities if v.get("severity") == "MEDIUM"),
            "low": sum(1 for v in vulnerabilities if v.get("severity") == "LOW"),
            "passed": critical == 0 and high == 0,
            "vulnerabilities": vulnerabilities[:50]  # Limit output
        }
    
    def _action_iac_validate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Infrastructure as Code."""
        iac_type = params.get("type", "terraform")  # terraform, cloudformation, ansible
        path = params.get("path", ".")
        
        issues = []
        
        if iac_type == "terraform":
            # Run terraform validate
            try:
                result = subprocess.run(
                    ["terraform", "validate", "-json"],
                    cwd=path,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode != 0:
                    validate_data = json.loads(result.stdout)
                    for diag in validate_data.get("diagnostics", []):
                        issues.append({
                            "severity": diag.get("severity"),
                            "summary": diag.get("summary"),
                            "detail": diag.get("detail"),
                            "range": diag.get("range")
                        })
            except FileNotFoundError:
                issues.append({
                    "severity": "error",
                    "summary": "Terraform not installed",
                    "detail": "Install Terraform to validate configurations"
                })
            except Exception as e:
                issues.append({
                    "severity": "error",
                    "summary": "Validation failed",
                    "detail": str(e)
                })
            
            # Run terraform fmt check
            try:
                result = subprocess.run(
                    ["terraform", "fmt", "-check", "-recursive"],
                    cwd=path,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode != 0:
                    issues.append({
                        "severity": "warning",
                        "summary": "Terraform files not formatted",
                        "detail": "Run 'terraform fmt -recursive' to fix",
                        "files": result.stdout.strip().split('\n') if result.stdout else []
                    })
            except:
                pass
        
        elif iac_type == "cloudformation":
            template_path = params.get("template_path")
            if template_path and os.path.exists(template_path):
                try:
                    import yaml
                    with open(template_path, 'r') as f:
                        template = yaml.safe_load(f)
                    
                    # Basic validation
                    if "Resources" not in template:
                        issues.append({
                            "severity": "error",
                            "summary": "Missing Resources section",
                            "detail": "CloudFormation template must have a Resources section"
                        })
                    
                    # Check for security groups allowing 0.0.0.0/0 on sensitive ports
                    for name, resource in template.get("Resources", {}).items():
                        if resource.get("Type") == "AWS::EC2::SecurityGroup":
                            ingress = resource.get("Properties", {}).get("SecurityGroupIngress", [])
                            for rule in ingress:
                                cidr = rule.get("CidrIp", "")
                                port = rule.get("FromPort", 0)
                                if cidr == "0.0.0.0/0" and port in [22, 3389, 3306, 5432]:
                                    issues.append({
                                        "severity": "critical",
                                        "summary": f"Security group {name} allows unrestricted access to port {port}",
                                        "detail": f"CidrIp 0.0.0.0/0 on port {port} is a security risk"
                                    })
                    
                except Exception as e:
                    issues.append({
                        "severity": "error",
                        "summary": "Failed to parse CloudFormation template",
                        "detail": str(e)
                    })
        
        return {
            "status": "validated",
            "iac_type": iac_type,
            "issues": issues,
            "passed": len([i for i in issues if i["severity"] == "error"]) == 0
        }
    
    def _action_docker_diagnostics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnose Docker issues."""
        diagnostics = []
        
        # Check Docker daemon
        try:
            result = subprocess.run(
                ["docker", "version", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                diagnostics.append({
                    "category": "daemon",
                    "status": "error",
                    "message": "Docker daemon not running",
                    "fix": "Start Docker service: sudo systemctl start docker"
                })
        except:
            diagnostics.append({
                "category": "daemon",
                "status": "error",
                "message": "Docker not installed or not in PATH",
                "fix": "Install Docker or add user to docker group"
            })
        
        # Check disk usage
        try:
            result = subprocess.run(
                ["docker", "system", "df", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                df_data = json.loads(result.stdout)
                for item in df_data:
                    if item.get("Size"):
                        size_str = item.get("Size", "0B")
                        # Extract numeric value
                        size_num = float(re.findall(r'[\d.]+', size_str)[0]) if re.findall(r'[\d.]+', size_str) else 0
                        if "GB" in size_str and size_num > 50:
                            diagnostics.append({
                                "category": "storage",
                                "status": "warning",
                                "message": f"High Docker {item.get('Type', 'storage')} usage: {size_str}",
                                "fix": f"Run 'docker {item.get('Type', '').lower()} prune' to clean up"
                            })
        except:
            pass
        
        # Check for exited containers
        try:
            result = subprocess.run(
                ["docker", "ps", "-aq", "-f", "status=exited"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.stdout.strip():
                count = len(result.stdout.strip().split('\n'))
                diagnostics.append({
                    "category": "containers",
                    "status": "info",
                    "message": f"{count} stopped containers found",
                    "fix": "Run 'docker container prune' to remove stopped containers"
                })
        except:
            pass
        
        return {
            "status": "success",
            "diagnostics": diagnostics,
            "healthy": len([d for d in diagnostics if d["status"] == "error"]) == 0
        }
    
    def _action_k8s_diagnostics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnose Kubernetes cluster."""
        namespace = params.get("namespace", "default")
        diagnostics = []
        
        # Check kubectl
        try:
            result = subprocess.run(
                ["kubectl", "version", "--client", "-o", "json"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                diagnostics.append({
                    "category": "client",
                    "status": "error",
                    "message": "kubectl not configured properly",
                    "fix": "Configure kubectl with valid kubeconfig"
                })
        except FileNotFoundError:
            diagnostics.append({
                "category": "client",
                "status": "error",
                "message": "kubectl not installed",
                "fix": "Install kubectl"
            })
            return {"status": "error", "diagnostics": diagnostics}
        
        # Check cluster connectivity
        try:
            result = subprocess.run(
                ["kubectl", "cluster-info"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                diagnostics.append({
                    "category": "cluster",
                    "status": "error",
                    "message": "Cannot connect to cluster",
                    "fix": "Check kubeconfig and cluster availability"
                })
        except:
            pass
        
        # Check pod status
        try:
            result = subprocess.run(
                ["kubectl", "get", "pods", "-n", namespace, "-o", "json"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                pods = json.loads(result.stdout)
                not_running = []
                for pod in pods.get("items", []):
                    status = pod.get("status", {}).get("phase", "Unknown")
                    if status != "Running":
                        not_running.append({
                            "name": pod.get("metadata", {}).get("name"),
                            "status": status
                        })
                
                if not_running:
                    diagnostics.append({
                        "category": "pods",
                        "status": "warning",
                        "message": f"{len(not_running)} pods not running in {namespace}",
                        "pods": not_running,
                        "fix": f"Run 'kubectl describe pods -n {namespace}' for details"
                    })
        except:
            pass
        
        return {
            "status": "success",
            "namespace": namespace,
            "diagnostics": diagnostics,
            "healthy": len([d for d in diagnostics if d["status"] == "error"]) == 0
        }
    
    def _action_network_check(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check network connectivity and latency."""
        hosts = params.get("hosts", ["8.8.8.8", "1.1.1.1", "google.com", "github.com"])
        port = params.get("port", 443)
        
        results = []
        for host in hosts:
            try:
                # Ping test
                ping_result = subprocess.run(
                    ["ping", "-c", "3", "-W", "2", host],
                    capture_output=True,
                    text=True,
                    timeout=15
                )
                
                pingable = ping_result.returncode == 0
                latency = None
                packet_loss = 100
                
                if pingable:
                    # Extract avg latency
                    match = re.search(r'min/avg/max.*?=\s*[\d.]+/([\d.]+)', ping_result.stdout)
                    if match:
                        latency = float(match.group(1))
                    
                    # Extract packet loss
                    loss_match = re.search(r'(\d+)% packet loss', ping_result.stdout)
                    if loss_match:
                        packet_loss = int(loss_match.group(1))
                
                results.append({
                    "host": host,
                    "pingable": pingable,
                    "latency_ms": round(latency, 1) if latency else None,
                    "packet_loss_percent": packet_loss,
                    "status": "healthy" if pingable and packet_loss < 10 else "degraded" if pingable else "unhealthy"
                })
            except Exception as e:
                results.append({
                    "host": host,
                    "pingable": False,
                    "error": str(e),
                    "status": "error"
                })
        
        return {
            "status": "success",
            "hosts_tested": len(hosts),
            "results": results,
            "healthy": sum(1 for r in results if r.get("status") == "healthy"),
            "degraded": sum(1 for r in results if r.get("status") == "degraded"),
            "unhealthy": sum(1 for r in results if r.get("status") in ["unhealthy", "error"])
        }
    
    def _action_dependency_check(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check project dependencies for updates and conflicts."""
        manifest_type = params.get("type", "pip")  # pip, npm, cargo, poetry
        path = params.get("path", ".")
        
        outdated = []
        conflicts = []
        
        if manifest_type == "pip":
            req_file = os.path.join(path, "requirements.txt")
            if os.path.exists(req_file):
                try:
                    result = subprocess.run(
                        ["pip", "list", "--outdated", "--format=json"],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    if result.returncode == 0:
                        outdated = json.loads(result.stdout)
                        # Filter to only requirements.txt packages
                        with open(req_file) as f:
                            req_packages = [line.strip().split('==')[0].split('>=')[0].lower() 
                                          for line in f if line.strip() and not line.startswith('#')]
                        outdated = [p for p in outdated if p.get("name", "").lower() in req_packages]
                except:
                    pass
        
        elif manifest_type == "npm":
            package_json = os.path.join(path, "package.json")
            if os.path.exists(package_json):
                try:
                    result = subprocess.run(
                        ["npm", "outdated", "--json"],
                        cwd=path,
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    if result.stdout:
                        outdated_data = json.loads(result.stdout)
                        for pkg, info in outdated_data.items():
                            outdated.append({
                                "name": pkg,
                                "current": info.get("current"),
                                "wanted": info.get("wanted"),
                                "latest": info.get("latest")
                            })
                except:
                    pass
        
        return {
            "status": "success",
            "manifest_type": manifest_type,
            "outdated_count": len(outdated),
            "outdated_packages": outdated[:20],
            "conflicts": conflicts,
            "recommendation": f"Run 'pip install --upgrade <package>' to update {len(outdated)} packages" if outdated else "All dependencies up to date"
        }


# API Integration
class DevOpsAPI:
    """FastAPI endpoints for DevOps diagnostics."""
    
    @staticmethod
    def get_routes(agent: 'DevOpsAgent'):
        """Get FastAPI routes."""
        from fastapi import APIRouter, HTTPException
        from pydantic import BaseModel, Field
        from typing import List, Optional
        
        router = APIRouter(prefix="/devops", tags=["devops"])
        
        class HealthCheckRequest(BaseModel):
            services: List[dict] = Field(default_factory=list, description="Services to check")
            include_docker: bool = True
            include_network: bool = True
        
        class DockerfileAuditRequest(BaseModel):
            path: str = "Dockerfile"
            strict: bool = False
        
        class PipelineRequest(BaseModel):
            platform: str = "github"
            language: str = "python"
            enable_docker: bool = True
            enable_security: bool = True
        
        class LogAnalysisRequest(BaseModel):
            path: str
            service: str = "application"
        
        class SecurityScanRequest(BaseModel):
            type: str = "dependencies"  # dependencies, code, container
            path: str = "."
            image: Optional[str] = None
        
        @router.post("/health")
        async def health_check(request: HealthCheckRequest):
            """Run comprehensive health check."""
            result = agent.run({
                "action": "health_check",
                **request.dict()
            })
            if result["status"] != "success":
                raise HTTPException(status_code=500, detail=result.get("message"))
            return result
        
        @router.post("/audit/dockerfile")
        async def audit_dockerfile(request: DockerfileAuditRequest):
            """Audit Dockerfile security."""
            return agent.run({
                "action": "audit_dockerfile",
                **request.dict()
            })
        
        @router.post("/pipeline/generate")
        async def generate_pipeline(request: PipelineRequest):
            """Generate CI/CD pipeline."""
            return agent.run({
                "action": "generate_pipeline",
                **request.dict()
            })
        
        @router.post("/logs/analyze")
        async def analyze_logs(request: LogAnalysisRequest):
            """Analyze log files."""
            return agent.run({
                "action": "analyze_logs",
                **request.dict()
            })
        
        @router.post("/security/scan")
        async def security_scan(request: SecurityScanRequest):
            """Run security scan."""
            return agent.run({
                "action": "security_scan",
                **request.dict()
            })
        
        @router.get("/templates/pipeline")
        async def list_pipeline_templates():
            """List available pipeline templates."""
            return {
                "templates": [
                    {"id": "github", "name": "GitHub Actions", "filename": ".github/workflows/ci.yml"},
                    {"id": "gitlab", "name": "GitLab CI", "filename": ".gitlab-ci.yml"},
                ]
            }
        
        return router
