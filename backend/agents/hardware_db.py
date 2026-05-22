"""
HardwareDB — MCU hardware spec loader.

Lookup order:
  1. In-process cache (same process lifetime)
  2. Supabase `hardware_db` table  (keyed by mcu_key, e.g. "esp32", "stm32f405")
  3. Local YAML files in hardware_db/ (fallback for offline / migration period)
  4. LLM generation for completely unknown MCUs → cached back to Supabase

All credentials come from environment variables. No hardcoding.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

_HARDWARE_DB_DIR = Path(__file__).parent.parent / "hardware_db"
_CACHE: Dict[str, Dict[str, Any]] = {}


# ── Key normalisation ────────────────────────────────────────────────────────

def _normalise_key(mcu: str) -> str:
    """'STM32F405RG' → 'stm32f405rg', 'ESP32-WROOM-32' → 'esp32_wroom32'"""
    return re.sub(r"[^a-z0-9]", "_", mcu.lower()).strip("_")


# ── Supabase helpers ─────────────────────────────────────────────────────────

def _supabase_headers() -> Dict[str, str]:
    key = os.environ.get("SUPABASE_SERVICE_KEY", "")
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def _supabase_url() -> str:
    return os.environ.get("SUPABASE_URL", "").rstrip("/")


def _fetch_from_supabase(mcu_key: str) -> Optional[Dict[str, Any]]:
    """Return spec dict or None if not found / Supabase unavailable."""
    base = _supabase_url()
    if not base:
        return None
    try:
        import requests
        url = f"{base}/rest/v1/hardware_db?mcu_key=eq.{mcu_key}&select=spec&limit=1"
        r = requests.get(url, headers=_supabase_headers(), timeout=5)
        if r.status_code == 200:
            rows = r.json()
            if rows:
                return rows[0]["spec"]
        elif r.status_code not in (404, 406):
            logger.warning("[HardwareDB] Supabase query failed %s: %s", r.status_code, r.text[:200])
    except Exception as exc:
        logger.warning("[HardwareDB] Supabase unavailable: %s", exc)
    return None


def _save_to_supabase(mcu_key: str, mcu_name: str, family: str, spec: Dict[str, Any], source: str = "llm_generated") -> None:
    """Upsert generated spec to Supabase for caching."""
    base = _supabase_url()
    if not base:
        return
    try:
        import requests
        payload = {
            "mcu_key": mcu_key,
            "mcu_name": mcu_name,
            "family": family,
            "spec": spec,
            "source": source,
        }
        url = f"{base}/rest/v1/hardware_db"
        r = requests.post(
            url,
            headers={**_supabase_headers(), "Prefer": "resolution=merge-duplicates"},
            json=payload,
            timeout=10,
        )
        if r.status_code not in (200, 201, 204):
            logger.warning("[HardwareDB] Supabase upsert failed %s: %s", r.status_code, r.text[:200])
        else:
            logger.info("[HardwareDB] Cached %s to Supabase", mcu_key)
    except Exception as exc:
        logger.warning("[HardwareDB] Supabase write failed: %s", exc)


# ── YAML fallback ────────────────────────────────────────────────────────────

def _fetch_from_yaml(mcu_key: str) -> Optional[Dict[str, Any]]:
    """Search seed and generated YAML files."""
    candidates = [
        _HARDWARE_DB_DIR / f"{mcu_key}.yaml",
        _HARDWARE_DB_DIR / "generated" / f"{mcu_key}.yaml",
        _HARDWARE_DB_DIR / "custom" / f"{mcu_key}.yaml",
    ]
    # Also try fuzzy match: e.g. "esp32" matches "esp32_wroom32.yaml"
    for f in _HARDWARE_DB_DIR.glob("**/*.yaml"):
        norm = _normalise_key(f.stem)
        if norm == mcu_key or mcu_key in norm:
            candidates.append(f)

    for path in candidates:
        if path.exists():
            try:
                data = yaml.safe_load(path.read_text())
                return _yaml_to_spec(data)
            except Exception as exc:
                logger.warning("[HardwareDB] Failed to load %s: %s", path, exc)
    return None


def _yaml_to_spec(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise YAML file format to the canonical spec dict stored in Supabase."""
    spec: Dict[str, Any] = {}
    for key in ("clock_mhz", "flash_kb", "flash_mb", "sram_kb", "adc_channels", "dac_channels"):
        if key in data:
            spec[key] = data[key]

    # Pins: map {pin: {restrictions: [...], electrical: {...}}} → flat
    pins: Dict[str, Any] = {}
    for pin, info in (data.get("pins") or {}).items():
        pins[pin] = {}
        if "restrictions" in info:
            pins[pin]["restrictions"] = info["restrictions"]
        if "electrical" in info:
            pins[pin]["electrical"] = info["electrical"]
        if "note" in info:
            pins[pin]["note"] = info["note"]
    spec["pins"] = pins

    # Peripherals
    periph = data.get("peripherals") or {}
    spec_periph: Dict[str, Any] = {}

    for bus_type in ("i2c", "spi", "uart", "can"):
        if bus_type in periph:
            spec_periph[bus_type] = periph[bus_type]

    if "pwm" in periph:
        spec_periph["pwm"] = periph["pwm"]

    if "gpio_cs_pool" in periph:
        spec_periph["gpio_cs_pool"] = periph["gpio_cs_pool"]

    spec["peripherals"] = spec_periph
    return spec


# ── LLM fallback ─────────────────────────────────────────────────────────────

_LLM_SCHEMA_PROMPT = """You are an embedded systems expert. Generate a hardware specification for the MCU: {mcu_name}

Return ONLY valid JSON matching this exact schema (no prose, no markdown fences):
{{
  "clock_mhz": <number>,
  "flash_kb": <number or null>,
  "flash_mb": <number or null>,
  "sram_kb": <number>,
  "adc_channels": <number>,
  "dac_channels": <number>,
  "pins": {{
    "<PIN_NAME>": {{
      "restrictions": ["strapping_pin"|"debug_pin"|"input_only"|"internal_flash"|"boot_mode"],
      "electrical": {{"input_only": <bool>, "five_volt_tolerant": <bool>}}
    }}
  }},
  "peripherals": {{
    "i2c": [{{"name": "I2C0", "scl": "<PIN>", "sda": "<PIN>", "max_freq_khz": 400}}],
    "spi": [{{"name": "SPI0", "sclk": "<PIN>", "miso": "<PIN>", "mosi": "<PIN>", "cs_pool": ["<PIN>", ...]}}],
    "uart": [{{"name": "UART0", "tx": "<PIN>", "rx": "<PIN>"}}],
    "can": [{{"name": "CAN0", "tx": "<PIN>", "rx": "<PIN>"}}],
    "pwm": {{"name": "TIM_PWM", "pins": ["<PIN>", ...]}},
    "gpio_cs_pool": ["<PIN>", ...]
  }}
}}

Only include pins with restrictions/special electrical properties in "pins" dict.
Include ALL peripheral instances actually present on this MCU.
Exclude boot/flash/debug pins from pwm pins list.
"""


def _generate_from_llm(mcu_name: str, llm_provider: Any) -> Optional[Dict[str, Any]]:
    """Ask LLM to synthesise a hardware spec. Returns validated spec or None."""
    try:
        prompt = _LLM_SCHEMA_PROMPT.format(mcu_name=mcu_name)
        raw = llm_provider.generate_json(prompt) if hasattr(llm_provider, "generate_json") else None
        if raw is None:
            # Fallback: plain generate
            raw_text = llm_provider.generate(prompt)
            raw = json.loads(raw_text)
        spec = _validate_spec(raw)
        return spec
    except Exception as exc:
        logger.error("[HardwareDB] LLM generation failed for %s: %s", mcu_name, exc)
        return None


def _validate_spec(spec: Any) -> Dict[str, Any]:
    """Minimal schema validation. Raises ValueError on bad structure."""
    if not isinstance(spec, dict):
        raise ValueError("spec must be a dict")
    required = ("sram_kb",)
    for field in required:
        if field not in spec:
            raise ValueError(f"spec missing required field: {field}")
    if "peripherals" not in spec:
        spec["peripherals"] = {}
    if "pins" not in spec:
        spec["pins"] = {}
    return spec


# ── Adapter: spec → PLATFORM_DEFS-compatible dict ────────────────────────────

def _spec_to_platform_def(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert canonical spec (from Supabase / YAML) to the flat dict format
    that _allocate_pins() in codegen_agent expects.
    """
    periph = spec.get("peripherals", {})
    pins_meta = spec.get("pins", {})

    # PWM pins: exclude restricted pins at this stage too
    pwm_data = periph.get("pwm", {})
    if isinstance(pwm_data, list):
        pwm_data = pwm_data[0] if pwm_data else {}
    raw_pwm = pwm_data.get("pins", [])
    restricted = {
        p for p, meta in pins_meta.items()
        if any(r in ("debug_pin", "internal_flash", "boot_mode", "strapping_pin", "input_only")
               for r in (meta.get("restrictions") or []))
    }
    pwm_pins = [p for p in raw_pwm if p not in restricted]

    # I2C interfaces: [(name, scl, sda), ...]
    i2c_interfaces = []
    for bus in (periph.get("i2c") or []):
        i2c_interfaces.append((bus["name"], bus["scl"], bus["sda"]))

    # SPI interfaces: [(name, sclk, miso, mosi, cs0), ...]
    spi_interfaces = []
    for bus in (periph.get("spi") or []):
        cs0 = (bus.get("cs_pool") or [None])[0]
        spi_interfaces.append((bus["name"], bus["sclk"], bus["miso"], bus["mosi"], cs0))

    # UART interfaces: [(name, tx, rx), ...]
    uart_interfaces = []
    for bus in (periph.get("uart") or []):
        uart_interfaces.append((bus["name"], bus["tx"], bus["rx"]))

    # CAN interfaces: [(name, tx, rx), ...]
    can_interfaces = []
    for bus in (periph.get("can") or []):
        can_interfaces.append((bus["name"], bus["tx"], bus["rx"]))

    gpio_cs_pool = periph.get("gpio_cs_pool", [])

    plat_def = {
        "clock_mhz":       spec.get("clock_mhz"),
        "flash_kb":        spec.get("flash_kb"),
        "flash_mb":        spec.get("flash_mb"),
        "sram_kb":         spec.get("sram_kb"),
        "adc_channels":    spec.get("adc_channels", 0),
        "dac_channels":    spec.get("dac_channels", 0),
        "pwm_pins":        pwm_pins,
        "i2c_interfaces":  i2c_interfaces,
        "spi_interfaces":  spi_interfaces,
        "uart_interfaces": uart_interfaces,
        "can_interfaces":  can_interfaces,
        "gpio_cs_pool":    gpio_cs_pool,
        # Raw restricted pin metadata for electrical checks
        "_pins_meta":      pins_meta,
        "_raw_spec":       spec,
    }
    return plat_def


# ── Public API ───────────────────────────────────────────────────────────────

class HardwareDB:
    """
    Load MCU hardware specs from Supabase (primary) or YAML files (fallback).
    Unknown MCUs are synthesised by LLM and cached back to Supabase.
    """

    @classmethod
    def load(cls, mcu: str, llm_provider: Any = None) -> Dict[str, Any]:
        """
        Return a platform-def dict compatible with codegen_agent._allocate_pins().

        Args:
            mcu: MCU identifier string, e.g. "ESP32", "STM32F405", "STM32G474RE"
            llm_provider: Optional LLM client with .generate_json() or .generate()

        Returns:
            Platform def dict. Empty dict if MCU is completely unknown and LLM is unavailable.
        """
        mcu_key = _normalise_key(mcu)

        # 1. In-process cache
        if mcu_key in _CACHE:
            return _CACHE[mcu_key]

        spec = None

        # 2. Supabase
        spec = _fetch_from_supabase(mcu_key)
        if spec:
            logger.info("[HardwareDB] Loaded %s from Supabase", mcu_key)

        # 3. Local YAML fallback
        if spec is None:
            spec = _fetch_from_yaml(mcu_key)
            if spec:
                logger.info("[HardwareDB] Loaded %s from local YAML", mcu_key)

        # 4. LLM synthesis
        if spec is None and llm_provider is not None:
            logger.info("[HardwareDB] Generating spec for unknown MCU: %s", mcu)
            spec = _generate_from_llm(mcu, llm_provider)
            if spec:
                family = mcu_key.split("_")[0]
                _save_to_supabase(mcu_key, mcu, family, spec, source="llm_generated")
                logger.info("[HardwareDB] Cached LLM-generated spec for %s", mcu_key)

        if spec is None:
            logger.warning("[HardwareDB] No spec found for %s — returning empty platform def", mcu_key)
            plat_def: Dict[str, Any] = {
                "pwm_pins": [], "i2c_interfaces": [], "spi_interfaces": [],
                "uart_interfaces": [], "can_interfaces": [], "gpio_cs_pool": [],
                "_pins_meta": {}, "_raw_spec": {},
            }
            _CACHE[mcu_key] = plat_def
            return plat_def

        plat_def = _spec_to_platform_def(spec)
        _CACHE[mcu_key] = plat_def
        return plat_def

    @classmethod
    def clear_cache(cls) -> None:
        _CACHE.clear()
