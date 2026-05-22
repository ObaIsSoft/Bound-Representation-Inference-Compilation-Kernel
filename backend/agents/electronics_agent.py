"""
Production ElectronicsAgent v2 — Goal-Directed Electronics Design

Architecture:
  TopologyLibrary  → loads YAML topology "constitution", evaluates selection rules
  ComponentSizer   → first-principles design equations, E-series value rounding
  NgSpiceEngine    → raw subprocess ngspice-46 (bypasses broken PySpice API entirely)
  ComponentCatalog → DigiKey v4 / Mouser v2 / Octopart v4 REST APIs (from .env)
  KiCadExporter    → KiCad v2 .net netlist format

Design flow (power supply / motor driver goals):
  1. Parse goal  →  2. Select topology  →  3. Size components (Phase 1 ideal)
  4. Run ngspice  →  5. Validate  →  6. Adjust if failing  (iterate ≤ 5×)
  7. Catalog lookup  →  8. Run ngspice Phase 2 (real R_ds_on / ESR / DCR)
  9. Export KiCad netlist  →  10. Return full design package

Legacy operations (simulate_circuit, analyze_pcb, si_analysis, pi_analysis,
thermal_analysis, drc_check) are preserved for orchestrator backward compat.

API keys (all from os.environ, set by main.py via load_dotenv):
  DIGIKEY_CLIENT_ID, DIGIKEY_SECRET, DIGIKEY_API_PATH
  MOUSER_API_KEY, MOUSER_PARTNER_ID
  OCTOPART_API_KEY
  Graceful failure when DNS unreachable — catalog skipped, Phase 2 omitted.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import math
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ── Specialized analysis engines ───────────────────────────────────────────────
try:
    from agents.electronics_engines import (
        SignalIntegrityEngine, PowerIntegrityEngine, ElectronicsThermalEngine,
        PCBGeometryEngine, AnalogDesignEngine, DigitalDesignEngine,
        RFDesignEngine, GerberWriter, EMCEngine,
        MagneticsDesignEngine, ControlLoopDesignEngine, ExtendedGerberWriter,
        TraceSpec, ViaSpec,
    )
    _ENGINES_AVAILABLE = True
except ImportError:
    try:
        from electronics_engines import (
            SignalIntegrityEngine, PowerIntegrityEngine, ElectronicsThermalEngine,
            PCBGeometryEngine, AnalogDesignEngine, DigitalDesignEngine,
            RFDesignEngine, GerberWriter, EMCEngine,
            MagneticsDesignEngine, ControlLoopDesignEngine, ExtendedGerberWriter,
            TraceSpec, ViaSpec,
        )
        _ENGINES_AVAILABLE = True
    except ImportError:
        _ENGINES_AVAILABLE = False
        logger.warning("electronics_engines.py not found — advanced analysis unavailable")

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DesignGoal:
    v_in: float           # Input voltage (V)
    v_out: float          # Target output voltage (V)
    i_out: float          # Rated output current (A)
    ripple_v: float       # Max peak-to-peak output ripple (V)
    f_sw: float           # Switching frequency (Hz)
    efficiency_min: float # Minimum acceptable efficiency
    t_amb: float          # Ambient temperature (°C)
    goal_type: str        # "power_supply" | "motor_driver"
    project_name: str     # Human-readable label
    # motor driver extras
    r_motor: float = 0.0  # Motor winding resistance (Ω); 0 = auto from v_out/i_out
    l_motor: float = 1e-3 # Motor winding inductance (H)


@dataclass
class Component:
    id: str
    type: str
    value: Optional[float] = None
    unit: Optional[str] = None
    footprint: Optional[str] = None
    model: Optional[str] = None
    pins: Dict[str, str] = field(default_factory=dict)
    connections: Dict[str, str] = field(default_factory=dict)
    thermal: Dict[str, float] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Net:
    name: str
    nodes: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class Circuit:
    name: str
    components: Dict[str, Component] = field(default_factory=dict)
    nets: Dict[str, Net] = field(default_factory=dict)
    signals: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TopologyLibrary
# ---------------------------------------------------------------------------

class TopologyLibrary:
    """
    Loads electronics_topologies.yaml and provides topology selection,
    design-equation evaluation, and SPICE template access.
    """

    _MATH_NS: Dict[str, Any] = {
        "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
        "exp": math.exp, "abs": abs, "min": min, "max": max,
        "pi": math.pi, "__builtins__": {},
    }

    def __init__(self, yaml_path: Optional[Path] = None):
        if yaml_path is None:
            yaml_path = Path(__file__).parent.parent / "config" / "electronics_topologies.yaml"
        with open(yaml_path) as fh:
            self._data: Dict[str, Any] = yaml.safe_load(fh)
        self._topologies: Dict[str, Any] = self._data.get("topologies", {})
        self._sim_cfg: Dict[str, Any] = self._data.get("simulation", {
            "cycles_total": 8, "cycles_skip": 3, "steps_per_cycle": 200,
            "motor_cycles_total": 20,
        })
        self._e_series: Dict[str, str] = self._data.get("e_series", {
            "resistor": "E96", "capacitor": "E24", "inductor": "E12",
        })

    # ------------------------------------------------------------------
    def select_topology(self, goal: DesignGoal) -> Tuple[str, Dict[str, Any], float]:
        """
        Evaluate selection rules for every topology and return
        (key, topology_dict, score) for the best match.
        Raises ValueError if no topology satisfies the rules.
        """
        ns = self._goal_namespace(goal)
        best_key: Optional[str] = None
        best_score = -1e99
        best_topo: Optional[Dict] = None

        for key, topo in self._topologies.items():
            sel = topo.get("selection", {})
            rules: List[str] = sel.get("rules", [])
            # Merge topology default params into namespace for rule evaluation
            params_ns = {
                name: spec["default"]
                for name, spec in topo.get("parameters", {}).items()
            }
            eval_ns = {**ns, **params_ns}
            try:
                if not all(self._eval_expr(r, eval_ns) for r in rules):
                    continue
                score = float(self._eval_expr(sel.get("score", "1.0"), eval_ns))
            except Exception as exc:
                logger.debug("Topology %s rule evaluation error: %s", key, exc)
                continue
            if score > best_score:
                best_score = score
                best_key = key
                best_topo = topo

        if best_key is None:
            raise ValueError(
                f"No topology found for goal V_in={goal.v_in}V → V_out={goal.v_out}V "
                f"@ {goal.i_out}A. Check selection rules in electronics_topologies.yaml."
            )
        logger.info("Topology selected: %s (score=%.4f)", best_key, best_score)
        return best_key, best_topo, best_score  # type: ignore[return-value]

    def evaluate_equations(
        self, topology_key: str, base_namespace: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate all design equations in definition order.
        Each equation result is added to the namespace so later equations can
        reference earlier ones. Returns the full accumulated namespace delta.
        """
        topo = self._topologies[topology_key]
        # Start with default params
        ns = {
            name: spec["default"]
            for name, spec in topo.get("parameters", {}).items()
        }
        ns.update(base_namespace)
        ns.update(self._MATH_NS)

        equations: Dict[str, str] = topo.get("design_equations", {})
        results: Dict[str, float] = {}
        for var_name, expr in equations.items():
            try:
                self._validate_expr(expr)
                val = float(eval(expr, {"__builtins__": {}}, ns))  # noqa: S307
                ns[var_name] = val
                results[var_name] = val
            except Exception as exc:
                logger.warning("Equation '%s = %s' failed: %s", var_name, expr, exc)
        return results

    def get_spice_template(self, topology_key: str, phase: int) -> str:
        topo = self._topologies[topology_key]
        key = f"spice_phase{phase}"
        return topo.get(key, "")

    def get_bom_roles(self, topology_key: str) -> List[Dict[str, Any]]:
        return self._topologies[topology_key].get("bom_roles", [])

    def get_validation(self, topology_key: str) -> Dict[str, Any]:
        return self._topologies[topology_key].get("validation", {})

    def get_sim_config(self) -> Dict[str, Any]:
        return self._sim_cfg

    def get_e_series(self) -> Dict[str, str]:
        return self._e_series

    # ------------------------------------------------------------------
    def _goal_namespace(self, goal: DesignGoal) -> Dict[str, Any]:
        return {
            "v_in": goal.v_in,
            "v_out": goal.v_out,
            "i_out": goal.i_out,
            "ripple_v": goal.ripple_v,
            "f_sw": goal.f_sw,
            "efficiency_min": goal.efficiency_min,
            "t_amb": goal.t_amb,
            "goal_type": goal.goal_type,
        }

    # Allowed AST node types for safe eval of YAML expressions
    _SAFE_NODES = {
        ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.IfExp,
        ast.Compare, ast.Call, ast.Constant, ast.Name, ast.Attribute,
        ast.And, ast.Or, ast.Not,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv,
        ast.USub, ast.UAdd,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        ast.Num, ast.NameConstant,   # Python 3.7 compat aliases
    }

    @classmethod
    def _validate_expr(cls, expr: str) -> None:
        """Raise ValueError if expr contains any unsafe AST node."""
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Syntax error in expression: {exc}") from exc
        for node in ast.walk(tree):
            if type(node) not in cls._SAFE_NODES:
                raise ValueError(
                    f"Unsafe node {type(node).__name__!r} in expression: {expr!r}"
                )

    def _eval_expr(self, expr: str, ns: Dict[str, Any]) -> Any:
        self._validate_expr(expr)
        safe_ns = {**self._MATH_NS, **ns}
        return eval(expr, {"__builtins__": {}}, safe_ns)  # noqa: S307


# ---------------------------------------------------------------------------
# ComponentSizer
# ---------------------------------------------------------------------------

class ComponentSizer:
    """E-series value rounding and safe equation evaluation."""

    E12 = [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2]
    E24 = [
        1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
        3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1,
    ]
    E96 = [
        1.00, 1.02, 1.05, 1.07, 1.10, 1.13, 1.15, 1.18, 1.21, 1.24, 1.27, 1.30,
        1.33, 1.37, 1.40, 1.43, 1.47, 1.50, 1.54, 1.58, 1.62, 1.65, 1.69, 1.74,
        1.78, 1.82, 1.87, 1.91, 1.96, 2.00, 2.05, 2.10, 2.15, 2.21, 2.26, 2.32,
        2.37, 2.43, 2.49, 2.55, 2.61, 2.67, 2.74, 2.80, 2.87, 2.94, 3.01, 3.09,
        3.16, 3.24, 3.32, 3.40, 3.48, 3.57, 3.65, 3.74, 3.83, 3.92, 4.02, 4.12,
        4.22, 4.32, 4.42, 4.53, 4.64, 4.75, 4.87, 4.99, 5.11, 5.23, 5.36, 5.49,
        5.62, 5.76, 5.90, 6.04, 6.19, 6.34, 6.49, 6.65, 6.81, 6.98, 7.15, 7.32,
        7.50, 7.68, 7.87, 8.06, 8.25, 8.45, 8.66, 8.87, 9.09, 9.31, 9.53, 9.76,
    ]
    _SERIES = {"E12": E12, "E24": E24, "E96": E96}

    def round_to_eseries(
        self, value: float, series_name: str = "E24", direction: str = "up"
    ) -> float:
        """
        Round value to nearest preferred E-series value.
        direction='up' always picks the next higher value (conservative for power designs).
        direction='nearest' picks the closest.
        """
        if value <= 0:
            return value
        series = self._SERIES.get(series_name, self.E24)
        decade = 10 ** math.floor(math.log10(value))
        mantissa = value / decade
        if direction == "up":
            for s in series:
                if s * decade >= value * 0.9999:
                    return round(s * decade, 12)
            return series[-1] * decade * 10
        # nearest
        closest = min(series, key=lambda s: abs(s * decade - value))
        return round(closest * decade, 12)


# ---------------------------------------------------------------------------
# NgSpiceEngine
# ---------------------------------------------------------------------------

class NgSpiceEngine:
    """
    Runs ngspice-46 via raw subprocess.  No PySpice involved.
    Writes a temporary .cir file, runs `ngspice -b`, parses .meas output.
    """

    _MEAS_RE = re.compile(
        r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)",
        re.MULTILINE,
    )

    def __init__(self):
        self._binary: Optional[str] = self._find_ngspice()

    def available(self) -> bool:
        return self._binary is not None

    def run_netlist(
        self, netlist_text: str, timeout: int = 60
    ) -> Dict[str, Any]:
        """
        Write netlist to a temp file, run ngspice -b, parse .meas results.

        Returns:
            {
                "success": bool,
                "measurements": {name: float},
                "stdout": str,
                "stderr": str,
                "error": str | None,
            }
        """
        if not self._binary:
            return {"success": False, "error": "ngspice binary not found", "measurements": {}}

        with tempfile.NamedTemporaryFile(
            suffix=".cir", mode="w", delete=False, prefix="brick_spice_"
        ) as fh:
            fh.write(netlist_text)
            cir_path = fh.name

        try:
            result = subprocess.run(
                [self._binary, "-b", cir_path],
                capture_output=True, text=True, timeout=timeout,
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""

            # Detect fatal errors
            fatal = any(
                marker in stdout.lower() or marker in stderr.lower()
                for marker in ("fatal error", "simulation aborted", "could not find model")
            )
            measurements = self.parse_meas_output(stdout)
            success = bool(measurements) and not fatal

            return {
                "success": success,
                "measurements": measurements,
                "stdout": stdout,
                "stderr": stderr,
                "error": (stderr[:500] if not success else None),
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False, "measurements": {},
                "error": f"ngspice timed out after {timeout}s",
                "stdout": "", "stderr": "",
            }
        except Exception as exc:
            return {
                "success": False, "measurements": {},
                "error": str(exc), "stdout": "", "stderr": "",
            }
        finally:
            try:
                os.unlink(cir_path)
            except OSError:
                pass

    def fill_template(self, template: str, ctx: Dict[str, Any]) -> str:
        """
        Safe format substitution: only replaces {keys} present in ctx.
        Leaves unmatched {keys} unchanged (avoids KeyError on partial templates).
        """
        # Replace {key:.Xf}, {key:.Xg}, {key:.Xg}, {key} etc.
        def replacer(m: re.Match) -> str:
            full = m.group(0)         # e.g. "{vin:.6g}"
            key = m.group(1)          # e.g. "vin"
            fmt = m.group(2) or ""    # e.g. ":.6g"
            if key in ctx:
                if fmt:
                    return format(ctx[key], fmt.lstrip(":"))
                return str(ctx[key])
            return full               # leave unchanged

        return re.sub(r"\{(\w+)(:[^}]*)?\}", replacer, template)

    def parse_meas_output(self, stdout: str) -> Dict[str, float]:
        results: Dict[str, float] = {}
        for m in self._MEAS_RE.finditer(stdout):
            name = m.group(1).lower()
            try:
                results[name] = float(m.group(2))
            except ValueError:
                pass
        # ngspice sometimes prints "failed" for measurements that couldn't complete
        failed_re = re.compile(r"^([a-zA-Z_]\w*)\s*=\s*failed", re.MULTILINE | re.IGNORECASE)
        for m in failed_re.finditer(stdout):
            results.pop(m.group(1).lower(), None)
        return results

    def _find_ngspice(self) -> Optional[str]:
        # Try env first, then common paths
        candidates = [
            os.environ.get("NGSPICE_BIN", ""),
            "/usr/local/bin/ngspice",
            "/usr/bin/ngspice",
            "/opt/homebrew/bin/ngspice",
        ]
        for path in candidates:
            if path and Path(path).exists():
                return path
        # Try PATH
        try:
            result = subprocess.run(
                ["which", "ngspice"], capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# ComponentCatalog
# ---------------------------------------------------------------------------

class ComponentCatalog:
    """
    Searches DigiKey v4, Mouser v2, and Octopart v4 for real component parts.
    All API keys come from os.environ.
    All network calls have a 10s timeout; DNS failures are caught silently.
    Returns None when all distributors are unreachable.
    """

    def __init__(self):
        self._dk_client_id = os.environ.get("DIGIKEY_CLIENT_ID", "")
        self._dk_secret = os.environ.get("DIGIKEY_SECRET", "")
        self._dk_base = os.environ.get("DIGIKEY_API_PATH", "https://api.digikey.com")
        self._mouser_key = os.environ.get("MOUSER_API_KEY", "")
        self._octopart_key = os.environ.get("OCTOPART_API_KEY", "")
        self._dk_token: Optional[str] = None
        self._dk_token_expiry: float = 0.0

    async def search_all_roles(
        self,
        bom_roles: List[Dict[str, Any]],
        computed: Dict[str, Any],
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """Search catalog for every BOM role. Returns {role: part_dict | None}."""
        results: Dict[str, Optional[Dict]] = {}
        for role_spec in bom_roles:
            role_name = role_spec.get("role", "unknown")
            part = await self.search(role_spec, computed)
            results[role_name] = part
        return results

    async def search(
        self,
        role_spec: Dict[str, Any],
        computed: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Try DigiKey → Mouser → Octopart in order. Return first result."""
        keywords = self._format_keywords(role_spec, computed)
        required_fields: List[str] = role_spec.get("datasheet_fields", [])
        part_type = role_spec.get("type", "")
        filters = {"type": part_type, "v_rated": self._get_v_rated(role_spec, computed),
                   "i_rated": self._get_i_rated(role_spec, computed)}

        logger.info("Catalog search [%s]: %s", role_spec.get("role"), keywords)

        for searcher in [self._search_digikey, self._search_mouser, self._search_octopart]:
            try:
                part = await searcher(keywords, filters, required_fields)
                if part:
                    return part
            except Exception as exc:
                logger.debug("Catalog searcher %s failed: %s", searcher.__name__, exc)

        logger.warning("No part found for role '%s'", role_spec.get("role"))
        return None

    async def _get_digikey_token(self) -> Optional[str]:
        if not self._dk_client_id or not self._dk_secret:
            return None
        if self._dk_token and time.time() < self._dk_token_expiry - 60:
            return self._dk_token
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._dk_base}/v1/oauth2/token",
                    data={
                        "grant_type": "client_credentials",
                        "client_id": self._dk_client_id,
                        "client_secret": self._dk_secret,
                    },
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self._dk_token = data.get("access_token")
                        self._dk_token_expiry = time.time() + data.get("expires_in", 3600)
                        return self._dk_token
        except Exception as exc:
            logger.debug("DigiKey OAuth failed: %s", exc)
        return None

    async def _search_digikey(
        self, keywords: str, filters: Dict, required_fields: List[str]
    ) -> Optional[Dict[str, Any]]:
        token = await self._get_digikey_token()
        if not token:
            return None
        try:
            import aiohttp
            headers = {
                "Authorization": f"Bearer {token}",
                "X-DIGIKEY-Client-Id": self._dk_client_id,
                "Content-Type": "application/json",
                "X-DIGIKEY-Locale-Language": "en",
                "X-DIGIKEY-Locale-Currency": "USD",
            }
            body = {"keywords": keywords, "limit": 5, "offset": 0,
                    "sort": {"sortOption": "SortByDigiKeyPartNumber", "direction": "Ascending"}}
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._dk_base}/products/v4/search/keyword",
                    headers=headers, json=body,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()
            products = data.get("Products", [])
            for product in products:
                normalized = self._normalize_digikey_part(product, required_fields)
                if normalized:
                    return normalized
        except Exception as exc:
            logger.debug("DigiKey search error: %s", exc)
        return None

    async def _search_mouser(
        self, keywords: str, filters: Dict, required_fields: List[str]
    ) -> Optional[Dict[str, Any]]:
        if not self._mouser_key:
            return None
        try:
            import aiohttp
            body = {
                "SearchByKeywordRequest": {
                    "keyword": keywords, "records": 5, "startingRecord": 0,
                    "searchOptions": "None", "searchWithYourSignUpLanguage": "false",
                }
            }
            url = f"https://api.mouser.com/api/v2/search/keyword?apiKey={self._mouser_key}"
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=body, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()
            parts = (data.get("SearchResults") or {}).get("Parts") or []
            for part in parts:
                normalized = self._normalize_mouser_part(part, required_fields)
                if normalized:
                    return normalized
        except Exception as exc:
            logger.debug("Mouser search error: %s", exc)
        return None

    async def _search_octopart(
        self, keywords: str, filters: Dict, required_fields: List[str]
    ) -> Optional[Dict[str, Any]]:
        if not self._octopart_key:
            return None
        query = """
        query SearchParts($q: String!) {
          search(q: $q, limit: 5) {
            results {
              part {
                mpn manufacturer { name }
                short_description
                best_datasheet { url }
                specs { attribute { name shortname } display_value }
              }
            }
          }
        }
        """
        try:
            import aiohttp
            url = "https://octopart.com/api/v4/endpoint"
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    params={"token": self._octopart_key},
                    json={"query": query, "variables": {"q": keywords}},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()
            results = (
                data.get("data", {}).get("search", {}).get("results") or []
            )
            for item in results:
                part = item.get("part", {})
                normalized = self._normalize_octopart_part(part, required_fields)
                if normalized:
                    return normalized
        except Exception as exc:
            logger.debug("Octopart search error: %s", exc)
        return None

    # ------------------------------------------------------------------
    def _format_keywords(
        self, role_spec: Dict[str, Any], computed: Dict[str, Any]
    ) -> str:
        template = role_spec.get("search_keywords", "")
        v_rated = self._get_v_rated(role_spec, computed)
        i_rated = self._get_i_rated(role_spec, computed)
        val_h = computed.get(role_spec.get("value_var", ""), 0.0)
        val_uh = val_h * 1e6 if val_h else 0.0
        cap_f = computed.get(role_spec.get("value_var", ""), 0.0)
        cap_uf = cap_f * 1e6 if cap_f else 0.0
        try:
            return template.format(
                vds=v_rated, id=i_rated, vr=v_rated, iavg=i_rated,
                vin=computed.get("v_in", 0), vout=computed.get("v_out", 0),
                iout=computed.get("i_out", 0), val_uh=val_uh, val_uf=cap_uf,
                isat=computed.get(role_spec.get("i_sat_var", "l_i_sat_min"), 0),
                vrated=v_rated,
            )
        except KeyError:
            return template

    def _get_v_rated(self, role_spec: Dict, computed: Dict) -> float:
        var = role_spec.get("v_rated_var") or role_spec.get("v_in_var", "")
        return computed.get(var, 0.0)

    def _get_i_rated(self, role_spec: Dict, computed: Dict) -> float:
        var = role_spec.get("i_rated_var") or role_spec.get("i_out_var", "")
        return computed.get(var, 0.0)

    def _normalize_digikey_part(
        self, product: Dict, required_fields: List[str]
    ) -> Optional[Dict[str, Any]]:
        params_raw = product.get("Parameters", [])
        params: Dict[str, str] = {
            p["Parameter"].lower().replace(" ", "_"): p.get("Value", "")
            for p in params_raw
        }
        part: Dict[str, Any] = {
            "mpn": product.get("ManufacturerProductNumber", ""),
            "manufacturer": (product.get("Manufacturer") or {}).get("Name", ""),
            "description": product.get("ProductDescription", ""),
            "datasheet_url": product.get("DatasheetUrl", ""),
            "unit_price_usd": self._first_price(product.get("UnitPrice", 0)),
            "availability": product.get("QuantityAvailable", 0),
            "source": "digikey",
        }
        self._extract_electrical_params(params, part)
        return part if part.get("mpn") else None

    def _normalize_mouser_part(
        self, raw: Dict, required_fields: List[str]
    ) -> Optional[Dict[str, Any]]:
        specs_raw = raw.get("ProductAttributes", []) or []
        params: Dict[str, str] = {
            s.get("AttributeName", "").lower().replace(" ", "_"): s.get("AttributeValue", "")
            for s in specs_raw
        }
        part: Dict[str, Any] = {
            "mpn": raw.get("ManufacturerPartNumber", ""),
            "manufacturer": raw.get("Manufacturer", ""),
            "description": raw.get("Description", ""),
            "datasheet_url": raw.get("DataSheetUrl", ""),
            "unit_price_usd": self._parse_price(raw.get("PriceBreaks", [{}])[0].get("Price", "0")),
            "availability": raw.get("AvailabilityInStock", 0),
            "source": "mouser",
        }
        self._extract_electrical_params(params, part)
        return part if part.get("mpn") else None

    def _normalize_octopart_part(
        self, raw: Dict, required_fields: List[str]
    ) -> Optional[Dict[str, Any]]:
        specs_raw = raw.get("specs", []) or []
        params: Dict[str, str] = {
            s.get("attribute", {}).get("shortname", "").lower(): s.get("display_value", "")
            for s in specs_raw
        }
        part: Dict[str, Any] = {
            "mpn": raw.get("mpn", ""),
            "manufacturer": (raw.get("manufacturer") or {}).get("name", ""),
            "description": raw.get("short_description", ""),
            "datasheet_url": (raw.get("best_datasheet") or {}).get("url", ""),
            "unit_price_usd": None,
            "source": "octopart",
        }
        self._extract_electrical_params(params, part)
        return part if part.get("mpn") else None

    def _extract_electrical_params(
        self, params: Dict[str, str], out: Dict[str, Any]
    ) -> None:
        """
        Parse common electrical parameters from string representation.
        Handles formats like "50 mOhms", "10 nC", "30 V", "5 A".
        """
        unit_map = {
            # Rds(on) variants
            "rds(on)": "rds_on_mohm", "drain-source_on_resistance": "rds_on_mohm",
            "rds_on": "rds_on_mohm", "on_resistance": "rds_on_mohm",
            # Gate charge
            "gate_charge": "qg_nc", "qg": "qg_nc", "total_gate_charge": "qg_nc",
            # Threshold
            "vgs(th)": "v_th_v", "gate_threshold_voltage": "v_th_v",
            # Inductor DCR
            "dc_resistance_(dcr)": "dcr_mohm", "dcr": "dcr_mohm",
            "dc_resistance": "dcr_mohm", "dc_resistance_(max)": "dcr_mohm",
            # Capacitor ESR
            "esr_(equivalent_series_resistance)": "esr_mohm",
            "esr": "esr_mohm", "equivalent_series_resistance": "esr_mohm",
            # Saturation current
            "saturation_current_(isat)": "i_sat_a", "isat": "i_sat_a",
            "rated_current": "i_rms_a",
            # Voltage rating
            "voltage_-_rated": "v_rated_v", "voltage_rating": "v_rated_v",
            # Capacitance
            "capacitance": "capacitance_uf",
            # Inductance
            "inductance": "inductance_uh",
            # Dropout
            "dropout_voltage": "v_dropout_v",
            # Quiescent current
            "quiescent_current": "i_quiescent_ua",
            # Thermal
            "thermal_resistance_(junction_to_ambient)": "theta_ja_c_per_w",
        }
        for raw_key, out_key in unit_map.items():
            val_str = params.get(raw_key, "")
            if val_str:
                val = self._parse_value_with_unit(val_str)
                if val is not None:
                    # Normalize to expected units
                    if out_key in ("rds_on_mohm", "dcr_mohm", "esr_mohm"):
                        if "ohm" in val_str.lower() and "m" not in val_str.lower():
                            val *= 1000  # Ω → mΩ
                    elif out_key == "qg_nc":
                        if "nc" not in val_str.lower() and "n" not in val_str.lower():
                            val *= 1000  # C → nC
                    elif out_key == "capacitance_uf":
                        if "uf" not in val_str.lower() and "u" not in val_str.lower():
                            val *= 1e6  # F → µF
                    elif out_key == "inductance_uh":
                        if "uh" not in val_str.lower() and "u" not in val_str.lower():
                            val *= 1e6  # H → µH
                    out[out_key] = round(val, 6)

    def _parse_value_with_unit(self, s: str) -> Optional[float]:
        m = re.match(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([a-zA-Zµ]*)", s.strip())
        if not m:
            return None
        val = float(m.group(1))
        unit = m.group(2).lower()
        prefix_map = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6,
                      "m": 1e-3, "k": 1e3, "M": 1e6, "G": 1e9}
        if unit and unit[0] in prefix_map:
            val *= prefix_map[unit[0]]
        return val

    def _first_price(self, price_obj: Any) -> Optional[float]:
        if isinstance(price_obj, (int, float)):
            return float(price_obj)
        if isinstance(price_obj, list) and price_obj:
            return self._parse_price(str(price_obj[0].get("UnitPrice", 0)))
        return None

    def _parse_price(self, s: str) -> Optional[float]:
        try:
            return float(re.sub(r"[^\d.]", "", s))
        except (ValueError, TypeError):
            return None


# ---------------------------------------------------------------------------
# KiCadExporter
# ---------------------------------------------------------------------------

class KiCadExporter:
    """
    Exports a BRICK design result to KiCad v2 .net (netlist) format.
    This file can be imported directly into KiCad PCB editor.
    """

    def __init__(self, topo_lib: "Optional[TopologyLibrary]" = None) -> None:
        self._topo_lib = topo_lib

    def export_netlist(self, design: Dict[str, Any]) -> str:
        bom: Dict[str, Optional[Dict]] = design.get("bom", {})
        topology_key: str = design.get("topology_key", "unknown")
        goal: Dict[str, Any] = design.get("goal", {})

        lines = [
            "(export (version D)",
            f"  (design (source \"{topology_key}\")",
            f"   (date \"{time.strftime('%Y-%m-%d %H:%M:%S')}\")",
            f"   (tool \"BRICK OS Electronics Agent\")",
            "  )",
            "  (components",
        ]

        net_assignments: Dict[str, List[str]] = {}  # net_name → [ref.pin, ...]
        net_map = self._topology_nets(topology_key)

        for role_key, part in bom.items():
            ref = self._role_ref(design, role_key)
            value = part.get("mpn", role_key) if part else role_key
            desc = (part.get("description", "") if part else "No part found")[:60]
            footprint = self._default_footprint(role_key)
            lines.append(f"    (comp (ref \"{ref}\")")
            lines.append(f"     (value \"{value}\")")
            lines.append(f"     (description \"{desc}\")")
            lines.append(f"     (footprint \"{footprint}\")")
            lines.append("    )")
            # Register nets
            for pin_name, net_name in net_map.get(role_key, {}).items():
                net_assignments.setdefault(net_name, []).append(f"{ref}.{pin_name}")

        lines.append("  )")
        lines.append("  (nets")
        for i, (net_name, nodes) in enumerate(net_assignments.items(), start=1):
            lines.append(f"    (net (code \"{i}\") (name \"{net_name}\")")
            for node in nodes:
                ref, pin = node.split(".", 1)
                lines.append(f"     (node (ref \"{ref}\") (pin \"{pin}\"))")
            lines.append("    )")
        lines.append("  )")
        lines.append(")")
        return "\n".join(lines)

    def _role_ref(self, design: Dict, role_key: str) -> str:
        """Look up the schematic ref (Q1, L1, C1...) from YAML bom_roles or runtime meta."""
        topology_key = design.get("topology_key", "")
        # Sources in priority order: runtime meta (from design()), then YAML (via topo_lib)
        sources = [
            design.get("bom_roles_meta", []),
            (self._topo_lib.get_bom_roles(topology_key) if self._topo_lib else []),
        ]
        for source in sources:
            for r in source:
                if r.get("role") == role_key and r.get("ref"):
                    return r["ref"]
        # role not in any topology definition — return the role key itself so the
        # netlist is at least valid (KiCad accepts any alphanumeric ref)
        return role_key

    def _default_footprint(self, role_key: str) -> str:
        fp_map = {
            "mosfet_hs": "Package_TO_SOT_SMD:SOT-23",
            "mosfet_ls": "Package_TO_SOT_SMD:SOT-23",
            "mosfet_sw": "Package_TO_SOT_SMD:SOT-23",
            "mosfet_q1234": "Package_TO_SOT_SMD:SOT-23",
            "inductor": "Inductor_SMD:L_4.0x4.0",
            "cap_out": "Capacitor_SMD:C_0805",
            "cap_in": "Capacitor_SMD:C_0805",
            "cap_bulk": "Capacitor_SMD:C_1210",
            "ldo_ic": "Package_SO:SOIC-8",
            "diode": "Diode_SMD:D_SOD-123",
        }
        for key, fp in fp_map.items():
            if key in role_key:
                return fp
        return "Package_TO_SOT_SMD:SOT-23"

    def _topology_nets(self, topology_key: str) -> Dict[str, Dict[str, str]]:
        """
        Returns {role: {pin: net_name}} for standard topologies.
        This is a simplified fixed mapping — a full implementation would
        derive this from the netlist template.
        """
        if topology_key == "buck":
            return {
                "mosfet_hs": {"drain": "VIN", "source": "SW", "gate": "PWM"},
                "mosfet_ls": {"drain": "SW", "source": "GND", "gate": "PWM_N"},
                "inductor": {"1": "SW", "2": "VOUT"},
                "cap_out": {"1": "VOUT", "2": "GND"},
                "cap_in": {"1": "VIN", "2": "GND"},
            }
        if topology_key == "boost":
            return {
                "mosfet_sw": {"drain": "LX", "source": "GND", "gate": "PWM"},
                "inductor": {"1": "VIN", "2": "LX"},
                "diode": {"A": "LX", "K": "VOUT"},
                "cap_out": {"1": "VOUT", "2": "GND"},
                "cap_in": {"1": "VIN", "2": "GND"},
            }
        if topology_key == "ldo":
            return {
                "ldo_ic": {"IN": "VIN", "OUT": "VOUT", "GND": "GND"},
                "cap_out": {"1": "VOUT", "2": "GND"},
                "cap_in": {"1": "VIN", "2": "GND"},
            }
        if topology_key == "h_bridge":
            return {
                "mosfet_q1234": {"Q1_D": "VSUPPLY", "Q1_S": "MA",
                                  "Q2_D": "MB", "Q2_S": "GND",
                                  "Q3_D": "VSUPPLY", "Q3_S": "MB",
                                  "Q4_D": "MA", "Q4_S": "GND"},
                "cap_bulk": {"1": "VSUPPLY", "2": "GND"},
            }
        if topology_key == "flyback":
            return {
                "mosfet_sw":    {"drain": "SW", "source": "GND", "gate": "PWM"},
                "transformer":  {"Np+": "VIN", "Np-": "SW", "Ns+": "VOUT_RAW", "Ns-": "GND_SEC"},
                "diode":        {"A": "VOUT_RAW", "K": "VOUT"},
                "cap_out":      {"1": "VOUT", "2": "GND_SEC"},
                "cap_in":       {"1": "VIN", "2": "GND"},
                "resistor_snub":{"1": "SW", "2": "GND"},
            }
        if topology_key == "forward":
            return {
                "mosfet_sw":    {"drain": "VIN", "source": "SW", "gate": "PWM"},
                "transformer":  {"Np+": "VIN", "Np-": "SW", "Ns+": "RECT", "Ns-": "GND_SEC", "Nr+": "RESET", "Nr-": "VIN"},
                "diode_rect":   {"A": "RECT", "K": "SW_OUT"},
                "diode_fw":     {"A": "GND_SEC", "K": "SW_OUT"},
                "inductor":     {"1": "SW_OUT", "2": "VOUT"},
                "cap_out":      {"1": "VOUT", "2": "GND_SEC"},
                "cap_in":       {"1": "VIN", "2": "GND"},
            }
        if topology_key == "full_bridge":
            return {
                "mosfet_q1":    {"drain": "VBUS", "source": "HB_A", "gate": "PWM_HS_A"},
                "mosfet_q2":    {"drain": "HB_A", "source": "GND",  "gate": "PWM_LS_A"},
                "mosfet_q3":    {"drain": "VBUS", "source": "HB_B", "gate": "PWM_HS_B"},
                "mosfet_q4":    {"drain": "HB_B", "source": "GND",  "gate": "PWM_LS_B"},
                "transformer":  {"Np+": "HB_A", "Np-": "HB_B", "Ns+": "RECT_P", "Ns-": "RECT_N"},
                "diode_d1":     {"A": "RECT_P", "K": "VOUT"},
                "diode_d2":     {"A": "RECT_N", "K": "VOUT"},
                "inductor":     {"1": "VOUT", "2": "VOUT_F"},
                "cap_out":      {"1": "VOUT_F", "2": "GND_SEC"},
                "cap_in":       {"1": "VBUS", "2": "GND"},
            }
        if topology_key == "llc_resonant":
            return {
                "mosfet_hs":    {"drain": "VBUS", "source": "HB", "gate": "PWM_HS"},
                "mosfet_ls":    {"drain": "HB",   "source": "GND","gate": "PWM_LS"},
                "cap_res":      {"1": "HB", "2": "LR_A"},
                "inductor_lr":  {"1": "LR_A", "2": "LM_A"},
                "inductor_lm":  {"1": "LM_A", "2": "TX_P"},
                "transformer":  {"Np+": "TX_P", "Np-": "GND", "Ns+": "RECT_P", "Ns-": "RECT_N"},
                "diode_d1":     {"A": "RECT_P", "K": "VOUT"},
                "diode_d2":     {"A": "RECT_N", "K": "VOUT"},
                "cap_out":      {"1": "VOUT", "2": "GND_SEC"},
                "cap_in":       {"1": "VBUS", "2": "GND"},
            }
        if topology_key == "sepic":
            return {
                "mosfet_sw":    {"drain": "LX", "source": "GND", "gate": "PWM"},
                "inductor_l1":  {"1": "VIN", "2": "LX"},
                "cap_couple":   {"1": "LX", "2": "L2_A"},
                "inductor_l2":  {"1": "L2_A", "2": "DIODE_A"},
                "diode":        {"A": "DIODE_A", "K": "VOUT"},
                "cap_out":      {"1": "VOUT", "2": "GND"},
                "cap_in":       {"1": "VIN", "2": "GND"},
            }
        return {}


# ---------------------------------------------------------------------------
# Legacy SPICE + KiCad interfaces (preserved for backward compat)
# ---------------------------------------------------------------------------

class SpiceInterface:
    """Legacy PySpice wrapper — kept for orchestrator backward compat."""

    def __init__(self, simulator: str = "ngspice"):
        self.simulator = simulator
        self._available: Optional[bool] = None

    async def check(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            from PySpice.Spice.Netlist import Circuit as _C
            from PySpice.Unit import u_V, u_Ohm  # noqa: F401
            self._available = True
        except ImportError:
            self._available = False
        return self._available  # type: ignore[return-value]

    async def simulate(
        self, circuit: Circuit, analysis_type: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "Use NgSpiceEngine.run_netlist() directly. PySpice API is broken with ngspice-46."
        )


class KiCadInterface:
    """Legacy pcbnew wrapper."""

    def __init__(self, kicad_path: str = "/usr/share/kicad"):
        self.kicad_path = Path(kicad_path)
        self._available: Optional[bool] = None

    async def check(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import pcbnew  # noqa: F401
            self._available = True
        except ImportError:
            self._available = False
        return self._available  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# ElectronicsAgent
# ---------------------------------------------------------------------------

class ElectronicsAgent:
    """
    Production-grade electronics design and analysis agent.

    New capability: full design loop — takes a natural-language or structured
    goal (v_in, v_out, i_out, ripple, f_sw) and returns a complete design
    package: sized components, validated SPICE simulation, real parts from
    catalog, and a KiCad netlist.

    Legacy operations (simulate_circuit, analyze_pcb, si_analysis, pi_analysis,
    thermal_analysis, drc_check) are preserved for LangGraph orchestrator nodes.
    """

    # Verified unencrypted LTspice-compatible models (ngspice-readable, tested 2025-05).
    # Key: lowercase MPN with hyphens stripped.  Value: raw GitHub URL.
    _KNOWN_GOOD_SPICE_URLS: Dict[str, str] = {
        # Verified against ngspice-46: V_out=5.22V (target 5V), V_ref=1.30V (spec 1.25V)
        "lm317": "https://raw.githubusercontent.com/chrisnoisel/ltspice/master/sub/LM317.sub",
    }

    # Adjustable LDOs: set output voltage via external R1 (OUT→ADJ) and R2 (ADJ→GND).
    # Formula: V_out = V_ref × (1 + R2/R1); V_ref ≈ 1.25V for LM117/LM217/LM317 family.
    _ADJUSTABLE_LDO_MPNS: frozenset = frozenset({
        "lm317", "lm317l", "lm317hv", "lm317m", "lm317t",
        "lm117", "lm117hv", "lm217",
        "lm337", "lm337t",
    })

    # Fixed-output LDO Phase 2 template (2-pin + GND subcircuit: IN, OUT, AGND).
    # Used when _fetch_spice_model succeeds and device is NOT in _ADJUSTABLE_LDO_MPNS.
    _LDO_MANUFACTURER_TEMPLATE = (
        "* BRICK OS LDO — Phase 2 (manufacturer model, fixed output)\n"
        "* {project_name}  MPN={mpn_ic}\n"
        "*\n"
        ".include \"{spice_model_path}\"\n"
        "*\n"
        "Vin    vin_n  0      DC {vin:.6g}\n"
        "Cin    vin_n  0      {c_in_f:.11g}\n"
        "Xldo   vin_n  vout_n 0  {spice_model_name}\n"
        "Cout   vout_n cout_n {c_out_f:.11g}\n"
        "Resr   cout_n 0      {esr_cout:.11g}\n"
        "Rload  vout_n 0      {r_load:.6g}\n"
        "*\n"
        ".op\n"
        ".tran 1e-7 5e-4\n"
        ".meas tran VOUT_AVG avg v(vout_n) from=2.5e-4 to=5e-4\n"
        ".meas tran VOUT_PP  pp  v(vout_n) from=2.5e-4 to=5e-4\n"
        ".meas tran IIN_AVG  avg I(Vin)    from=2.5e-4 to=5e-4\n"
        ".end\n"
    )

    # Adjustable LDO Phase 2 template (3-pin: IN, ADJ, OUT subcircuit order).
    # PWL ramp on Vin is required — DC .op finds wrong multi-stable operating point
    # for complex feedback models (verified with LM317 25-BJT subcircuit, ngspice-46).
    # R1: OUT→ADJ (reference resistor, fixed).  R2: ADJ→GND (sets output voltage).
    # Simulates 10ms to allow feedback network to settle (LM317 settling ~3ms).
    _LDO_ADJ_MANUFACTURER_TEMPLATE = (
        "* BRICK OS LDO — Phase 2 (adjustable manufacturer model)\n"
        "* {project_name}  MPN={mpn_ic}\n"
        "* V_out={vout:.4f}V  R1={r1_adj:.1f}Ω (OUT→ADJ)  R2={r2_set:.1f}Ω (ADJ→GND)\n"
        "*\n"
        ".include \"{spice_model_path}\"\n"
        "*\n"
        "Vin    vin_n  0      PWL(0 0  0.5m {vin:.6g}  10m {vin:.6g})\n"
        "Cin    vin_n  0      {c_in_f:.11g}\n"
        "Xldo   vin_n  adj_n  vout_n  {spice_model_name}\n"
        "R1     vout_n adj_n  {r1_adj:.6g}\n"
        "R2     adj_n  0      {r2_set:.6g}\n"
        "Cout   vout_n cout_n {c_out_f:.11g}\n"
        "Resr   cout_n 0      {esr_cout:.11g}\n"
        "Rload  vout_n 0      {r_load:.6g}\n"
        "*\n"
        ".tran 10e-6 10e-3\n"
        ".meas tran VOUT_AVG avg v(vout_n) from=8e-3 to=10e-3\n"
        ".meas tran VOUT_PP  pp  v(vout_n) from=8e-3 to=10e-3\n"
        ".meas tran IIN_AVG  avg I(Vin)    from=8e-3 to=10e-3\n"
        ".end\n"
    )

    def __init__(self):
        self.name = "ElectronicsAgent"
        self._initialized = False
        self._topo_lib: Optional[TopologyLibrary] = None
        self._sizer = ComponentSizer()
        self._ngspice: Optional[NgSpiceEngine] = None
        self._catalog: Optional[ComponentCatalog] = None
        self._kicad_exporter = KiCadExporter()
        self.config = self._load_config()
        # Specialized engines
        if _ENGINES_AVAILABLE:
            self._si_engine   = SignalIntegrityEngine()
            self._pi_engine   = PowerIntegrityEngine()
            self._th_engine   = ElectronicsThermalEngine()
            self._pcb_engine  = PCBGeometryEngine()
            self._analog      = AnalogDesignEngine()
            self._digital     = DigitalDesignEngine()
            self._rf          = RFDesignEngine()
            self._gerber      = ExtendedGerberWriter()   # replaces base GerberWriter
            self._emc         = EMCEngine()
            self._magnetics   = MagneticsDesignEngine()
            self._control     = ControlLoopDesignEngine()
        else:
            self._si_engine = self._pi_engine = self._th_engine = None
            self._pcb_engine = self._analog = self._digital = None
            self._rf = self._gerber = self._emc = None
            self._magnetics = self._control = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _load_config(self) -> Dict[str, Any]:
        config_path = Path(__file__).parent.parent / "data" / "electronics_config.json"
        defaults: Dict[str, Any] = {
            "spice_simulator": "ngspice",
            "kicad_path": "/usr/share/kicad",
            "thermal_ambient_c": 25.0,
            "thermal_theta_ja_default": 50.0,
            "pcb_copper_thickness_oz": 1.0,
            "pcb_dielectric_constant": 4.5,
        }
        if config_path.exists():
            try:
                with open(config_path) as fh:
                    loaded = json.load(fh)
                    defaults.update(loaded.get("electronics", {}))
            except Exception as exc:
                logger.warning("Failed to load electronics config from %s: %s", config_path, exc)
        return defaults

    async def initialize(self):
        if self._initialized:
            return
        logger.info("[ElectronicsAgent] Initializing...")
        try:
            self._topo_lib = TopologyLibrary()
            self._kicad_exporter._topo_lib = self._topo_lib
            logger.info("TopologyLibrary loaded: %d topologies", len(self._topo_lib._topologies))
        except Exception as exc:
            logger.error("Failed to load TopologyLibrary: %s", exc)

        self._ngspice = NgSpiceEngine()
        logger.info("NgSpiceEngine: %s", "available" if self._ngspice.available() else "NOT FOUND")

        self._catalog = ComponentCatalog()
        self._initialized = True

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route by operation or goal type.

        Goal-based ops (new): params contains any of
          {"goal": {...}} | {"design_goal": {...}} | {"v_in": ..., "v_out": ...}

        Operation-based ops (legacy): params["operation"] = one of
          simulate_circuit | analyze_pcb | si_analysis | pi_analysis |
          thermal_analysis | drc_check | optimize_topology | design_topology
        """
        await self.initialize()
        operation = params.get("operation", "")

        # New goal-directed design flow
        if operation in ("design_topology", "design_power_supply", "design") or \
                "goal" in params or "design_goal" in params or \
                ("v_in" in params and "v_out" in params):
            return await self.design(params)

        # New specialized engine operations
        if operation in ("signal_integrity", "si_analysis_v2"):
            return await analyze_signal_integrity(params)
        if operation in ("power_integrity", "pi_analysis_v2", "pdn"):
            return await analyze_power_integrity(params)
        if operation in ("pcb_analysis", "si_pi"):
            return await analyze_pcb_si_pi(params)
        if operation in ("analog", "filter", "opamp", "adc_driver"):
            return await design_analog_circuit(params)
        if operation in ("digital", "fanout", "timing"):
            return await design_digital_interface(params)
        if operation in ("rf", "rf_design", "matching", "link_budget"):
            return await design_rf_circuit(params)
        if operation in ("emc", "emi", "cispr"):
            return await analyze_emc(params)
        if operation in ("manufacturing", "gerber", "fab"):
            return await generate_manufacturing_files(params)
        if operation in ("electronics_thermal", "junction_temp"):
            return await analyze_electronics_thermal(params)

        # Legacy operations
        dispatch = {
            "simulate_circuit": self._simulate_circuit,
            "analyze_pcb": self._analyze_pcb,
            "si_analysis": self._signal_integrity_analysis,
            "pi_analysis": self._power_integrity_analysis,
            "thermal_analysis": self._thermal_analysis,
            "drc_check": self._drc_check,
            "optimize_topology": self._optimize_topology,
        }
        handler = dispatch.get(operation)
        if handler:
            return await handler(params)

        # Fallback: power budget if components list provided
        if params.get("components"):
            return await self._power_budget_analysis(
                params["components"], params.get("power_supply", {}), params
            )

        return {"status": "error", "error": f"Unknown operation: '{operation}'"}

    # ------------------------------------------------------------------
    # Goal-directed design flow
    # ------------------------------------------------------------------

    async def design(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full design loop:
          parse → select topology → size → Phase-1 SPICE → validate/iterate
          → catalog lookup → Phase-2 SPICE → KiCad netlist
        """
        await self.initialize()
        t0 = time.time()
        logs: List[str] = []

        # 1. Parse goal
        try:
            goal = self._parse_goal(params)
        except Exception as exc:
            return {"status": "error", "error": f"Goal parsing failed: {exc}"}
        logs.append(f"Goal: {goal.project_name} | V_in={goal.v_in}V → V_out={goal.v_out}V "
                    f"@ {goal.i_out}A | f_sw={goal.f_sw/1e3:.0f}kHz | ΔV≤{goal.ripple_v*1000:.0f}mV")

        # 2. Select topology
        if not self._topo_lib:
            return {"status": "error", "error": "TopologyLibrary not loaded"}
        try:
            topo_key, topo, score = self._topo_lib.select_topology(goal)
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}
        logs.append(f"Topology: {topo_key} ({topo['name']}, score={score:.3f})")

        # 3. Size components
        base_ns = {
            "v_in": goal.v_in, "v_out": goal.v_out, "i_out": goal.i_out,
            "ripple_v": goal.ripple_v, "f_sw": goal.f_sw,
            "efficiency_min": goal.efficiency_min, "t_amb": goal.t_amb,
        }
        computed = self._topo_lib.evaluate_equations(topo_key, base_ns)
        computed.update(base_ns)
        logs.append(
            f"Sized: L={computed.get('l_value_uh', 0):.2f}µH  "
            f"C_out={computed.get('cap_out_uf', 0):.2f}µF  "
            f"duty={computed.get('duty', 0):.3f}"
        )

        # 4+5+6. Phase-1 SPICE with up to 5 iterations
        sim1_result: Optional[Dict[str, Any]] = None
        validation_report: List[str] = []

        meas: Dict[str, float] = {}
        if self._ngspice and self._ngspice.available():
            for attempt in range(5):
                netlist = self._build_phase1_netlist(goal, topo_key, computed)
                if not netlist:
                    logs.append("Phase 1: no SPICE template available for this topology")
                    break
                raw = await asyncio.to_thread(self._ngspice.run_netlist, netlist)
                if not raw["success"]:
                    logs.append(f"Phase 1 sim failed (attempt {attempt+1}): {raw.get('error','')}")
                    break
                meas = raw["measurements"]
                ok_checks, fail_checks = self._validate(topo_key, computed, meas)
                if fail_checks:
                    # Directly adjust component values — do NOT re-run evaluate_equations
                    # (that would overwrite the adjustments by recalculating from base formulas)
                    adj_log, computed = self._adjust_sizing(fail_checks, computed, meas)
                    logs.append(f"Phase 1 attempt {attempt+1}: {adj_log}")
                else:
                    sim1_result = {"measurements": meas, "attempt": attempt + 1}
                    logs.append(f"Phase 1: PASSED (attempt {attempt+1})")
                    validation_report = ok_checks
                    break
            else:
                logs.append("Phase 1: max iterations reached, proceeding with best sizing")
                sim1_result = {"measurements": meas, "attempt": 5}
        else:
            logs.append("Phase 1: ngspice not available — analytical sizing only")

        # 7. Catalog lookup
        bom_roles_meta = self._topo_lib.get_bom_roles(topo_key)
        bom: Dict[str, Optional[Dict]] = {}
        if self._catalog:
            logs.append("Catalog lookup: searching DigiKey / Mouser / Octopart...")
            try:
                bom = await self._catalog.search_all_roles(bom_roles_meta, computed)
                found = sum(1 for v in bom.values() if v)
                logs.append(f"Catalog: found {found}/{len(bom_roles_meta)} parts")
            except Exception as exc:
                logs.append(f"Catalog lookup failed: {exc}")

        # 8. Phase-2 SPICE (only if catalog found real component params)
        # For LDO: try to pull the manufacturer SPICE .lib before extracting params.
        # This runs regardless of catalog state — many TI/onsemi parts are downloadable
        # by MPN alone even without catalog pricing data.
        if topo_key == "ldo":
            ic_entry = bom.get("ldo_ic") or {}
            mpn_ic = ic_entry.get("mpn") or ""
            if mpn_ic and not ic_entry.get("_spice_model_path"):
                model_path = await self._fetch_spice_model(
                    mpn_ic, ic_entry.get("datasheet_url", "")
                )
                if model_path:
                    bom["ldo_ic"] = {**ic_entry, "_spice_model_path": model_path}
                    logs.append(f"LDO SPICE model: {Path(model_path).name}")
                else:
                    logs.append("LDO SPICE model: not found online — using behavioral model")

        sim2_result: Optional[Dict[str, Any]] = None
        phase2_params = self._extract_phase2_params(topo_key, bom, computed)
        if phase2_params and self._ngspice and self._ngspice.available():
            netlist2 = self._build_phase2_netlist(goal, topo_key, computed, phase2_params)
            if netlist2:
                raw2 = await asyncio.to_thread(self._ngspice.run_netlist, netlist2)
                if raw2["success"]:
                    sim2_result = {"measurements": raw2["measurements"]}
                    eff = self._compute_efficiency(topo_key, raw2["measurements"])
                    logs.append(f"Phase 2: PASSED | η={eff*100:.1f}%")
                else:
                    logs.append(f"Phase 2 sim failed: {raw2.get('error','')}")

        # 9a. Transformer magnetics design (for isolated topologies)
        magnetics_result: Optional[Dict[str, Any]] = None
        _TRANSFORMER_TOPOS = {"flyback", "forward", "full_bridge", "llc_resonant"}
        if topo_key in _TRANSFORMER_TOPOS and self._magnetics:
            try:
                if topo_key == "flyback":
                    magnetics_result = MagneticsDesignEngine.design_flyback_transformer(
                        v_in=goal.v_in, v_out=goal.v_out, i_out=goal.i_out,
                        f_sw=goal.f_sw, efficiency=goal.efficiency_min,
                    )
                elif topo_key == "forward":
                    magnetics_result = MagneticsDesignEngine.design_forward_transformer(
                        v_in=goal.v_in, v_out=goal.v_out, i_out=goal.i_out,
                        f_sw=goal.f_sw,
                    )
                elif topo_key == "full_bridge":
                    magnetics_result = MagneticsDesignEngine.design_full_bridge_transformer(
                        v_in=goal.v_in, v_out=goal.v_out, i_out=goal.i_out,
                        f_sw=goal.f_sw,
                    )
                elif topo_key == "llc_resonant":
                    magnetics_result = MagneticsDesignEngine.design_llc_resonant_tank(
                        v_in=goal.v_in, v_out=goal.v_out, i_out=goal.i_out,
                        f_sw=goal.f_sw,
                    )
                if magnetics_result:
                    logs.append(
                        f"Magnetics: Np={magnetics_result.get('np','?')}  "
                        f"Ns={magnetics_result.get('ns','?')}  "
                        f"n={magnetics_result.get('turns_ratio','?'):.4g}  "
                        f"Ap={magnetics_result.get('ap_cm4','?')}cm⁴"
                    )
            except Exception as exc:
                logs.append(f"Magnetics design failed: {exc}")

        # 9b. Inductor design for non-isolated CCM topologies
        elif topo_key in {"synchronous_buck", "boost"} and self._magnetics:
            try:
                magnetics_result = MagneticsDesignEngine.design_buck_inductor(
                    v_in=goal.v_in, v_out=goal.v_out, i_out=goal.i_out,
                    f_sw=goal.f_sw,
                )
                logs.append(
                    f"Inductor: L={magnetics_result['inductance_h']*1e6:.2f}µH  "
                    f"N={magnetics_result['n_turns']}T  "
                    f"AWG{magnetics_result['wire']['awg']}"
                )
            except Exception as exc:
                logs.append(f"Inductor design failed: {exc}")

        # 9c. Control loop compensation (buck and flyback voltage-mode)
        control_result: Optional[Dict[str, Any]] = None
        if topo_key in {"synchronous_buck", "flyback"} and self._control:
            try:
                l_h = computed.get("inductance_h", computed.get("inductance_uh", 10) * 1e-6)
                c_out_f = computed.get("cap_out_f", computed.get("cap_out_uf", 100) * 1e-6)
                # ESR: use catalog value if available, else conservative electrolytic estimate
                esr_catalog = (bom.get("cout") or {}).get("esr_mohm")
                esr_ohm = (esr_catalog / 1000.0) if esr_catalog else 0.050  # 50mΩ default
                r_load = goal.v_out / max(goal.i_out, 0.01)
                control_result = ControlLoopDesignEngine.recommend_compensator(
                    v_out=goal.v_out, v_ref=min(goal.v_out * 0.5, 2.5),
                    f_sw=goal.f_sw, l_h=l_h, c_out_f=c_out_f,
                    esr_ohm=esr_ohm, r_load_ohm=r_load,
                )
                logs.append(
                    f"Control loop: {control_result['compensator_type']}  "
                    f"fc={control_result['f_crossover_hz']:.0f}Hz  "
                    f"PM={control_result['phase_margin_deg']:.1f}°  "
                    f"{'STABLE' if control_result['stability_ok'] else 'MARGINAL'}"
                )
            except Exception as exc:
                logs.append(f"Control loop design failed: {exc}")

        # 9d. Build result dict
        design_result = {
            "topology_key": topo_key,
            "goal": {
                "v_in": goal.v_in, "v_out": goal.v_out, "i_out": goal.i_out,
                "ripple_v": goal.ripple_v, "f_sw": goal.f_sw,
            },
            "bom": bom,
            "bom_roles_meta": bom_roles_meta,
            "computed": computed,
        }
        kicad_netlist = self._kicad_exporter.export_netlist(design_result)

        elapsed = round((time.time() - t0) * 1000)
        return {
            "status": "success",
            "topology": topo_key,
            "topology_name": topo.get("name", topo_key),
            "goal": {
                "v_in_v": goal.v_in, "v_out_v": goal.v_out,
                "i_out_a": goal.i_out, "ripple_mv": goal.ripple_v * 1000,
                "f_sw_khz": goal.f_sw / 1e3,
            },
            "sizing": {
                "duty": round(computed.get("duty", 0), 4),
                "inductance_uh": round(computed.get("l_value_uh", 0), 3),
                "inductance_h": computed.get("inductance_h"),
                "cap_out_uf": round(computed.get("cap_out_uf", 0), 3),
                "cap_in_uf": round(computed.get("cap_in_f", 0) * 1e6, 3) if "cap_in_f" in computed else None,
                "mosfet_vds_min_v": round(computed.get("mosfet_vds_min", 0), 1),
                "mosfet_id_min_a": round(computed.get("mosfet_id_min", 0), 2),
                "efficiency_ideal_pct": round(computed.get("efficiency_ideal", 1.0) * 100, 1),
            },
            "simulation_phase1": sim1_result,
            "simulation_phase2": sim2_result,
            "validation": validation_report,
            "magnetics": magnetics_result,
            "control_loop": control_result,
            "bom": {
                role: {
                    "mpn": (p.get("mpn") if p else None),
                    "manufacturer": (p.get("manufacturer") if p else None),
                    "description": (p.get("description") if p else None),
                    "datasheet_url": (p.get("datasheet_url") if p else None),
                    "rds_on_mohm": (p.get("rds_on_mohm") if p else None),
                    "dcr_mohm": (p.get("dcr_mohm") if p else None),
                    "esr_mohm": (p.get("esr_mohm") if p else None),
                    "qg_nc": (p.get("qg_nc") if p else None),
                    "unit_price_usd": (p.get("unit_price_usd") if p else None),
                    "found": p is not None,
                }
                for role, p in bom.items()
            },
            "kicad_netlist": kicad_netlist,
            "logs": logs,
            "computation_ms": elapsed,
        }

    # ------------------------------------------------------------------
    # Goal parsing
    # ------------------------------------------------------------------

    def _parse_goal(self, params: Dict[str, Any]) -> DesignGoal:
        # Support nested goal dict or flat params; skip string values (NL descriptions)
        _raw = params.get("goal") or params.get("design_goal")
        g = _raw if isinstance(_raw, dict) else params
        def _f(key: str, default: float) -> float:
            return float(g.get(key, default))

        v_in = _f("v_in", _f("input_voltage", _f("supply_voltage", 12.0)))
        v_out = _f("v_out", _f("output_voltage", 5.0))
        i_out = _f("i_out", _f("output_current", _f("current_a", 1.0)))
        ripple_v = _f("ripple_v", _f("ripple_mv", 50.0) / 1000.0)
        f_sw = _f("f_sw", _f("switching_frequency_hz", _f("f_sw_khz", 500.0) * 1e3))
        efficiency_min = _f("efficiency_min", 0.80)
        t_amb = _f("t_amb", _f("ambient_temp_c", _f("ambient_temp", 25.0)))
        goal_type = str(g.get("goal_type", "power_supply")).lower()
        project_name = str(g.get("project_name", g.get("name",
            f"{v_in}V→{v_out}V@{i_out}A")))
        r_motor = _f("r_motor", v_out / max(i_out, 0.001) if goal_type == "motor_driver" else 0.0)
        l_motor = _f("l_motor", 1e-3)

        if v_in <= 0 or v_out <= 0 or i_out <= 0:
            raise ValueError(f"Invalid goal: v_in={v_in}, v_out={v_out}, i_out={i_out}")
        if ripple_v <= 0:
            ripple_v = v_out * 0.01  # default 1% ripple

        return DesignGoal(
            v_in=v_in, v_out=v_out, i_out=i_out,
            ripple_v=ripple_v, f_sw=f_sw,
            efficiency_min=efficiency_min, t_amb=t_amb,
            goal_type=goal_type, project_name=project_name,
            r_motor=r_motor, l_motor=l_motor,
        )

    # ------------------------------------------------------------------
    # Netlist building
    # ------------------------------------------------------------------

    def _build_spice_context(
        self, goal: DesignGoal, topo_key: str, computed: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build the format-string context dict for SPICE template substitution."""
        sim_cfg = self._topo_lib.get_sim_config() if self._topo_lib else {}
        cycles_total = sim_cfg.get("motor_cycles_total" if topo_key == "h_bridge"
                                   else "cycles_total", 8)
        cycles_skip = sim_cfg.get("cycles_skip", 3)
        steps_per_cycle = sim_cfg.get("steps_per_cycle", 200)

        t_period = 1.0 / goal.f_sw
        duty = computed.get("duty", goal.v_out / goal.v_in)
        t_on = duty * t_period
        t_step = t_period / steps_per_cycle
        t_end = cycles_total * t_period
        t_meas = cycles_skip * t_period

        ctx: Dict[str, Any] = {
            "project_name": goal.project_name,
            "vin": goal.v_in,
            "vout": goal.v_out,
            "iout": goal.i_out,
            "duty": duty,
            "t_on": t_on,
            "t_period": t_period,
            "t_step": t_step,
            "t_end": t_end,
            "t_meas": t_meas,
            "fsw_khz": goal.f_sw / 1e3,
            "vin_half": goal.v_in / 2.0,
            # Component values
            "l_h": computed.get("inductance_h", 0.0),
            "l_uh": computed.get("l_value_uh", 0.0),
            "c_out_f": computed.get("cap_out_f", 0.0),
            "c_out_uf": computed.get("cap_out_uf", 0.0),
            "c_in_f": computed.get("cap_in_f", 1e-6),
            "r_load": computed.get("r_load", goal.v_out / max(goal.i_out, 1e-6)),
            # Boost extras
            "iin_avg": computed.get("i_in_avg", goal.i_out),
            # Inductor initial conditions — exact steady-state values avoid LC transient
            # Buck: at HS turn-on, IL = valley (I_avg - ΔI/2)
            # Boost: at Sd turn-on (diode), IL = peak (I_in_avg + ΔI/2)
            "il_valley": max(0.0, goal.i_out - computed.get("ripple_i_l", 0.0) / 2),
            "il_peak": goal.i_out + computed.get("ripple_i_l", 0.0) / 2,
            "il_peak_boost": (computed.get("i_in_avg", goal.i_out)
                              + computed.get("ripple_i_l", 0.0) / 2),
            # LDO extras
            "v_ratio": goal.v_out / goal.v_in if goal.v_in > 0 else 1.0,
            "p_diss": computed.get("p_dissipated", (goal.v_in - goal.v_out) * goal.i_out),
            "eff_pct": computed.get("efficiency_ideal", goal.v_out / goal.v_in) * 100,
            # H-bridge extras
            "vsupply": goal.v_in,
            "imotor": goal.i_out,
            "r_motor": goal.r_motor if goal.r_motor > 0 else goal.v_out / max(goal.i_out, 1e-6),
            "l_motor": goal.l_motor,
            "bulk_cap_f": computed.get("bulk_cap_f", 1e-4),
            # Flyback / forward / full-bridge extras
            "turns_ratio": computed.get("turns_ratio", 1.0),
            "lp_h": computed.get("lp_h", 1e-5),
            "ls_h": computed.get("ls_h", 1e-5),
            "coupling": computed.get("coupling", 0.999),
            "ip_avg": computed.get("ip_avg", goal.i_out),
            # SEPIC extras
            "cs_f": computed.get("cs_f", 1e-6),
            "il1_avg": computed.get("il1_avg", goal.i_out),
            "il2_avg": computed.get("il2_avg", goal.i_out),
            # LLC extras
            "lr_h": computed.get("lr_h", 1e-5),
            "cr_f": computed.get("cr_f", 1e-8),
            "lm_h": computed.get("lm_h", 1e-4),
            "fn_nom": computed.get("fn_nom", 1.05),
            # Math constants for YAML expressions
            "pi": 3.14159265358979,
        }
        if extra:
            ctx.update(extra)
        return ctx

    def _build_phase1_netlist(
        self, goal: DesignGoal, topo_key: str, computed: Dict[str, Any]
    ) -> Optional[str]:
        if not self._topo_lib:
            return None
        template = self._topo_lib.get_spice_template(topo_key, 1)
        if not template.strip():
            return None
        ctx = self._build_spice_context(goal, topo_key, computed)
        netlist = self._ngspice.fill_template(template, ctx)  # type: ignore[union-attr]
        # Patch: replace single-PWM invert=TRUE pattern with dual-PWM (backward compat)
        netlist = self._patch_complementary_switches(netlist, ctx)
        return netlist

    def _build_phase2_netlist(
        self,
        goal: DesignGoal,
        topo_key: str,
        computed: Dict[str, Any],
        phase2_params: Dict[str, Any],
    ) -> Optional[str]:
        if not self._topo_lib:
            return None
        ctx = self._build_spice_context(goal, topo_key, computed, extra=phase2_params)
        # LDO with real manufacturer model: select fixed or adjustable template
        if topo_key == "ldo" and phase2_params.get("use_manufacturer_model"):
            tmpl = (
                self._LDO_ADJ_MANUFACTURER_TEMPLATE
                if phase2_params.get("is_adjustable")
                else self._LDO_MANUFACTURER_TEMPLATE
            )
            return self._ngspice.fill_template(tmpl, ctx)  # type: ignore[union-attr]
        template = self._topo_lib.get_spice_template(topo_key, 2)
        if not template.strip():
            return None
        return self._ngspice.fill_template(template, ctx)  # type: ignore[union-attr]

    @staticmethod
    def _detect_subckt_info(lib_path: str) -> Dict[str, Any]:
        """Return name, node list, and is_adjustable from the first .SUBCKT in a .lib file."""
        try:
            with open(lib_path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    stripped = line.strip().upper()
                    if stripped.startswith(".SUBCKT"):
                        parts = stripped.split()
                        if len(parts) >= 2:
                            name = parts[1]
                            nodes = parts[2:] if len(parts) > 2 else []
                            is_adj = any("ADJ" in n for n in nodes)
                            return {"name": name, "nodes": nodes, "is_adjustable": is_adj}
        except OSError:
            pass
        return {"name": "LDO1", "nodes": [], "is_adjustable": False}

    @staticmethod
    def _preprocess_ltspice_lib(raw_bytes: bytes) -> str:
        """
        Normalize LTspice .lib/.sub files for ngspice-46 compatibility.

        Fixes (all verified against ngspice-46 behavior):
          1. Latin-1 micro sign 0xB5 → 'u'  (ngspice chokes on non-UTF-8 byte sequences)
          2. Trailing 'load' keyword on current-source lines → removed
             (LTspice extension; ngspice reports "unknown parameter (load)")
          3. 'tol=N' on passive lines → removed  (LTspice tolerance annotation)
          4. Inline **…** comments on .SUBCKT/.ENDS lines → stripped
             (ngspice can mistake them for extra node names)
        """
        raw_bytes = raw_bytes.replace(b'\xb5', b'u')
        text = raw_bytes.decode("utf-8", errors="replace")
        text = re.sub(r'\s+load\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+tol=\S+', '', text, flags=re.IGNORECASE)
        # Non-greedy [^\n]*? stops at the leftmost '  **' on .SUBCKT/.ENDS lines.
        # The greedy (?:\s+\S+)* alternative wrongly consumed '**foo' as a node name.
        text = re.sub(
            r'^(\.(SUBCKT|ENDS)\b[^\n]*?)\s+\*\*[^\n]*$',
            r'\1', text, flags=re.IGNORECASE | re.MULTILINE,
        )
        return text

    async def _fetch_spice_model(self, mpn: str, datasheet_url: str = "") -> Optional[str]:
        """
        Download an ngspice-compatible SPICE .lib for the given MPN.

        Search order:
          1. BRICK_SPICE_MODEL_DIR (user-provided; ROHM downloads go here)
          2. Disk cache (/tmp/brick_spice_models/)
          3. _KNOWN_GOOD_SPICE_URLS — verified GitHub models (LTspice-preprocessed)
          4. onsemi confirmed URL pattern — fetched but rejected if **$ENCRYPTED_LIB

        LTspice compatibility: all downloaded content is passed through
        _preprocess_ltspice_lib() before caching to fix micro signs, 'load'
        keyword, tol= annotations, and inline ** comments on .SUBCKT/.ENDS lines.

        Manufacturer landscape (verified 2025-05):
          - onsemi: URL pattern confirmed but files are **$ENCRYPTED_LIB (PSpice only)
          - TI: opaque ZIP IDs, no predictable direct URL, many encrypted
          - ROHM: 3,500+ unencrypted LTspice models; requires per-page download
          - GitHub (chrisnoisel/ltspice, analogspice): community LTspice models,
            LM317 verified OK; others untested

        Returns local cache file path, or None (caller falls back to behavioral model).
        """
        try:
            import aiohttp
        except ImportError:
            return None

        mpn_lc = mpn.lower().replace("-", "")
        mpn_uc = mpn.upper()

        cache_dir = Path(tempfile.gettempdir()) / "brick_spice_models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{mpn_lc}.lib"

        def _save_and_return(raw: bytes, source: str) -> Optional[str]:
            text = self._preprocess_ltspice_lib(raw)
            if "**$ENCRYPTED_LIB" in text:
                logger.warning(
                    "SPICE model for %s from %s is Cadence-encrypted "
                    "(not usable with ngspice). Using behavioral model. "
                    "For a real model, download from rohm.com/support/design-model "
                    "and place in BRICK_SPICE_MODEL_DIR.",
                    mpn, source,
                )
                return None
            text_upper = text.upper()
            if ".SUBCKT" not in text_upper and ".MODEL" not in text_upper:
                logger.debug("SPICE model from %s has no subcircuit/model definition", source)
                return None
            cache_path.write_text(text, encoding="utf-8")
            logger.info("SPICE model for %s from %s → %s", mpn, source, cache_path)
            return str(cache_path)

        # 1. User-provided local directory (ROHM or manually downloaded models)
        user_model_dir = Path(os.environ.get("BRICK_SPICE_MODEL_DIR", ""))
        for candidate in (
            (user_model_dir / f"{mpn_lc}.lib") if user_model_dir.name else None,
            (user_model_dir / f"{mpn_uc}.lib") if user_model_dir.name else None,
        ):
            if candidate and candidate.exists() and candidate.stat().st_size > 100:
                result = _save_and_return(candidate.read_bytes(), str(candidate))
                if result:
                    return result

        # 2. Disk cache (already preprocessed from a prior run)
        if cache_path.exists() and cache_path.stat().st_size > 100:
            text = cache_path.read_text(errors="replace")
            if "**$ENCRYPTED_LIB" not in text:
                logger.info("SPICE model for %s from cache: %s", mpn, cache_path)
                return str(cache_path)
            cache_path.unlink()

        # 3 + 4. Network: known-good GitHub URLs first, then onsemi pattern
        known_url = self._KNOWN_GOOD_SPICE_URLS.get(mpn_lc)
        onsemi_candidates: List[str] = [
            f"https://www.onsemi.com/download/models/lib/{mpn_lc}%20simulation%20model.lib",
            f"https://www.onsemi.com/download/models/lib/{mpn_lc}%20(spice%20model).lib",
            f"https://www.onsemi.com/download/models/lib/{mpn_lc}%20pspice%20model.lib",
            f"https://www.onsemi.com/download/models/lib/{mpn_lc}_model.lib",
            f"https://www.onsemi.com/download/models/lib/{mpn_lc}.lib",
        ]
        url_candidates: List[str] = ([known_url] if known_url else []) + onsemi_candidates

        headers = {"User-Agent": "Mozilla/5.0 (compatible; BRICK-OS-EDA/1.0)"}
        timeout = aiohttp.ClientTimeout(total=15)
        try:
            async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
                for url in url_candidates:
                    try:
                        async with session.get(url, allow_redirects=True) as resp:
                            if resp.status != 200:
                                continue
                            raw = await resp.read()
                            result = _save_and_return(raw, url)
                            if result:
                                return result
                    except Exception as exc:
                        logger.debug("SPICE model URL %s: %s", url, exc)
        except Exception as exc:
            logger.debug("SPICE model fetch failed for %s: %s", mpn, exc)

        logger.debug("No ngspice-compatible SPICE model found for %s; using behavioral model", mpn)
        return None

    def _patch_complementary_switches(
        self, netlist: str, ctx: Dict[str, Any]
    ) -> str:
        """
        If the YAML template uses the 'invert=TRUE' pattern (which doesn't
        reliably produce complementary behavior in ngspice-46), rewrite to
        use an explicit complementary PWM behavioral source.
        """
        if "invert=TRUE" not in netlist and "invert=true" not in netlist:
            return netlist
        vin = ctx["vin"]
        vin_half = ctx["vin_half"]
        t_on = ctx["t_on"]
        t_period = ctx["t_period"]
        # Insert a complementary behavioral source after the first Vpwm line
        comp_line = (
            f"\n* Complementary PWM (auto-generated by BRICK patch)\n"
            f"Vpwm_n pwm_n 0 PULSE({vin:.6g} 0 0 1e-9 1e-9 {t_on:.11g} {t_period:.11g})\n"
        )
        # Replace invert=TRUE switch control to use pwm_n
        netlist = re.sub(r"(S\w+\s+\w+\s+\w+\s+)0\s+pwm(\s+\w+)", r"\1pwm_n\2", netlist)
        # Remove invert=TRUE from model
        netlist = re.sub(r"\s+invert=TRUE", "", netlist)
        # Lower Vt to vin/2 in all models
        netlist = re.sub(r"Vt=0\.5", f"Vt={vin_half:.6g}", netlist)
        # Insert complement source after first PULSE line
        netlist = re.sub(
            r"(Vpwm\s+pwm\s+0\s+PULSE[^\n]*\n)", r"\1" + comp_line, netlist, count=1
        )
        return netlist

    # ------------------------------------------------------------------
    # Validation and adjustment
    # ------------------------------------------------------------------

    def _validate(
        self,
        topo_key: str,
        computed: Dict[str, Any],
        meas: Dict[str, float],
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Run validation checks from YAML.
        Returns (ok_messages, fail_list).
        fail_list items: {"check": name, "action": action_on_fail, ...}
        """
        if not self._topo_lib:
            return [], []
        checks = self._topo_lib.get_validation(topo_key)
        ok: List[str] = []
        fail: List[Dict[str, Any]] = []

        safe_ns = {
            **computed,
            "vout_avg": meas.get("vout_avg", computed.get("v_out", 0)),
            "vout_pp": meas.get("vout_pp", 0),
            "il_avg": meas.get("il_avg", 0),
            "il_pp": meas.get("il_pp", 0),
            "il_min": meas.get("il_min", 0),
            "imotor_avg": meas.get("imotor_avg", 0),
            "vsupply_pp": meas.get("vsupply_pp", 0),
            "p_in": meas.get("p_in", 0),
            "p_out": meas.get("p_out", 0),
            "sqrt": math.sqrt, "abs": abs, "max": max, "min": min,
            "__builtins__": {},
        }

        for check_name, spec in checks.items():
            expr = spec.get("expr", "0")
            try:
                TopologyLibrary._validate_expr(expr)
                val = float(eval(expr, {"__builtins__": {}}, safe_ns))  # noqa: S307
            except Exception as exc:
                logger.warning("Validation check %r: eval failed for %r: %s", check_name, expr, exc)
                continue

            passed = True
            limit_str = ""

            if "max" in spec:
                limit = float(spec["max"])
                if val > limit:
                    passed = False
                    limit_str = f"max={limit}"
            if "max_from_goal" in spec:
                try:
                    mgexpr = spec["max_from_goal"]
                    TopologyLibrary._validate_expr(mgexpr)
                    limit = float(eval(mgexpr, {"__builtins__": {}}, safe_ns))  # noqa: S307
                    if val > limit:
                        passed = False
                        limit_str = f"max={limit:.3f}"
                except Exception as exc:
                    logger.warning("max_from_goal eval failed for check %r: %s", check_name, exc)
            if "min" in spec:
                limit = float(spec["min"])
                if val < limit:
                    passed = False
                    limit_str = f"min={limit}"

            unit = spec.get("unit", "")
            if passed:
                ok.append(f"✓ {check_name}: {val:.4g} {unit}")
            else:
                fail.append({
                    "check": check_name,
                    "value": val,
                    "limit_str": limit_str,
                    "unit": unit,
                    "action": spec.get("action_on_fail", ""),
                })

        return ok, fail

    def _adjust_sizing(
        self,
        fail_checks: List[Dict[str, Any]],
        computed: Dict[str, Any],
        meas: Dict[str, float],
    ) -> Tuple[str, Dict[str, Any]]:
        """Apply correction strategy for each failed check. Returns (log, updated_computed)."""
        updated = dict(computed)
        msgs: List[str] = []

        for fc in fail_checks:
            action = fc.get("action", "")
            val = fc.get("value", 0.0)

            if action == "increase_cap_out":
                target_ripple = computed.get("ripple_v", 0.05)
                actual_ripple = meas.get("vout_pp", val)
                if actual_ripple > 0 and target_ripple > 0:
                    factor = min((actual_ripple / target_ripple) * 1.3, 3.0)
                    old_c = updated.get("cap_out_f", 1e-6)
                    updated["cap_out_f"] = old_c * factor
                    updated["cap_out_uf"] = updated["cap_out_f"] * 1e6
                    msgs.append(f"cap_out ×{factor:.2f} → {updated['cap_out_uf']:.2f}µF")

            elif action == "increase_inductance":
                ripple_i = updated.get("ripple_i_l", computed.get("i_out", 1) * 0.3)
                il_min = meas.get("il_min", 0.0)
                if il_min < 0 and ripple_i > 0:
                    factor = min(1.0 + abs(il_min) / (ripple_i * 0.5), 2.5)
                    old_l = updated.get("inductance_h", 1e-6)
                    updated["inductance_h"] = old_l * factor
                    updated["l_value_uh"] = updated["inductance_h"] * 1e6
                    updated["l_i_sat_min"] = (
                        (updated.get("i_out", 1) + ripple_i / 2)
                        * updated.get("inductor_sat_margin", 1.25)
                    )
                    msgs.append(f"inductance ×{factor:.2f} → {updated['l_value_uh']:.2f}µH")

            elif action == "adjust_duty":
                # In ideal Phase 1, duty is exact by V-s balance — do not adjust.
                # A vout error here means startup transient, not a design error.
                vout_avg = meas.get("vout_avg", 0)
                v_out = computed.get("v_out", 0)
                if vout_avg > 0 and v_out > 0:
                    err_pct = abs(vout_avg - v_out) / v_out * 100
                    msgs.append(f"vout_avg={vout_avg:.3f}V vs target {v_out:.3f}V "
                                f"({err_pct:.1f}% — likely startup transient, duty unchanged)")

            elif action == "increase_bulk_cap":
                vsupply_pp = meas.get("vsupply_pp", val)
                supply_ripple_ratio = updated.get("supply_ripple_ratio", 0.05)
                v_in = updated.get("v_in", computed.get("v_in", 12))
                target_ripple = v_in * supply_ripple_ratio
                if vsupply_pp > 0 and target_ripple > 0:
                    factor = min((vsupply_pp / target_ripple) * 1.3, 3.0)
                    old_c = updated.get("bulk_cap_f", 1e-4)
                    updated["bulk_cap_f"] = old_c * factor
                    updated["bulk_cap_uf"] = updated["bulk_cap_f"] * 1e6
                    msgs.append(f"bulk_cap ×{factor:.2f} → {updated['bulk_cap_uf']:.1f}µF")

        log = " | ".join(msgs) if msgs else "no adjustment found"
        return log, updated

    # ------------------------------------------------------------------
    # Phase-2 parameter extraction
    # ------------------------------------------------------------------

    def _extract_phase2_params(
        self,
        topo_key: str,
        bom: Dict[str, Optional[Dict]],
        computed: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Extract R_ds_on, ESR, DCR etc. from catalog parts for Phase-2 netlist.
        Returns None if insufficient data to run Phase 2.
        """
        p: Dict[str, Any] = {}

        if topo_key == "buck":
            hs = bom.get("mosfet_hs") or {}
            ls = bom.get("mosfet_ls") or hs  # same part for synchronous
            ind = bom.get("inductor") or {}
            cap = bom.get("cap_out") or {}
            ron_hs = (hs.get("rds_on_mohm", 0) or 0) / 1000.0
            ron_ls = (ls.get("rds_on_mohm", 0) or 0) / 1000.0
            dcr_l = (ind.get("dcr_mohm", 0) or 0) / 1000.0
            esr_cout = (cap.get("esr_mohm", 0) or 0) / 1000.0
            if not (ron_hs > 0 and ron_ls > 0 and dcr_l >= 0 and esr_cout >= 0):
                return None
            p.update(ron_hs=ron_hs, ron_ls=ron_ls, dcr_l=dcr_l, esr_cout=esr_cout,
                     mpn_hs=hs.get("mpn", ""), mpn_ls=ls.get("mpn", ""),
                     mpn_l=ind.get("mpn", ""), mpn_cout=cap.get("mpn", ""))

        elif topo_key == "boost":
            sw = bom.get("mosfet_sw") or {}
            d = bom.get("diode") or {}
            ind = bom.get("inductor") or {}
            cap = bom.get("cap_out") or {}
            ron_sw = (sw.get("rds_on_mohm", 0) or 0) / 1000.0
            ron_d = (d.get("esr_mohm", 50) or 50) / 1000.0  # diode forward resistance approx
            dcr_l = (ind.get("dcr_mohm", 0) or 0) / 1000.0
            esr_cout = (cap.get("esr_mohm", 0) or 0) / 1000.0
            if not ron_sw > 0:
                return None
            p.update(ron_sw=ron_sw, ron_d=ron_d, dcr_l=dcr_l, esr_cout=esr_cout,
                     mpn_sw=sw.get("mpn", ""), mpn_d=d.get("mpn", ""),
                     mpn_l=ind.get("mpn", ""), mpn_cout=cap.get("mpn", ""))

        elif topo_key == "ldo":
            ic  = bom.get("ldo_ic") or {}
            cap = bom.get("cap_out") or {}
            # Dropout voltage — from catalog datasheet field or 0.5V conservative default
            # (TI LP2985: 178mV typ; AMS1117: 1.3V; generous default covers most LDOs)
            v_do = float(ic.get("v_dropout_v") or 0.5)
            # Quiescent current — from catalog or 50µA typical for low-power LDOs
            # Source: TI SLVA71B §2.1 "Quiescent current"
            i_q_ua = float(ic.get("i_quiescent_ua") or 50.0)
            i_q = i_q_ua * 1e-6
            # Output cap ESR — ceramic default 5mΩ, electrolytic 50mΩ
            # Source: Murata GRM series (ceramic); Panasonic FM series (electrolytic)
            esr_cout = (cap.get("esr_mohm") or 5.0) / 1000.0
            p.update(
                v_do=v_do, i_q=i_q, i_q_ua=i_q_ua,
                esr_cout=esr_cout, esr_cout_mohm=esr_cout * 1000.0,
                c_out_f=computed.get("cap_out_f", 1e-6),
                c_in_f=computed.get("cap_in_f", 1e-7),
            )
            # If _fetch_spice_model succeeded in design(), use the real manufacturer model.
            spice_path = ic.get("_spice_model_path", "")
            if spice_path:
                info = self._detect_subckt_info(spice_path)
                subckt_name = info["name"]
                # Detect adjustable by MPN name (more reliable than node-name scan
                # since many adjustable LDO subcircuits use numeric node labels).
                mpn_ic = ic.get("mpn", "")
                mpn_key = mpn_ic.lower().replace("-", "")
                is_adjustable = mpn_key in self._ADJUSTABLE_LDO_MPNS or info["is_adjustable"]
                # R1/R2 voltage-divider for adjustable LDOs (LM317 family).
                # R1: OUT→ADJ (reference resistor, standard 240Ω per TI SLVS044).
                # R2: ADJ→GND, sized to produce desired V_out.
                # V_out = V_ref × (1 + R2/R1); V_ref = 1.25V for LM117/LM217/LM317.
                v_ref_ldo = 1.25
                r1_adj = 240.0
                vout_val = computed.get("v_out") or p.get("vout", 5.0)
                r2_set = r1_adj * max(0.0, (vout_val / v_ref_ldo) - 1.0)
                p.update(
                    use_manufacturer_model=True,
                    spice_model_path=spice_path,
                    spice_model_name=subckt_name,
                    mpn_ic=mpn_ic,
                    is_adjustable=is_adjustable,
                    r1_adj=r1_adj,
                    r2_set=r2_set,
                )

        elif topo_key == "h_bridge":
            q = bom.get("mosfet_q1234") or {}
            ron = (q.get("rds_on_mohm", 0) or 0) / 1000.0
            if not ron > 0:
                return None
            p.update(ron=ron, mpn_q=q.get("mpn", ""))

        return p if p else None

    def _compute_efficiency(self, topo_key: str, meas: Dict[str, float]) -> float:
        p_in = abs(meas.get("p_in", 0))
        p_out = abs(meas.get("p_out", 0))
        if p_in > 0:
            return min(p_out / p_in, 1.0)
        return 0.0

    # ------------------------------------------------------------------
    # Legacy circuit simulation
    # ------------------------------------------------------------------

    async def _simulate_circuit(self, params: Dict[str, Any]) -> Dict[str, Any]:
        circuit_data = params.get("circuit", {})
        analysis_type = params.get("analysis_type", "op")
        circuit = self._parse_circuit(circuit_data)

        raw_components = params.get("components", [])
        if raw_components:
            return await self._power_budget_analysis(
                raw_components, params.get("power_supply", {}), params
            )
        return await self._analytical_mna(circuit, analysis_type, params)

    async def _analytical_mna(
        self, circuit: Circuit, analysis_type: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        t0 = time.time()
        raw_components = params.get("components", [])
        if raw_components:
            return await self._power_budget_analysis(
                raw_components, params.get("power_supply", {}), params
            )

        components = circuit.components
        if not components:
            return {"status": "no_circuit", "fidelity": "analytical",
                    "message": "No components or netlist provided."}

        nodes: set = {"GND"}
        for comp in components.values():
            for pin in getattr(comp, "connections", {}).values():
                nodes.add(str(pin))
        node_list = ["GND"] + sorted(n for n in nodes if n != "GND")
        n_nodes = len(node_list)
        node_idx = {n: i for i, n in enumerate(node_list)}

        G = np.zeros((n_nodes, n_nodes))
        I_vec = np.zeros(n_nodes)
        voltage_sources: List[Any] = []

        for cid, comp in components.items():
            ctype = (getattr(comp, "type", "") or "").lower()
            val = getattr(comp, "value", 0.0) or 0.0
            conns = getattr(comp, "connections", {})
            keys = list(conns.keys())

            if ctype == "resistor" and val > 0 and len(keys) >= 2:
                n1 = node_idx.get(str(conns[keys[0]]), 0)
                n2 = node_idx.get(str(conns[keys[1]]), 0)
                g = 1.0 / val
                G[n1, n1] += g; G[n2, n2] += g
                G[n1, n2] -= g; G[n2, n1] -= g
            elif ctype in ("voltage_source", "vdc", "battery") and len(keys) >= 2:
                voltage_sources.append((cid, val, conns, keys))
            elif ctype in ("current_source", "idc") and len(keys) >= 2:
                n1 = node_idx.get(str(conns[keys[0]]), 0)
                n2 = node_idx.get(str(conns[keys[1]]), 0)
                I_vec[n1] -= val; I_vec[n2] += val

        n_vs = len(voltage_sources)
        try:
            if n_vs:
                size = n_nodes + n_vs
                A = np.zeros((size, size)); b = np.zeros(size)
                A[:n_nodes, :n_nodes] = G; b[:n_nodes] = I_vec
                for k, (cid, vs_val, conns, keys) in enumerate(voltage_sources):
                    row = n_nodes + k
                    n1 = node_idx.get(str(conns[keys[0]]), 0)
                    n2 = node_idx.get(str(conns[keys[1]]), 0)
                    A[row, n1] = 1; A[row, n2] = -1
                    A[n1, row] = 1; A[n2, row] = -1
                    b[row] = vs_val
                A[0, :] = 0; A[0, 0] = 1; b[0] = 0
                x = np.linalg.solve(A, b)
                node_voltages = {node_list[i]: round(float(x[i]), 6) for i in range(n_nodes)}
                vs_currents = {voltage_sources[k][0]: round(float(x[n_nodes + k]), 6)
                               for k in range(n_vs)}
            else:
                G[0, :] = 0; G[0, 0] = 1; I_vec[0] = 0
                x = np.linalg.solve(G, I_vec)
                node_voltages = {node_list[i]: round(float(x[i]), 6) for i in range(n_nodes)}
                vs_currents = {}
        except np.linalg.LinAlgError:
            node_voltages = {n: 0.0 for n in node_list}
            vs_currents = {}

        power_per_comp: Dict[str, float] = {}
        total_power_w = 0.0
        for cid, comp in components.items():
            ctype = (getattr(comp, "type", "") or "").lower()
            val = getattr(comp, "value", 0.0) or 0.0
            conns = getattr(comp, "connections", {})
            keys = list(conns.keys())
            if ctype == "resistor" and val > 0 and len(keys) >= 2:
                v1 = node_voltages.get(str(conns[keys[0]]), 0.0)
                v2 = node_voltages.get(str(conns[keys[1]]), 0.0)
                p = ((v1 - v2) ** 2) / val
                power_per_comp[cid] = round(p, 6)
                total_power_w += p

        return {
            "status": "success",
            "fidelity": "analytical_mna",
            "analysis_type": analysis_type,
            "node_voltages": node_voltages,
            "voltage_source_currents": vs_currents,
            "power_dissipation_w": power_per_comp,
            "total_power_w": round(total_power_w, 4),
            "node_count": n_nodes,
            "component_count": len(components),
            "computation_ms": round((time.time() - t0) * 1000, 2),
        }

    async def _power_budget_analysis(
        self,
        components: List[Dict],
        power_supply: Dict,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        supply_voltage = float(power_supply.get("voltage", 5.0))
        supply_cap_mah = float(power_supply.get("capacity_mah", 0))
        total_current_ma = 0.0
        total_power_mw = 0.0
        comp_results: List[Dict] = []
        warnings: List[str] = []

        for comp in components:
            name = comp.get("name", comp.get("type", "unknown"))
            voltage = float(comp.get("voltage", supply_voltage))
            current = float(comp.get("current_ma", comp.get("current", 0)))
            power = voltage * current
            if voltage > supply_voltage * 1.05:
                warnings.append(
                    f"{name}: requires {voltage}V but supply is {supply_voltage}V"
                )
            total_current_ma += current
            total_power_mw += power
            comp_results.append({
                "name": name, "voltage_v": voltage,
                "current_ma": current, "power_mw": round(power, 2),
            })

        supply_current_ma = float(
            power_supply.get("max_current_ma", power_supply.get("current_ma", 0))
        )
        if supply_current_ma and total_current_ma > supply_current_ma:
            warnings.append(
                f"Total draw {total_current_ma:.0f} mA exceeds supply {supply_current_ma:.0f} mA"
            )

        battery_life_h = None
        if supply_cap_mah and total_current_ma:
            battery_life_h = round(supply_cap_mah / total_current_ma, 2)

        signal_notes: List[str] = []
        for comp in components:
            iface = (comp.get("interface") or "").upper()
            if iface == "SPI":
                signal_notes.append(f"{comp.get('name','?')}: SPI — max trace ~150 mm at 10 MHz")
            elif iface == "I2C":
                signal_notes.append(f"{comp.get('name','?')}: I²C — add 4.7 kΩ pull-ups on SDA/SCL")
            elif iface == "UART":
                signal_notes.append(f"{comp.get('name','?')}: UART — max 15 m at 9600 baud")
            elif iface in ("PWM", "PWM/PPM"):
                signal_notes.append(f"{comp.get('name','?')}: PWM — 100 Ω series resistor to damp ringing")

        return {
            "status": "success",
            "fidelity": "analytical_power_budget",
            "supply_voltage_v": supply_voltage,
            "total_current_ma": round(total_current_ma, 2),
            "total_power_w": round(total_power_mw / 1000.0, 4),
            "total_power_mw": round(total_power_mw, 2),
            "battery_life_hours": battery_life_h,
            "components": comp_results,
            "signal_integrity": signal_notes,
            "warnings": warnings,
            "component_count": len(components),
        }

    # ------------------------------------------------------------------
    # Legacy PCB / SI / PI / thermal / DRC
    # ------------------------------------------------------------------

    async def _analyze_pcb(self, params: Dict[str, Any]) -> Dict[str, Any]:
        pcb_data = params.get("pcb", {})
        traces = pcb_data.get("traces", [])
        vias = pcb_data.get("vias", [])
        power_planes = pcb_data.get("power_planes", [])
        trace_results = [self._analyze_trace(t) for t in traces]
        via_results = [self._analyze_via(v) for v in vias]
        plane_results = [self._analyze_power_plane(p) for p in power_planes]
        return {
            "status": "success",
            "method": "pcb_analysis",
            "trace_count": len(traces),
            "via_count": len(vias),
            "trace_analyses": trace_results,
            "via_analyses": via_results,
            "power_plane_analyses": plane_results,
            "recommendations": self._generate_pcb_recommendations(trace_results, via_results),
        }

    def _analyze_trace(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        width_mil = trace.get("width_mil", 10)
        thickness_mil = trace.get("thickness_mil", 1.4)
        length_mil = trace.get("length_mil", 1000)
        layer = trace.get("layer", "external")
        delta_T = trace.get("temperature_rise_c", 10)
        k = 0.048 if layer == "external" else 0.024
        A = width_mil * thickness_mil
        I_max = k * (delta_T ** 0.44) * (A ** 0.725)
        rho = 0.688
        R = rho * (length_mil / 1000) / A
        h = trace.get("dielectric_height_mil", 10)
        er = trace.get("dielectric_constant", self.config.get("pcb_dielectric_constant", 4.5))
        Z0 = (87 / math.sqrt(er + 1.41)) * math.log(5.98 * h / (0.8 * width_mil + thickness_mil))
        return {
            "trace_id": trace.get("id", "unknown"),
            "current_capacity_a": float(I_max),
            "resistance_ohm": float(R),
            "impedance_ohm": float(Z0),
            "layer": layer,
            "width_mil": width_mil,
        }

    def _analyze_via(self, via: Dict[str, Any]) -> Dict[str, Any]:
        diameter_mil = via.get("diameter_mil", 10)
        plating_mil = via.get("plating_thickness_mil", 1.0)
        board_thickness_mil = via.get("board_thickness_mil", 62)
        r_outer = diameter_mil / 2
        r_inner = r_outer - plating_mil
        A_barrel = math.pi * (r_outer ** 2 - r_inner ** 2)
        rho = 0.688
        R = rho * (board_thickness_mil / 1000) / A_barrel
        h = board_thickness_mil * 0.0254
        d = diameter_mil * 0.0254
        L_nh = 2 * h * (math.log(4 * h / max(d, 1e-9)) + 1)
        return {
            "via_id": via.get("id", "unknown"),
            "resistance_mohm": float(R * 1000),
            "current_capacity_a": float(A_barrel),
            "inductance_nh": float(L_nh),
        }

    def _analyze_power_plane(self, plane: Dict[str, Any]) -> Dict[str, Any]:
        thickness_oz = plane.get("copper_weight_oz", 1.0)
        current_a = plane.get("current_a", 1.0)
        R_sheet = 0.5 / thickness_oz  # mΩ/sq for 1 oz copper
        V_drop = current_a * R_sheet / 1000
        return {
            "plane_id": plane.get("id", "unknown"),
            "sheet_resistance_mohm_sq": float(R_sheet),
            "estimated_ir_drop_mv": float(V_drop * 1000),
        }

    async def _signal_integrity_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        nets = params.get("nets", [])
        frequency_mhz = params.get("frequency_mhz", 100)
        c = 11.8  # inch/ns
        er = self.config.get("pcb_dielectric_constant", 4.5)
        wavelength_inch = c / (frequency_mhz / 1000 * math.sqrt(er))
        results = []
        for net in nets:
            length_inch = net.get("length_inch", 0)
            is_critical = length_inch > wavelength_inch / 10
            net_result: Dict[str, Any] = {
                "net_name": net.get("name"),
                "is_critical_length": is_critical,
                "wavelength_inch": round(wavelength_inch, 2),
            }
            if is_critical:
                Z_trace = net.get("trace_impedance_ohm", 50)
                Z_receiver = net.get("receiver_impedance_ohm", float("inf"))
                gamma = (Z_receiver - Z_trace) / (Z_receiver + Z_trace)
                net_result["reflection_coefficient"] = round(gamma, 3)
                net_result["reflection_percent"] = round(abs(gamma) * 100, 1)
                if abs(gamma) > 0.1:
                    net_result["recommendation"] = "Add termination — reflections >10%"
            results.append(net_result)
        return {
            "status": "success",
            "method": "signal_integrity_analysis",
            "frequency_mhz": frequency_mhz,
            "critical_nets": sum(1 for r in results if r.get("is_critical_length")),
            "net_analyses": results,
        }

    async def _power_integrity_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        voltage_v = params.get("voltage_v", 3.3)
        current_a = params.get("current_a", 1.0)
        max_ripple_mv = params.get("max_ripple_mv", 50)
        I_transient = current_a * 0.5
        Z_target = (max_ripple_mv / 1000) / max(I_transient, 1e-9)
        caps = self._calculate_decoupling_caps(voltage_v, Z_target)
        plane_resistance = params.get("plane_resistance_mohm", 10)
        plane_inductance = params.get("plane_inductance_nh", 10)
        frequencies = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
        pdn_impedance = []
        for f in frequencies:
            omega = 2 * math.pi * f
            Z_caps_inv = complex(0)
            for cap in caps:
                C = cap["value_f"]
                ESR = cap.get("esr_ohm", 0.1)
                ESL = cap.get("esl_nh", 1) * 1e-9
                Z_cap = ESR + 1j * omega * ESL + 1 / (1j * omega * C)
                if Z_cap != 0:
                    Z_caps_inv += 1 / Z_cap
            Z_caps = 1 / Z_caps_inv if Z_caps_inv != 0 else complex(1e9)
            Z_plane = plane_resistance / 1000 + 1j * omega * plane_inductance * 1e-9
            Z_total = Z_plane + Z_caps
            pdn_impedance.append({
                "frequency_hz": f,
                "impedance_mohm": float(abs(Z_total) * 1000),
            })
        max_z = max(z["impedance_mohm"] for z in pdn_impedance)
        meets_target = max_z < Z_target * 1000
        return {
            "status": "success",
            "method": "power_integrity_analysis",
            "voltage_v": voltage_v,
            "target_impedance_mohm": round(Z_target * 1000, 2),
            "max_measured_impedance_mohm": round(max_z, 2),
            "meets_target": meets_target,
            "decoupling_recommendations": caps,
            "pdn_impedance_vs_frequency": pdn_impedance,
        }

    def _calculate_decoupling_caps(
        self, voltage_v: float, Z_target: float
    ) -> List[Dict[str, Any]]:
        return [
            {"value_f": 10e-6, "type": "tantalum_bulk",
             "voltage_rating_v": voltage_v * 1.5, "esr_ohm": 0.5, "esl_nh": 2,
             "purpose": "Bulk (1kHz–100kHz)"},
            {"value_f": 100e-9, "type": "ceramic_X7R",
             "voltage_rating_v": voltage_v * 1.5, "esr_ohm": 0.01, "esl_nh": 0.5,
             "purpose": "Mid-freq (100kHz–10MHz)"},
            {"value_f": 10e-9, "type": "ceramic_X7R",
             "voltage_rating_v": voltage_v * 1.5, "esr_ohm": 0.01, "esl_nh": 0.3,
             "purpose": "HF (10MHz–100MHz)"},
        ]

    async def _thermal_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        components = params.get("components", [])
        ambient_c = params.get("ambient_temp_c", self.config.get("thermal_ambient_c", 25))
        results: List[Dict] = []
        issues: List[str] = []
        for comp in components:
            comp_id = comp.get("id", "unknown")
            power_w = comp.get("power_dissipation_w", 0)
            theta_ja = comp.get("theta_ja_c_w", self.config.get("thermal_theta_ja_default", 50))
            t_max = comp.get("max_junction_temp_c", 150)
            t_j = ambient_c + power_w * theta_ja
            margin = t_max - t_j
            status = "ok" if margin > 20 else ("warning" if margin > 0 else "critical")
            results.append({
                "component_id": comp_id,
                "power_dissipation_w": power_w,
                "junction_temp_c": round(t_j, 1),
                "thermal_margin_c": round(margin, 1),
                "status": status,
            })
            if margin < 0:
                issues.append(f"{comp_id}: over-temp by {abs(margin):.1f}°C")
            elif margin < 20:
                issues.append(f"{comp_id}: low margin ({margin:.1f}°C)")
        return {
            "status": "success",
            "method": "thermal_analysis",
            "ambient_temp_c": ambient_c,
            "component_analyses": results,
            "issues": issues,
            "recommendations": self._generate_thermal_recommendations(results),
        }

    def _generate_thermal_recommendations(self, results: List[Dict]) -> List[str]:
        recs: List[str] = []
        critical = sum(1 for r in results if r.get("status") == "critical")
        warning = sum(1 for r in results if r.get("status") == "warning")
        if critical:
            recs.append(f"{critical} components over temperature — add heatsinks or thermal vias")
        if warning:
            recs.append(f"{warning} components with low thermal margin — improve airflow")
        if any(r.get("power_dissipation_w", 0) > 1 for r in results):
            recs.append("High-power components: use thermal via arrays to improve heat spreading")
        return recs

    async def _drc_check(self, params: Dict[str, Any]) -> Dict[str, Any]:
        pcb_data = params.get("pcb", {})
        rules = params.get("design_rules", self._default_design_rules())
        violations: List[Dict] = []
        traces = pcb_data.get("traces", [])
        for i, t1 in enumerate(traces):
            for t2 in traces[i + 1:]:
                clearance = self._calculate_clearance(t1, t2)
                min_c = rules.get("min_trace_clearance_mil", 5)
                if clearance < min_c:
                    violations.append({
                        "type": "clearance", "severity": "error",
                        "objects": [t1.get("id"), t2.get("id")],
                        "clearance_mil": round(clearance, 2),
                        "required_mil": min_c,
                    })
        for trace in traces:
            width = trace.get("width_mil", 0)
            min_w = rules.get("min_trace_width_mil", 5)
            if width < min_w:
                violations.append({
                    "type": "trace_width", "severity": "error",
                    "object": trace.get("id"), "width_mil": width, "required_mil": min_w,
                })
        return {
            "status": "success" if not violations else "violations_found",
            "method": "design_rule_check",
            "violation_count": len(violations),
            "violations": violations,
        }

    def _default_design_rules(self) -> Dict[str, Any]:
        return {
            "min_trace_width_mil": 5, "min_trace_clearance_mil": 5,
            "min_via_size_mil": 10, "min_via_drill_mil": 5,
            "max_vias_per_net": 10, "min_annular_ring_mil": 5,
        }

    def _calculate_clearance(self, t1: Dict, t2: Dict) -> float:
        path1 = t1.get("path", [])
        path2 = t2.get("path", [])
        if not path1 or not path2:
            return 1000.0
        x1_min = min(p[0] for p in path1); x1_max = max(p[0] for p in path1)
        y1_min = min(p[1] for p in path1); y1_max = max(p[1] for p in path1)
        x2_min = min(p[0] for p in path2); x2_max = max(p[0] for p in path2)
        y2_min = min(p[1] for p in path2); y2_max = max(p[1] for p in path2)
        dx = max(x1_min - x2_max, x2_min - x1_max, 0)
        dy = max(y1_min - y2_max, y2_min - y1_max, 0)
        return math.sqrt(dx ** 2 + dy ** 2)

    async def _optimize_topology(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delegates to the real design flow when possible."""
        reqs = params.get("requirements", {})
        if reqs:
            design_params = {
                "v_in": reqs.get("input_voltage", 12.0),
                "v_out": reqs.get("output_voltage", 5.0),
                "i_out": reqs.get("max_current_a", reqs.get("current_a", 1.0)),
                "ripple_v": reqs.get("max_ripple_mv", 50) / 1000,
                "efficiency_min": reqs.get("min_efficiency", 0.80),
                "f_sw": reqs.get("switching_frequency_hz", 500e3),
            }
            return await self.design(design_params)
        return {"status": "error", "error": "No requirements provided for topology optimization"}

    def _parse_circuit(self, circuit_data: Dict[str, Any]) -> Circuit:
        circuit = Circuit(name=circuit_data.get("name", "unnamed"))
        for comp_data in circuit_data.get("components", []):
            comp = Component(
                id=comp_data["id"], type=comp_data["type"],
                value=comp_data.get("value"), unit=comp_data.get("unit"),
                footprint=comp_data.get("footprint"), model=comp_data.get("model"),
                pins=comp_data.get("pins", {}),
                connections=comp_data.get("connections", comp_data.get("pins", {})),
                thermal=comp_data.get("thermal", {}),
                params=comp_data.get("params", {}),
            )
            circuit.components[comp.id] = comp
        for net_data in circuit_data.get("nets", []):
            net = Net(
                name=net_data["name"],
                nodes=[(n["component"], n["pin"]) for n in net_data.get("nodes", [])],
            )
            circuit.nets[net.name] = net
        return circuit

    def _generate_pcb_recommendations(
        self, traces: List[Dict], vias: List[Dict]
    ) -> List[str]:
        recs: List[str] = []
        high_r = [t for t in traces if t.get("resistance_ohm", 0) > 1.0]
        if high_r:
            recs.append(f"{len(high_r)} traces have high resistance (>1Ω) — widen or use thicker copper")
        if len(vias) > 50:
            recs.append(f"High via count ({len(vias)}) — consider via stitching")
        return recs


# ---------------------------------------------------------------------------
# Module-level convenience wrappers
# ---------------------------------------------------------------------------

async def quick_circuit_check(circuit_data: Dict[str, Any]) -> Dict[str, Any]:
    agent = ElectronicsAgent()
    return await agent.run({"operation": "simulate_circuit", "circuit": circuit_data})


async def quick_pcb_check(pcb_data: Dict[str, Any]) -> Dict[str, Any]:
    agent = ElectronicsAgent()
    return await agent.run({"operation": "analyze_pcb", "pcb": pcb_data})


async def design_power_supply(
    v_in: float, v_out: float, i_out: float,
    ripple_mv: float = 50.0, f_sw_khz: float = 500.0,
) -> Dict[str, Any]:
    agent = ElectronicsAgent()
    return await agent.design({
        "v_in": v_in, "v_out": v_out, "i_out": i_out,
        "ripple_v": ripple_mv / 1000.0, "f_sw": f_sw_khz * 1e3,
    })


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC ENGINE METHODS — callable by orchestrator or API
# ═══════════════════════════════════════════════════════════════════════════

async def analyze_signal_integrity(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Full signal integrity analysis for a trace or set of traces.

    params:
      trace:   {width_mm, length_mm, layer: 'microstrip'|'stripline'}
      stackup: {h_mm, t_mm, er}
      signal:  {rise_time_ps, z_source_ohm, z_load_ohm, frequency_mhz}
      adjacent_trace: optional crosstalk neighbor {width_mm, spacing_mm}
    """
    if not _ENGINES_AVAILABLE:
        return {"error": "electronics_engines not loaded"}

    si = SignalIntegrityEngine()
    trace   = params.get("trace", {"width_mm": 0.15, "length_mm": 50, "layer": "microstrip"})
    stackup = params.get("stackup", {"h_mm": 0.2, "t_mm": 0.035, "er": 4.5})
    signal  = params.get("signal", {"rise_time_ps": 1000, "z_source_ohm": 50,
                                     "z_load_ohm": 1e6, "frequency_mhz": 100})
    result = si.analyze_trace(trace, stackup, signal)

    # Crosstalk if adjacent trace specified
    adj = params.get("adjacent_trace")
    if adj:
        xt = si.crosstalk_coefficients(
            trace["width_mm"], adj.get("spacing_mm", 0.2),
            stackup["h_mm"], result["er_eff"],
            trace["length_mm"], signal.get("rise_time_ps", 1000)
        )
        result["crosstalk"] = xt

    return result


async def analyze_power_integrity(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    PDN analysis and decoupling strategy.

    params:
      v_rail: float           — power rail voltage (V)
      i_transient: float      — worst-case transient current (A)
      ripple_pct: float       — allowed ripple (%, default 3)
      f_transient_mhz: float  — transient frequency
      n_ics: int              — number of ICs on rail
      bulk_caps: [{c_f, esr, esl_h, qty}]
      bypass_caps: [{c_f, esr, esl_h, qty}]
    """
    if not _ENGINES_AVAILABLE:
        return {"error": "electronics_engines not loaded"}

    pi = PowerIntegrityEngine()
    v_rail       = params.get("v_rail", 3.3)
    i_transient  = params.get("i_transient", 1.0)
    ripple_pct   = params.get("ripple_pct", 3.0)
    f_mhz        = params.get("f_transient_mhz", 10.0)
    n_ics        = params.get("n_ics", 1)
    bulk_caps    = params.get("bulk_caps", [])
    bypass_caps  = params.get("bypass_caps", [])

    strategy = pi.decoupling_strategy(v_rail, i_transient, ripple_pct, f_mhz, n_ics)

    pdn_curve: Dict[str, Any] = {}
    if bulk_caps or bypass_caps:
        pdn_curve = pi.pdn_impedance_curve(bulk_caps, bypass_caps)
        # Trim to avoid huge serialised lists
        pdn_curve.pop("frequencies_hz", None)
        pdn_curve.pop("impedance_ohm", None)

    z_tgt = pi.target_impedance(v_rail, ripple_pct, i_transient)

    return {
        "z_target_ohm": z_tgt,
        "decoupling_strategy": strategy,
        "pdn_summary": pdn_curve,
        "ground_bounce": (pi.ground_bounce(5e-9, i_transient, 1e-9)
                          if i_transient > 0.1 else None),
    }


async def analyze_pcb_si_pi(params: Dict[str, Any]) -> Dict[str, Any]:
    """Combined PCB SI + PI + thermal + DRC in one call."""
    si_result = await analyze_signal_integrity(params.get("si", {}))
    pi_result = await analyze_power_integrity(params.get("pi", {}))

    if _ENGINES_AVAILABLE:
        # DRC
        traces_raw = params.get("traces", [])
        vias_raw   = params.get("vias",   [])
        traces = [TraceSpec(**t) for t in traces_raw] if traces_raw else []
        vias   = [ViaSpec(**v)   for v in vias_raw]   if vias_raw   else []
        rules  = params.get("drc_rules", {"min_trace_width_mm": 0.1,
                                           "min_clearance_mm": 0.15,
                                           "min_annular_ring_mm": 0.1})
        drc = PCBGeometryEngine().run_drc(traces, vias, rules)
    else:
        drc = {"error": "engines not loaded"}

    return {"signal_integrity": si_result,
            "power_integrity": pi_result,
            "drc": drc}


async def design_analog_circuit(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analog circuit design: filters, op-amp stages, ADC drivers, voltage references.

    params.goal_type: "filter" | "opamp" | "adc_driver" | "vref"
    """
    if not _ENGINES_AVAILABLE:
        return {"error": "electronics_engines not loaded"}

    ag = AnalogDesignEngine()
    goal_type = params.get("goal_type", "filter").lower()

    if goal_type == "filter":
        topology = params.get("topology", "butterworth").lower()
        f_c = params.get("f_cutoff_hz", 1000.0)
        order = params.get("order", 2)
        ftype = params.get("filter_type", "lowpass")

        if topology in ("rc", "passive_rc"):
            result = ag.design_rc_filter(f_c, ftype, min(order, 4))
        elif topology == "lc":
            result = ag.design_lc_filter(f_c, params.get("z0_ohm", 50.0), ftype, order)
        elif topology == "sallen_key":
            result = ag.design_sallen_key_lpf(f_c, params.get("q", 0.7071),
                                              params.get("gain", 1.0))
        else:
            result = ag.design_butterworth_filter(order, f_c, ftype)

    elif goal_type == "opamp":
        result = ag.design_opamp_gain_stage(
            params.get("gain", 10.0),
            params.get("bandwidth_hz", 100e3),
            params.get("topology", "non_inverting"),
        )

    elif goal_type == "adc_driver":
        result = ag.design_adc_driver(
            params.get("resolution_bits", 12),
            params.get("sample_rate_sps", 1e6),
            params.get("input_range_v", 3.3),
        )

    elif goal_type == "vref":
        result = ag.design_voltage_reference(
            params.get("v_ref", 2.5),
            params.get("v_supply", 5.0),
            params.get("load_ma", 1.0),
        )
    else:
        result = {"error": f"Unknown analog goal_type: {goal_type}"}

    return {"goal_type": goal_type, "design": result}


async def design_digital_interface(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Digital interface analysis: fanout, decoupling, timing, crystal.
    """
    if not _ENGINES_AVAILABLE:
        return {"error": "electronics_engines not loaded"}

    dg = DigitalDesignEngine()
    results: Dict[str, Any] = {}

    if "fanout" in params:
        fo = params["fanout"]
        results["fanout"] = dg.fanout_analysis(
            fo.get("driver", "LVCMOS"),
            fo.get("load", "LVCMOS"),
            fo.get("n_loads", 4),
            fo.get("trace_length_mm", 20),
        )

    if "decoupling" in params:
        dc = params["decoupling"]
        results["decoupling"] = dg.decoupling_cap_per_ic(
            dc.get("v_supply", 3.3),
            dc.get("max_current_ma", 100),
            dc.get("rise_time_ns", 2),
        )

    if "crystal" in params:
        xtal = params["crystal"]
        results["crystal_caps"] = dg.crystal_load_capacitors(
            xtal.get("c_l_pf", 12),
            xtal.get("c_stray_pf", 3),
        )

    if "timing" in params:
        tm = params["timing"]
        results["timing"] = dg.timing_budget(
            tm.get("t_clk_ns", 10),
            tm.get("t_setup_ns", 0.5),
            tm.get("t_hold_ns", 0.2),
            tm.get("t_prop_ns", 3),
            tm.get("t_skew_ns", 0.1),
        )

    return results


async def design_rf_circuit(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    RF circuit design: impedance matching, link budget, patch antenna.
    """
    if not _ENGINES_AVAILABLE:
        return {"error": "electronics_engines not loaded"}

    rf = RFDesignEngine()
    results: Dict[str, Any] = {}
    freq = params.get("freq_hz", 2.4e9)

    if "match" in params:
        m = params["match"]
        results["matching_network"] = rf.design_l_match(
            m.get("z_source", 50), m.get("z_load", 50), freq,
            m.get("high_pass", False)
        )

    if "link_budget" in params:
        lb = params["link_budget"]
        results["link_budget"] = rf.link_budget(
            lb.get("p_tx_dbm", 20), lb.get("g_tx_dbi", 2),
            lb.get("g_rx_dbi", 2), freq,
            lb.get("distance_m", 100),
            lb.get("cable_loss_db", 0),
            lb.get("other_losses_db", 0),
            lb.get("rx_sensitivity_dbm", -90),
        )

    if "noise_figure" in params:
        nf = params["noise_figure"]
        results["noise_figure"] = rf.noise_figure_cascade(
            nf.get("nf_db", [3, 10]), nf.get("gain_db", [15, 10])
        )

    if "patch_antenna" in params:
        pa = params["patch_antenna"]
        results["patch_antenna"] = rf.patch_antenna_dimensions(
            freq, pa.get("er", 4.4), pa.get("h_mm", 1.6)
        )

    if "trace_z0" in params:
        tz = params["trace_z0"]
        results["trace_impedance"] = SignalIntegrityEngine.microstrip_z0(
            tz.get("width_mm", 2.9), tz.get("h_mm", 1.6),
            tz.get("t_mm", 0.035), tz.get("er", 4.4)
        )

    return results


async def analyze_emc(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    EMC pre-compliance estimation (CISPR 25/32).

    params:
      loops: [{loop_area_cm2, current_ma, freq_mhz}]
      shielding: {material, thickness_mm, freq_mhz}
      cm_filter: {noise_current_ma, freq_mhz, target_insertion_loss_db}
      ferrite: {target_z_ohm, dc_current_a, freq_mhz}
    """
    if not _ENGINES_AVAILABLE:
        return {"error": "electronics_engines not loaded"}

    emc = EMCEngine()
    results: Dict[str, Any] = {}

    if "loops" in params:
        results["radiated_emissions"] = emc.check_cispr32_compliance(
            params["loops"], params.get("margin_db", 6)
        )

    if "shielding" in params:
        sh = params["shielding"]
        results["shielding"] = emc.shielding_effectiveness(
            sh.get("material", "aluminum"),
            sh.get("thickness_mm", 1.0),
            sh.get("freq_mhz", 100),
        )

    if "cm_filter" in params:
        cf = params["cm_filter"]
        results["cm_filter"] = emc.cm_filter_design(
            cf.get("noise_current_ma", 1),
            cf.get("freq_mhz", 30),
            cf.get("target_insertion_loss_db", 30),
        )

    if "ferrite" in params:
        fb = params["ferrite"]
        results["ferrite_bead"] = emc.ferrite_bead_selection(
            fb.get("target_z_ohm", 100),
            fb.get("dc_current_a", 0.5),
            fb.get("freq_mhz", 100),
        )

    return results


async def generate_manufacturing_files(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate Gerber + drill + BOM + CPL files.

    params:
      layer_name: str
      traces: [TraceSpec dicts]
      vias:   [ViaSpec dicts]
      board_outline: [(x,y), ...]  mm coordinates
      bom: {ref: {value, mpn, description, footprint}}
      components: [{ref, value, package, x_mm, y_mm, rotation_deg, layer}]
    """
    if not _ENGINES_AVAILABLE:
        return {"error": "electronics_engines not loaded"}

    gw = ExtendedGerberWriter()   # includes solder mask, silkscreen, paste layers
    traces  = [TraceSpec(**t) for t in params.get("traces", [])]
    vias    = [ViaSpec(**v)   for v in params.get("vias",   [])]
    outline = [tuple(p) for p in params.get("board_outline", [])]
    pads    = params.get("pads")   # optional: [{x, y, size_mm, shape, w_mm, h_mm}]
    refs    = params.get("silkscreen_refs")     # optional: [{ref, x, y, angle}]
    outlines = params.get("silkscreen_outlines")  # optional: [[(x,y),...]]

    files = gw.generate_fab_package(
        layer_name=params.get("layer_name", "design"),
        traces=traces,
        vias=vias,
        board_outline=outline if outline else None,
        bom=params.get("bom"),
        components=params.get("components"),
        pads=pads,
        silkscreen_refs=refs,
        silkscreen_outlines=outlines,
    )

    return {
        "files_generated": list(files.keys()),
        "layers": {
            "copper": [k for k in files if k.endswith((".gtl", ".gbl"))],
            "mask":   [k for k in files if k.endswith((".gm5", ".gm6"))],
            "silk":   [k for k in files if k.endswith((".gto", ".gbo"))],
            "paste":  [k for k in files if k.endswith(".gtp")],
            "drill":  [k for k in files if k.endswith(".drl")],
        },
        "file_contents": files,
        "layer_name": params.get("layer_name", "design"),
    }


async def analyze_electronics_thermal(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Electronics-specific thermal analysis (junction temperature, via array, heatsink).

    params:
      p_diss_w: float
      theta_jc: float
      theta_cs: float   (default 0.5)
      theta_sa: float   (0 = no heatsink)
      t_amb_c:  float   (default 25)
      via_array: {n_vias, via_dia_mm, via_height_mm}
      heatsink_req: bool  — compute required theta_sa
      transient: {p_pulse_w, pulse_width_s, c_th_j}
    """
    if not _ENGINES_AVAILABLE:
        return {"error": "electronics_engines not loaded"}

    th = ElectronicsThermalEngine()
    p_diss    = params.get("p_diss_w", 1.0)
    theta_jc  = params.get("theta_jc", 50.0)
    theta_cs  = params.get("theta_cs", 0.5)
    theta_sa  = params.get("theta_sa", 0.0)
    t_amb     = params.get("t_amb_c", 25.0)

    result = th.junction_temperature(p_diss, theta_jc, theta_cs, theta_sa, t_amb)

    if "via_array" in params:
        va = params["via_array"]
        result["via_thermal_resistance"] = th.via_array_thermal_resistance(
            va.get("n_vias", 9), va.get("via_dia_mm", 0.3),
            va.get("via_height_mm", 1.6)
        )

    if params.get("heatsink_req"):
        result["heatsink_requirement"] = th.heatsink_requirement(
            p_diss, theta_jc, t_amb_c=t_amb
        )

    if "transient" in params:
        tr = params["transient"]
        result["transient"] = th.transient_thermal(
            tr.get("p_pulse_w", p_diss),
            tr.get("pulse_width_s", 1e-3),
            theta_jc,
            tr.get("c_th_j", 0.01),
            t_amb,
        )

    return result


async def design_magnetics(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transformer / inductor magnetics design.

    params:
      topology: 'flyback' | 'forward' | 'full_bridge' | 'llc_resonant' | 'buck_inductor'
      v_in, v_out, i_out: electrical spec
      f_sw: switching frequency [Hz]
      efficiency: optional (default 0.85)
      duty_max: optional (default 0.45 for isolated topologies)
      b_max_t: optional peak flux density [T] (default 0.25)
      core_material: optional ('3C97'|'N87'|'N97'|'PC95'|'generic')

    Returns: np, ns, turns_ratio, Lp_h, Ls_h, Ap_cm4, wire gauges, losses.
    """
    if not _ENGINES_AVAILABLE:
        return {"error": "electronics_engines not loaded"}

    topology  = params.get("topology", "buck_inductor").lower()
    v_in  = float(params["v_in"])
    v_out = float(params["v_out"])
    i_out = float(params["i_out"])
    f_sw  = float(params.get("f_sw", params.get("f_sw_hz", 200e3)))
    mat   = params.get("core_material", "generic")

    if topology == "flyback":
        return MagneticsDesignEngine.design_flyback_transformer(
            v_in=v_in, v_out=v_out, i_out=i_out, f_sw=f_sw,
            efficiency=float(params.get("efficiency", 0.85)),
            duty_max=float(params.get("duty_max", 0.45)),
            b_max_t=float(params.get("b_max_t", 0.25)),
            core_material=mat,
        )
    elif topology == "forward":
        return MagneticsDesignEngine.design_forward_transformer(
            v_in=v_in, v_out=v_out, i_out=i_out, f_sw=f_sw,
            duty_max=float(params.get("duty_max", 0.45)),
            core_material=mat,
        )
    elif topology == "full_bridge":
        return MagneticsDesignEngine.design_full_bridge_transformer(
            v_in=v_in, v_out=v_out, i_out=i_out, f_sw=f_sw,
            core_material=mat,
        )
    elif topology == "llc_resonant":
        return MagneticsDesignEngine.design_llc_resonant_tank(
            v_in=v_in, v_out=v_out, i_out=i_out, f_sw=f_sw,
            quality_factor=float(params.get("quality_factor", 0.5)),
            fn=float(params.get("fn", 1.05)),
        )
    else:  # buck_inductor / boost (same formula)
        return MagneticsDesignEngine.design_buck_inductor(
            v_in=v_in, v_out=v_out, i_out=i_out, f_sw=f_sw,
            ripple_ratio=float(params.get("ripple_ratio", 0.3)),
            core_material=mat,
        )


async def design_control_loop(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Closed-loop voltage-mode compensator design.

    params:
      v_out: output voltage [V]
      v_ref: reference voltage [V] (default 1.25 for TL431 or 0.8 for internal ref)
      f_sw:  switching frequency [Hz]
      l_h:   output filter inductance [H]
      c_out_f: output capacitance [F]
      esr_ohm: output capacitor ESR [Ω]
      r_load_ohm: nominal load resistance [Ω] (or compute from v_out/i_out)
      i_out: optional — used to derive r_load if r_load_ohm not provided
      v_ramp: PWM ramp amplitude [V] (default 1.0)
      compensator_type: 'auto' | 'type2' | 'type3' (default 'auto')
      bode: bool — include Bode data in response (default False)

    Returns: component values (R, C), crossover frequency, phase margin, stability flag.
    """
    if not _ENGINES_AVAILABLE:
        return {"error": "electronics_engines not loaded"}

    v_out   = float(params["v_out"])
    v_ref   = float(params.get("v_ref", min(v_out * 0.4, 2.5)))
    f_sw    = float(params.get("f_sw", params.get("f_sw_hz", 500e3)))
    l_h     = float(params["l_h"])
    c_out_f = float(params["c_out_f"])
    esr_ohm = float(params.get("esr_ohm", 0.050))
    i_out   = float(params.get("i_out", 1.0))
    r_load  = float(params.get("r_load_ohm", v_out / max(i_out, 0.001)))
    v_ramp  = float(params.get("v_ramp", 1.0))
    comp_type = params.get("compensator_type", "auto").lower()

    if comp_type == "type2":
        result = ControlLoopDesignEngine.design_type2_compensator(
            v_out, v_ref, f_sw, l_h, c_out_f, esr_ohm, r_load, v_ramp
        )
    elif comp_type == "type3":
        result = ControlLoopDesignEngine.design_type3_compensator(
            v_out, v_ref, f_sw, l_h, c_out_f, esr_ohm, r_load, v_ramp
        )
    else:
        result = ControlLoopDesignEngine.recommend_compensator(
            v_out, v_ref, f_sw, l_h, c_out_f, esr_ohm, r_load, v_ramp
        )

    if params.get("bode"):
        result["bode"] = ControlLoopDesignEngine.bode_data(
            f_sw, l_h, c_out_f, esr_ohm, r_load,
            v_in=float(params.get("v_in", v_out * 3)),
            v_ramp=v_ramp,
            compensator=result,
        )

    return result


# ---------------------------------------------------------------------------
# FastAPI routes (preserved for backward compat)
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    router = None

if HAS_FASTAPI:
    router = APIRouter(prefix="/electronics", tags=["electronics"])

    class DesignRequest(BaseModel):
        v_in: float = Field(..., description="Input voltage (V)")
        v_out: float = Field(..., description="Target output voltage (V)")
        i_out: float = Field(..., description="Output current (A)")
        ripple_mv: float = Field(default=50.0, description="Max output ripple (mV)")
        f_sw_khz: float = Field(default=500.0, description="Switching frequency (kHz)")
        efficiency_min: float = Field(default=0.80, description="Minimum efficiency")
        project_name: Optional[str] = Field(default=None)

    class CircuitDesignRequest(BaseModel):
        components: List[Dict[str, Any]] = Field(default_factory=list)
        connections: List[Dict[str, Any]] = Field(default_factory=list)
        fidelity: Optional[str] = Field(default=None)

    class PowerAnalysisRequest(BaseModel):
        voltage: float
        components: List[Dict[str, Any]]

    @router.post("/design")
    async def design_circuit(request: DesignRequest):
        """Run full electronics design loop — topology selection, SPICE, catalog."""
        try:
            agent = ElectronicsAgent()
            params: Dict[str, Any] = request.model_dump()
            params["ripple_v"] = params.pop("ripple_mv") / 1000.0
            params["f_sw"] = params.pop("f_sw_khz") * 1e3
            return await agent.design(params)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/design/circuit")
    async def design_circuit_legacy(request: CircuitDesignRequest):
        """Legacy endpoint — runs power budget analysis."""
        try:
            agent = ElectronicsAgent()
            return await agent.run({
                "operation": "simulate_circuit",
                "circuit": {"components": request.components, "connections": request.connections},
                "fidelity": request.fidelity,
            })
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/simulate")
    async def simulate_circuit(request: CircuitDesignRequest):
        try:
            agent = ElectronicsAgent()
            return await agent.run({
                "operation": "simulate_circuit",
                "circuit": {"components": request.components, "connections": request.connections},
                "fidelity": request.fidelity,
            })
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/analyze/power")
    async def analyze_power(request: PowerAnalysisRequest):
        try:
            agent = ElectronicsAgent()
            return await agent.run({
                "operation": "simulate_circuit",
                "components": request.components,
                "power_supply": {"voltage": request.voltage},
            })
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/run")
    async def run_electronics_agent(params: Dict[str, Any]):
        try:
            agent = ElectronicsAgent()
            return await agent.run(params)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.get("/library/components")
    async def get_component_library():
        """Returns topology capabilities rather than a hardcoded list."""
        return {
            "status": "success",
            "message": "Component catalog queries DigiKey/Mouser/Octopart at design time.",
            "supported_topologies": [
                "buck", "boost", "ldo", "h_bridge",
                "flyback", "sepic", "full_bridge", "llc_resonant", "forward"
            ],
            "catalog_sources": ["DigiKey v4 API", "Mouser v2 API", "Octopart v4 GraphQL"],
            "analysis_capabilities": [
                "signal_integrity", "power_integrity", "pcb_drc",
                "analog_filter", "opamp_stage", "adc_driver",
                "digital_fanout", "timing_analysis", "crystal_oscillator",
                "rf_matching", "link_budget", "patch_antenna",
                "emc_cispr32", "shielding_effectiveness",
                "gerber_export", "excellon_drill", "bom_csv",
                "junction_temperature", "via_thermal", "heatsink_sizing",
            ],
        }

    # ── New engine routes ────────────────────────────────────────────────────

    @router.post("/analyze/signal-integrity")
    async def api_si(params: Dict):
        return await analyze_signal_integrity(params)

    @router.post("/analyze/power-integrity")
    async def api_pi(params: Dict):
        return await analyze_power_integrity(params)

    @router.post("/analyze/pcb")
    async def api_pcb(params: Dict):
        return await analyze_pcb_si_pi(params)

    @router.post("/design/analog")
    async def api_analog(params: Dict):
        return await design_analog_circuit(params)

    @router.post("/design/digital")
    async def api_digital(params: Dict):
        return await design_digital_interface(params)

    @router.post("/design/rf")
    async def api_rf(params: Dict):
        return await design_rf_circuit(params)

    @router.post("/analyze/emc")
    async def api_emc(params: Dict):
        return await analyze_emc(params)

    @router.post("/manufacturing/gerber")
    async def api_gerber(params: Dict):
        return await generate_manufacturing_files(params)

    @router.post("/analyze/thermal")
    async def api_thermal(params: Dict):
        return await analyze_electronics_thermal(params)

    @router.post("/design/magnetics")
    async def api_magnetics(params: Dict):
        """
        Transformer / inductor design.
        topology: 'flyback'|'forward'|'full_bridge'|'llc_resonant'|'buck_inductor'
        v_in, v_out, i_out, f_sw required.
        """
        return await design_magnetics(params)

    @router.post("/design/control-loop")
    async def api_control_loop(params: Dict):
        """
        Voltage-mode compensator design (Type-2 or Type-3).
        v_out, l_h, c_out_f, esr_ohm, f_sw required.
        compensator_type: 'auto'|'type2'|'type3'
        bode: true to include Bode plot data.
        """
        return await design_control_loop(params)

    @router.get("/design/core-materials")
    async def api_core_materials():
        """Return available ferrite core material grades and their properties."""
        if not _ENGINES_AVAILABLE:
            return {"error": "electronics_engines not loaded"}
        return {
            "materials": {
                k: {
                    "note": v["note"],
                    "b_sat_t": v["b_sat"],
                    "steinmetz_alpha": v["alpha"],
                    "steinmetz_beta": v["beta"],
                }
                for k, v in MagneticsDesignEngine._CORE_MATERIALS.items()
            }
        }

    @router.get("/design/wire-table")
    async def api_wire_table():
        """Return AWG wire gauge table (diameter, resistance, current capacity)."""
        if not _ENGINES_AVAILABLE:
            return {"error": "electronics_engines not loaded"}
        return {
            "awg_table": {
                str(awg): {
                    "diameter_mm": d, "resistance_mohm_per_m": r, "current_max_a": i
                }
                for awg, (d, r, i) in MagneticsDesignEngine._AWG_TABLE.items()
            },
            "note": "Current ratings at 300 A/cm² (natural convection). Multiply by 1.5 for forced air."
        }
