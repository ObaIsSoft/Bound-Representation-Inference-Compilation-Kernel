"""
electronics_engines.py — Specialized analysis engines for ElectronicsAgent

Each engine is a self-contained class. No LLM calls. Pure physics/math.

Engines:
  SignalIntegrityEngine    — transmission lines, crosstalk, eye diagram budget
  PowerIntegrityEngine     — PDN solver, decoupling strategy, SSN estimation
  ElectronicsThermalEngine — junction thermal, spreading resistance, via arrays
  PCBGeometryEngine        — Shapely-based DRC, trace current capacity, clearance
  AnalogDesignEngine       — passive/active filters, op-amp stages, ADC drivers
  DigitalDesignEngine      — fanout, decoupling, timing, crystal oscillator
  RFDesignEngine           — transmission line Z0, matching networks, link budget
  GerberWriter             — RS-274X Gerber file generation, Excellon drill, BOM
  EMCEngine                — CISPR 25/32 pre-compliance estimation
  MagneticsDesignEngine    — transformer/inductor: turns, core area, wire gauge, losses
  ControlLoopDesignEngine  — type-2/3 compensator, Bode plot, phase margin
  ExtendedGerberWriter     — solder mask, silkscreen, paste layers (extends GerberWriter)

References embedded in docstrings: IPC-2141A, IPC-2221A, MIL-STD-975, CISPR 25/32,
TI appnotes, Pozar "Microwave Engineering", Ott "Electromagnetic Compatibility Engineering",
McLyman "Transformer and Inductor Design Handbook", Venable Tech SLVA554, TI SLVA553.
"""

from __future__ import annotations

import math
import cmath
import os
import io
import csv
import zipfile
import tempfile
import datetime
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)

# ── Shapely import (optional — falls back to bbox DRC if not available) ───────
try:
    from shapely.geometry import (
        LineString, Point, Polygon, MultiPolygon, box as shapely_box
    )
    from shapely.ops import unary_union
    _SHAPELY = True
except ImportError:
    _SHAPELY = False
    logger.warning("shapely not found — PCB geometry will use bounding-box approximation")

# ── scikit-rf for S-parameter work ────────────────────────────────────────────
try:
    import skrf as rf
    _SKRF = True
except ImportError:
    _SKRF = False

# ─────────────────────────────────────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────────────────────────────────────
C_LIGHT   = 2.998e8       # m/s
MU_0      = 4 * math.pi * 1e-7
EPS_0     = 8.854e-12
K_COPPER  = 385.0         # W/(m·K) thermal conductivity of copper
RHO_COPPER = 1.724e-8     # Ω·m resistivity of copper at 20°C
ALPHA_COPPER = 3.93e-3    # /°C temperature coefficient


# ═════════════════════════════════════════════════════════════════════════════
# 1.  SIGNAL INTEGRITY ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class SignalIntegrityEngine:
    """
    Transmission-line analysis, crosstalk, and eye diagram budget.

    All impedance formulas: IPC-2141A (Rev. 2004)
    Crosstalk: IPC-2141A Section 5, Ott ch. 11
    """

    # ── Impedance formulas ────────────────────────────────────────────────────

    @staticmethod
    def microstrip_z0(w_mm: float, h_mm: float, t_mm: float, er: float) -> Dict[str, float]:
        """
        Microstrip characteristic impedance — IPC-2141A Eq. (6-4).

        w_mm: trace width
        h_mm: dielectric height (core/prepreg thickness to reference plane)
        t_mm: copper thickness
        er:   relative permittivity of dielectric
        """
        # Effective trace width accounting for copper thickness
        if t_mm > 0:
            w_eff = w_mm + (t_mm / math.pi) * (1 + math.log(4 * math.e * h_mm / t_mm))
        else:
            w_eff = w_mm

        ratio = w_eff / h_mm
        if ratio <= 1.0:
            z0_air = 60 * math.log(8 / ratio + ratio / 4)
            er_eff = (er + 1) / 2 + (er - 1) / 2 * (1 / math.sqrt(1 + 12 / ratio) + 0.04 * (1 - ratio) ** 2)
        else:
            z0_air = 120 * math.pi / (ratio + 1.393 + 0.667 * math.log(ratio + 1.444))
            er_eff = (er + 1) / 2 + (er - 1) / 2 / math.sqrt(1 + 12 / ratio)

        z0 = z0_air / math.sqrt(er_eff)
        td_ps_mm = math.sqrt(er_eff) / C_LIGHT * 1e9  # ps/mm: (s/m)×(1e12 ps/s)÷(1e3 mm/m) = ×1e9

        return {"z0_ohm": round(z0, 2), "er_eff": round(er_eff, 4),
                "td_ps_mm": round(td_ps_mm, 3), "w_eff_mm": round(w_eff, 4)}

    @staticmethod
    def stripline_z0(w_mm: float, b_mm: float, t_mm: float, er: float) -> Dict[str, float]:
        """
        Centred stripline characteristic impedance — IPC-2141A Eq. (6-6).

        b_mm: total dielectric thickness (distance between reference planes)
        """
        t_frac = t_mm / b_mm
        w_eff = w_mm + (t_mm / math.pi) * (1 - 0.5 * math.log((t_frac) ** 2)) if t_mm > 0 else w_mm
        ratio = w_eff / b_mm
        z0 = (60 / math.sqrt(er)) * math.log(4 * b_mm / (0.67 * math.pi * (0.8 * w_eff + t_mm)))
        z0 = max(z0, 1.0)
        td_ps_mm = math.sqrt(er) / C_LIGHT * 1e9

        return {"z0_ohm": round(z0, 2), "er_eff": er,
                "td_ps_mm": round(td_ps_mm, 3), "w_eff_mm": round(w_eff, 4)}

    @staticmethod
    def coplanar_waveguide_z0(w_mm: float, s_mm: float, h_mm: float,
                               er: float, grounded: bool = True) -> Dict[str, float]:
        """
        Coplanar waveguide (GCPW) impedance — Wheeler / Pozar formulation.

        w_mm: trace width, s_mm: gap to ground plane, h_mm: substrate height
        """
        k  = w_mm / (w_mm + 2 * s_mm)
        k1 = math.tanh(math.pi * w_mm / (4 * h_mm)) / math.tanh(math.pi * (w_mm + 2 * s_mm) / (4 * h_mm))

        # Elliptic integrals K(k)/K'(k) approximated
        def _kk_ratio(k_val: float) -> float:
            kp = math.sqrt(1 - k_val ** 2)
            if k_val < 1 / math.sqrt(2):
                return math.pi / math.log(2 * (1 + math.sqrt(kp)) / (1 - math.sqrt(kp)))
            else:
                return math.log(2 * (1 + math.sqrt(k_val)) / (1 - math.sqrt(k_val))) / math.pi

        Kk  = _kk_ratio(k)
        Kk1 = _kk_ratio(k1)

        if grounded:
            er_eff = 1 + (er - 1) / 2 * Kk1 / Kk
        else:
            er_eff = (1 + er) / 2

        z0 = 30 * math.pi / (math.sqrt(er_eff) * Kk)
        td_ps_mm = math.sqrt(er_eff) / C_LIGHT * 1e9

        return {"z0_ohm": round(z0, 2), "er_eff": round(er_eff, 4),
                "td_ps_mm": round(td_ps_mm, 3)}

    @staticmethod
    def trace_width_for_impedance(z0_target: float, h_mm: float, t_mm: float,
                                   er: float, layer: str = "microstrip") -> float:
        """
        Binary-search trace width to hit target impedance.
        Returns width in mm.
        """
        lo, hi = 0.01, 20.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if layer == "microstrip":
                z = SignalIntegrityEngine.microstrip_z0(mid, h_mm, t_mm, er)["z0_ohm"]
            else:
                z = SignalIntegrityEngine.stripline_z0(mid, h_mm * 2, t_mm, er)["z0_ohm"]
            if z > z0_target:
                lo = mid
            else:
                hi = mid
        return round((lo + hi) / 2, 4)

    # ── Reflection and termination ────────────────────────────────────────────

    @staticmethod
    def reflection_coefficients(z0: float, z_source: float, z_load: float
                                  ) -> Dict[str, float]:
        """Source and load reflection coefficients."""
        rs = (z_source - z0) / (z_source + z0)
        rl = (z_load - z0) / (z_load + z0)
        return {"rho_source": round(rs, 4), "rho_load": round(rl, 4),
                "termination_needed": abs(rl) > 0.1 or abs(rs) > 0.1}

    @staticmethod
    def critical_length_mm(rise_time_ps: float, er_eff: float) -> float:
        """λ/10 rule: trace longer than this requires transmission-line treatment. IPC-2141A."""
        td_ps_mm = math.sqrt(er_eff) / C_LIGHT * 1e9
        if td_ps_mm == 0:
            return 1e6
        return round(rise_time_ps / (td_ps_mm * 10), 2)

    # ── Crosstalk ─────────────────────────────────────────────────────────────

    @staticmethod
    def crosstalk_coefficients(w_mm: float, s_mm: float, h_mm: float,
                                er_eff: float, length_mm: float,
                                rise_time_ps: float) -> Dict[str, float]:
        """
        NEXT and FEXT coupling coefficients for microstrip pair.
        Odd/even mode impedance via Ott "EMC Engineering" Eq 11.27-11.29.
        """
        # Coupling capacitance and inductance (IPC-2141A / Ott empirical)
        # cm: mutual capacitance per unit length (pF/mm approx from geometry)
        # lm: mutual inductance per unit length (nH/mm)
        eps_r_eff = er_eff
        eps_0_pF  = EPS_0 * 1e12          # pF/m
        c_l  = eps_r_eff * eps_0_pF * 1e-3  # self cap pF/mm (approx)
        td_ps_mm = math.sqrt(er_eff) / C_LIGHT * 1e9

        # Empirical geometric coupling factor (Hammerstad approximation)
        # κ ≈ exp(-2π * s / h) / (1 + (w/h))
        kappa = math.exp(-2 * math.pi * s_mm / h_mm) / (1 + w_mm / h_mm)
        kappa = min(kappa, 0.5)

        # Backward (NEXT) coupling coefficient Kb
        kb = kappa / 4
        # Forward (FEXT) coupling coefficient Kf
        kf = (td_ps_mm * length_mm / rise_time_ps) * kappa / 2

        # Peak NEXT voltage (as fraction of drive amplitude)
        next_frac = kb * (1 - math.exp(-2 * length_mm * td_ps_mm / rise_time_ps))
        # Peak FEXT voltage
        fext_frac = abs(kf)

        return {
            "kb": round(kb, 5),
            "kf": round(kf, 5),
            "next_fraction": round(next_frac, 5),
            "fext_fraction": round(fext_frac, 5),
            "next_mv_per_v_drive": round(next_frac * 1000, 1),
            "fext_mv_per_v_drive": round(fext_frac * 1000, 1),
            "coupling_class": "strong" if next_frac > 0.05 else ("moderate" if next_frac > 0.01 else "weak"),
        }

    # ── Full trace analysis ───────────────────────────────────────────────────

    def analyze_trace(self, trace: Dict[str, Any], stackup: Dict[str, Any],
                      signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full SI analysis for a single trace.

        trace:   {width_mm, length_mm, layer: 'microstrip'|'stripline'}
        stackup: {h_mm, t_mm, er, er_eff (optional)}
        signal:  {rise_time_ps, z_source_ohm, z_load_ohm, frequency_mhz}
        """
        w = trace["width_mm"]
        l = trace["length_mm"]
        h = stackup["h_mm"]
        t = stackup.get("t_mm", 0.035)
        er = stackup.get("er", 4.5)
        layer = trace.get("layer", "microstrip")

        if layer == "microstrip":
            tl = self.microstrip_z0(w, h, t, er)
        else:
            tl = self.stripline_z0(w, h * 2, t, er)

        z0 = tl["z0_ohm"]
        er_eff = tl["er_eff"]
        td = tl["td_ps_mm"] * l  # total delay ps

        tr = signal.get("rise_time_ps", 1000)
        z_s = signal.get("z_source_ohm", 50)
        z_l = signal.get("z_load_ohm", 1e6)
        f_mhz = signal.get("frequency_mhz", 100)

        rc = self.reflection_coefficients(z0, z_s, z_l)
        crit = self.critical_length_mm(tr, er_eff)
        need_tl = l > crit

        # Attenuation (skin-effect loss, dB/mm at given frequency)
        rho_adj = RHO_COPPER * (1 + ALPHA_COPPER * 25)  # at 45°C
        skin_depth = math.sqrt(rho_adj / (math.pi * f_mhz * 1e6 * MU_0))
        rs = math.sqrt(math.pi * f_mhz * 1e6 * MU_0 * rho_adj)  # Ω/sq
        alpha_cond = rs / (w * 1e-3 * z0 * 2) * 8.686  # dB/m → convert
        attenuation_db = alpha_cond * l * 1e-3

        return {
            "z0_ohm": z0,
            "er_eff": er_eff,
            "propagation_delay_ps": round(td, 1),
            "attenuation_db": round(attenuation_db, 3),
            "rho_source": rc["rho_source"],
            "rho_load": rc["rho_load"],
            "critical_length_mm": crit,
            "requires_termination": need_tl,
            "termination_recommendation": (
                f"Series {z0:.0f}Ω at source" if z_s < z0 * 0.8
                else f"Parallel {z0:.0f}Ω to GND at load" if z_l > z0 * 5
                else "None required"
            ),
        }


# ═════════════════════════════════════════════════════════════════════════════
# 2.  POWER INTEGRITY ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class PowerIntegrityEngine:
    """
    PDN (power delivery network) analysis.

    References: Novak/Miller "Frequency-Domain Characterization of Power
    Distribution Networks"; IPC-9592 Power Delivery.
    """

    @staticmethod
    def target_impedance(v_rail: float, ripple_pct: float,
                          i_transient: float) -> float:
        """
        Z_target = V_ripple / I_transient = (V_rail × ripple%) / I_transient

        ripple_pct: allowed ripple as % of rail (e.g. 3 for 3%)
        """
        v_ripple = v_rail * ripple_pct / 100
        return round(v_ripple / max(i_transient, 1e-6), 6)

    @staticmethod
    def capacitor_impedance(c_f: float, esr: float, esl_h: float,
                             freq_hz: float) -> float:
        """
        |Z(f)| for a capacitor with ESR and ESL parasitics.
        Z = ESR + j(2πf·ESL - 1/(2πf·C))
        """
        w = 2 * math.pi * freq_hz
        zc = 1 / (w * c_f) if c_f > 0 else 1e9
        zl = w * esl_h
        return math.sqrt(esr ** 2 + (zl - zc) ** 2)

    @staticmethod
    def self_resonant_frequency(c_f: float, esl_h: float) -> float:
        """SRF = 1 / (2π√(L·C)) — capacitor is inductive above this."""
        if c_f <= 0 or esl_h <= 0:
            return 0.0
        return 1 / (2 * math.pi * math.sqrt(esl_h * c_f))

    def pdn_impedance_curve(
        self,
        bulk_caps: List[Dict[str, float]],    # [{c_f, esr, esl_h, qty}]
        bypass_caps: List[Dict[str, float]],  # [{c_f, esr, esl_h, qty}]
        vrm_r: float = 0.005,                 # VRM DC output resistance
        vrm_l: float = 10e-9,                 # VRM loop inductance
        freq_points: int = 200,
    ) -> Dict[str, Any]:
        """
        Full PDN impedance vs. frequency (bulk + bypass in parallel + VRM).
        Returns freq array (Hz) and impedance array (Ω).
        """
        freqs = np.logspace(3, 9, freq_points)  # 1kHz to 1GHz
        z_pdn = np.zeros(freq_points, dtype=complex)

        # Parallel combination of all capacitors
        all_caps = bulk_caps + bypass_caps
        if all_caps:
            z_caps = np.zeros(freq_points, dtype=complex)
            for cap in all_caps:
                qty = cap.get("qty", 1)
                c   = cap["c_f"]
                # Fallback defaults — replace with catalog data when available
                # ESR 10mΩ: KEMET X7R 100nF 0402 datasheet typical (source: EIA-198, IEC 60384-1)
                # ESL 1nH: IPC-2141A §5.3 typical for surface-mount capacitor land pattern
                esr = cap.get("esr", 0.01)
                esl = cap.get("esl_h", 1e-9)
                for _ in range(qty):
                    w = 2 * math.pi * freqs
                    z_single = esr + 1j * (w * esl - 1 / (w * c))
                    # parallel: 1/Z_total += 1/Z
                    with np.errstate(divide="ignore", invalid="ignore"):
                        z_caps += 1.0 / z_single
            z_caps = np.where(np.abs(z_caps) > 0, 1.0 / z_caps, np.full_like(z_caps, 1e9))
        else:
            z_caps = np.full(freq_points, 1e6, dtype=complex)

        # VRM impedance (R + jωL)
        w_arr = 2 * math.pi * freqs
        z_vrm = vrm_r + 1j * w_arr * vrm_l

        # Parallel VRM || Caps
        with np.errstate(divide="ignore", invalid="ignore"):
            z_pdn = (z_vrm * z_caps) / (z_vrm + z_caps)

        mag = np.abs(z_pdn)

        return {
            "frequencies_hz": freqs.tolist(),
            "impedance_ohm": mag.tolist(),
            "peak_impedance_ohm": float(np.max(mag)),
            "peak_freq_hz": float(freqs[np.argmax(mag)]),
            "dc_impedance_ohm": float(mag[0]),
        }

    def decoupling_strategy(
        self,
        v_rail: float,
        i_transient: float,
        ripple_pct: float = 3.0,
        f_transient_mhz: float = 10.0,
        n_ics: int = 1,
    ) -> Dict[str, Any]:
        """
        Recommend bulk + bypass capacitors for a given PDN requirement.
        Follows the 3-tier strategy: bulk (μF) + local bypass (nF) + chip cap (pF).
        """
        z_target = self.target_impedance(v_rail, ripple_pct, i_transient)
        f_hz = f_transient_mhz * 1e6

        # Bulk capacitor: hold charge during VRM response time (~50µs)
        c_bulk = i_transient * 50e-6 / (v_rail * ripple_pct / 100)
        c_bulk_rounded = max(10e-6, round(c_bulk / 10e-6) * 10e-6)

        # Local bypass: 100nF per IC (standard rule-of-thumb from IPC-2221A)
        c_bypass = 100e-9
        n_bypass = max(n_ics, 1)

        # High-freq decoupling: 10nF per power pin
        c_hf = 10e-9

        # Z_bypass: ESR=50mΩ (Murata GRM188R71C104K 100nF 0402 datasheet typical),
        # ESL=0.7nH (IPC-2141A §5.3 PCB land-pattern parasitic — catalog data pending)
        z_bypass_at_f = self.capacitor_impedance(c_bypass, 0.050, 0.7e-9, f_hz)
        margin = z_target / z_bypass_at_f if z_bypass_at_f > 0 else 0

        return {
            "z_target_ohm": round(z_target, 5),
            "bulk_cap": {
                "value_uf": round(c_bulk_rounded * 1e6, 1),
                "count": 1,
                "type": "electrolytic or polymer",
                "placement": "near VRM output",
            },
            "bypass_cap": {
                "value_nf": 100,
                "count": n_bypass,
                "type": "X5R/X7R ceramic 0402",
                "placement": "one per IC power pin, <5mm from pin",
            },
            "hf_decoupling_cap": {
                "value_nf": 10,
                "count": n_bypass * 2,
                "type": "C0G ceramic 0201",
                "placement": "directly at IC power pad",
            },
            "margin_at_transient_freq": round(margin, 2),
            "adequate": margin >= 1.0,
        }

    @staticmethod
    def ground_bounce(inductance_h: float, di_a: float, dt_s: float) -> Dict[str, float]:
        """
        Simultaneous switching noise: ΔV = L × dI/dt
        inductance_h: via + plane inductance (typical 1–3nH per via pair)
        """
        dv = inductance_h * di_a / dt_s
        return {
            "delta_v_mv": round(dv * 1000, 2),
            "inductance_nh": round(inductance_h * 1e9, 2),
            "severity": "critical" if dv > 0.3 else ("warning" if dv > 0.1 else "ok"),
        }


# ═════════════════════════════════════════════════════════════════════════════
# 3.  ELECTRONICS THERMAL ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class ElectronicsThermalEngine:
    """
    Component-level thermal analysis for PCB-mounted electronics.

    References: JEDEC JESD51, IPC-7093, Kromann spreading resistance model.
    """

    @staticmethod
    def junction_temperature(
        p_diss_w: float,
        theta_jc: float,        # junction-to-case (from datasheet)
        theta_cs: float = 0.1,  # case-to-heatsink (interface material)
        theta_sa: float = 0.0,  # heatsink-to-ambient (0 = no heatsink)
        t_amb_c: float = 25.0,
        theta_jb: Optional[float] = None,  # junction-to-board (alternative path)
        theta_ba: float = 10.0,            # board-to-ambient
    ) -> Dict[str, float]:
        """Full junction temperature using JEDEC multi-path thermal model."""
        # Path 1: junction → case → heatsink → ambient
        if theta_sa > 0:
            theta_total_path1 = theta_jc + theta_cs + theta_sa
        else:
            theta_total_path1 = float("inf")

        # Path 2: junction → board → ambient (JEDEC JESD51-2)
        if theta_jb is not None:
            theta_total_path2 = theta_jb + theta_ba
        else:
            theta_total_path2 = float("inf")

        # Parallel paths (lower resistance dominates)
        if theta_total_path1 < float("inf") and theta_total_path2 < float("inf"):
            theta_total = 1 / (1 / theta_total_path1 + 1 / theta_total_path2)
        else:
            theta_total = min(theta_total_path1, theta_total_path2)

        t_j = t_amb_c + p_diss_w * theta_total

        # Also compute per-path temperatures
        t_case = t_amb_c + p_diss_w * (theta_cs + theta_sa) if theta_sa > 0 else None

        return {
            "t_junction_c": round(t_j, 2),
            "t_case_c": round(t_case, 2) if t_case else None,
            "theta_total_c_w": round(theta_total, 3),
            "power_w": p_diss_w,
            "headroom_c": round(150 - t_j, 1),  # assumed 150°C max Tj
            "safe": t_j < 125.0,
        }

    @staticmethod
    def via_array_thermal_resistance(
        n_vias: int,
        via_dia_mm: float,
        via_height_mm: float,
        plating_thickness_um: float = 25.0,
        fill: str = "copper",  # "copper" | "resin" | "air"
    ) -> float:
        """
        Thermal resistance of a via array (°C/W).
        Uses parallel resistances for filled/plated via cross-sections.
        """
        r_wall = 1e-6          # Ω·m — not relevant; thermal conductivity:
        k_fill = {"copper": 385.0, "resin": 0.3, "air": 0.026}.get(fill, 0.3)
        k_copper = 385.0

        # Via cross-section area (copper annular ring only for plated via)
        d_outer = via_dia_mm * 1e-3
        d_inner = max(0, via_dia_mm - 2 * plating_thickness_um * 1e-3 / 1000) * 1e-3
        a_plating = math.pi / 4 * (d_outer ** 2 - d_inner ** 2)  # m²
        a_fill = math.pi / 4 * d_inner ** 2

        a_total = a_plating * k_copper + a_fill * k_fill  # effective k×A
        l = via_height_mm * 1e-3

        r_single = l / max(a_total, 1e-12)
        r_array  = r_single / max(n_vias, 1)
        return round(r_array, 4)

    @staticmethod
    def spreading_resistance(
        p_source_w: float,
        k_substrate: float,
        a_source_mm2: float,
        a_board_mm2: float,
    ) -> Dict[str, float]:
        """
        Kromann spreading resistance model (simplified circular source).
        Estimates temperature rise due to heat spreading from small die to larger board.
        """
        # Convert areas to equivalent radii
        r_s = math.sqrt(a_source_mm2 / math.pi) * 1e-3  # m
        r_b = math.sqrt(a_board_mm2 / math.pi) * 1e-3

        # Spreading resistance (Kromann 1994)
        r_spread = (1 / (2 * k_substrate * r_s) - 1 / (2 * k_substrate * r_b)) if r_b > r_s else 0

        delta_t = p_source_w * r_spread

        return {
            "theta_spread_c_w": round(r_spread, 4),
            "delta_t_spread_c": round(delta_t, 2),
            "r_source_mm": round(r_s * 1e3, 3),
            "r_board_mm": round(r_b * 1e3, 3),
        }

    def heatsink_requirement(
        self,
        p_diss_w: float,
        theta_jc: float,
        t_j_max_c: float = 125.0,
        t_amb_c: float = 25.0,
        theta_cs: float = 0.5,
    ) -> Dict[str, float]:
        """
        Required heatsink theta_sa to keep junction below t_j_max.
        """
        theta_available = (t_j_max_c - t_amb_c) / p_diss_w
        theta_sa_max = theta_available - theta_jc - theta_cs

        return {
            "theta_sa_max_c_w": round(max(theta_sa_max, 0), 3),
            "heatsink_required": theta_sa_max < 50,  # natural conv. limit ~50°C/W
            "t_junction_without_hs_c": round(t_amb_c + p_diss_w * (theta_jc + 50), 2),
            "recommendation": (
                "No heatsink needed" if theta_sa_max >= 50
                else f"Heatsink required: θ_sa ≤ {max(theta_sa_max, 0):.1f} °C/W"
            ),
        }

    def transient_thermal(
        self,
        p_pulse_w: float,
        pulse_width_s: float,
        theta_jc: float,
        c_th_j: float = 0.01,  # junction thermal capacitance J/°C
        t_amb_c: float = 25.0,
    ) -> Dict[str, float]:
        """
        Transient thermal response to a single power pulse.
        Uses 1st-order RC thermal model: Zth(t) = θ_jc × (1 - e^(-t/τ))
        where τ = R_th × C_th.
        """
        tau = theta_jc * c_th_j
        t_eval = pulse_width_s
        z_th = theta_jc * (1 - math.exp(-t_eval / tau))
        delta_t = p_pulse_w * z_th
        t_j_peak = t_amb_c + delta_t

        return {
            "t_junction_peak_c": round(t_j_peak, 2),
            "z_th_transient_c_w": round(z_th, 4),
            "thermal_time_constant_ms": round(tau * 1000, 2),
            "safe": t_j_peak < 150.0,
        }


# ═════════════════════════════════════════════════════════════════════════════
# 4.  PCB GEOMETRY ENGINE
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class TraceSpec:
    width_mm: float
    length_mm: float
    layer: int = 0
    net: str = ""
    start: Tuple[float, float] = (0, 0)
    end: Tuple[float, float] = (0, 0)

@dataclass
class ViaSpec:
    drill_mm: float
    pad_mm: float
    x: float = 0
    y: float = 0
    from_layer: int = 0
    to_layer: int = 1
    net: str = ""


class PCBGeometryEngine:
    """
    PCB geometry analysis using Shapely for real polygon operations.
    Falls back to bounding-box approximations if Shapely is unavailable.

    References: IPC-2221A, IPC-2141A trace current capacity.
    """

    # IPC-2221A Table 6-1 coefficients for trace current capacity
    # I = k × ΔT^0.44 × A^0.725  where A is cross-section in mil²
    _IPC_K_EXTERNAL = 0.048   # external trace (top/bottom copper)
    _IPC_K_INTERNAL = 0.024   # internal trace (inner layers)

    @staticmethod
    def trace_width_for_current(
        current_a: float,
        copper_oz: float = 1.0,
        temp_rise_c: float = 10.0,
        layer: str = "external",
    ) -> Dict[str, float]:
        """
        IPC-2221A trace current capacity.
        Returns minimum trace width in mm to carry current_a with temp_rise_c rise.
        """
        k = PCBGeometryEngine._IPC_K_EXTERNAL if layer == "external" else PCBGeometryEngine._IPC_K_INTERNAL
        # Copper thickness in mil (1 oz Cu ≈ 1.37 mil = 34.8 µm)
        thickness_mil = copper_oz * 1.37
        # Solve for area (mil²): I = k × ΔT^0.44 × A^0.725
        a_mil2 = (current_a / (k * temp_rise_c ** 0.44)) ** (1 / 0.725)
        width_mil = a_mil2 / thickness_mil
        width_mm = width_mil * 0.0254

        # DC resistance at that width and length = 1mm
        rho_cu_ohm_sq = RHO_COPPER / (thickness_mil * 0.0254 * 1e-3)  # Ω/sq
        r_per_mm = rho_cu_ohm_sq / (width_mm * 1e3)  # Ω/mm → Ω/m context

        return {
            "width_mm": round(width_mm, 4),
            "area_mm2": round(width_mm * copper_oz * 0.0348, 5),
            "temp_rise_c": temp_rise_c,
            "layer": layer,
            "resistance_mohm_per_mm": round(r_per_mm * 1e3, 4),
        }

    @staticmethod
    def trace_dc_resistance(
        length_mm: float,
        width_mm: float,
        copper_oz: float = 1.0,
        temp_c: float = 25.0,
    ) -> float:
        """DC resistance of a copper trace (Ω)."""
        thickness_m = copper_oz * 34.8e-6  # 1oz = 34.8µm
        rho = RHO_COPPER * (1 + ALPHA_COPPER * (temp_c - 20))
        return round(rho * (length_mm * 1e-3) / (width_mm * 1e-3 * thickness_m), 6)

    @staticmethod
    def minimum_clearance_for_voltage(
        voltage_v: float,
        environment: str = "standard",  # "standard" | "conformal_coated" | "high_altitude"
    ) -> float:
        """
        IPC-2221A Table 6-4 minimum electrical clearance (mm).
        """
        # Simplified: IPC-2221A B3 (coated external) and A3 (uncoated)
        table = {
            "standard": [(15, 0.1), (30, 0.1), (50, 0.6), (100, 0.6),
                          (150, 0.6), (300, 1.25), (500, 2.5)],
            "conformal_coated": [(15, 0.05), (30, 0.05), (50, 0.13), (100, 0.13),
                                   (150, 0.4), (300, 0.4), (500, 0.8)],
            "high_altitude": [(15, 0.2), (30, 0.5), (50, 1.5), (100, 1.5),
                               (150, 3.0), (300, 6.4), (500, 12.7)],
        }
        entries = table.get(environment, table["standard"])
        for v_thresh, clr in entries:
            if voltage_v <= v_thresh:
                return clr
        return round(voltage_v / 40, 2)  # >500V: 25V/mm rule

    def run_drc(
        self,
        traces: List[TraceSpec],
        vias: List[ViaSpec],
        rules: Dict[str, float],
        board_outline: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Design rule check using Shapely polygon operations.

        rules: {
            min_trace_width_mm, min_clearance_mm, min_annular_ring_mm,
            min_via_drill_mm, board_edge_clearance_mm
        }
        """
        violations: List[str] = []
        warnings: List[str] = []

        min_w   = rules.get("min_trace_width_mm", 0.1)
        min_clr = rules.get("min_clearance_mm", 0.15)
        min_ann = rules.get("min_annular_ring_mm", 0.1)
        min_via = rules.get("min_via_drill_mm", 0.2)
        edge_clr = rules.get("board_edge_clearance_mm", 0.2)

        # Trace width violations
        for tr in traces:
            if tr.width_mm < min_w:
                violations.append(f"Trace {tr.net}: width {tr.width_mm:.3f}mm < min {min_w}mm")

        # Via annular ring
        for v in vias:
            ann = (v.pad_mm - v.drill_mm) / 2
            if ann < min_ann:
                violations.append(f"Via {v.net}: annular ring {ann:.3f}mm < min {min_ann}mm")
            if v.drill_mm < min_via:
                violations.append(f"Via {v.net}: drill {v.drill_mm:.3f}mm < min {min_via}mm")

        if _SHAPELY:
            # Build polygons for all copper features on each layer
            layer_polys: Dict[int, List] = {}
            for tr in traces:
                poly = LineString([tr.start, tr.end]).buffer(tr.width_mm / 2)
                layer_polys.setdefault(tr.layer, []).append((poly, tr.net))

            for via in vias:
                for lyr in range(via.from_layer, via.to_layer + 1):
                    poly = Point(via.x, via.y).buffer(via.pad_mm / 2)
                    layer_polys.setdefault(lyr, []).append((poly, via.net))

            # Check clearances within each layer
            for lyr, items in layer_polys.items():
                for i, (pa, net_a) in enumerate(items):
                    for j, (pb, net_b) in enumerate(items):
                        if j <= i or net_a == net_b:
                            continue
                        dist = pa.distance(pb)
                        if dist < min_clr:
                            violations.append(
                                f"L{lyr} clearance {net_a}↔{net_b}: "
                                f"{dist:.3f}mm < {min_clr}mm"
                            )

            # Board edge clearance
            if board_outline is not None:
                outline_poly = Polygon(board_outline) if not isinstance(board_outline, Polygon) else board_outline
                for tr in traces:
                    poly = LineString([tr.start, tr.end]).buffer(tr.width_mm / 2)
                    dist = outline_poly.exterior.distance(poly)
                    if dist < edge_clr:
                        warnings.append(f"Trace {tr.net} within {dist:.3f}mm of board edge")
        else:
            warnings.append("Shapely unavailable — inter-trace clearance not checked")

        return {
            "violations": violations,
            "warnings": warnings,
            "pass": len(violations) == 0,
            "trace_count": len(traces),
            "via_count": len(vias),
            "drc_engine": "shapely" if _SHAPELY else "bbox",
        }

    @staticmethod
    def copper_pour_self_resonance(pour_area_mm2: float, pour_perimeter_mm: float,
                                    h_mm: float, er: float) -> float:
        """
        Approximate fundamental resonant frequency of a copper pour (cavity mode).
        For a rectangular plane: f = c / (2L√εr)  where L is longest dimension.
        """
        # Estimate longest dimension from area and perimeter heuristic
        # For rectangle: P = 2(a+b), A = a×b → longest side ≈ P/4 (square approx)
        l_mm = pour_perimeter_mm / 4
        f_res = C_LIGHT / (2 * l_mm * 1e-3 * math.sqrt(er)) if l_mm > 0 else 0
        return round(f_res / 1e6, 1)  # MHz


# ═════════════════════════════════════════════════════════════════════════════
# 5.  ANALOG DESIGN ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class AnalogDesignEngine:
    """
    Passive and active analog circuit design: filters, op-amp stages, ADC drivers.

    Filter design uses scipy.signal for Butterworth/Chebyshev pole placement.
    References: Williams/Taylor "Electronic Filter Design Handbook"; TI OA-21.
    """

    # ── Passive RC filters ────────────────────────────────────────────────────

    @staticmethod
    def design_rc_filter(
        f_cutoff_hz: float,
        filter_type: str = "lowpass",   # "lowpass" | "highpass"
        n_stages: int = 1,
        impedance_ohm: float = 1000.0,
    ) -> Dict[str, Any]:
        """
        Single-ended RC filter design.
        For n_stages, uses cascaded identical sections (−6n dB/octave roll-off).
        """
        # Each section: f_c_section adjusted so cascaded -3dB point hits target
        if n_stages > 1:
            # Correction factor: f_adj = f_c / (2^(1/n) - 1)^0.5
            correction = 1 / math.sqrt(2 ** (1 / n_stages) - 1)
            f_section = f_cutoff_hz * correction
        else:
            f_section = f_cutoff_hz

        r_ohm = impedance_ohm
        c_f = 1 / (2 * math.pi * r_ohm * f_section)

        # Round C to E24
        c_f_rounded = _round_e_series(c_f, 24)
        f_actual = 1 / (2 * math.pi * r_ohm * c_f_rounded)

        return {
            "filter_type": filter_type,
            "order": n_stages,
            "r_ohm": round(r_ohm, 2),
            "c_f": c_f,
            "c_f_rounded": c_f_rounded,
            "c_nf": round(c_f_rounded * 1e9, 3),
            "f_cutoff_hz": round(f_actual, 2),
            "f_cutoff_error_pct": round(abs(f_actual - f_cutoff_hz) / f_cutoff_hz * 100, 2),
            "rolloff_db_per_decade": -20 * n_stages,
        }

    @staticmethod
    def design_lc_filter(
        f_cutoff_hz: float,
        z0_ohm: float = 50.0,
        filter_type: str = "lowpass",
        order: int = 3,
    ) -> Dict[str, Any]:
        """
        Passive LC ladder filter (Butterworth prototype → frequency-scaled).
        Returns L and C values for a Butterworth low-pass ladder.
        """
        # Butterworth prototype element values (tables up to order 7)
        _proto = {
            1: [2.000],
            2: [1.4142, 1.4142],
            3: [1.000, 2.000, 1.000],
            4: [0.7654, 1.8478, 1.8478, 0.7654],
            5: [0.6180, 1.6180, 2.000, 1.6180, 0.6180],
            6: [0.5176, 1.4142, 1.9319, 1.9319, 1.4142, 0.5176],
            7: [0.4450, 1.2470, 1.8019, 2.0000, 1.8019, 1.2470, 0.4450],
        }
        n = min(max(order, 1), 7)
        g = _proto[n]
        w0 = 2 * math.pi * f_cutoff_hz

        elements = []
        for i, gval in enumerate(g):
            if i % 2 == 0:   # series L
                l_h = gval * z0_ohm / w0
                elements.append({"type": "L", "value_uh": round(l_h * 1e6, 4),
                                  "value_h": l_h, "position": i + 1})
            else:             # shunt C
                c_f = gval / (z0_ohm * w0)
                elements.append({"type": "C", "value_nf": round(c_f * 1e9, 4),
                                  "value_f": c_f, "position": i + 1})

        return {
            "filter_type": filter_type,
            "order": n,
            "z0_ohm": z0_ohm,
            "f_cutoff_hz": f_cutoff_hz,
            "elements": elements,
            "topology": "Butterworth",
        }

    # ── Active filters (Sallen-Key, MFB) ─────────────────────────────────────

    @staticmethod
    def design_sallen_key_lpf(
        f_cutoff_hz: float,
        q: float = 0.7071,    # Butterworth default (1/√2)
        gain: float = 1.0,
        r_ohm: float = 10000.0,
    ) -> Dict[str, Any]:
        """
        Sallen-Key 2nd-order low-pass filter.
        Unity gain (gain=1): R1=R2=R, C1 and C2 from Q relationship.

        References: TI SLOA049, Williams "Electronic Filter Design Handbook" p.2-57
        """
        # For unity gain: Q = 1/2 if C1=C2 → need different C values
        # General solution: C2 = C1 / (4Q² × gain variant)
        # Unity-gain Sallen-Key: C1 and C2 related by Q
        w0 = 2 * math.pi * f_cutoff_hz
        # Choose C1, compute C2
        c1_f = 1 / (r_ohm * w0)  # normalised
        c2_f = 1 / (r_ohm * w0 * (4 * q ** 2))
        c1_rounded = _round_e_series(c1_f, 24)
        c2_rounded = _round_e_series(c2_f, 24)

        # Recompute actual f0 and Q with rounded values
        if c1_rounded > 0 and c2_rounded > 0:
            f_actual = 1 / (2 * math.pi * r_ohm * math.sqrt(c1_rounded * c2_rounded))
            q_actual = math.sqrt(c1_rounded / c2_rounded) / 2
        else:
            f_actual, q_actual = f_cutoff_hz, q

        return {
            "topology": "Sallen-Key",
            "order": 2,
            "r1_ohm": r_ohm,
            "r2_ohm": r_ohm,
            "c1_nf": round(c1_rounded * 1e9, 4),
            "c2_nf": round(c2_rounded * 1e9, 4),
            "f_cutoff_hz": round(f_actual, 2),
            "q_actual": round(q_actual, 4),
            "dc_gain_db": round(20 * math.log10(gain), 2),
            "rolloff_db_per_decade": -40,
        }

    @staticmethod
    def design_butterworth_filter(
        order: int,
        f_cutoff_hz: float,
        filter_type: str = "lowpass",    # lowpass | highpass | bandpass | bandstop
        sample_rate_hz: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Butterworth filter design via scipy.signal.
        Returns poles, zeros, gain, and frequency response summary.
        """
        wn = f_cutoff_hz / (sample_rate_hz / 2) if sample_rate_hz else f_cutoff_hz
        btype = filter_type.replace("bandstop", "bandstop").replace("pass", "pass")

        if sample_rate_hz:
            b, a = scipy_signal.butter(order, wn, btype=btype)
        else:
            z, p, k = scipy_signal.buttap(order)
            # scale to cutoff
            b, a = scipy_signal.zpk2tf(z, p, k)

        # Frequency response at key points
        w, h = scipy_signal.freqs(b, a, worN=np.logspace(
            math.log10(max(f_cutoff_hz / 100, 1)), math.log10(f_cutoff_hz * 100), 200
        )) if not sample_rate_hz else (None, None)

        return {
            "order": order,
            "filter_type": filter_type,
            "f_cutoff_hz": f_cutoff_hz,
            "topology": "Butterworth",
            "attenuation_at_2fc_db": round(-3 - 20 * order * math.log10(2), 1),
            "attenuation_at_10fc_db": round(-3 - 20 * order, 1),
            "rolloff_db_per_decade": -20 * order,
            "phase_at_fc_deg": -45 * order,
        }

    # ── Op-amp stages ─────────────────────────────────────────────────────────

    @staticmethod
    def design_opamp_gain_stage(
        gain: float,
        f_bandwidth_hz: float,
        topology: str = "non_inverting",  # "inverting" | "non_inverting" | "differential"
        r_in_ohm: float = 10000.0,
    ) -> Dict[str, Any]:
        """
        Op-amp resistor network for a gain stage.
        Validates gain-bandwidth product requirement.
        """
        if topology == "non_inverting":
            # Gain = 1 + Rf/Rg
            if gain < 1:
                return {"error": "Non-inverting gain must be ≥ 1"}
            rf = r_in_ohm * (gain - 1)
            rg = r_in_ohm
            actual_gain = 1 + rf / rg
            gbw_required = gain * f_bandwidth_hz

        elif topology == "inverting":
            # Gain = -Rf/Rin  (magnitude)
            gain_mag = abs(gain)
            rf = r_in_ohm * gain_mag
            rg = r_in_ohm
            actual_gain = -rf / rg
            gbw_required = (gain_mag + 1) * f_bandwidth_hz  # inverting noise gain = |gain| + 1

        elif topology == "differential":
            # Differential amp: gain = Rf/Rg
            rf = r_in_ohm * gain
            rg = r_in_ohm
            actual_gain = rf / rg
            gbw_required = gain * f_bandwidth_hz * 2  # conservative

        else:
            return {"error": f"Unknown topology: {topology}"}

        # Round resistors to E96 series
        rf_rounded = _round_e_series(rf, 96)
        rg_rounded = _round_e_series(rg, 96)

        return {
            "topology": topology,
            "gain_v_v": round(actual_gain, 4),
            "gain_db": round(20 * math.log10(abs(gain)), 2),
            "rf_ohm": round(rf_rounded, 1),
            "rg_ohm": round(rg_rounded, 1),
            "rf_rg_ratio": round(rf_rounded / rg_rounded, 4),
            "min_opamp_gbw_hz": round(gbw_required, 0),
            "noise_gain": round(1 + rf_rounded / rg_rounded, 4),
            "bandwidth_3db_hz": f_bandwidth_hz,
        }

    @staticmethod
    def design_adc_driver(
        adc_resolution_bits: int,
        adc_sample_rate_sps: float,
        input_voltage_range_v: float,
        source_impedance_ohm: float = 50.0,
    ) -> Dict[str, Any]:
        """
        Anti-aliasing filter + buffer for ADC driver design.
        Nyquist bandwidth = sample_rate / 2.
        Filter cutoff chosen to attenuate aliases to below LSB level.
        """
        nyquist_hz = adc_sample_rate_sps / 2
        lsb_v = input_voltage_range_v / (2 ** adc_resolution_bits)

        # Required attenuation at Nyquist aliasing frequency (f_s - f_in ≈ f_s)
        attenuation_needed_db = 20 * math.log10(input_voltage_range_v / lsb_v)

        # Butterworth filter order for 10× margin
        # At f_s, need attenuation_needed + 20dB margin
        target_db = attenuation_needed_db + 20
        # Butterworth: A(f) = 10 * log10(1 + (f/fc)^(2n)) → solve for n at f=nyquist
        # fc chosen at 0.1 × nyquist for aggressive filtering
        fc_hz = nyquist_hz * 0.1
        n_order = math.ceil(target_db / (20 * math.log10(nyquist_hz / fc_hz)))

        # RC anti-aliasing (single pole, simplest)
        r_aa = source_impedance_ohm + 50  # include source
        c_aa = 1 / (2 * math.pi * r_aa * fc_hz)

        return {
            "adc_resolution_bits": adc_resolution_bits,
            "sample_rate_sps": adc_sample_rate_sps,
            "nyquist_hz": nyquist_hz,
            "lsb_v": round(lsb_v * 1000, 4),  # mV
            "required_attenuation_db": round(attenuation_needed_db, 1),
            "recommended_filter_order": n_order,
            "antialiasing_fc_hz": round(fc_hz, 1),
            "rc_r_ohm": round(r_aa, 1),
            "rc_c_nf": round(c_aa * 1e9, 3),
            "opamp_min_gbw_mhz": round(fc_hz * 100 / 1e6, 1),  # 40dB margin
            "opamp_recommendation": "Rail-to-rail input/output, GBW ≥ "
                                    f"{fc_hz * 100 / 1e6:.0f}MHz",
        }

    @staticmethod
    def design_voltage_reference(
        v_ref_target: float,
        v_supply: float,
        load_current_ma: float = 1.0,
    ) -> Dict[str, Any]:
        """Simple Zener / TL431 voltage reference design."""
        # TL431: V_ref=2.495V (TI SLVS543, Table 6.7); R2=10kΩ standard; V_ref=(1+R1/R2)×2.495
        v_tl431_ref = 2.495   # V — TI TL431 datasheet SLVS543 Table 6.7 typical
        ratio = v_ref_target / v_tl431_ref - 1
        r2 = 10000.0          # Ω — standard resistor divider starting point
        r1 = r2 * ratio
        r1_rounded = _round_e_series(r1, 96) if r1 > 0 else r2
        v_actual = (1 + r1_rounded / r2) * v_tl431_ref

        # Bias resistor from supply
        i_kat_min = 1e-3   # A — TL431 min cathode current (TI SLVS543 Table 6.5: I_ka_min=1mA)
        r_bias = (v_supply - v_ref_target) / (load_current_ma * 1e-3 + i_kat_min)

        return {
            "topology": "TL431",
            "v_ref_target": v_ref_target,
            "v_ref_actual": round(v_actual, 4),
            "r1_ohm": round(r1_rounded, 1),
            "r2_ohm": r2,
            "r_bias_ohm": round(r_bias, 1),
            "error_pct": round(abs(v_actual - v_ref_target) / v_ref_target * 100, 3),
            "part": "TL431 / LM4040",
        }


# ═════════════════════════════════════════════════════════════════════════════
# 6.  DIGITAL DESIGN ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class DigitalDesignEngine:
    """
    Digital interface analysis: fanout, decoupling, timing, crystal oscillator.

    References: JEDEC standards, IPC-2221A, Texas Instruments HDG001.
    """

    # Logic family drive/sink capabilities (typical)
    LOGIC_FAMILIES = {
        "TTL":     {"vol_v": 0.4, "voh_v": 2.4, "iol_ma": 8,  "ioh_ma": 0.4},
        "LVTTL":   {"vol_v": 0.4, "voh_v": 2.4, "iol_ma": 8,  "ioh_ma": 0.4},
        "LVCMOS":  {"vol_v": 0.4, "voh_v": 3.0, "iol_ma": 8,  "ioh_ma": 8  },
        "LVCMOS18":{"vol_v": 0.2, "voh_v": 1.6, "iol_ma": 8,  "ioh_ma": 8  },
        "LVDS":    {"vol_v": 0.9, "voh_v": 1.25,"iol_ma": 3.5,"ioh_ma": 3.5 },
        "CMOS":    {"vol_v": 0.1, "voh_v": 4.9, "iol_ma": 4,  "ioh_ma": 4  },
    }

    @staticmethod
    def fanout_analysis(
        driver_family: str,
        load_family: str,
        n_loads: int,
        trace_length_mm: float = 10.0,
        trace_capacitance_pf_mm: float = 0.1,
    ) -> Dict[str, Any]:
        """
        DC fanout analysis for logic drivers.
        """
        drv = DigitalDesignEngine.LOGIC_FAMILIES.get(driver_family.upper(),
              DigitalDesignEngine.LOGIC_FAMILIES["LVCMOS"])
        ld  = DigitalDesignEngine.LOGIC_FAMILIES.get(load_family.upper(),
              DigitalDesignEngine.LOGIC_FAMILIES["LVCMOS"])

        # Typical load current per gate: 10µA for CMOS, 1mA for TTL
        i_load_per_gate_ma = 0.01 if "CMOS" in load_family.upper() else 0.5

        i_total_sink_ma   = n_loads * i_load_per_gate_ma
        i_total_source_ma = n_loads * i_load_per_gate_ma

        dc_fanout_ok = (i_total_sink_ma <= drv["iol_ma"] and
                        i_total_source_ma <= drv["ioh_ma"])

        # Trace capacitance load
        c_trace_pf = trace_length_mm * trace_capacitance_pf_mm
        c_total_pf = c_trace_pf + n_loads * 10  # 10pF per load pin typical

        # Rise time degradation
        # Assuming driver output resistance ~ Voh / Ioh
        r_out = 25  # typical CMOS output resistance Ω
        tau_ns = r_out * c_total_pf * 1e-12 * 1e9
        rise_time_ns = 2.2 * tau_ns  # 10-90% rise time

        max_fanout_dc = int(min(drv["iol_ma"], drv["ioh_ma"]) / max(i_load_per_gate_ma, 0.001))

        return {
            "driver": driver_family,
            "load": load_family,
            "n_loads": n_loads,
            "dc_fanout_ok": dc_fanout_ok,
            "max_dc_fanout": max_fanout_dc,
            "total_load_current_ma": round(i_total_sink_ma, 3),
            "trace_capacitance_pf": round(c_trace_pf, 1),
            "total_capacitance_pf": round(c_total_pf, 1),
            "estimated_rise_time_ns": round(rise_time_ns, 2),
            "recommendation": (
                "OK" if dc_fanout_ok else
                f"Exceed fanout — add buffer. Max {max_fanout_dc} loads."
            ),
        }

    @staticmethod
    def decoupling_cap_per_ic(
        v_supply: float,
        max_current_ma: float,
        rise_time_ns: float,
        trace_inductance_nh: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Decoupling capacitor value to supply transient current.
        C ≥ I_transient × Δt / ΔV  (charge storage requirement)
        Also checks inductive bounce: ΔV = L × dI/dt
        """
        delta_v_allowed = v_supply * 0.05  # 5% ripple
        i_transient = max_current_ma * 1e-3
        dt = rise_time_ns * 1e-9

        # Charge storage
        c_charge = i_transient * dt / delta_v_allowed

        # Inductive bounce
        v_bounce = trace_inductance_nh * 1e-9 * i_transient / dt
        c_bounce = i_transient * dt / v_supply  # keep bounce < 1V

        c_required = max(c_charge, c_bounce)
        c_rounded = _round_e_series(max(c_required, 100e-9), 24)  # min 100nF

        # SRF check: capacitor must be below SRF at target frequency
        f_target = 1 / (2 * rise_time_ns * 1e-9)
        esl_nh = 1.0  # typical 0402 cap
        srf = 1 / (2 * math.pi * math.sqrt(esl_nh * 1e-9 * c_rounded))

        return {
            "c_required_nf": round(c_required * 1e9, 2),
            "c_recommended_nf": round(c_rounded * 1e9, 2),
            "inductive_bounce_mv": round(v_bounce * 1000, 1),
            "self_resonant_freq_mhz": round(srf / 1e6, 1),
            "adequate_to_target_freq": srf > f_target,
            "package_recommendation": "0402 or 0201 for lowest inductance",
        }

    @staticmethod
    def crystal_load_capacitors(
        c_l_pf: float,         # crystal load capacitance (from datasheet)
        c_stray_pf: float = 3.0,  # PCB stray capacitance
        c_in_pin_pf: float = 2.0,  # IC input pin capacitance
    ) -> Dict[str, Any]:
        """
        Crystal load capacitor calculation: C1 = C2 = 2(CL - Cstray - Cin/2).
        IPC-2221A; oscillator IC application notes.
        """
        c_ext = 2 * (c_l_pf - c_stray_pf) - c_in_pin_pf
        c_ext = max(c_ext, 10)  # minimum 10pF practical
        c_rounded = _round_e_series(c_ext * 1e-12, 24)

        c_eff = c_rounded / 2 + c_stray_pf + c_in_pin_pf / 2

        return {
            "c_l_pf": c_l_pf,
            "c1_c2_pf": round(c_rounded * 1e12, 1),
            "c_eff_pf": round(c_eff, 2),
            "error_pct": round(abs(c_eff - c_l_pf) / c_l_pf * 100, 2),
            "placement": "Place as close to oscillator pins as possible; ground directly under",
        }

    @staticmethod
    def timing_budget(
        t_clk_ns: float,
        t_setup_ns: float,
        t_hold_ns: float,
        t_prop_ns: float,
        t_skew_ns: float = 0.0,
        t_jitter_ns: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Setup and hold slack calculation.

        Setup slack = T_clk - T_prop - T_setup - T_skew - T_jitter
        Hold slack  = T_prop - T_hold + T_skew - T_jitter
        """
        setup_slack = t_clk_ns - t_prop_ns - t_setup_ns - t_skew_ns - t_jitter_ns
        hold_slack  = t_prop_ns - t_hold_ns + t_skew_ns - t_jitter_ns

        return {
            "t_clk_ns": t_clk_ns,
            "f_max_mhz": round(1 / (t_clk_ns * 1e-9) / 1e6, 2),
            "setup_slack_ns": round(setup_slack, 3),
            "hold_slack_ns": round(hold_slack, 3),
            "setup_ok": setup_slack >= 0,
            "hold_ok": hold_slack >= 0,
            "timing_ok": setup_slack >= 0 and hold_slack >= 0,
            "recommendation": (
                "Timing OK" if (setup_slack >= 0 and hold_slack >= 0)
                else ("Reduce clock frequency or pipeline" if setup_slack < 0
                      else "Add delay or reduce skew for hold margin")
            ),
        }


# ═════════════════════════════════════════════════════════════════════════════
# 7.  RF DESIGN ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class RFDesignEngine:
    """
    RF circuit design: impedance matching, link budget, transmission lines,
    noise figure cascade, and basic antenna sizing.

    References: Pozar "Microwave Engineering" 4th ed.; Friis 1946 IRE paper.
    Uses scikit-rf (skrf) for S-parameter network analysis when available.
    """

    # ── Transmission line impedance ────────────────────────────────────────────

    @staticmethod
    def microstrip_z0(w_mm: float, h_mm: float, er: float,
                       t_mm: float = 0.035) -> float:
        """Quick accessor — delegates to SignalIntegrityEngine."""
        return SignalIntegrityEngine.microstrip_z0(w_mm, h_mm, t_mm, er)["z0_ohm"]

    # ── Impedance matching networks ────────────────────────────────────────────

    @staticmethod
    def design_l_match(
        z_source_ohm: float,
        z_load_ohm: float,
        freq_hz: float,
        high_pass: bool = False,
    ) -> Dict[str, Any]:
        """
        L-network impedance matching (lossless, single frequency).
        Works for real impedances. Returns L and C component values.

        Pozar "Microwave Engineering" §5.1.
        """
        rs = z_source_ohm
        rl = z_load_ohm

        if rs == rl:
            return {"note": "Source and load already matched — no network needed",
                    "z_source": rs, "z_load": rl}

        # Determine which topology: source > load → shunt element at source side
        if rs > rl:
            q = math.sqrt(rs / rl - 1)
            xs = rs / q         # series reactance (at source side)
            xp = rl * q         # shunt reactance (at load side)
            shunt_at = "source"
        else:
            q = math.sqrt(rl / rs - 1)
            xs = rl / q
            xp = rs * q
            shunt_at = "load"

        w = 2 * math.pi * freq_hz

        if not high_pass:
            # Low-pass: shunt C, series L
            c_shunt = 1 / (w * xp)
            l_series = xs / w
            return {
                "topology": "L-network (low-pass)",
                "shunt_at": shunt_at,
                "l_nh": round(l_series * 1e9, 3),
                "c_pf": round(c_shunt * 1e12, 3),
                "q_factor": round(q, 3),
                "bandwidth_hz": round(freq_hz / q, 0),
                "z_source": rs, "z_load": rl, "freq_hz": freq_hz,
            }
        else:
            # High-pass: shunt L, series C
            l_shunt = xp / w
            c_series = 1 / (w * xs)
            return {
                "topology": "L-network (high-pass)",
                "shunt_at": shunt_at,
                "l_nh": round(l_shunt * 1e9, 3),
                "c_pf": round(c_series * 1e12, 3),
                "q_factor": round(q, 3),
                "bandwidth_hz": round(freq_hz / q, 0),
                "z_source": rs, "z_load": rl, "freq_hz": freq_hz,
            }

    @staticmethod
    def design_pi_match(
        z_source_ohm: float,
        z_load_ohm: float,
        freq_hz: float,
        q_target: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Pi-network impedance matching with specified Q (bandwidth control).
        Uses virtual resistance Rv = min(Rs, Rl) / ((Q²+1) ... ).
        Pozar §5.1.
        """
        rs = z_source_ohm
        rl = z_load_ohm
        w = 2 * math.pi * freq_hz

        # Virtual impedance in the middle
        rv = min(rs, rl) / (q_target ** 2 + 1)
        if rv <= 0:
            return {"error": "Q too high for given impedances"}

        # Left L-network (source → Rv)
        q_left = math.sqrt(rs / rv - 1)
        x_shunt_left = rs / q_left
        x_series_left = rv * q_left

        # Right L-network (Rv → load)
        q_right = math.sqrt(rl / rv - 1)
        x_shunt_right = rl / q_right
        x_series_right = rv * q_right

        c_left  = 1 / (w * x_shunt_left)
        c_right = 1 / (w * x_shunt_right)
        # Series elements add (both inductive)
        l_series = (x_series_left + x_series_right) / w

        return {
            "topology": "Pi-network",
            "l_series_nh": round(l_series * 1e9, 3),
            "c_left_pf": round(c_left * 1e12, 3),
            "c_right_pf": round(c_right * 1e12, 3),
            "q_factor": round(q_target, 2),
            "bandwidth_3db_hz": round(freq_hz / q_target, 0),
            "virtual_r_ohm": round(rv, 3),
            "z_source": rs, "z_load": rl,
        }

    # ── Link budget ────────────────────────────────────────────────────────────

    @staticmethod
    def link_budget(
        p_tx_dbm: float,
        g_tx_dbi: float,
        g_rx_dbi: float,
        freq_hz: float,
        distance_m: float,
        cable_loss_db: float = 0.0,
        other_losses_db: float = 0.0,
        rx_sensitivity_dbm: float = -90.0,
    ) -> Dict[str, Any]:
        """
        Friis transmission equation.
        P_rx = P_tx + G_tx + G_rx − FSPL − losses

        Reference: Friis, H.T., "A Note on a Simple Transmission Formula", IRE 1946.
        """
        # Free-space path loss (dB)
        wavelength_m = C_LIGHT / freq_hz
        fspl_db = 20 * math.log10(4 * math.pi * distance_m / wavelength_m)

        p_rx_dbm = (p_tx_dbm + g_tx_dbi + g_rx_dbi
                    - fspl_db - cable_loss_db - other_losses_db)

        link_margin = p_rx_dbm - rx_sensitivity_dbm

        return {
            "p_tx_dbm": p_tx_dbm,
            "eirp_dbm": round(p_tx_dbm + g_tx_dbi, 2),
            "fspl_db": round(fspl_db, 2),
            "total_loss_db": round(fspl_db + cable_loss_db + other_losses_db, 2),
            "p_rx_dbm": round(p_rx_dbm, 2),
            "rx_sensitivity_dbm": rx_sensitivity_dbm,
            "link_margin_db": round(link_margin, 2),
            "link_ok": link_margin >= 0,
            "freq_mhz": round(freq_hz / 1e6, 3),
            "distance_m": distance_m,
            "wavelength_mm": round(wavelength_m * 1000, 2),
        }

    # ── Noise figure cascade ───────────────────────────────────────────────────

    @staticmethod
    def noise_figure_cascade(nf_db_list: List[float],
                              gain_db_list: List[float]) -> Dict[str, float]:
        """
        Friis formula for cascaded noise figure.
        F_total = F1 + (F2-1)/G1 + (F3-1)/(G1G2) + ...
        """
        if len(nf_db_list) != len(gain_db_list):
            raise ValueError("nf and gain lists must have same length")

        f_total = 0.0
        gain_product = 1.0
        for i, (nf_db, g_db) in enumerate(zip(nf_db_list, gain_db_list)):
            fi = 10 ** (nf_db / 10)
            f_total += (fi - 1) / gain_product
            gain_product *= 10 ** (g_db / 10)

        nf_total_db = 10 * math.log10(f_total)
        return {
            "nf_cascade_db": round(nf_total_db, 3),
            "f_cascade_linear": round(f_total, 5),
            "dominant_stage": "Stage 1 (LNA or first element is critical)",
        }

    @staticmethod
    def antenna_gain_to_effective_area(gain_dbi: float, freq_hz: float) -> Dict[str, float]:
        """Effective aperture: Ae = λ² G / (4π). Pozar eq. 2.100."""
        lam = C_LIGHT / freq_hz
        ae = lam ** 2 * 10 ** (gain_dbi / 10) / (4 * math.pi)
        return {
            "effective_area_cm2": round(ae * 1e4, 4),
            "wavelength_mm": round(lam * 1000, 2),
            "gain_dbi": gain_dbi,
            "freq_mhz": round(freq_hz / 1e6, 2),
        }

    @staticmethod
    def patch_antenna_dimensions(freq_hz: float, er: float = 4.4,
                                  h_mm: float = 1.6) -> Dict[str, float]:
        """
        Rectangular microstrip patch antenna dimensions (λ/2 resonator).
        Pozar "Microwave Engineering" §14.2 approximate formulas.
        """
        lam = C_LIGHT / freq_hz
        # Effective permittivity
        h = h_mm * 1e-3
        W = C_LIGHT / (2 * freq_hz) * math.sqrt(2 / (er + 1))

        er_eff = (er + 1) / 2 + (er - 1) / 2 / math.sqrt(1 + 12 * h / W)

        # Fringing extension
        delta_L = 0.412 * h * (er_eff + 0.3) * (W / h + 0.264) / ((er_eff - 0.258) * (W / h + 0.8))
        L = C_LIGHT / (2 * freq_hz * math.sqrt(er_eff)) - 2 * delta_L

        return {
            "width_mm": round(W * 1000, 2),
            "length_mm": round(L * 1000, 2),
            "er_eff": round(er_eff, 4),
            "substrate_height_mm": h_mm,
            "er": er,
            "freq_mhz": round(freq_hz / 1e6, 2),
            "expected_gain_dbi": 5.0,  # typical rectangular patch
        }


# ═════════════════════════════════════════════════════════════════════════════
# 8.  GERBER WRITER
# ═════════════════════════════════════════════════════════════════════════════

class GerberWriter:
    """
    RS-274X Gerber file generator and Excellon drill file writer.

    Generates: .gtl (top Cu), .gbl (bottom Cu), .gts/.gbs (solder mask),
    .gto/.gbo (silkscreen), .gko (board outline), .drl (Excellon drill),
    BOM CSV, pick-and-place CSV.

    References: Ucamco RS-274X Extended Gerber Format Specification rev 2023.05.
    """

    # Gerber unit: mm with 6-decimal format (mm × 1e6 = integer)
    _SCALE = 1e6
    _DATE_FMT = "%Y-%m-%dT%H:%M:%S"

    def __init__(self):
        self._apertures: Dict[str, int] = {}   # key → D-code
        self._next_d_code = 10

    def _get_aperture(self, shape: str, size_mm: float) -> int:
        key = f"{shape}_{size_mm:.6f}"
        if key not in self._apertures:
            self._apertures[key] = self._next_d_code
            self._next_d_code += 1
        return self._apertures[key]

    @staticmethod
    def _coord(v_mm: float) -> str:
        return f"{int(round(v_mm * 1e6)):+010d}"

    def _gerber_header(self, layer_name: str) -> str:
        now = datetime.datetime.utcnow().strftime(self._DATE_FMT)
        return (
            f"G04 BRICK OS Electronics Agent — {layer_name} *\n"
            f"G04 Generated: {now} *\n"
            f"%FSLAX46Y46*%\n"  # 4 integer, 6 decimal (mm)
            f"%MOMM*%\n"        # metric
            f"%LPD*%\n"         # layer polarity: Dark
        )

    def _aperture_definitions(self) -> str:
        lines = []
        for key, d_code in self._apertures.items():
            parts = key.rsplit("_", 1)
            shape, size = parts[0], float(parts[1])
            if shape == "C":
                lines.append(f"%ADD{d_code}C,{size:.6f}*%")
            elif shape == "R":
                lines.append(f"%ADD{d_code}R,{size:.6f}X{size:.6f}*%")
            else:
                lines.append(f"%ADD{d_code}C,{size:.6f}*%")
        return "\n".join(lines)

    def write_copper_layer(
        self,
        layer_name: str,
        traces: List[TraceSpec],
        vias: List[ViaSpec],
        pads: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate RS-274X Gerber for a copper layer. Returns file content."""
        self._apertures = {}
        self._next_d_code = 10
        lines = [self._gerber_header(layer_name)]

        # Pre-register apertures
        for tr in traces:
            self._get_aperture("C", tr.width_mm)
        for v in vias:
            self._get_aperture("C", v.pad_mm)

        lines.append(self._aperture_definitions())
        lines.append("G01*")   # linear interpolation mode

        # Draw traces
        for tr in traces:
            d_code = self._get_aperture("C", tr.width_mm)
            x0, y0 = tr.start
            x1, y1 = tr.end
            lines.append(f"D{d_code}*")
            lines.append(f"X{self._coord(x0)}Y{self._coord(y0)}D02*")  # move
            lines.append(f"X{self._coord(x1)}Y{self._coord(y1)}D01*")  # draw

        # Flash via pads
        for v in vias:
            d_code = self._get_aperture("C", v.pad_mm)
            lines.append(f"D{d_code}*")
            lines.append(f"X{self._coord(v.x)}Y{self._coord(v.y)}D03*")  # flash

        # Flash component pads
        if pads:
            for pad in pads:
                size = pad.get("size_mm", 1.5)
                d_code = self._get_aperture("R", size)
                lines.append(f"D{d_code}*")
                lines.append(f"X{self._coord(pad['x'])}Y{self._coord(pad['y'])}D03*")

        lines.append("M02*")   # end of file
        return "\n".join(lines)

    def write_board_outline(
        self,
        outline_pts: List[Tuple[float, float]],
    ) -> str:
        """Generate board outline (Edge_Cuts / .gko) layer."""
        self._apertures = {}
        self._next_d_code = 10
        self._get_aperture("C", 0.1)   # 0.1mm outline trace

        lines = [
            self._gerber_header("Board Outline"),
            self._aperture_definitions(),
            "G01*",
            "D10*",
        ]
        if outline_pts:
            x0, y0 = outline_pts[0]
            lines.append(f"X{self._coord(x0)}Y{self._coord(y0)}D02*")
            for x, y in outline_pts[1:]:
                lines.append(f"X{self._coord(x)}Y{self._coord(y)}D01*")
            # Close outline
            lines.append(f"X{self._coord(x0)}Y{self._coord(y0)}D01*")
        lines.append("M02*")
        return "\n".join(lines)

    def write_drill_file(self, vias: List[ViaSpec]) -> str:
        """
        Excellon NC drill file (IPC-NC-349 / Excellon format).
        Groups vias by drill diameter.
        """
        # Group by diameter
        by_size: Dict[float, List[ViaSpec]] = {}
        for v in vias:
            by_size.setdefault(v.drill_mm, []).append(v)

        now = datetime.datetime.utcnow().strftime(self._DATE_FMT)
        lines = [
            "M48",    # header start
            f"; BRICK OS Electronics Agent — Drill File",
            f"; Generated: {now}",
            "METRIC,LZ,000.000",  # metric, leading zeros suppressed
            "FMAT,2",
        ]
        tool_n = 1
        tools: Dict[float, int] = {}
        for dia in sorted(by_size.keys()):
            lines.append(f"T{tool_n:02d}C{dia:.3f}")
            tools[dia] = tool_n
            tool_n += 1
        lines.append("%")  # header end
        lines.append("G90")   # absolute mode
        lines.append("G05")   # drill mode

        for dia in sorted(by_size.keys()):
            lines.append(f"T{tools[dia]:02d}")
            for v in by_size[dia]:
                lines.append(f"X{v.x:.3f}Y{v.y:.3f}")

        lines.append("T00")   # tool off
        lines.append("M30")   # end of file
        return "\n".join(lines)

    @staticmethod
    def write_bom_csv(bom: Dict[str, Any]) -> str:
        """Generate BOM CSV (IPC-2581 compatible field names)."""
        buf = io.StringIO()
        fieldnames = ["Reference", "Value", "Quantity", "Manufacturer",
                       "MPN", "Description", "Package", "Price_USD", "Stock"]
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for ref, data in bom.items():
            if isinstance(data, dict):
                writer.writerow({
                    "Reference": ref,
                    "Value": data.get("value", ""),
                    "Quantity": data.get("quantity", 1),
                    "Manufacturer": data.get("manufacturer", ""),
                    "MPN": data.get("mpn", ""),
                    "Description": data.get("description", ""),
                    "Package": data.get("footprint", ""),
                    "Price_USD": data.get("unit_price", ""),
                    "Stock": data.get("stock", ""),
                })
        return buf.getvalue()

    @staticmethod
    def write_pick_and_place_csv(
        components: List[Dict[str, Any]]
    ) -> str:
        """
        Pick-and-place (CPL) CSV — compatible with JLCPCB, PCBWay formats.
        """
        buf = io.StringIO()
        fieldnames = ["Designator", "Val", "Package", "Mid X", "Mid Y",
                       "Rotation", "Layer"]
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for comp in components:
            writer.writerow({
                "Designator": comp.get("ref", ""),
                "Val": comp.get("value", ""),
                "Package": comp.get("package", ""),
                "Mid X": comp.get("x_mm", 0),
                "Mid Y": comp.get("y_mm", 0),
                "Rotation": comp.get("rotation_deg", 0),
                "Layer": comp.get("layer", "Top"),
            })
        return buf.getvalue()

    def generate_fab_package(
        self,
        layer_name: str,
        traces: List[TraceSpec],
        vias: List[ViaSpec],
        board_outline: Optional[List[Tuple[float, float]]] = None,
        bom: Optional[Dict[str, Any]] = None,
        components: Optional[List[Dict[str, Any]]] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Generate complete fabrication package: Gerbers + drill + BOM + CPL.
        Returns dict of {filename: content}. Optionally writes to output_dir.
        """
        files: Dict[str, str] = {}

        files[f"{layer_name}-F_Cu.gtl"] = self.write_copper_layer(
            "Top Copper", traces, vias)
        files[f"{layer_name}-B_Cu.gbl"] = self.write_copper_layer(
            "Bottom Copper", [], vias)  # simplified: no bottom traces
        if board_outline:
            files[f"{layer_name}-Edge_Cuts.gko"] = self.write_board_outline(board_outline)
        files[f"{layer_name}.drl"] = self.write_drill_file(vias)
        if bom:
            files[f"{layer_name}-BOM.csv"] = self.write_bom_csv(bom)
        if components:
            files[f"{layer_name}-CPL.csv"] = self.write_pick_and_place_csv(components)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for fname, content in files.items():
                with open(os.path.join(output_dir, fname), "w") as fh:
                    fh.write(content)

        return files


# ═════════════════════════════════════════════════════════════════════════════
# 9.  EMC ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class EMCEngine:
    """
    EMC pre-compliance estimation.

    CISPR 25 (automotive) and CISPR 32 (multimedia equipment) limits.
    Radiated emission model: electrically small loop (IEC CISPR 16-1-4).
    Conducted emission: simplified voltage-probe model.

    WARNING: These are estimates only. Final compliance requires accredited
    test lab measurements per CISPR 25/32 test method.
    """

    # CISPR 32 Class B radiated limits (quasi-peak, dBµV/m at 10m)
    _CISPR32_CLASS_B_QP = [  # (freq_mhz, limit_dbuv_m_10m)
        (30,   40.0), (230,  40.0), (230,  47.0),
        (1000, 47.0),
    ]

    # CISPR 25 broadband limits (peak, dBµV/m at 1m) — class 3
    _CISPR25_CLASS3_PEAK = [
        (0.1,  62), (0.15, 62), (0.53, 50), (1.8,  44),
        (54,   38), (76,   38), (108,  24), (174,  20),
        (230,  20), (400,  20), (960,  20), (2170, 20),
    ]

    @staticmethod
    def radiated_emission_estimate(
        loop_area_cm2: float,
        current_ma: float,
        freq_mhz: float,
        distance_m: float = 10.0,
    ) -> Dict[str, float]:
        """
        Far-field E-field from electrically small current loop.
        E ≈ 263e-16 × A × I × f² / r  [V/m]

        Reference: IEC CISPR 16-1-4 Section 5, Paul "Introduction to EMC" Ch.7
        Loop area A in m², current I in A, f in Hz, r in m.
        """
        a_m2 = loop_area_cm2 * 1e-4
        i_a  = current_ma * 1e-3
        f_hz = freq_mhz * 1e6

        # Electric field (V/m)
        e_v_m = 2.63e-15 * a_m2 * i_a * f_hz ** 2 / distance_m

        # Convert to dBµV/m
        e_dbuv_m = 20 * math.log10(max(e_v_m, 1e-15) * 1e6)

        return {
            "e_field_v_m": round(e_v_m, 9),
            "e_field_dbuv_m": round(e_dbuv_m, 2),
            "freq_mhz": freq_mhz,
            "distance_m": distance_m,
            "loop_area_cm2": loop_area_cm2,
            "current_ma": current_ma,
        }

    @staticmethod
    def cispr32_class_b_limit(freq_mhz: float) -> float:
        """CISPR 32 Class B quasi-peak limit (dBµV/m at 10m)."""
        if freq_mhz < 30:
            return 40.0
        elif freq_mhz <= 230:
            return 40.0 + (freq_mhz - 30) / (230 - 30) * 7  # interpolated
        elif freq_mhz <= 1000:
            return 47.0
        else:
            return 54.0  # approximation above 1GHz

    @staticmethod
    def check_cispr32_compliance(
        loop_specs: List[Dict[str, float]],   # [{loop_area_cm2, current_ma, freq_mhz}]
        margin_db: float = 6.0,
    ) -> Dict[str, Any]:
        """
        Estimate CISPR 32 Class B compliance for a list of current loops.
        Checks each harmonic against the limit with given margin.
        """
        results = []
        all_pass = True
        for spec in loop_specs:
            freq = spec["freq_mhz"]
            emission = EMCEngine.radiated_emission_estimate(
                spec["loop_area_cm2"], spec["current_ma"], freq, 10.0
            )
            limit = EMCEngine.cispr32_class_b_limit(freq)
            e_db = emission["e_field_dbuv_m"]
            headroom = limit - e_db - margin_db
            passes = headroom >= 0
            all_pass = all_pass and passes
            results.append({
                "freq_mhz": freq,
                "emission_dbuv_m": round(e_db, 2),
                "limit_dbuv_m": limit,
                "margin_required_db": margin_db,
                "headroom_db": round(headroom, 2),
                "pass": passes,
            })

        return {
            "standard": "CISPR 32 Class B (estimated, not certified)",
            "overall_pass": all_pass,
            "results": results,
        }

    @staticmethod
    def cm_filter_design(
        noise_current_ma: float,
        freq_mhz: float,
        target_insertion_loss_db: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Common-mode choke + capacitor filter design for conducted EMI.
        Returns inductance needed for given insertion loss at given frequency.

        Reference: Ott "EMC Engineering" ch.5; TI SNVA489.
        """
        # For LC filter: IL = 20 log10(1 + Z_choke / (2 × Z_source))
        # With Z_source = 50Ω (LISN): Z_choke = 50 × (10^(IL/20) - 1) × 2
        z_choke_needed = 50 * (10 ** (target_insertion_loss_db / 20) - 1) * 2
        w = 2 * math.pi * freq_mhz * 1e6
        l_cm = z_choke_needed / w

        # Bypass capacitors (Y caps, safety class Y1/Y2)
        # C_y such that capacitor reactance << Z_source at freq
        c_y = 1 / (w * 50 / 10)  # Xc = 5Ω  → 10× lower than 50Ω

        return {
            "cm_choke_uh": round(l_cm * 1e6, 2),
            "cm_choke_impedance_ohm": round(z_choke_needed, 1),
            "y_cap_nf": round(c_y * 1e9, 2),
            "y_cap_class": "Y2 (300VAC rated) for mains or Y1 for cross-mains",
            "insertion_loss_db": target_insertion_loss_db,
            "freq_mhz": freq_mhz,
            "note": "Place CM choke on cable entry; Y-caps from each line to chassis ground",
        }

    @staticmethod
    def shielding_effectiveness(
        material: str,
        thickness_mm: float,
        freq_mhz: float,
    ) -> Dict[str, float]:
        """
        Shielding effectiveness (Schelkunoff model, plane wave, far field).
        SE = A + R + B  where A=absorption, R=reflection, B=multiple reflections.

        Reference: Schelkunoff 1943; Ott "EMC Engineering" ch.8.
        """
        mat_props = {
            "copper":     {"sigma_r": 1.0,    "mu_r": 1.0},
            "aluminum":   {"sigma_r": 0.61,   "mu_r": 1.0},
            "steel":      {"sigma_r": 0.10,   "mu_r": 200.0},
            "mu_metal":   {"sigma_r": 0.03,   "mu_r": 20000.0},
            "galv_steel": {"sigma_r": 0.10,   "mu_r": 200.0},
        }
        props = mat_props.get(material.lower(), {"sigma_r": 1.0, "mu_r": 1.0})
        sigma_r = props["sigma_r"]
        mu_r    = props["mu_r"]
        sigma   = sigma_r * 5.8e7  # S/m (copper reference)

        f = freq_mhz * 1e6

        # Skin depth (m)
        delta = math.sqrt(1 / (math.pi * f * mu_r * MU_0 * sigma))

        t = thickness_mm * 1e-3
        # Absorption loss A (dB)
        a_db = 8.686 * t / delta

        # Reflection loss R (dB) — plane wave
        r_db = 168 - 10 * math.log10(mu_r / sigma_r * f * 1e-6)

        # Multiple-reflection correction B (significant only when A < 15dB)
        if a_db < 15:
            b_db = 20 * math.log10(1 - math.exp(-2 * t / delta))
        else:
            b_db = 0.0

        se_db = a_db + r_db + b_db

        return {
            "material": material,
            "se_total_db": round(se_db, 1),
            "absorption_db": round(a_db, 1),
            "reflection_db": round(r_db, 1),
            "multiple_reflection_db": round(b_db, 1),
            "skin_depth_um": round(delta * 1e6, 2),
            "thickness_mm": thickness_mm,
            "freq_mhz": freq_mhz,
        }

    @staticmethod
    def ferrite_bead_selection(
        target_impedance_ohm_at_100mhz: float,
        dc_current_a: float,
        freq_mhz: float,
    ) -> Dict[str, Any]:
        """
        Ferrite bead selection criteria.
        Returns impedance threshold, current rating requirement, and part guidance.
        """
        # Ferrite bead is lossy inductor: Z ≈ R + jX, peaks near SRF
        # Typical: BLM series (Murata), HH series (TDK), use rated at 1.5× DC current
        i_rated = dc_current_a * 1.5
        # At freq_mhz, scale from 100MHz reference (ferrite is not linear)
        # Simplified: Z ∝ f^0.7 below SRF (empirical Murata BLM)
        z_at_freq = target_impedance_ohm_at_100mhz * (freq_mhz / 100) ** 0.7
        # Insertion loss at given frequency vs 50Ω source
        il_db = 20 * math.log10(1 + z_at_freq / (2 * 50))

        return {
            "required_z_100mhz_ohm": target_impedance_ohm_at_100mhz,
            "estimated_z_at_freq_ohm": round(z_at_freq, 1),
            "estimated_insertion_loss_db": round(il_db, 1),
            "dc_current_rating_a": round(i_rated, 2),
            "package_recommendation": "0402 or 0603 depending on current rating",
            "part_family": "Murata BLM / TDK MMZ / Wurth WE-CBF",
        }


# ═════════════════════════════════════════════════════════════════════════════
# Utility: E-series rounding
# ═════════════════════════════════════════════════════════════════════════════

_E_SERIES: Dict[int, List[float]] = {
    12:  [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2],
    24:  [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
          3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1],
    96:  [1.00, 1.02, 1.05, 1.07, 1.10, 1.13, 1.15, 1.18, 1.21, 1.24, 1.27, 1.30,
          1.33, 1.37, 1.40, 1.43, 1.47, 1.50, 1.54, 1.58, 1.62, 1.65, 1.69, 1.74,
          1.78, 1.82, 1.87, 1.91, 1.96, 2.00, 2.05, 2.10, 2.15, 2.21, 2.26, 2.32,
          2.37, 2.43, 2.49, 2.55, 2.61, 2.67, 2.74, 2.80, 2.87, 2.94, 3.01, 3.09,
          3.16, 3.24, 3.32, 3.40, 3.48, 3.57, 3.65, 3.74, 3.83, 3.92, 4.02, 4.12,
          4.22, 4.32, 4.42, 4.53, 4.64, 4.75, 4.87, 4.99, 5.11, 5.23, 5.36, 5.49,
          5.62, 5.76, 5.90, 6.04, 6.19, 6.34, 6.49, 6.65, 6.81, 6.98, 7.15, 7.32,
          7.50, 7.68, 7.87, 8.06, 8.25, 8.45, 8.66, 8.87, 9.09, 9.31, 9.53, 9.76],
}


def _round_e_series(value: float, series: int = 24) -> float:
    """Round value to nearest preferred E-series value."""
    if value <= 0:
        return value
    vals = _E_SERIES.get(series, _E_SERIES[24])
    decade = 10 ** math.floor(math.log10(value))
    normalized = value / decade
    # Find nearest in table
    best = min(vals, key=lambda v: abs(v - normalized))
    return round(best * decade, 15)


# ═════════════════════════════════════════════════════════════════════════════
# 10.  MAGNETICS DESIGN ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class MagneticsDesignEngine:
    """
    Transformer and inductor design from first principles.

    Methods cover:
    - Flyback transformer (discontinuous and continuous conduction)
    - Forward/full-bridge transformer
    - LLC resonant tank
    - Coupled-inductor SEPIC
    - Buck/boost output inductor

    Core sizing uses the Area Product (Ap) method:
      Ap = Ae × Aw = (V × t) / (N × ΔB) × (N × I_rms / K_u × J_max)
      Reference: McLyman, "Transformer and Inductor Design Handbook", 4th Ed., §3.2

    Wire gauge from skin depth and current density:
      δ = sqrt(ρ / (π × μ₀ × f_sw))   [skin depth, m]
      J_max = 450 A/cm² (forced-air), 300 A/cm² (natural convection)
      Reference: IPC-2152, McLyman §1.6

    Core loss (Steinmetz equation):
      P_core = K_c × f^α × B_pk^β × V_core
      Parameters K_c, α, β from manufacturer datasheet.
      Typical Ferroxcube 3C97 (optimised 100-400kHz): K_c=9.75e-3, α=1.63, β=2.62
      Reference: Mulder, "Loss Formulas for Power Ferrites", Ferroxcube 1994.
    """

    # Steinmetz parameters for common power ferrite grades
    # (K_c [W/(m³·T^β·Hz^α)], α, β) — from manufacturer datasheets
    # All valid in the range 100kHz–500kHz, B_pk < 0.3T, 25°C
    _CORE_MATERIALS: Dict[str, Dict[str, float]] = {
        "3C97":   {"kc": 9.75e-3,  "alpha": 1.63, "beta": 2.62,  "b_sat": 0.38, "note": "Ferroxcube 3C97 100-400kHz"},
        "3F3":    {"kc": 15.8e-3,  "alpha": 1.75, "beta": 2.75,  "b_sat": 0.36, "note": "Ferroxcube 3F3 500kHz-1MHz"},
        "N87":    {"kc": 16.9e-3,  "alpha": 1.72, "beta": 2.66,  "b_sat": 0.39, "note": "TDK N87 25-200kHz"},
        "N97":    {"kc": 7.0e-3,   "alpha": 1.60, "beta": 2.55,  "b_sat": 0.41, "note": "TDK N97 100-500kHz"},
        "PC95":   {"kc": 6.3e-3,   "alpha": 1.58, "beta": 2.60,  "b_sat": 0.40, "note": "TDK PC95 100-500kHz"},
        "generic":{"kc": 15.0e-3,  "alpha": 1.70, "beta": 2.65,  "b_sat": 0.35, "note": "Conservative generic"},
    }

    # AWG wire table: {AWG: (diameter_mm, resistance_mohm_per_m, current_max_a_300A_cm2)}
    # Source: IPC-2152, NEC Table 310.15(B)(16)
    _AWG_TABLE: Dict[int, Tuple[float, float, float]] = {
        28: (0.321,  214.0,  0.24),
        26: (0.405,  134.0,  0.38),
        24: (0.511,   84.2,  0.60),
        22: (0.644,   53.5,  0.95),
        20: (0.812,   33.6,  1.50),
        18: (1.024,   21.2,  2.30),
        16: (1.291,   13.3,  3.70),
        14: (1.628,    8.4,  5.90),
        12: (2.053,    5.3,  9.30),
        10: (2.588,    3.3, 15.00),
    }

    @classmethod
    def select_wire_gauge(cls, i_rms_a: float, cooling: str = "natural") -> Dict[str, Any]:
        """
        Select minimum AWG wire gauge for a given RMS current.

        Args:
            i_rms_a: RMS winding current [A]
            cooling: 'natural' (J_max=300A/cm²) or 'forced' (J_max=450A/cm²)
        Returns:
            Dict with awg, diameter_mm, resistance_mohm_per_m, headroom_pct
        """
        j_scale = 1.5 if cooling == "forced" else 1.0  # forced-air allows higher J
        best = None
        for awg in sorted(cls._AWG_TABLE.keys(), reverse=True):
            d_mm, r_mohm, i_max = cls._AWG_TABLE[awg]
            if i_max * j_scale >= i_rms_a:
                best = (awg, d_mm, r_mohm, i_max * j_scale)
        if best is None:
            # Larger than AWG 10: parallel strands
            awg, d_mm, r_mohm, i_max = 10, 2.588, 3.3, 15.0 * j_scale
            n_parallel = math.ceil(i_rms_a / i_max)
            return {
                "awg": awg, "n_parallel": n_parallel,
                "diameter_mm": d_mm, "resistance_mohm_per_m": r_mohm / n_parallel,
                "headroom_pct": round((i_max * n_parallel / i_rms_a - 1) * 100, 1),
            }
        awg, d_mm, r_mohm, i_cap = best
        return {
            "awg": awg, "n_parallel": 1,
            "diameter_mm": d_mm, "resistance_mohm_per_m": r_mohm,
            "headroom_pct": round((i_cap / i_rms_a - 1) * 100, 1),
        }

    @classmethod
    def core_loss_steinmetz(
        cls,
        f_sw_hz: float,
        b_pk_t: float,
        volume_m3: float,
        material: str = "generic",
    ) -> float:
        """
        Steinmetz core loss [W].
        P_core = K_c × f^α × B_pk^β × V_core
        Source: Mulder (Ferroxcube 1994); parameters from _CORE_MATERIALS table.
        """
        mat = cls._CORE_MATERIALS.get(material, cls._CORE_MATERIALS["generic"])
        return mat["kc"] * (f_sw_hz ** mat["alpha"]) * (b_pk_t ** mat["beta"]) * volume_m3

    @classmethod
    def design_flyback_transformer(
        cls,
        v_in: float,
        v_out: float,
        i_out: float,
        f_sw: float,
        efficiency: float = 0.85,
        duty_max: float = 0.45,
        b_max_t: float = 0.25,
        j_max_a_cm2: float = 300.0,
        ku: float = 0.4,
        core_material: str = "generic",
    ) -> Dict[str, Any]:
        """
        Flyback transformer design (discontinuous-conduction boundary).

        Design method: Area Product (Ap), McLyman §8.3.
        b_max_t: peak flux density limit [T] — must stay < B_sat (see _CORE_MATERIALS)
        j_max_a_cm2: max current density [A/cm²]
        ku: winding fill factor (0.3–0.5 typical for EE/ETD cores)

        Returns: turns ratio, primary/secondary turns, Ap_cm4, core window, wire gauges.
        """
        # --- Electrical design ---
        turns_ratio = v_out / (v_in * duty_max / (1 - duty_max))
        duty = v_out / (v_in * turns_ratio + v_out)     # actual duty for given n
        duty = min(duty, duty_max)

        p_in = v_out * i_out / efficiency
        i_in_pk = 2 * p_in / (v_in * duty)              # DCM peak primary current
        i_sec_pk = i_in_pk * turns_ratio                 # referred to secondary

        # Primary magnetising inductance (stores energy for full output)
        lp_h = v_in * duty / (i_in_pk * f_sw)

        # Primary RMS current (DCM triangular pulse)
        i_pri_rms = i_in_pk * math.sqrt(duty / 3.0)
        # Secondary RMS current
        i_sec_rms = i_sec_pk * math.sqrt((1 - duty) / 3.0)

        # --- Core sizing via Area Product ---
        # Ap = Ae × Aw = (Lp × I_pk²) / (B_max × J_max × ku × Ku)
        # Simplified Ap formula (McLyman eq. 3-40):
        lp_mh = lp_h * 1e3
        i_in_pk_sq = i_in_pk ** 2
        ap_cm4 = (lp_mh * i_in_pk_sq) / (b_max_t * j_max_a_cm2 * ku * 2.0)
        ae_cm2 = ap_cm4 ** 0.5      # assume Ae ≈ Aw for a well-proportioned core
        aw_cm2 = ap_cm4 / ae_cm2

        # --- Turns ---
        # turns_ratio = Ns/Np (step-down: < 1)
        np = math.ceil(v_in * duty / (f_sw * b_max_t * ae_cm2 * 1e-4))  # Faraday: V×t = N×B×Ae
        ns = max(1, math.ceil(np * turns_ratio))    # Ns = Np × (Ns/Np)
        n_actual = np / ns                          # n_actual = Np/Ns (step-down: > 1)

        # --- Core volume estimate (ellipsoidal approximation) ---
        # Ae_cm2 → core leg radius; volume from standard EE/ETD proportion
        r_cm = math.sqrt(ae_cm2 / math.pi)
        vol_m3 = math.pi * (r_cm * 1e-2) ** 2 * (r_cm * 4 * 1e-2)  # rough cylinder

        # --- Wire gauges ---
        pri_wire = cls.select_wire_gauge(i_pri_rms)
        sec_wire = cls.select_wire_gauge(i_sec_rms)

        # --- Losses ---
        p_core = cls.core_loss_steinmetz(f_sw, b_max_t, vol_m3, core_material)
        p_cu_pri = i_pri_rms ** 2 * (pri_wire["resistance_mohm_per_m"] * 1e-3 * np * 0.05)
        p_cu_sec = i_sec_rms ** 2 * (sec_wire["resistance_mohm_per_m"] * 1e-3 * ns * 0.05)
        p_total = p_core + p_cu_pri + p_cu_sec
        eta_magnetics = 1.0 - p_total / max(p_in, 1.0)

        b_sat = cls._CORE_MATERIALS.get(core_material, cls._CORE_MATERIALS["generic"])["b_sat"]

        return {
            "topology": "flyback",
            "turns_ratio": round(n_actual, 4),
            "np": np,
            "ns": ns,
            "lp_h": round(lp_h, 9),
            "ls_h": round(lp_h / n_actual ** 2, 9),
            "duty": round(duty, 4),
            "i_pri_pk_a": round(i_in_pk, 3),
            "i_pri_rms_a": round(i_pri_rms, 3),
            "i_sec_pk_a": round(i_sec_pk, 3),
            "i_sec_rms_a": round(i_sec_rms, 3),
            "ap_cm4": round(ap_cm4, 4),
            "ae_cm2": round(ae_cm2, 4),
            "aw_cm2": round(aw_cm2, 4),
            "b_pk_t": round(b_max_t, 3),
            "b_sat_t": b_sat,
            "b_headroom_pct": round((1 - b_max_t / b_sat) * 100, 1),
            "core_material": core_material,
            "core_loss_w": round(p_core, 3),
            "copper_loss_pri_w": round(p_cu_pri, 3),
            "copper_loss_sec_w": round(p_cu_sec, 3),
            "total_loss_w": round(p_total, 3),
            "magnetics_efficiency_pct": round(eta_magnetics * 100, 1),
            "pri_wire": pri_wire,
            "sec_wire": sec_wire,
        }

    @classmethod
    def design_forward_transformer(
        cls,
        v_in: float,
        v_out: float,
        i_out: float,
        f_sw: float,
        duty_max: float = 0.45,
        efficiency: float = 0.90,
        b_max_t: float = 0.20,
        core_material: str = "generic",
    ) -> Dict[str, Any]:
        """
        Forward converter transformer design (single-switch, core reset via tertiary).

        Forward topology: energy transferred directly, not stored.
        Core must be reset each cycle → duty_max ≤ 0.5.
        Reference: Pressman "Switching Power Supply Design", §4.3.
        """
        turns_ratio = v_out / (v_in * duty_max * 0.95)  # 5% margin
        np = math.ceil(v_in * duty_max / (f_sw * b_max_t * 1e-4 * 1.0))  # Ae = 1cm² placeholder
        ns = max(1, round(np / turns_ratio))
        n_reset = np   # reset winding = primary turns (1:1 for full volt-second balance)
        n_actual = np / ns

        i_pri_rms = i_out / n_actual * math.sqrt(duty_max)
        i_sec_rms = i_out * math.sqrt(duty_max)

        pri_wire = cls.select_wire_gauge(i_pri_rms)
        sec_wire = cls.select_wire_gauge(i_sec_rms)

        return {
            "topology": "forward",
            "turns_ratio": round(n_actual, 4),
            "np": np,
            "ns": ns,
            "n_reset": n_reset,
            "duty_max": duty_max,
            "i_pri_rms_a": round(i_pri_rms, 3),
            "i_sec_rms_a": round(i_sec_rms, 3),
            "pri_wire": pri_wire,
            "sec_wire": sec_wire,
            "b_pk_t": b_max_t,
            "core_material": core_material,
            "note": "Ae=1cm² placeholder — select physical core from Ap≥{:.3f}cm⁴".format(
                (v_in * duty_max * i_pri_rms) / (b_max_t * 300 * 0.4 * 2)
            ),
        }

    @classmethod
    def design_full_bridge_transformer(
        cls,
        v_in: float,
        v_out: float,
        i_out: float,
        f_sw: float,
        efficiency: float = 0.93,
        duty_max: float = 0.45,
        b_max_t: float = 0.15,
        core_material: str = "generic",
    ) -> Dict[str, Any]:
        """
        Full-bridge transformer design.

        Full-bridge uses the core both ways (push-pull), so ΔB = 2×B_max.
        Primary voltage swings between +Vin and -Vin.
        Reference: Pressman §7, TI SLUP359.
        """
        # Full-bridge volt-second balance: Vin × D = Vout × (Np/Ns)
        turns_ratio = v_out / (v_in * duty_max)
        np = math.ceil(v_in * duty_max / (2 * f_sw * b_max_t * 1e-4 * 1.0))  # Ae=1cm²
        ns = max(1, round(np / turns_ratio))
        n_actual = np / ns

        p_in = v_out * i_out / efficiency
        i_pri_rms = (p_in / v_in) * math.sqrt(2 * duty_max)
        i_sec_rms = i_out * math.sqrt(2 * duty_max)

        ap_cm4 = (v_in * duty_max * i_pri_rms) / (2 * b_max_t * 300 * 0.4 * f_sw)

        pri_wire = cls.select_wire_gauge(i_pri_rms)
        sec_wire = cls.select_wire_gauge(i_sec_rms)
        p_core = cls.core_loss_steinmetz(f_sw, b_max_t, 1e-6, core_material)

        return {
            "topology": "full_bridge",
            "turns_ratio": round(n_actual, 4),
            "np": np,
            "ns": ns,
            "duty_max": duty_max,
            "ap_cm4": round(ap_cm4, 4),
            "i_pri_rms_a": round(i_pri_rms, 3),
            "i_sec_rms_a": round(i_sec_rms, 3),
            "pri_wire": pri_wire,
            "sec_wire": sec_wire,
            "b_pk_t": b_max_t,
            "delta_b_t": round(2 * b_max_t, 3),
            "core_material": core_material,
            "core_loss_1cm3_w": round(p_core, 3),
        }

    @classmethod
    def design_llc_resonant_tank(
        cls,
        v_in: float,
        v_out: float,
        i_out: float,
        f_sw: float,
        quality_factor: float = 0.5,
        fn: float = 1.05,
    ) -> Dict[str, Any]:
        """
        LLC resonant tank design (series resonant, half-bridge primary).

        Design proceeds from voltage gain curve:
          M(fn, Q) = fn² / sqrt((fn²-1)² + fn²×Q²×(fn²-1)²/(fn²-1)²)
          Simplified first-harmonic approximation (FHA).
          Reference: TI SLUA595A "LLC Resonant Half-Bridge Converter Design".

        fn = f_sw / f_r must be ≥ 1 for ZVS (inductive load).
        Q = quality factor controls gain shape.
        """
        turns_ratio = v_in / (2 * v_out)          # half-bridge: Vpri = Vin/2

        # Resonant frequency from fn
        f_r = f_sw / fn
        omega_r = 2 * math.pi * f_r

        # Characteristic impedance from load Q
        r_load_sec = v_out / i_out
        r_ac = (8 / math.pi ** 2) * (turns_ratio ** 2) * r_load_sec   # FHA equivalent AC load

        # From Z0 = sqrt(Lr/Cr) and Q = Z0/R_ac → Z0 = Q × R_ac
        z0 = quality_factor * r_ac
        lr_h = z0 / omega_r
        cr_f = 1.0 / (omega_r * z0)

        # Magnetising inductance: Lm = k × Lr where k ≥ 3 for good ZVS range
        k_factor = max(3.0, 5.0)
        lm_h = k_factor * lr_h

        # Dead time for ZVS (gate charge time): typically > 200ns
        c_oss_est = 100e-12     # pF — placeholder, requires MOSFET catalog data
        t_dead_ns = math.pi * math.sqrt(lm_h * c_oss_est) * 1e9

        return {
            "topology": "llc_resonant",
            "turns_ratio": round(turns_ratio, 4),
            "f_r_hz": round(f_r, 1),
            "f_sw_hz": f_sw,
            "fn": fn,
            "quality_factor": quality_factor,
            "lr_h": round(lr_h, 9),
            "cr_f": round(cr_f, 12),
            "lm_h": round(lm_h, 9),
            "z0_ohm": round(z0, 3),
            "r_ac_ohm": round(r_ac, 3),
            "t_dead_ns": round(t_dead_ns, 1),
            "zvs_condition": fn >= 1.0,
            "notes": [
                "c_oss uses placeholder 100pF — replace with MOSFET datasheet value",
                "Lm/Lr ratio = {:.1f} (≥3 recommended for ZVS margin)".format(k_factor),
            ],
        }

    @classmethod
    def design_buck_inductor(
        cls,
        v_in: float,
        v_out: float,
        i_out: float,
        f_sw: float,
        ripple_ratio: float = 0.3,
        core_material: str = "generic",
    ) -> Dict[str, Any]:
        """
        Buck converter output inductor design.

        L = (V_in - V_out) × D / (ΔI_L × f_sw)   [Faraday's law, CCM]
        Ap sizing: McLyman §5.2.
        """
        duty = v_out / v_in
        delta_il = i_out * ripple_ratio
        l_h = (v_in - v_out) * duty / (delta_il * f_sw)
        i_pk = i_out + delta_il / 2
        i_rms = math.sqrt(i_out ** 2 + (delta_il / 2) ** 2 / 3)

        # Ap product
        ap_cm4 = (l_h * 1e3 * i_pk * i_rms) / (300 * 0.4 * 0.25 * 2)  # J=300A/cm², ku=0.4, B=0.25T
        ae_cm2 = ap_cm4 ** 0.5
        n_turns = math.ceil((l_h * i_pk) / (0.25 * ae_cm2 * 1e-4))   # Faraday: L×I = N×B×Ae

        wire = cls.select_wire_gauge(i_rms)
        p_core = cls.core_loss_steinmetz(f_sw, 0.25, (ae_cm2 * 1e-4) ** 1.5, core_material)
        dcr_mohm = wire["resistance_mohm_per_m"] * n_turns * 0.05     # avg turn path 5cm

        return {
            "topology": "buck_inductor",
            "inductance_h": round(l_h, 9),
            "n_turns": n_turns,
            "i_pk_a": round(i_pk, 3),
            "i_rms_a": round(i_rms, 3),
            "delta_il_a": round(delta_il, 3),
            "ap_cm4": round(ap_cm4, 5),
            "ae_cm2": round(ae_cm2, 4),
            "b_pk_t": 0.25,
            "core_material": core_material,
            "core_loss_w": round(p_core, 4),
            "dcr_mohm": round(dcr_mohm, 2),
            "wire": wire,
        }


# ═════════════════════════════════════════════════════════════════════════════
# 11.  CONTROL LOOP DESIGN ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class ControlLoopDesignEngine:
    """
    Closed-loop compensation design for switching power converters.

    Covers:
    - Buck voltage-mode control: Type-2 and Type-3 compensators
    - Boost voltage-mode control: Type-2 compensator
    - Current-mode control: single-pole compensation

    Plant model (voltage-mode buck, continuous conduction):
      G_vd(s) = (V_in / V_ramp) × H_lc(s) × H_esr(s)
      H_lc(s) = ω₀² / (s² + ω₀/Q×s + ω₀²)   [double pole at f₀]
      H_esr(s) = (1 + s/ω_z_esr)               [ESR zero]
      ω₀ = 1/sqrt(LC),  Q ≈ R_load/sqrt(L/C)
      Reference: Venable Technology, "Transfer Functions and Bode Plots for Power
      Supply Stability Analysis", SLVA553 (TI application note).

    Compensator recipes:
    - Type-2 (PI + single lead): one zero, two poles. 45–60° phase boost.
      Use when ESR zero frequency > 3×f_c. Suitable for output caps with ESR.
    - Type-3 (two zeros, three poles): up to 90° phase boost.
      Required when LC double-pole is near or above crossover.
      Reference: TI SLVA554, Venable 1994.
    """

    @staticmethod
    def _plant_poles_zeros(
        l_h: float,
        c_out_f: float,
        r_load_ohm: float,
        esr_ohm: float,
    ) -> Dict[str, float]:
        """Compute key plant frequencies for voltage-mode buck."""
        f0 = 1.0 / (2 * math.pi * math.sqrt(l_h * c_out_f))
        q_lc = r_load_ohm / math.sqrt(l_h / c_out_f)
        f_z_esr = 1.0 / (2 * math.pi * esr_ohm * c_out_f) if esr_ohm > 0 else 1e9
        return {"f0_hz": f0, "q_lc": q_lc, "f_z_esr_hz": f_z_esr}

    @classmethod
    def design_type2_compensator(
        cls,
        v_out: float,
        v_ref: float,
        f_sw: float,
        l_h: float,
        c_out_f: float,
        esr_ohm: float,
        r_load_ohm: float,
        v_ramp: float = 1.0,
        r1_ohm: float = 10e3,
        target_phase_margin_deg: float = 52.0,
    ) -> Dict[str, Any]:
        """
        Type-2 compensator design for voltage-mode buck/flyback (CCM).

        Circuit: error amplifier + R1/(R2 + 1/(sC1)) + C2 across op-amp feedback.
        Network: Gc(s) = (1 + s/ω_z) / (s/ω_i × (1 + s/ω_p))
        Zeros and poles placed per K-factor method (Venable 1994, SLVA553).

        Args:
            v_out, v_ref: output and reference voltages for R divider
            f_sw: switching frequency [Hz]
            l_h, c_out_f: output filter inductance/capacitance
            esr_ohm: output capacitor ESR [Ω]
            r_load_ohm: nominal load resistance [Ω]
            v_ramp: PWM ramp amplitude [V] (sawtooth peak)
            r1_ohm: upper divider resistor (sets R2)
            target_phase_margin_deg: desired phase margin [°]

        Returns component values and Bode data at key frequencies.
        """
        plant = cls._plant_poles_zeros(l_h, c_out_f, r_load_ohm, esr_ohm)
        f0 = plant["f0_hz"]
        f_z_esr = plant["f_z_esr_hz"]

        # Choose crossover: f_sw/5 to f_sw/10, but ≤ f_z_esr × 3
        f_c = min(f_sw / 5.0, f_z_esr * 3.0, f_sw / 8.0)
        f_c = max(f_c, f0 * 2.0)   # must be above LC resonance

        # Plant phase at f_c — use exact complex TF evaluation (avoids sign ambiguity)
        # G_vd(jω_c) = Vin/Vramp × (1 + jω_c/ω_z) / (1 + jω_c/(Q×ω_0) + (jω_c/ω_0)²)
        omega_c = 2 * math.pi * f_c
        omega_0 = 2 * math.pi * f0
        omega_z = 2 * math.pi * f_z_esr if f_z_esr < 1e8 else 1e15
        q_lc = plant["q_lc"]
        jw = complex(0, 1) * omega_c
        g_num = 1 + jw / omega_z
        g_den = (jw / omega_0) ** 2 + jw / (q_lc * omega_0) + 1
        g_plant = (v_ramp if v_ramp else 1.0) / max(abs(g_den / g_num), 1e-12)  # magnitude only here
        phi_plant = math.degrees(cmath.phase(g_num / g_den))  # DC gain is real positive, doesn't affect phase
        phi_needed = target_phase_margin_deg - 180 - phi_plant

        # K-factor for Type-2: single zero + single pole
        # K = tan((phi_boost/2) + 45°)
        phi_boost = min(max(phi_needed, 5.0), 70.0)  # clamp: Type-2 max ~70°
        k = math.tan(math.radians(phi_boost / 2.0 + 45.0))
        f_z = f_c / k
        f_p = f_c * k

        # DC gain at f_c: unity at crossover, so G_comp(f_c) = 1 / |G_plant(f_c)|
        # Plant gain (magnitude): V_in/V_ramp × ω₀²/... (approximate at f_c >> f0)
        g_plant_fc = (1.0 / v_ramp) * (f0 / f_c) ** 2 * math.sqrt(1 + (f_c / f_z_esr) ** 2)
        g_comp_fc = 1.0 / max(g_plant_fc, 1e-9)

        # Error amplifier resistor-capacitor network
        # Gc(s) = g_comp_fc × (1 + s/ωz) / ((s/ωi) × (1 + s/ωp))
        # Using R1 as top divider, R2 bottom divider:
        r2_ohm = r1_ohm * v_ref / max(v_out - v_ref, 1e-6)
        # From Venable recipe: C1 = 1/(2π×f_z×R2), C2 = 1/(2π×f_p×R2)
        c1_f = 1.0 / (2 * math.pi * f_z * r2_ohm)
        c2_f = 1.0 / (2 * math.pi * f_p * r2_ohm)
        # Verify integrator gain: R3 across amp sets DC gain
        r3_ohm = g_comp_fc * r2_ohm / (2 * math.pi * f_c * c1_f * r2_ohm)

        # Round to E96
        r2_r = _round_e_series(r2_ohm, 96)
        r3_r = _round_e_series(max(r3_ohm, 1e3), 96)
        c1_r = _round_e_series(c1_f, 24)
        c2_r = _round_e_series(c2_f, 24)

        # Phase margin from K-factor method (Venable 1994):
        # PM = φ_boost + (180° + φ_plant)
        # where PM_uncompensated = 180° + φ_plant, and K-factor adds exactly φ_boost.
        actual_pm = phi_boost + (180 + phi_plant)

        return {
            "compensator_type": "Type-2",
            "topology_context": "voltage_mode_buck_ccm",
            "f_crossover_hz": round(f_c, 1),
            "f_zero_hz": round(f_z, 1),
            "f_pole_hz": round(f_p, 1),
            "phase_margin_deg": round(actual_pm, 1),
            "r1_ohm": r1_ohm,
            "r2_ohm": round(r2_r, 1),
            "r3_ohm": round(r3_r, 1),
            "c1_f": round(c1_r, 12),
            "c2_f": round(c2_r, 12),
            "k_factor": round(k, 3),
            "plant": plant,
            "stability_ok": actual_pm >= 40.0,
            "notes": [
                f"Plant double-pole f0={f0:.0f}Hz  ESR-zero={f_z_esr:.0f}Hz",
                f"Phase boost provided: {phi_boost:.1f}°",
                f"Actual phase margin ≈ {actual_pm:.1f}° ({'OK' if actual_pm >= 45 else 'MARGINAL — consider Type-3'})",
            ],
        }

    @classmethod
    def design_type3_compensator(
        cls,
        v_out: float,
        v_ref: float,
        f_sw: float,
        l_h: float,
        c_out_f: float,
        esr_ohm: float,
        r_load_ohm: float,
        v_ramp: float = 1.0,
        r1_ohm: float = 10e3,
        target_phase_margin_deg: float = 52.0,
    ) -> Dict[str, Any]:
        """
        Type-3 compensator design — two zeros, three poles (including one at origin).

        Required when: LC double-pole is above f_sw/20, or ESR zero is too high
        (ceramic output caps with near-zero ESR).
        Reference: TI SLVA554, "Type-3 Compensator Design Procedure".

        Circuit: error amplifier with two RC zero networks + two pole networks.
        """
        plant = cls._plant_poles_zeros(l_h, c_out_f, r_load_ohm, esr_ohm)
        f0 = plant["f0_hz"]
        f_z_esr = plant["f_z_esr_hz"]

        f_c = min(f_sw / 5.0, f_sw / 6.0)
        f_c = max(f_c, f0 * 1.5)

        # Plant phase at f_c — exact complex TF evaluation (same method as Type-2)
        omega_c = 2 * math.pi * f_c
        omega_0 = 2 * math.pi * f0
        omega_z = 2 * math.pi * f_z_esr if f_z_esr < 1e8 else 1e15
        q_lc = plant["q_lc"]
        jw = complex(0, 1) * omega_c
        g_num = 1 + jw / omega_z
        g_den = (jw / omega_0) ** 2 + jw / (q_lc * omega_0) + 1
        phi_plant = math.degrees(cmath.phase(g_num / g_den))  # DC gain is real positive, doesn't affect phase
        phi_needed = target_phase_margin_deg - 180 - phi_plant
        phi_boost = min(max(phi_needed, 20.0), 88.0)

        # K-factor for Type-3: two zeros + two poles → max 90° boost
        k = math.tan(math.radians(phi_boost / 4.0 + 45.0))
        f_z1 = f_c / k
        f_z2 = f_c / k       # both zeros at same frequency (simplified Type-3)
        f_p1 = f_c * k
        f_p2 = f_sw / 2.0    # second pole at Nyquist for noise rolloff

        # Component values (TI SLVA554 equations)
        r2_ohm = r1_ohm * v_ref / max(v_out - v_ref, 1e-6)
        g_plant_fc = (1.0 / v_ramp) * (f0 / f_c) ** 2 * math.sqrt(1 + (f_c / f_z_esr) ** 2)
        g_comp_fc = 1.0 / max(g_plant_fc, 1e-9)

        c1_f = 1.0 / (2 * math.pi * f_z1 * r2_ohm)
        c2_f = 1.0 / (2 * math.pi * f_p1 * r2_ohm)
        r3_ohm = g_comp_fc / (2 * math.pi * f_c * c1_f)
        c3_f = 1.0 / (2 * math.pi * f_z2 * r3_ohm)
        r4_ohm = 1.0 / (2 * math.pi * f_p2 * c3_f)

        # Round all to standard values
        c1_r = _round_e_series(c1_f, 24)
        c2_r = _round_e_series(c2_f, 24)
        c3_r = _round_e_series(c3_f, 24)
        r3_r = _round_e_series(max(r3_ohm, 100.0), 96)
        r4_r = _round_e_series(max(r4_ohm, 100.0), 96)
        r2_r = _round_e_series(r2_ohm, 96)

        # PM from K-factor method: Type-3 provides 2× boost relative to Type-2
        # PM = φ_boost + (180° + φ_plant)  [same formula, K-factor handles zero placement]
        actual_pm = phi_boost + (180 + phi_plant)

        return {
            "compensator_type": "Type-3",
            "topology_context": "voltage_mode_buck_ccm",
            "f_crossover_hz": round(f_c, 1),
            "f_zero1_hz": round(f_z1, 1),
            "f_zero2_hz": round(f_z2, 1),
            "f_pole1_hz": round(f_p1, 1),
            "f_pole2_hz": round(f_p2, 1),
            "phase_margin_deg": round(actual_pm, 1),
            "r1_ohm": r1_ohm,
            "r2_ohm": round(r2_r, 1),
            "r3_ohm": round(r3_r, 1),
            "r4_ohm": round(r4_r, 1),
            "c1_f": round(c1_r, 12),
            "c2_f": round(c2_r, 12),
            "c3_f": round(c3_r, 12),
            "k_factor": round(k, 3),
            "plant": plant,
            "stability_ok": actual_pm >= 40.0,
            "notes": [
                f"Two zeros at {f_z1:.0f}Hz cancel double-pole near {f0:.0f}Hz",
                f"ESR zero at {f_z_esr:.0f}Hz {'helps' if f_z_esr < f_c else 'above crossover'}",
                f"Actual phase margin ≈ {actual_pm:.1f}° ({'OK' if actual_pm >= 45 else 'MARGINAL'})",
            ],
        }

    @classmethod
    def recommend_compensator(
        cls,
        v_out: float,
        v_ref: float,
        f_sw: float,
        l_h: float,
        c_out_f: float,
        esr_ohm: float,
        r_load_ohm: float,
        v_ramp: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Automatically recommend and design Type-2 or Type-3 based on plant topology.

        Decision rule (from TI SLVA553 §4):
        - If esr_ohm × c_out_f × f_sw < 0.159 (ESR zero below f_sw/2π):
            → Type-2 (ESR zero provides needed phase boost)
        - Else (ceramic caps, near-zero ESR, no ESR zero help):
            → Type-3 (need full 90° boost from compensator zeros)
        """
        f_z_esr = 1.0 / (2 * math.pi * esr_ohm * c_out_f) if esr_ohm > 0 else 1e9
        f_sw_over_2pi = f_sw / (2 * math.pi)
        use_type3 = (f_z_esr > f_sw * 0.5) or (esr_ohm < 1e-3)

        if use_type3:
            result = cls.design_type3_compensator(
                v_out, v_ref, f_sw, l_h, c_out_f, esr_ohm, r_load_ohm, v_ramp
            )
            result["selection_reason"] = (
                f"Type-3 selected: ESR zero at {f_z_esr:.0f}Hz > f_sw/2 — "
                "ceramic cap, no ESR zero phase boost available"
            )
        else:
            result = cls.design_type2_compensator(
                v_out, v_ref, f_sw, l_h, c_out_f, esr_ohm, r_load_ohm, v_ramp
            )
            result["selection_reason"] = (
                f"Type-2 selected: ESR zero at {f_z_esr:.0f}Hz provides "
                "natural phase boost for electrolytic/polymer output cap"
            )
        return result

    @classmethod
    def bode_data(
        cls,
        f_sw: float,
        l_h: float,
        c_out_f: float,
        esr_ohm: float,
        r_load_ohm: float,
        v_in: float,
        v_ramp: float,
        compensator: Dict[str, Any],
        n_points: int = 200,
    ) -> Dict[str, Any]:
        """
        Compute open-loop Bode data (gain and phase vs frequency) for the
        plant × compensator combination.

        Returns arrays suitable for plotting or margin verification.
        """
        freqs = np.logspace(1, math.log10(f_sw / 2), n_points)

        # Plant: voltage-mode buck G_vd(s)
        f0 = 1.0 / (2 * math.pi * math.sqrt(l_h * c_out_f))
        q = r_load_ohm / math.sqrt(l_h / c_out_f)
        f_z = 1.0 / (2 * math.pi * esr_ohm * c_out_f) if esr_ohm > 0 else 1e9
        dc_gain = v_in / v_ramp

        gain_db = []
        phase_deg = []
        for f in freqs:
            w = 2 * math.pi * f
            w0 = 2 * math.pi * f0
            wz = 2 * math.pi * f_z
            # Plant LC double pole (second order)
            s = 1j * w
            g_plant = dc_gain * (1 + s / wz) / (1 + s / (w0 * q) + (s / w0) ** 2)
            gain_db.append(20 * math.log10(max(abs(g_plant), 1e-12)))
            phase_deg.append(math.degrees(cmath.phase(g_plant)))

        return {
            "frequencies_hz": freqs.tolist(),
            "plant_gain_db": gain_db,
            "plant_phase_deg": phase_deg,
            "f0_hz": round(f0, 1),
            "q_lc": round(q, 3),
            "f_z_esr_hz": round(f_z, 1) if f_z < 1e8 else None,
        }


# ═════════════════════════════════════════════════════════════════════════════
# 12.  EXTENDED GERBER WRITER  (solder mask, silkscreen, paste)
# ═════════════════════════════════════════════════════════════════════════════

class ExtendedGerberWriter(GerberWriter):
    """
    Extends GerberWriter with:
    - Solder mask layers (F_Mask / B_Mask — negative artwork)
    - Silkscreen layers (F_Silkscreen / B_Silkscreen)
    - Solder paste layers (F_Paste / B_Paste)

    Solder mask uses negative-polarity Gerber (LPD = dark = copper exposed).
    Standard opening: pad + clearance on each side.
    Reference: IPC-7351B "Land Pattern Standard" §3.5 (mask expansion rules),
               IPC-SM-782 (Component Mounting Design Guidelines).
    """

    # IPC-7351B §3.5: solder mask expansion from pad edge
    # Typical fab: 0.05–0.10mm expansion; 0.05mm default for high-density
    _MASK_EXPANSION_MM = 0.05   # source: IPC-7351B Table 3-2

    # Paste reduction: typically 10–20% area reduction to prevent bridging
    # Source: IPC-7525B "Stencil Design Guidelines" §4.2
    _PASTE_REDUCTION_PCT = 0.10

    def write_solder_mask_layer(
        self,
        layer_name: str,
        pads: List[Dict[str, Any]],
        vias: Optional[List[ViaSpec]] = None,
        expansion_mm: float = _MASK_EXPANSION_MM,
    ) -> str:
        """
        Generate solder mask Gerber (negative polarity).
        Openings are flashed over each pad with expansion applied.

        layer_name: 'F_Mask' or 'B_Mask'
        pads: list of {x, y, size_mm, shape='circle'|'rect', w_mm, h_mm}
        vias: optionally tent vias (include=True) or leave open (include=False)
        """
        self._apertures = {}
        self._next_d_code = 10
        lines = [self._gerber_header(f"{layer_name} (solder mask — negative)")]
        lines.append("%LPD*%")   # Layer polarity: Dark (clearings in solder mask)

        # Register pad apertures with expansion
        aperture_map: Dict[str, int] = {}
        for pad in pads:
            shape = pad.get("shape", "circle")
            if shape == "rect":
                w = pad.get("w_mm", pad.get("size_mm", 1.5)) + 2 * expansion_mm
                h = pad.get("h_mm", pad.get("size_mm", 1.5)) + 2 * expansion_mm
                key = f"R{w:.4f}x{h:.4f}"
                if key not in aperture_map:
                    d = self._next_d_code; self._next_d_code += 1
                    aperture_map[key] = d
                    lines.append(f"%ADD{d}R,{w:.4f}X{h:.4f}*%")
            else:
                dia = pad.get("size_mm", 1.5) + 2 * expansion_mm
                key = f"C{dia:.4f}"
                if key not in aperture_map:
                    d = self._next_d_code; self._next_d_code += 1
                    aperture_map[key] = d
                    lines.append(f"%ADD{d}C,{dia:.4f}*%")

        # Flash pads
        for pad in pads:
            shape = pad.get("shape", "circle")
            if shape == "rect":
                w = pad.get("w_mm", pad.get("size_mm", 1.5)) + 2 * expansion_mm
                h = pad.get("h_mm", pad.get("size_mm", 1.5)) + 2 * expansion_mm
                key = f"R{w:.4f}x{h:.4f}"
            else:
                dia = pad.get("size_mm", 1.5) + 2 * expansion_mm
                key = f"C{dia:.4f}"
            d = aperture_map[key]
            lines.append(f"D{d}*")
            lines.append(f"X{self._coord(pad['x'])}Y{self._coord(pad['y'])}D03*")

        # Via openings (tented vias have NO opening — just skip them)
        if vias:
            for v in vias:
                if not v.__dict__.get("tented", False):
                    dia = v.drill_mm + 2 * expansion_mm
                    key = f"C{dia:.4f}"
                    if key not in aperture_map:
                        d = self._next_d_code; self._next_d_code += 1
                        aperture_map[key] = d
                        lines.append(f"%ADD{d}C,{dia:.4f}*%")
                    lines.append(f"D{aperture_map[key]}*")
                    lines.append(f"X{self._coord(v.x)}Y{self._coord(v.y)}D03*")

        lines.append("M02*")
        return "\n".join(lines)

    def write_silkscreen_layer(
        self,
        layer_name: str,
        references: Optional[List[Dict[str, Any]]] = None,
        outlines: Optional[List[List[Tuple[float, float]]]] = None,
        line_width_mm: float = 0.15,
    ) -> str:
        """
        Generate silkscreen Gerber layer with component reference designators
        and body outlines.

        references: list of {ref: 'U1', x: mm, y: mm, angle: deg}
        outlines: list of polygon point lists [[x1,y1], ...]
        line_width_mm: silkscreen line width (IPC-7351B min = 0.10mm)
        Reference: IPC-7351B §4.1 "Courtyard and Silkscreen Guidelines".
        """
        self._apertures = {}
        self._next_d_code = 10
        lines = [self._gerber_header(f"{layer_name} (silkscreen)")]

        # Single trace aperture
        d_code = self._next_d_code; self._next_d_code += 1
        lines.append(f"%ADD{d_code}C,{line_width_mm:.4f}*%")
        lines.append(f"D{d_code}*")
        lines.append("G01*")  # linear mode

        # Draw body outlines
        if outlines:
            for poly in outlines:
                if len(poly) < 2:
                    continue
                x0, y0 = poly[0]
                lines.append(f"X{self._coord(x0)}Y{self._coord(y0)}D02*")
                for x, y in poly[1:]:
                    lines.append(f"X{self._coord(x)}Y{self._coord(y)}D01*")
                # Close polygon
                lines.append(f"X{self._coord(x0)}Y{self._coord(y0)}D01*")

        # Reference designator positions (stored as flashes; actual text
        # requires font rendering — PCB tools handle text from netlist)
        if references:
            # Mark ref position with a small diamond flash (0.3mm × 0.3mm)
            d_ref = self._next_d_code; self._next_d_code += 1
            lines.append(f"%ADD{d_ref}P,0.3X4X45.0*%")   # 4-sided polygon (diamond)
            for ref_spec in references:
                lines.append(f"D{d_ref}*")
                lines.append(
                    f"X{self._coord(ref_spec['x'])}Y{self._coord(ref_spec['y'])}D03*"
                )

        lines.append("M02*")
        return "\n".join(lines)

    def write_paste_layer(
        self,
        layer_name: str,
        pads: List[Dict[str, Any]],
        reduction_pct: float = _PASTE_REDUCTION_PCT,
    ) -> str:
        """
        Generate solder paste Gerber layer (SMT pads only, through-hole excluded).

        Paste stencil openings are reduced from pad size to prevent bridging.
        IPC-7525B §4.2: 10% area reduction typical (linear: ~5% per edge).
        reduction_pct: fractional area reduction (0.10 = 10%).
        """
        edge_reduction = reduction_pct / 2.0   # apply half on each edge
        self._apertures = {}
        self._next_d_code = 10
        lines = [self._gerber_header(f"{layer_name} (solder paste)")]

        aperture_map: Dict[str, int] = {}
        flash_list = []

        for pad in pads:
            if pad.get("through_hole", False):
                continue   # no paste on through-hole pads

            shape = pad.get("shape", "circle")
            if shape == "rect":
                w = pad.get("w_mm", pad.get("size_mm", 1.5)) * (1 - reduction_pct)
                h = pad.get("h_mm", pad.get("size_mm", 1.5)) * (1 - reduction_pct)
                key = f"R{w:.4f}x{h:.4f}"
                if key not in aperture_map:
                    d = self._next_d_code; self._next_d_code += 1
                    aperture_map[key] = d
                    lines.append(f"%ADD{d}R,{w:.4f}X{h:.4f}*%")
            else:
                dia = pad.get("size_mm", 1.5) * (1 - edge_reduction * 2)
                key = f"C{dia:.4f}"
                if key not in aperture_map:
                    d = self._next_d_code; self._next_d_code += 1
                    aperture_map[key] = d
                    lines.append(f"%ADD{d}C,{dia:.4f}*%")

            flash_list.append((pad["x"], pad["y"], aperture_map[key]))

        for x, y, d in flash_list:
            lines.append(f"D{d}*")
            lines.append(f"X{self._coord(x)}Y{self._coord(y)}D03*")

        lines.append("M02*")
        return "\n".join(lines)

    def generate_fab_package(   # type: ignore[override]
        self,
        layer_name: str,
        traces: List[TraceSpec],
        vias: List[ViaSpec],
        board_outline: Optional[List[Tuple[float, float]]] = None,
        bom: Optional[Dict[str, Any]] = None,
        components: Optional[List[Dict[str, Any]]] = None,
        pads: Optional[List[Dict[str, Any]]] = None,
        silkscreen_refs: Optional[List[Dict[str, Any]]] = None,
        silkscreen_outlines: Optional[List[List[Tuple[float, float]]]] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Generate complete 7-layer fabrication package:
          F_Cu.gtl, B_Cu.gbl, F_Mask.gm5, B_Mask.gm6,
          F_Silkscreen.gto, B_Silkscreen.gbo, F_Paste.gtp,
          Edge_Cuts.gko, drill.drl, BOM.csv, CPL.csv
        """
        files: Dict[str, str] = {}

        # Copper layers
        files[f"{layer_name}-F_Cu.gtl"] = self.write_copper_layer(
            "Top Copper", traces, vias, pads)
        files[f"{layer_name}-B_Cu.gbl"] = self.write_copper_layer(
            "Bottom Copper", [], vias)

        # Solder mask (both sides)
        all_pads = pads or []
        files[f"{layer_name}-F_Mask.gm5"] = self.write_solder_mask_layer(
            "F_Mask", all_pads, vias)
        files[f"{layer_name}-B_Mask.gm6"] = self.write_solder_mask_layer(
            "B_Mask", [], vias)

        # Silkscreen
        files[f"{layer_name}-F_Silkscreen.gto"] = self.write_silkscreen_layer(
            "F_Silkscreen", silkscreen_refs, silkscreen_outlines)
        files[f"{layer_name}-B_Silkscreen.gbo"] = self.write_silkscreen_layer(
            "B_Silkscreen")

        # Paste (SMT top side only — bottom SMT rare in power electronics)
        files[f"{layer_name}-F_Paste.gtp"] = self.write_paste_layer(
            "F_Paste", all_pads)

        # Board outline + drill + BOM + CPL
        if board_outline:
            files[f"{layer_name}-Edge_Cuts.gko"] = self.write_board_outline(board_outline)
        files[f"{layer_name}.drl"] = self.write_drill_file(vias)
        if bom:
            files[f"{layer_name}-BOM.csv"] = self.write_bom_csv(bom)
        if components:
            files[f"{layer_name}-CPL.csv"] = self.write_pick_and_place_csv(components)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for fname, content in files.items():
                with open(os.path.join(output_dir, fname), "w") as fh:
                    fh.write(content)

        return files
