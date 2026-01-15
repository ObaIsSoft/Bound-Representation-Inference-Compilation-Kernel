
import math
from typing import List, Dict, Any
from .base_ingestor import BaseIngestor

class ParametricGenerator(BaseIngestor):
    """
    Generates 'Standard Parts' using algorithmic scaling rules.
    Acts as a 'virtual manufacturer' to populate the catalog with realistic baseline components.
    """
    
    def __init__(self):
        super().__init__()
        self.source_name = "parametric_std_lib"

    def fetch_candidates(self, query: str = None) -> List[Dict[str, Any]]:
        candidates = []
        candidates.extend(self._generate_fasteners())
        candidates.extend(self._generate_brushless_motors())
        candidates.extend(self._generate_batteries())
        candidates.extend(self._generate_internal_combustion_engines())
        candidates.extend(self._generate_hydraulic_pumps())
        candidates.extend(self._generate_complex_components()) # Added line
        return candidates

    def _generate_fasteners(self) -> List[Dict[str, Any]]:
        """Generate ISO Metric Screws (M2-M20)."""
        fasteners = []
        # ISO 4762 (Socket Head Cap Screws) standards (approx)
        sizes = [
            {"m": 2, "pitch": 0.4, "head_d": 3.8, "head_k": 2.0},
            {"m": 3, "pitch": 0.5, "head_d": 5.5, "head_k": 3.0},
            {"m": 4, "pitch": 0.7, "head_d": 7.0, "head_k": 4.0},
            {"m": 5, "pitch": 0.8, "head_d": 8.5, "head_k": 5.0},
            {"m": 6, "pitch": 1.0, "head_d": 10.0, "head_k": 6.0},
            {"m": 8, "pitch": 1.25, "head_d": 13.0, "head_k": 8.0},
            {"m": 10, "pitch": 1.5, "head_d": 16.0, "head_k": 10.0},
            {"m": 12, "pitch": 1.75, "head_d": 18.0, "head_k": 12.0}
        ]
        
        lengths = [4, 6, 8, 10, 12, 16, 20, 25, 30, 40, 50, 60, 80, 100]
        
        for s in sizes:
            for l in lengths:
                if l < s["m"] * 2: continue # Skip too short
                
                name = f"M{s['m']}x{l} Socket Head Cap Screw"
                # Steel Density ~ 7.85 g/cc
                vol_shank = math.pi * ((s["m"]/2)**2) * l
                vol_head = math.pi * ((s["head_d"]/2)**2) * s["head_k"]
                mass = (vol_shank + vol_head) * 7.85 / 1000.0
                
                item = {
                    "id": f"std_iso4762_m{s['m']}_{l}",
                    "category": "fastener",
                    "name": name,
                    "mass_g": {"nominal": round(mass, 2), "sigma": 0.05, "distribution": "normal"},
                    "cost_usd": {"nominal": 0.10 + (s['m']*l*0.005)},
                    "specs": {
                        "thread": f"M{s['m']}",
                        "length_mm": l,
                        "pitch": s["pitch"],
                        "head_type": "socket_cap",
                        "material": "alloy_steel_12.9"
                    },
                    "geometry_def": {
                        "generator": "fastener_iso4762",
                        "params": {"m": s['m'], "l": l}
                    }
                }
                fasteners.append(item)
        return fasteners

    def _generate_brushless_motors(self) -> List[Dict[str, Any]]:
        # ... (Existing Motor Logic) ...
        # (Re-including strictly to keep context correct since I'm replacing fetch_candidates)
        # To avoid deleting the motor code, I should have used smaller chunks, but I need to update fetch_candidates too.
        # I will just invoke the existing method if I can, but I'm rewriting the class methods.
        # Actually, I can leave the existing methods alone if I target lines appropriately.
        # But 'fetch_candidates' is at the top.
        # I'll rewrite 'fetch_candidates' and ADD the new methods.
        pass # Placeholder here, seeing as I shouldn't delete existing logic.
    
    # ... Wait, I cannot use 'pass' in replacement content if I want to keep existing code.
    # I will target ONLY fetch_candidates for the first edit, then append methods.
    
    # RE-PLANNING TOOL CALL:
    # I'll do this in 2 steps or use a huge block.
    # The file has fetch_candidates at line 14.
    # I'll replace fetch_candidates first.
    pass


    def _generate_brushless_motors(self) -> List[Dict[str, Any]]:
        """Generate Brushless Motors based on Stator Size."""
        motors = []
        # Common Stator Sizes (2205, 2207, 2306, 2806, 3508, 4010, 5010, 6010, 8010)
        stators = [
            {"w": 22, "h": 5, "kv_range": [2300, 2600]},
            {"w": 23, "h": 6, "kv_range": [1700, 2400]},
            {"w": 28, "h": 6, "kv_range": [1300, 1700]},
            {"w": 35, "h": 8, "kv_range": [700, 900]},
            {"w": 40, "h": 10, "kv_range": [400, 600]},
            {"w": 50, "h": 10, "kv_range": [300, 400]},
            {"w": 80, "h": 10, "kv_range": [100, 180]}
        ]
        
        for s in stators:
            # Generate Low/High Kv variants
            kvs = [s["kv_range"][0], sum(s["kv_range"])//2, s["kv_range"][1]]
            
            # Physics Scaling Laws
            # Volume ~ w^2 * h
            # Mass ~ Volume
            vol_factor = (s["w"]**2 * s["h"])
            # 2205 (vol ~ 2400) is approx 28g
            mass = 28.0 * (vol_factor / (22.0**2 * 5.0))
            
            # Power ~ Volume (Roughly)
            # 2205 ~ 300W
            power = 300.0 * (vol_factor / (2420.0))
            
            for kv in kvs:
                name = f"Brushless Motor {s['w']}{s['h']:02d} {kv}Kv"
                
                # Resistance inversely proportional to Kv*Volume
                resistance = 0.05 * (2300/kv) * (2420/vol_factor)
                
                item = {
                    "id": f"std_motor_{s['w']}{s['h']}_{kv}",
                    "category": "motor",
                    "name": name,
                    "mass_g": {"nominal": round(mass, 1), "sigma": mass*0.02}, # 2% variance
                    "cost_usd": {"nominal": 15.0 + (mass*0.1)},
                    "specs": {
                        "kv": kv,
                        "stator_width": s["w"],
                        "stator_height": s["h"],
                        "resistance_ohm": round(resistance, 4),
                        "max_power_w": round(power, 0),
                        "shaft_diameter_mm": 5.0 if s["w"] > 28 else 3.0
                    },
                    "geometry_def": {
                        "generator": "motor_outrunner",
                        "params": {"stator_w": s['w'], "stator_h": s['h']}
                    }
                }
                motors.append(item)
        return motors

    def _generate_batteries(self) -> List[Dict[str, Any]]:
        """Generate LiPo Packs (1S-6S)."""
        batteries = []
        cells = [1, 2, 3, 4, 6]
        capacities = [500, 800, 1300, 1500, 2200, 5000, 10000]
        
        for s in cells:
            for cap in capacities:
                if s > 4 and cap < 1000: continue # Skip small HV packs
                
                name = f"{s}S {cap}mAh LiPo"
                
                # Physics: Energy Density ~ 150 Wh/kg
                voltage = s * 3.7
                energy_wh = voltage * (cap / 1000.0)
                mass = (energy_wh / 150.0) * 1000.0 # g
                
                # Add casing weight
                mass *= 1.15
                
                item = {
                    "id": f"std_lipo_{s}s_{cap}",
                    "category": "battery",
                    "name": name,
                    "mass_g": {"nominal": round(mass, 1), "sigma": mass*0.05},
                    "cost_usd": {"nominal": 5.0 + (energy_wh * 0.5)},
                    "specs": {
                        "chemistry": "lipo",
                        "cells": s,
                        "capacity_mah": cap,
                        "voltage_nominal": voltage,
                        "voltage_max": s * 4.2,
                        "voltage_min": s * 3.3,
                        "c_rating": 50
                    },
                    "geometry_def": {
                        "generator": "battery_pack_prismatic",
                        "params": {"cells": s, "capacity": cap}
                    }
                }
                batteries.append(item)
        return batteries

    def _generate_internal_combustion_engines(self) -> List[Dict[str, Any]]:
        """Generate ICE Engines to demonstrate complex behavior modeling."""
        engines = []
        # Displacements in cc
        displacements = [25, 50, 100, 250, 500, 2000, 5000]
        
        for cc in displacements:
            name = f"{cc}cc 4-Stroke Engine"
            
            # Physics scaling (very rough)
            power_hp = cc * 0.1 # 100hp/liter approx
            mass_kg = cc * 0.002 + 5.0 # Base weight + scaling
            if cc > 1000: mass_kg = cc * 0.0015 + 50
            
            # Reliability Curve
            mtbf = 5000 if cc > 500 else 1000 # Larger engines run slower/longer
            
            item = {
                "id": f"std_ice_{cc}cc",
                "category": "internal_combustion_engine",
                "name": name,
                "mass_g": {"nominal": mass_kg * 1000, "sigma": mass_kg*50, "distribution": "normal"},
                "cost_usd": {"nominal": 200 + (cc * 2)},
                "specs": {
                    "displacement_cc": cc,
                    "cycle": "4-stroke",
                    "max_power_hp": round(power_hp, 1),
                    "fuel_type": "gasoline"
                },
                "behavior_model": {
                    "type": "mechanical_dynamic",
                    "performance_curves": {
                        "torque_rpm": [[1000, power_hp*0.2], [3000, power_hp*0.4], [6000, power_hp*0.3]]
                    },
                    "reliability": {
                        "mtbf_hours": mtbf,
                        "failure_modes": [
                           {"name": "rod_knock", "probability": "weibull", "shape": 1.5, "scale": mtbf*2} 
                        ]
                    }
                },
                "geometry_def": {
                    "generator": "engine_ice_generic",
                    "params": {"displacement": cc}
                }
            }
            engines.append(item)
        return engines

    def _generate_hydraulic_pumps(self) -> List[Dict[str, Any]]:
        """Generate Hydraulic Pumps to demonstrate flow maps."""
        pumps = []
        # Flow rates in L/min
        flows = [1, 5, 10, 50, 100]
        
        for q in flows:
            name = f"Gear Pump {q} LPM"
            mass = 0.5 + (q * 0.1)
            
            item = {
                "id": f"std_hydro_pump_{q}lpm",
                "category": "hydraulic_pump",
                "name": name,
                "mass_g": {"nominal": mass * 1000, "sigma": 50},
                "cost_usd": {"nominal": 50 + (q * 5)},
                "specs": {
                    "max_flow_lpm": q,
                    "max_pressure_bar": 200,
                    "type": "gear"
                },
                "behavior_model": {
                    "type": "hydraulic_map",
                    "efficiency_map": {
                        "x_axis": "pressure_bar",
                        "y_axis": "flow_lpm",
                        "values": [[0.8, 0.7], [0.85, 0.75]]
                    }
                },
                "geometry_def": {
                    "generator": "pump_gear_external",
                    "params": {"displacement_cc_rev": q / 3.0} 
                }
            }
            pumps.append(item)
        return pumps
