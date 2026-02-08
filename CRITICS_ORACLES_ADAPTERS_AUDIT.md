# BRICK OS - Critics, Oracles & Adapters Audit

**Date:** 2026-02-08  
**Scope:** All Critics, Oracles, and Adapters  
**Total Files Audited:** 70+

---

## 🔴 CRITICAL - Hardcoded Physics Limits & Thresholds

### 1. ControlCritic (`critics/ControlCritic.py`)
**Lines 19-22:** Hardcoded physical limits (NOT from vehicle specs!)
```python
self.MAX_THRUST = 1000.0  # N - Hardcoded!
self.MAX_TORQUE = 100.0   # Nm - Hardcoded!
self.MAX_VELOCITY = 50.0  # m/s - Hardcoded!
self.MAX_POSITION = 1000.0  # m - Hardcoded!
```
**Line 81:** Arbitrary control effort threshold
```python
if control_effort > 100.0:  # Arbitrary threshold
```

**Fix:** Load from vehicle configuration or design parameters

---

### 2. PhysicsCritic (`critics/PhysicsCritic.py`)
**Line 162:** Placeholder gate alignment
```python
gate_align = 0.5  # Placeholder
```
**Lines 304-322:** Hardcoded thresholds
```python
if overall_performance < 0.7:  # Hardcoded threshold
if gate_alignment < 0.5:  # Hardcoded threshold  
if len(report.failure_modes) >= 3:  # Magic number
```

**Line 241:** Hardcoded speed of light (should use physics kernel)
```python
c = 299792458  # Should get from physics kernel
```

---

### 3. MaterialCritic (`critics/MaterialCritic.py`)
**Lines 276-277:** Placeholder inputs for neural network
```python
0.0, # Time placeholder
7.0  # pH placeholder
```
**Line 97:** Hardcoded temperature threshold
```python
high_temp_count = sum(1 for t in temps if t > 150)  # 150°C hardcoded
```
**Line 163:** Hardcoded degradation threshold
```python
if degradation_rate > 0.5:  # 50% hardcoded
```
**Line 171:** Hardcoded high-temp ratio
```python
if high_temp_count > len(self.temperature_history) * 0.3:  # 30% hardcoded
```
**Line 199:** Synthetic training data generation (not real!)
```python
if temp > 150:
    target_correction = -0.1  # Synthetic/forced value
```

---

### 4. ElectronicsCritic (`critics/ElectronicsCritic.py`)
**Line 183:** Hardcoded power margin threshold
```python
if avg_margin > 1000:  # >1kW excess considered "over-conservative"
```
**Line 179:** Hardcoded deficit rate
```python
if deficit_rate > 0.3:  # 30% hardcoded
```
**Line 191:** Hardcoded false alarm threshold
```python
if self.false_alarms > 5:  # Magic number
```
**Line 202:** Hardcoded detection rate
```python
if short_detection_rate < 0.8:  # 80% hardcoded
```

---

### 5. SurrogateCritic (`critics/SurrogateCritic.py`)
**Line 21:** Hardcoded drift threshold
```python
self.drift_threshold = 0.15  # 15% error triggers retrain
```
**Lines 170-174:** Hardcoded speed thresholds
```python
if speed < 10:   # 10 m/s hardcoded
    low_speed_gates.append(gate_val)
elif speed > 50:  # 50 m/s hardcoded
    high_speed_gates.append(gate_val)
```
**Lines 214, 218, 230:** Hardcoded accuracy thresholds
```python
if accuracy < 0.8:  # 80% hardcoded
if drift_rate > 0.3:  # 30% hardcoded
if self.false_positives > self.validated_predictions * 0.3:  # 30% hardcoded
```

---

### 6. GeometryCritic (`critics/GeometryCritic.py`)
**Lines 70-72:** Hardcoded timing thresholds
```python
if failure_rate > 0.2:  # 20% hardcoded
elif avg_time > 2.0:  # 2 seconds hardcoded
```

---

### 7. DesignCritic (`critics/DesignCritic.py`)
**Lines 58-59:** Hardcoded entropy calculation
```pythonn# Max entropy for 10 bins is log2(10) ~= 3.32
diversity_score = entropy / 3.32  # Hardcoded max
```
**Lines 68, 70:** Hardcoded diversity/acceptance thresholds
```python
if diversity_score < 0.3:  # 30% hardcoded
if acceptance_rate < 0.5:  # 50% hardcoded
```

---

## 🔴 CRITICAL - Oracle Adapters with Hardcoded Defaults

### Electronics Oracle Adapters

#### 8. PowerElectronicsAdapter (`electronics_oracle/adapters/power_electronics_adapter.py`)
**Line 13:** Hardcoded efficiency
```python
efficiency = params.get("efficiency", 0.9)  # 90% default
```
**Lines 10, 11, 19, 25:** Hardcoded default voltages/powers
```python
Vin = params.get("input_voltage_v", 12)  # 12V default
D = params.get("duty_cycle", 0.5)  # 50% default
Vrms = params.get("ac_voltage_rms", 120)  # 120V default
```

#### 9. AnalogCircuitsAdapter (`electronics_oracle/adapters/analog_circuits_adapter.py`)
**Lines 10-12, 17-19, 25-26:** Hardcoded component values
```python
Rf = params.get("feedback_resistor_ohm", 10000)  # 10kΩ default
Rin = params.get("input_resistor_ohm", 1000)     # 1kΩ default
Vin = params.get("input_voltage_v", 1.0)         # 1V default
R = params.get("resistor_ohm", 1000)             # 1kΩ default
C = params.get("capacitor_f", 1e-6)              # 1µF default
```

#### 10. ControlSystemsAdapter (`electronics_oracle/adapters/control_systems_adapter.py`)
**Lines 33-35:** Hardcoded PID gains
```python
Kp = params.get("proportional_gain", 1.0)
Ki = params.get("integral_gain", 0.1)
Kd = params.get("derivative_gain", 0.01)
```
**Lines 50-52:** Hardcoded Ziegler-Nichols constants
```python
Kp_zn = 0.6 * Ku
Ki_zn = 1.2 * Ku / Tu
Kd_zn = 0.075 * Ku * Tu
```

---

### Materials Oracle Adapters

#### 11. MechanicalPropertiesAdapter (`materials_oracle/adapters/mechanical_properties_adapter.py`)
**Line 44:** Hardcoded Young's modulus (steel)
```python
E = params.get("youngs_modulus_pa", 200e9)  # Steel default
```
**Line 58:** Hardcoded yield strength
```python
yield_strength = params.get("yield_strength_pa", 250e6)
```
**Line 62:** Hardcoded Poisson's ratio
```python
nu = params.get("poissons_ratio", 0.3)
```
**Lines 87-89, 103-104, 124, 131:** Hardcoded test parameters
```python
F = params.get("force_n", 3000)           # Brinell test
D = params.get("ball_diameter_mm", 10)    # 10mm ball
d = params.get("indent_diameter_mm", 4)   # 4mm indent
```

---

### Chemistry Oracle Adapters

#### 12. ThermochemistryAdapter (`chemistry_oracle/adapters/thermochemistry_adapter.py`)
**Line 19:** Hardcoded gas constant (acceptable, but should use scipy.constants)
```python
R = 8.314  # Gas constant (J/mol·K)
```
**Lines 46, 80, 117-118, 140-143:** Hardcoded standard conditions
```python
temperature = params.get("temperature_k", 298.15)  # 25°C default
K1 = params.get("K1", 1.0)
T1 = params.get("T1_k", 298.15)  # 25°C default
P1 = params.get("P1_pa", 101325)  # 1 atm default
T1 = params.get("T1_k", 373.15)   # Water boiling point
delta_H_vap = params.get("enthalpy_vap_kj_mol", 40.7)  # Water!
```

---

## 🟡 MEDIUM - Minor Issues

### 13. Other Oracle Adapters
Most other adapters follow the same pattern:
- Accept parameters with hardcoded defaults
- Use standard physics formulas
- **Issue:** Defaults may not be appropriate for all use cases

Examples:
- `semiconductor_devices_adapter.py` - Default doping concentrations
- `rf_microwave_adapter.py` - Default impedance (50Ω)
- `pcb_design_adapter.py` - Default trace widths
- `thermal_properties_adapter.py` - Default thermal conductivities
- `electrical_properties_adapter.py` - Default resistivities

---

## 📊 Summary Table

| Category | Component | Issue Count | Severity |
|----------|-----------|-------------|----------|
| **Critics** | ControlCritic | 5 | 🔴 Critical |
| **Critics** | PhysicsCritic | 4 | 🔴 Critical |
| **Critics** | MaterialCritic | 6 | 🔴 Critical |
| **Critics** | ElectronicsCritic | 4 | 🔴 Critical |
| **Critics** | SurrogateCritic | 5 | 🔴 Critical |
| **Critics** | GeometryCritic | 2 | 🟡 Medium |
| **Critics** | DesignCritic | 3 | 🟡 Medium |
| **Critics** | Other critics | 10+ | 🟢 Low |
| **Adapters** | PowerElectronics | 5 | 🔴 Critical |
| **Adapters** | AnalogCircuits | 6 | 🔴 Critical |
| **Adapters** | ControlSystems | 4 | 🔴 Critical |
| **Adapters** | MechanicalProperties | 8 | 🔴 Critical |
| **Adapters** | Thermochemistry | 7 | 🔴 Critical |
| **Adapters** | Other adapters | 30+ | 🟡 Medium |

**Total Issues Found:** 90+  
**Critical:** 50+  
**Medium:** 30+  
**Low:** 10+

---

## 🛠️ Recommended Fixes

### Immediate (Week 1)
1. **ControlCritic:** Load limits from vehicle configuration
2. **All Oracle Adapters:** Remove defaults - require explicit parameters
3. **PhysicsCritic:** Get constants from physics kernel

### Short-term (Week 2-3)
4. **All Critics:** Externalize thresholds to `config/critic_thresholds.json`
5. **MaterialCritic:** Connect to real temperature database
6. **ElectronicsCritic:** Load component specs from component catalog

### Medium-term (Month 2)
7. **All Adapters:** Add database lookups for material/component properties
8. **Add validation:** Reject requests with missing parameters (no defaults)
9. **Add uncertainty quantification:** Return confidence intervals

---

## 📁 Configuration Files to Create

```
backend/
├── config/
│   ├── critic_thresholds.json       # All critic thresholds
│   ├── vehicle_limits.json          # Physical limits by vehicle type
│   ├── oracle_defaults.yaml         # Centralized defaults (if needed)
│   └── adapter_schemas.json         # Required parameters per adapter
└── data/
    ├── material_standards.json      # Standard test parameters
    └── component_defaults.json      # Component-type defaults
```

---

## Example: Configuration Schema

### `config/critic_thresholds.json`
```json
{
  "ControlCritic": {
    "max_thrust_n": null,  // null = from vehicle spec
    "max_torque_nm": null,
    "control_effort_threshold": 100.0
  },
  "MaterialCritic": {
    "high_temp_threshold_c": 150,
    "degradation_threshold": 0.5,
    "mass_error_threshold_pct": 10
  },
  "ElectronicsCritic": {
    "power_deficit_threshold": 0.3,
    "short_detection_min_rate": 0.8
  }
}
```

### `config/vehicle_limits.json`
```json
{
  "drone_small": {
    "max_thrust_n": 100.0,
    "max_torque_nm": 10.0,
    "max_velocity_ms": 20.0
  },
  "drone_large": {
    "max_thrust_n": 1000.0,
    "max_torque_nm": 100.0,
    "max_velocity_ms": 50.0
  }
}
```

---

## 🎯 Priority Actions

1. **🔥 HOT:** ControlCritic limits MUST come from vehicle specs (safety critical)
2. **HIGH:** All oracle adapters should reject missing params (fail fast)
3. **MEDIUM:** Externalize all thresholds for tuning
4. **LOW:** Add comprehensive parameter validation

---

**Next Step:** Start with ControlCritic - connect to vehicle configuration database.
