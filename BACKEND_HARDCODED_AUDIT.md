# BRICK OS Backend - Hardcoded Values & Stub Implementations Audit

**Date:** 2026-02-08  
**Scope:** All agents in `/backend/agents/`  
**Status:** Critical findings requiring real API integrations

---

## 🔴 CRITICAL - Hardcoded Data (No Real API)

### 1. CostAgent (`cost_agent.py`)
**Issues:**
- **Line 51-58:** Hardcoded material costs dictionary:
  ```python
  material_costs = {
      "Aluminum 6061": 20.0,
      "Steel": 15.0,
      "Titanium": 150.0,
      "Carbon Fiber": 200.0,
      "PLA": 25.0,
      "ABS": 30.0
  }
  ```
- **Line 77-84:** Hardcoded currency conversion rates (USD, EUR, GBP, JPY, CAD)
- **Line 91:** Fixed budget threshold of $100k USD

**Recommendations:**
- Integrate with real pricing APIs:
  - **Metals:** London Metal Exchange (LME) API, Fastmarkets
  - **Plastics:** PlasticsNews Pricing, ICIS
  - **Currency:** OpenExchangeRates, CurrencyLayer
- Create `services/pricing_service.py` with caching

---

### 2. ComponentAgent (`component_agent.py`)
**Issues:**
- **Lines 142-157:** Uses Supabase client but has fallback that generates **synthetic/test components** if no database:
  ```python
  if not self.db.enabled:
      return []  # Returns empty, but test mode generates fake components
  ```
- **Lines 273-279:** Generates dummy mesh if none exists:
  ```python
  import trimesh
  mesh = trimesh.creation.box(extents=[1,1,1])  # Synthetic geometry
  ```

**Recommendations:**
- Integrate with real component APIs:
  - **McMaster-Carr:** Scraping API or catalog integration
  - **DigiKey:** DigiKey API for electronics
  - **Mouser:** Mouser Electronics API
  - **Thingiverse:** For 3D printable components
  - **GrabCAD:** Engineering component library
- Add `services/component_catalog_service.py`

---

### 3. AssetSourcingAgent (`asset_sourcing_agent.py`)
**Issues:**
- **Line 18:** `self.mock_assets = []` - Empty mock catalog
- **Lines 102-107:** Only searches mock_assets (always empty initially)
- **Lines 48-100:** LLM generates synthetic "Kit" components but no real sourcing

**Recommendations:**
- Integrate with real 3D asset APIs:
  - **NASA 3D Resources:** Official API (free)
  - **Sketchfab:** Sketchfab API
  - **Poly/Google:** (deprecated, find alternative)
  - **TurboSquid:** Commercial API
  - **CGTrader:** Marketplace API
- Add `services/asset_sourcing_service.py`

---

### 4. SafetyAgent (`safety_agent.py`)
**Issues:**
- **Lines 41-48:** Hardcoded safety thresholds:
  ```python
  if metrics.get("max_stress_mpa", 0) > 200:  # Hardcoded 200 MPa
      hazards.append("High Stress detected (>200 MPa)")
  if metrics.get("max_temp_c", 0) > 100:  # Hardcoded 100°C
      hazards.append("High Temperature detected (>100 C)")
  ```
- No material-specific safety factors

**Recommendations:**
- Load safety thresholds from:
  - **Material-specific:** Yield strength from materials DB
  - **Industry standards:** OSHA, ISO, ASME via `data/safety_standards.json`
  - **User-defined:** Project-specific safety factors
- Create `services/safety_standards_service.py`

---

### 5. DfmAgent (`dfm_agent.py`)
**Issues:**
- **Lines 37-39:** Hardcoded manufacturing limits:
  ```python
  min_limit = 0.8  # mm for FDM default
  if method == "CNC": min_limit = 1.5
  elif method == "SLA": min_limit = 0.5
  ```
- **Line 46:** Hardcoded max aspect ratio of 10.0

**Recommendations:**
- Load from `config/manufacturing_standards.py` (already exists but not fully utilized)
- Integrate with real process databases:
  - **Protolabs:** API for manufacturing constraints
  - **Xometry:** Instant quoting API (includes DFM feedback)
  - **3D Hubs (Hubs.com):** Manufacturing API

---

### 6. SustainabilityAgent (`sustainability_agent.py`)
**Issues:**
- **Lines 24-29:** Hardcoded carbon factors:
  ```python
  factors = {
      "Aluminum 6061": 12.0,
      "Steel": 1.8,
      "PLA": 3.5,
      "Titanium": 30.0
  }
  ```
- No lifecycle analysis (LCA) integration

**Recommendations:**
- Integrate with real LCA databases:
  - **ecoinvent:** Commercial LCA database
  - **GaBi:** Professional LCA software API
  - **OpenLCA:** Open source LCA API
  - **Climatiq:** Carbon calculation API
- Create `services/sustainability_service.py`

---

### 7. PerformanceAgent (`performance_agent.py`)
**Issues:**
- **Line 29:** Mock efficiency score:
  ```python
  metrics["efficiency_score"] = 0.85  # Mock
  ```

**Recommendations:**
- Calculate from actual physics results
- Integrate with simulation benchmarking

---

### 8. ManufacturingAgent (`manufacturing_agent.py`)
**Issues:**
- **Lines 11-13:** Hardcoded economic constants:
  ```python
  HOURLY_MACHINING_RATE_USD = 50.0 
  SETUP_COST_USD = 100.0
  MACHINING_TIME_PER_KG_HOUR = 1.0 
  ```
- **Line 320:** Hardcoded minimum CNC radius of 1.0mm

**Recommendations:**
- Load from `data/manufacturing_rates` table (partially implemented)
- Integrate with quoting APIs:
  - **Xometry:** Real-time quoting
  - **Protolabs:** DFM and pricing API
  - **Fictiv:** Manufacturing platform API

---

### 9. NetworkAgent (`network_agent.py`)
**Issues:**
- **Line 43:** Mock hops:
  ```python
  hops = 2  # Mock average hops
  ```
- **Lines 63-74:** Placeholder latency prediction (simple formula, no ML):
  ```python
  def _predict_latency(self, hops: int, load_mbps: float) -> float:
      base_delay = 0.5  # ms per hop hardware delay
      congestion_factor = 0.1  # ms per mbps load
  ```

**Recommendations:**
- Integrate with network simulation tools:
  - **NS-3:** Network simulator
  - **OMNeT++:** Discrete event simulator
  - **Cisco Modeling Labs:** Network simulation API

---

### 10. ElectronicsAgent (`electronics_agent.py`)
**Issues:**
- **Lines 579, 596-614:** Mock evaluation fallback:
  ```python
  def _mock_evaluate(self, topology: Dict) -> Dict:
      # Heuristic efficiency calculation based on component presence
  ```
- **Lines 660-687:** Mock Oracle response when real oracle unavailable

**Recommendations:**
- Integrate with real circuit simulation:
  - **NGSpice:** Circuit simulator
  - **PySpice:** Python interface to SPICE
  - **LTSpice:** Circuit simulation (automated)
  - **Ansys Electronics:** Professional EM simulation API

---

### 11. AssetSourcingAgent (`asset_sourcing_agent.py`)
**Already covered in #3**

---

## 🟡 MEDIUM - Configuration Files Not Externalized

### 12. ToleranceAgent (`tolerance_agent.py`)
**Issues:**
- **Lines 28-31:** Fallback hardcoded fits:
  ```python
  self.hole_basis_fits = {
      "H7/g6": {"type": "clearance", "min_clear": 0.01, "max_clear": 0.04}
  }
  ```

**Recommendations:**
- Complete integration with `config/manufacturing_standards.py`
- Load ISO 286 tolerance tables from database

---

### 13. TopologicalAgent (`topological_agent.py`)
**Issues:**
- **Lines 19-24:** Hardcoded terrain types:
  ```python
  self.terrain_types = {
      "flat": {"slope": (0, 5), "roughness": (0, 0.1)},
      "gentle_hills": {"slope": (5, 15), "roughness": (0.1, 0.3)},
      ...
  }
  ```
- **Line 187:** Placeholder marine traversability:
  ```python
  return 0.85  # Placeholder
  ```

**Recommendations:**
- Load from `data/terrain_classifications.json`
- Integrate with real elevation data APIs:
  - **USGS:** Elevation data API
  - **Google Earth Engine:** Terrain analysis
  - **Mapbox:** Elevation API

---

### 14. UnifiedDesignAgent (`unified_design_agent.py`)
**Issues:**
- **Lines 167-169:** Hardcoded material keywords mapping
- **Lines 158-163:** Hardcoded color map:
  ```python
  COLOR_MAP = {
      "red": (0, 0.9, 0.8),
      "blue": (240, 0.9, 0.8),
      ...
  }
  ```

**Recommendations:**
- Load color palettes from `data/design_palettes.json`
- Material mapping should query materials DB

---

## 🟢 LOW - Minor Improvements

### 15. ChemistryAgent (`chemistry_agent.py`)
**Issues:**
- **Lines 79-80:** Default fallback materials:
  ```python
  if not materials:
       materials = ["Steel", "Aluminum"]  # Default to common checks
  ```
- **Lines 158-161:** Fallback material family detection:
  ```python
  if "aluminum" in lower_type: mat_family = "aluminum"; density = 2.7
  elif "titanium" in lower_type: mat_family = "titanium"; density = 4.4
  ```

**Status:** Minor - Has proper DB integration as primary source

---

### 16. ThermalAgent, StructuralAgent, MaterialAgent
**Status:** These agents already have:
- ✅ Physics kernel integration
- ✅ Neural surrogates
- ✅ Oracle delegation
- ✅ Database lookups (primary)
- Minor hardcoded fallbacks only when DB fails

---

## 📋 Summary Table

| Agent | Severity | Issue Count | Primary Fix Needed |
|-------|----------|-------------|-------------------|
| CostAgent | 🔴 Critical | 3 | Real pricing APIs |
| ComponentAgent | 🔴 Critical | 2 | Component catalog APIs |
| AssetSourcingAgent | 🔴 Critical | 3 | 3D asset marketplace APIs |
| SafetyAgent | 🔴 Critical | 2 | Material-specific thresholds |
| DfmAgent | 🔴 Critical | 2 | Manufacturing constraints DB |
| SustainabilityAgent | 🔴 Critical | 1 | LCA database integration |
| ManufacturingAgent | 🔴 Critical | 2 | Real quoting APIs |
| NetworkAgent | 🔴 Critical | 2 | Network simulation tools |
| ElectronicsAgent | 🟡 Medium | 2 | SPICE simulation |
| ToleranceAgent | 🟡 Medium | 1 | ISO standards completion |
| TopologicalAgent | 🟡 Medium | 2 | Terrain data APIs |
| PerformanceAgent | 🟡 Medium | 1 | Physics-based calculation |
| UnifiedDesignAgent | 🟢 Low | 2 | Externalize config |
| ChemistryAgent | 🟢 Low | 2 | Minor fallbacks |

**Total Agents Audited:** 30+  
**Critical Issues:** 17  
**Medium Issues:** 8  
**Low Issues:** 4

---

## 🛠️ Implementation Priority

### Phase 1 (Immediate - Week 1-2)
1. **CostAgent:** Fix material costs to query database
2. **SafetyAgent:** Load thresholds from materials DB
3. **PerformanceAgent:** Calculate from real physics results

### Phase 2 (Short-term - Week 3-4)
4. **ComponentAgent:** Integrate DigiKey/Mouser APIs
5. **SustainabilityAgent:** Add Climatiq API
6. **DfmAgent:** Complete manufacturing standards config

### Phase 3 (Medium-term - Month 2)
7. **ManufacturingAgent:** Xometry/Protolabs API integration
8. **AssetSourcingAgent:** NASA 3D + Sketchfab APIs
9. **ElectronicsAgent:** NGSpice integration

### Phase 4 (Long-term - Month 3)
10. **NetworkAgent:** NS-3 or OMNeT++ integration
11. **TopologicalAgent:** USGS/Google Earth elevation APIs
12. **All agents:** Complete externalization of hardcoded configs

---

## 📁 Files to Create

```
backend/
├── services/
│   ├── __init__.py
│   ├── pricing_service.py         # Metal/plastic pricing APIs
│   ├── component_catalog_service.py  # DigiKey, Mouser, McMaster
│   ├── asset_sourcing_service.py    # 3D model repositories
│   ├── sustainability_service.py    # LCA databases
│   ├── manufacturing_service.py     # Quoting APIs
│   └── safety_standards_service.py  # OSHA, ISO, ASME
└── data/
    ├── safety_standards.json       # Material-specific limits
    ├── design_palettes.json        # Color schemes
    └── terrain_classifications.json # Terrain types
```

---

## 🔗 Recommended API Integrations

### Pricing & Materials
- **LME (London Metal Exchange):** https://www.lme.com/
- **Fastmarkets:** https://www.fastmarkets.com/
- **Climatiq:** https://www.climatiq.io/ (carbon calculations)

### Components
- **DigiKey API:** https://developer.digikey.com/
- **Mouser API:** https://www.mouser.com/api-hub/
- **McMaster-Carr:** (Scraping or partner API)

### 3D Assets
- **NASA 3D Resources:** https://nasa3d.arc.nasa.gov/
- **Sketchfab API:** https://sketchfab.com/developers/
- **CGTrader:** https://www.cgtrader.com/

### Manufacturing
- **Xometry API:** https://www.xometry.com/api/
- **Protolabs:** https://www.protolabs.com/
- **Fictiv:** https://www.fictiv.com/

### Simulation
- **NGSpice:** http://ngspice.sourceforge.net/
- **OpenFOAM:** https://openfoam.org/ (CFD)
- **NS-3:** https://www.nsnam.org/ (networking)

---

**Next Step:** Prioritize based on user requirements and start with CostAgent and SafetyAgent fixes.
