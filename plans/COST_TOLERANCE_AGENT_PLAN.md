# CostAgent & ToleranceAgent Implementation Plan

**Date:** 2026-02-26  
**Status:** Production-Ready Implementation  
**Research Basis:** 2019-2026 Modern Methods

---

## Phase 1: CostAgent Implementation

### 1.1 Architecture Overview

```
CostAgent
├── Core: Activity-Based Costing (ABC) - Classical foundation
├── ML Layer: XGBoost/Random Forest cost prediction - Modern (2022-2024)
├── Uncertainty: Monte Carlo simulation - Modern (2023)
├── Data Sources:
│   ├── Material prices (real-time APIs + fallback database)
│   ├── Manufacturing rates (database-driven)
│   ├── Regional labor rates (configurable)
│   └── Historical cost data (for ML training)
└── Output: Point estimate + confidence intervals
```

### 1.2 Components to Implement

#### A. Cost Models (Classical - Well Validated)

**Activity-Based Costing Formula:**
```
Total Cost = Material Cost + Manufacturing Cost + Overhead

Material Cost = Σ(mass_i × price_per_kg_i)

Manufacturing Cost = Setup Cost + (Cycle Time × Hourly Rate × Quantity)

Overhead = (Material + Manufacturing) × Overhead Rate
```

**Uncertainty Propagation:**
```
Cost Range = [μ - 2σ, μ + 2σ]  (95% confidence)

Where σ comes from:
- Material price volatility
- Manufacturing rate uncertainty
- Cycle time variation
```

#### B. ML Cost Prediction (Modern - 2022-2024 Research)

**Features for ML Model:**
- Geometry complexity metrics (surface area, volume, feature count)
- Material properties (hardness, machinability index)
- Process type (CNC, casting, 3D printing)
- Tolerance requirements (tight tolerances = higher cost)
- Surface finish requirements
- Quantity (economies of scale)

**Model Architecture:**
- XGBoost (gradient boosting) - Industry standard for tabular data
- Random Forest (ensemble) - Baseline comparison
- Uncertainty via quantile regression or Monte Carlo dropout

### 1.3 Implementation Requirements

**Dependencies:**
```python
# Core
numpy>=1.24.0
pandas>=2.0.0

# ML
xgboost>=2.0.0
scikit-learn>=1.3.0

# API
aiohttp>=3.8.0  # Async HTTP for pricing APIs
```

### 1.4 Production Features

**Required:**
- [x] Activity-based costing with full cost breakdown
- [x] Uncertainty quantification (Monte Carlo)
- [x] ML cost prediction with confidence intervals
- [x] Regional cost variations (labor rates)
- [x] Quantity break analysis (economies of scale)
- [x] Caching for performance
- [x] Validation against historical data
- [x] Error handling with graceful degradation

---

## Phase 2: ToleranceAgent Implementation

### 2.1 Architecture Overview

```
ToleranceAgent
├── Core: Statistical Tolerance Analysis
│   ├── RSS (Root Sum Square) - Standard method
│   ├── Monte Carlo simulation - Modern (2020)
│   └── Worst-case analysis - Conservative backup
├── ML Layer: Neural surrogate for fast stack-up (2023)
├── GD&T: ISO/ASME standard compliance
└── Output: Pass/fail probability + sensitivity analysis
```

### 2.2 Components to Implement

#### A. Tolerance Stack-Up Methods

**1. RSS (Root Sum Square):**
```
T_stack = √(Σ(T_i²))
Yield: ~99.73% (6σ) if normal distribution
```

**2. Monte Carlo Simulation:**
```
For n=10,000 simulations:
  1. Sample each dimension from distribution
  2. Calculate assembly dimension
  3. Check against spec

Pass rate = (n_pass / n_total) × 100%
```

**3. Worst-Case:**
```
T_stack = Σ|T_i|
Yield: 100% if within spec (over-design)
```

### 2.3 Production Features

**Required:**
- [x] RSS tolerance stack-up
- [x] Monte Carlo (10,000+ iterations)
- [x] Worst-case analysis
- [x] GD&T position with MMC
- [x] 1D/2D/3D stack-ups
- [x] Sensitivity analysis
- [x] Statistical distributions
- [x] ML surrogate for fast analysis

---

## Success Criteria

**CostAgent:**
- Estimates within ±15% of actual costs
- 95% confidence intervals validated
- ML model R² > 0.8
- API response < 500ms (cached)

**ToleranceAgent:**
- RSS matches Monte Carlo
- GD&T compliant with ISO/ASME
- Monte Carlo converges (CV < 1%)
- ML surrogate error < 2%

---

## No-Fallback Policy

**Hard Requirements:**
1. No `pass` statements in methods
2. No `return None` as success path
3. No silent error swallowing
4. No undocumented hardcoded constants
5. All methods have docstrings + type hints
6. All public methods have unit tests
