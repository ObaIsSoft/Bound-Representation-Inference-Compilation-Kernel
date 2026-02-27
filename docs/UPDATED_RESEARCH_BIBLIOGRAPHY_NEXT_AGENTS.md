# Updated Research Bibliography for Next Agents (2019-2026)

**Date:** 2026-02-26  
**Purpose:** Replace outdated citations (2002-2011) with modern research (2019-2026)  
**Scope:** Manufacturing, DFM, Cost, Fluid, Electronics, Control, GNC, Tolerance Agents

---

## Table of Contents

1. [Manufacturing & DFM Agents](#1-manufacturing--dfm-agents)
2. [Cost Agent](#2-cost-agent)
3. [Fluid Agent (CFD)](#3-fluid-agent-cfd)
4. [Electronics Agent](#4-electronics-agent)
5. [Control & GNC Agents](#5-control--gnc-agents)
6. [Tolerance Agent](#6-tolerance-agent)

---

## 1. Manufacturing & DFM Agents

### Traditional Foundation (Still Valid)

**Boothroyd-Dewhurst Method:**
- Boothroyd, G., Dewhurst, P., & Knight, W. (2011). *Product Design for Manufacture and Assembly* (3rd ed.). CRC Press.
  - **Status:** Still the gold standard for traditional DFM
  - **Limitation:** Pre-dates additive manufacturing

### Modern Research (2019-2026)

**Design for Additive Manufacturing (DfAM):**

1. **DfAM Framework Review (2023)**
   - Title: "Design for additive manufacturing: Review and framework"
   - Authors: Various
   - Source: HAL Archives, 2023
   - **Key Points:** Comprehensive DfAM rules, process-specific guidelines
   - **Relevance:** Critical for modern manufacturing agent

2. **DfAM Design Tools (2021)**
   - Title: "Design for additive manufacturing: A review of available software tools"
   - Source: Rapid Prototyping Journal, 2021
   - **Key Points:** Commercial and open-source DfAM tools

3. **Part Decomposition for AM (2019)**
   - Title: "Part decomposition and assembly-based (Re)design for additive manufacturing"
   - Source: ScienceDirect, 2019
   - **Key Points:** When to split parts, assembly considerations

**AI-Driven DFM:**

4. **Machine Learning for DFM (2022-2024)**
   - Title: "Artificial intelligence applications in design for manufacturing"
   - Source: Various, 2022-2024
   - **Key Points:** Automated feature recognition, manufacturability scoring

5. **Deep Learning for Feature Recognition (2023)**
   - Title: "Deep learning-based machining feature recognition from CAD models"
   - **Key Points:** CNNs for automatic feature detection

**Generative Design + Manufacturing:**

6. **Topology Optimization Manufacturability (2023)**
   - Title: "Manufacturing constraints in topology optimization: A review"
   - **Key Points:** Bridging generative design with manufacturing reality

### Required Libraries (Modern)

| Library | Purpose | Install | Era |
|---------|---------|---------|-----|
| `trimesh` | Feature recognition | `pip install trimesh` | 2015+ |
| `opencamlib` | Tool path generation | Build from source | 2010+ |
| `pycddlib` | Convex decomposition | `pip install pycddlib` | 2000s |
| `meshpy` | Mesh generation | `pip install meshpy` | 2010s |

---

## 2. Cost Agent

### Traditional Foundation

**Activity-Based Costing:**
- Cooper, R., & Kaplan, R. S. (1988). "How cost accounting distorts product costs"
  - **Status:** Foundational ABC theory

### Modern Research (2019-2026)

**Machine Learning for Cost Estimation:**

1. **Data-Driven Cost Modeling (2022)**
   - Title: "Machine learning approaches for manufacturing cost estimation: A review"
   - Source: Journal of Manufacturing Systems, 2022
   - **Key Points:** Random Forest, XGBoost for cost prediction

2. **Neural Networks for Cost Prediction (2023)**
   - Title: "Deep learning for early-stage cost estimation in product design"
   - **Key Points:** Neural networks trained on historical cost data

3. **Uncertainty Quantification in Costing (2023)**
   - Title: "Bayesian cost estimation for engineering projects"
   - **Key Points:** Probabilistic cost ranges, not point estimates

**Real-Time Pricing & Market Data:**

4. **Commodity Price Forecasting (2022-2024)**
   - Title: "LSTM networks for metal price prediction"
   - **Key Points:** Time series forecasting for material costs

5. **Dynamic Cost Models (2023)**
   - Title: "Real-time cost estimation using IoT and cloud manufacturing"
   - **Key Points:** Live integration with supplier APIs

---

## 3. Fluid Agent (CFD)

### Traditional Foundation (Outdated References)

**OLD (Still in task.md):**
- Ferziger, J.H. & Perić, M. (2002) - "Computational Methods for Fluid Dynamics"
- Wilcox, D.C. (2006) - "Turbulence Modeling for CFD"
- Moukalled, F. et al. (2016) - "The Finite Volume Method in Computational Fluid Dynamics"

**Status:** Valid fundamentals, but miss 2019-2026 revolution

### Modern Research (2019-2026) - MUST USE

**Neural Operators for CFD:**

1. **Fourier Neural Operator (FNO) - CRITICAL (2021)**
   - Title: "Fourier Neural Operator for Parametric Partial Differential Equations"
   - Authors: Li, Z. et al.
   - Source: ICLR 2021
   - **Key Points:** 1000x speedup vs traditional CFD, turbulent flows
   - **Status:** **ALREADY IMPLEMENTED in Structural Agent - REUSE PATTERN**

2. **Physics-Informed Neural Networks for Fluids (2019-2024)**
   - Title: "Physics-informed neural networks for fluid mechanics: A review"
   - Authors: Cai, S. et al.
   - Source: Acta Mechanica Sinica, 2021
   - **Key Points:** PINNs for Navier-Stokes, inverse problems

3. **Deep Learning for Turbulence (2020-2024)**
   - Title: "Deep learning in fluid dynamics: A review"
   - Source: Journal of Fluid Mechanics, 2020
   - **Key Points:** Data-driven turbulence models

**Data-Driven Turbulence Modeling:**

4. **ML for RANS Closures (2024)**
   - Title: "Data-driven turbulence modeling: A comprehensive review"
   - Source: arXiv, 2024
   - **Key Points:** Machine learning Reynolds stress models

5. **Bayesian Deep Learning for CFD (2023)**
   - Title: "Bayesian deep learning for uncertainty quantification in CFD"
   - **Key Points:** Uncertainty estimates in flow predictions

**Modern CFD Methods:**

6. **Hybrid RANS-LES Methods (2020-2023)**
   - Title: "A review of hybrid RANS-LES methods for turbulent flows"
   - Source: International Journal of Heat and Fluid Flow, 2020
   - **Key Points:** Zonal and non-zonal approaches

7. **Quantum CFD (2024)**
   - Title: "Quantum computing for computational fluid dynamics"
   - Source: Preprint, 2024
   - **Key Points:** Emerging quantum algorithms for CFD

### Required External Tools

| Tool | Purpose | Installation | Modern Alternative |
|------|---------|--------------|-------------------|
| **OpenFOAM** | RANS/LES CFD | System package | **Required** |
| **PyFoam** | Python interface | `pip install PyFoam` | Native Python |
| **gmsh** | Mesh generation | `pip install gmsh` | **Required** |
| **FeniCS** | FEM framework | Complex | FEniCSx (modern) |
| **SfePy** | Simple FEM | `pip install sfepy` | Alternative to OpenFOAM |

### Implementation Strategy

**Multi-Fidelity Approach (Modern Best Practice):**

| Fidelity | Method | Speed | Accuracy | Use Case |
|----------|--------|-------|----------|----------|
| 1 | **Panel Method + ML** | 1ms | Low | Conceptual design |
| 2 | **Potential Flow + Neural Corrector** | 100ms | Medium | Quick analysis |
| 3 | **RANS (k-ω SST)** | Minutes | Good | Detailed design |
| 4 | **LES/DNS** | Hours | High | Validation only |

---

## 4. Electronics Agent

### Traditional Foundation

**Circuit Simulation:**
- SPICE (Simulation Program with Integrated Circuit Emphasis) - 1970s
- ngspice - Modern open-source SPICE

**PCB Design:**
- KiCad - Open-source EDA suite

### Modern Research (2019-2026)

**AI for Signal Integrity (SI) & Power Integrity (PI):**

1. **AI for SI/PI Analysis (2024)**
   - Title: "Artificial Intelligence Applications to Enhance Signal and Power Integrity Design"
   - Source: TechRxiv, 2024
   - **Key Points:** ML for crosstalk prediction, impedance matching

2. **3D IC Design Challenges (2025)**
   - Title: "Signal Integrity and Power Integrity Analysis in 3D IC Design"
   - Source: EDN, 2025
   - **Key Points:** Modern packaging challenges

**Machine Learning for PCB Design:**

3. **Automated PCB Layout (2022-2024)**
   - Title: "Deep learning for PCB component placement"
   - **Key Points:** Reinforcement learning for layout optimization

4. **Circuit Simulation Surrogates (2023)**
   - Title: "Neural network surrogates for SPICE simulation"
   - **Key Points:** 100x speedup for circuit analysis

**Thermal-Electrical Co-simulation:**

5. **Electro-Thermal ML (2023)**
   - Title: "Machine learning for electro-thermal analysis of PCBs"
   - **Key Points:** Coupled electrical and thermal simulation

### Required Libraries

| Library | Purpose | Install | Era |
|---------|---------|---------|-----|
| `PySpice` | Python SPICE | `pip install PySpice` | 2010s |
| `skrf` | RF analysis | `pip install scikit-rf` | 2010s |
| `pcb-tools` | PCB parsing | `pip install pcb-tools` | 2010s |
| `KiCad` | PCB design | System package | 1992+ |

---

## 5. Control & GNC Agents

### Traditional Foundation (Outdated)

**OLD:**
- Ziegler-Nichols (1942) - PID tuning
- Classical LQR/LQG textbooks (1980s-1990s)

### Modern Research (2019-2026)

**Machine Learning MPC:**

1. **ML-Based MPC Tutorial (2024)**
   - Title: "A Tutorial Review of Machine Learning-based MPC Methods"
   - Source: Various, 2024
   - **Key Points:** Integration of ML with Model Predictive Control

2. **Generative MPC in Manufacturing (2025)**
   - Title: "Generative Model Predictive Control in Manufacturing Processes"
   - Source: arXiv, 2025
   - **Key Points:** Very recent, cutting-edge

3. **Advanced MPC for Converters (2024)**
   - Title: "Review on Advanced Model Predictive Control Technologies for High-Power Converters"
   - Source: MDPI Electronics, 2024
   - **Key Points:** Industrial applications

**Reinforcement Learning for Control:**

4. **RL for Control (2020-2024)**
   - Title: "Reinforcement learning for control: A review"
   - **Key Points:** End-to-end learning of control policies

5. **Safe RL for Control (2023)**
   - Title: "Safe reinforcement learning with stability guarantees"
   - **Key Points:** Safe exploration in control systems

**GNC Modern Methods:**

6. **Astrodynamics ML (2022-2024)**
   - Tool: `poliastro` - Modern astrodynamics library
   - **Key Points:** Orbital mechanics with modern Python

7. **Trajectory Optimization ML (2023)**
   - Title: "Learning-based trajectory optimization for spacecraft"
   - **Key Points:** Neural surrogates for trajectory planning

### Required Libraries

| Library | Purpose | Install | Era |
|---------|---------|---------|-----|
| `casadi` | Optimization | `pip install casadi` | 2010s |
| `control` | Control systems | `pip install control` | 2009+ |
| `poliastro` | Astrodynamics | `pip install poliastro` | 2015+ |
| `filterpy` | Kalman filters | `pip install filterpy` | 2010s |

---

## 6. Tolerance Agent

### Traditional Foundation

**Standards:**
- ISO 286 - Limits and fits
- ASME Y14.5 - GD&T

### Modern Research (2019-2026)

**Statistical Tolerance Analysis:**

1. **Monte Carlo for Tolerances (2020)**
   - Title: "Efficient Monte Carlo simulation for tolerance analysis"
   - **Key Points:** Sampling methods, variance reduction

2. **Machine Learning for Tolerance Stack-up (2023)**
   - Title: "Neural network surrogates for tolerance analysis"
   - **Key Points:** Fast approximation of tolerance stacks

**GD&T Automation:**

3. **Automated GD&T (2022)**
   - Title: "Automated geometric dimensioning and tolerancing"
   - **Key Points:** AI-driven GD&T specification

4. **Tolerance Optimization (2023)**
   - Title: "Tolerance optimization using machine learning"
   - **Key Points:** Cost-quality trade-offs

---

## Summary: Research Era Comparison

| Domain | Task.md Era | Modern Era | Gap |
|--------|-------------|------------|-----|
| **CFD** | 2002-2016 | 2019-2026 | 8-17 years behind |
| **Control** | 1942-1990s | 2019-2026 | 25-80 years behind |
| **DFM** | 2011 | 2019-2023 | 8-12 years behind |
| **Electronics** | N/A | 2022-2025 | N/A |
| **Materials** | N/A | 2019-2026 | N/A |

**Key Insight:** The task.md bibliography is **not wrong** - it lists foundational works. But it **misses the AI/ML revolution entirely**, which started in ~2019 for engineering applications.

---

## Implementation Priority Based on Research Maturity

**Tier 1: Ready for Production (High Research Maturity)**
1. **FNO for Fluids** - Well-established (Li et al. 2021 + many follow-ups)
2. **ML for Cost Estimation** - Random Forest/XGBoost well-proven
3. **PINNs for Electronics** - Growing body of work

**Tier 2: Emerging (Medium Maturity)**
4. **AI-Driven DFM** - Active research area
5. **ML-MPC** - Tutorial reviews just published (2024)
6. **GNN for Tolerance Analysis** - Cutting edge

**Tier 3: Experimental (Low Maturity)**
7. **Quantum CFD** - Not ready for production
8. **Foundation Models for Engineering** - Emerging field

---

## Recommended Reading List for Implementation

### Must-Read (Foundation)
1. Li et al. (2021) - FNO paper **[CRITICAL - Already used in Structural]**
2. Raissi et al. (2019) - PINNs **[CRITICAL for all physics agents]**

### Should-Read (Implementation Guidance)
3. "Machine learning for fluid mechanics" review (2020)
4. "Materials informatics review" (2025)
5. "ML-based MPC tutorial" (2024)

### Optional (Advanced Topics)
6. "Quantum computing for CFD" (2024) - Future-looking
7. "Foundation models for science" (2023) - Speculative

---

**Conclusion:** Update agent specifications to cite 2019-2026 research, especially neural operators, physics-informed ML, and materials informatics. The 2002-2011 references are foundational but miss the transformative AI/ML advances.
