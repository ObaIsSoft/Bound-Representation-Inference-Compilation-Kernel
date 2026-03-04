# BRICK OS: Agent Implementation Checklist

## Critical Gaps Identified in Research

### 1. Thermal Agents
**thermal_solver_3d.py** - ✅ PRODUCTION READY
- [x] 3D FVM validated against NAFEMS T1 (18% error, converges)
- [x] 7-point stencil, conjugate gradient solver
- [ ] Add anisotropic material support (k_x, k_y, k_z)
- [ ] Add transient solver

**thermal_agent.py** - NEEDS ML UPGRADE
- [ ] Integrate Fourier Neural Operator (Li et al. 2021)
- [ ] Generate training data from OpenFOAM (10k samples)
- [ ] Train FNO with physics-informed loss
- [ ] Validate against NAFEMS conjugate heat transfer benchmarks

---

### 2. Fluid Agents
**fluid_agent.py** - CORRELATION MODE WORKS
- [x] Cd(Re) correlations (Schiller-Naumann, White)
- [x] Prandtl-Glauert compressibility correction
- [ ] Train FNO on OpenFOAM data (Re: 0.1-10M, Ma: 0-0.8)
- [ ] Implement AGMF-Net multi-fidelity (Zhan et al. 2024)
- [ ] Add OpenFOAM integration
- [ ] Validate on NACA airfoil benchmarks

**fno_fluid.py** - ARCHITECTURE ONLY
- [ ] Complete training pipeline
- [ ] Generate synthetic training data
- [ ] Train for 1000 epochs
- [ ] Export TorchScript model
- [ ] Validate L2 error < 5%

---

### 3. Structural Agent - MOST CRITICAL
**structural_agent.py** - FALLBACKS EVERYWHERE
- [ ] **HARD REQUIREMENT: Train FNO on CalculiX simulations**
  - Generate 10k parametric geometry simulations
  - Extract stress/strain fields
  - Train FNO: 4 Fourier layers, width=64, modes=12
  - Validate against ASME V&V 20
- [ ] **HARD REQUIREMENT: Build ROM database**
  - Run 1000 FEA simulations for POD extraction
  - Store 99% energy retention modes
  - Implement proper subspace projection
- [ ] Fix CalculiX integration (currently untested)
- [ ] Add mesh convergence study
- [ ] Implement ASME V&V 20 verification metrics

**Research Required:**
- Li et al. (2021) FNO for Parametric PDEs
- Kovachki et al. (2021) Universal approximation of FNO
- ASME V&V 20 (2006)

---

### 4. Geometry Agents
**geometry_agent.py** - MANIFOLD3D WORKS, OCP BROKEN
- [ ] Fix OpenCASCADE environment (bus error)
- [ ] Add STEP/IGES import via OCP
- [ ] Implement neural implicit geometry (DeepSDF)
- [ ] Add text-to-3D generation pipeline

**manifold_agent.py** - ✅ WORKS
- [x] Watertightness checking
- [x] Self-intersection detection
- [x] SDF reconstruction
- [ ] Add mesh simplification

---

### 5. Manufacturing Agents
**dfm_agent.py** - FALSE POSITIVES
- [ ] **CRITICAL: Fix hole detection false positives**
  - Filter by circularity > 0.8
  - Add aspect ratio checks
  - Validate on real manufactured parts
- [ ] Add CNN-based feature recognition
- [ ] Train manufacturability scoring model
- [ ] Partner with machine shop for ground truth
- [ ] Implement Boothroyd-Dewhurst handling/insertion times

**Research Required:**
- Boothroyd et al. (2011) "Product Design for Manufacture and Assembly"

**cost_agent.py** - DATABASE DEPENDENT
- [ ] Implement XGBoost cost predictor
- [ ] Train on historical manufacturing data
- [ ] Add real-time market price integration
- [ ] Fix confidence calculation (currently 0.5)

**tolerance_agent.py** - BASIC
- [ ] Implement ML surrogate for fast stack-up
- [ ] Train on Monte Carlo samples
- [ ] Add cost-quality trade-off optimization
- [ ] Integrate GD&T specification

---

### 6. Materials Agents
**material_agent.py** - DATABASE ONLY
- [ ] **REQUIRED: Integrate GNN for property prediction**
  - Use ALIGNN or MatterTune
  - Load pre-trained ORB-V2 or JMP-S
  - Predict: E, σ_y, k, ρ
  - Validate on Matbench
- [ ] Add materials informatics pipeline
- [ ] Connect to GNoME dataset

**Research Required:**
- Xie & Grossman (2018) CGCNN
- Choudhary & DeCost (2021) ALIGNN
- MatterTune (2025)

---

### 7. Electronics Agent
**electronics_agent.py** - MOCK IMPLEMENTATIONS
- [ ] **DELETE mock efficiency calculation**
- [ ] Integrate PySpice for circuit simulation
- [ ] Add SPICE netlist generation
- [ ] Implement ML surrogate for SI/PI
- [ ] Add component library integration
- [ ] Implement signal integrity analysis

**Research Required:**
- "AI for SI/PI Analysis" (TechRxiv 2024)
- Neural circuit surrogates (2023)

---

### 8. Control Agents
**control_agent.py** - LQR ONLY
- [ ] **DELETE ML-MPC claims if not implemented**
- [ ] Implement true RL-MPC with TD3
- [ ] Train CEM policy on control tasks
- [ ] Add differentiable MPC layer
- [ ] Implement policy ensemble

**Research Required:**
- Arroyo et al. (2022) RL-MPC
- Amos et al. (2018) Differentiable MPC
- Lin et al. (2024) TD3-based RL-MPC

---

### 9. AI/ML Infrastructure
**surrogate_training.py** - FRAMEWORK ONLY
- [ ] Implement automated data generation pipeline
- [ ] Add active learning with uncertainty
- [ ] Implement multi-fidelity data fusion
- [ ] Add hyperparameter optimization

---

### 10. STUB AGENTS TO DELETE OR REPLACE

| Agent | Action | Rationale |
|-------|--------|-----------|
| generic_agent.py | DELETE | Placeholder, 43 lines |
| performance_agent.py | REWRITE | Hardcoded 0.85 efficiency |
| visual_validator_agent.py | REWRITE | Score=1.0, no analysis |
| safety_agent.py | REWRITE | Mock analysis only |
| topological_agent.py | REWRITE | Hardcoded 0.85 score |
| standards_agent.py | INTEGRATE | Use real standards DB |

---

## Implementation Timeline

### Sprint 1 (Weeks 1-2): Critical Fixes
- [ ] Fix DFM false positives
- [ ] Fix OpenCASCADE environment
- [ ] Delete generic_agent.py
- [ ] Rewrite performance_agent.py

### Sprint 2 (Weeks 3-4): FNO Training Data
- [ ] Generate 1000 OpenFOAM simulations (fluid)
- [ ] Generate 1000 CalculiX simulations (structural)
- [ ] Set up training infrastructure

### Sprint 3 (Weeks 5-6): FNO Training
- [ ] Train FluidFNO
- [ ] Train StructuralFNO
- [ ] Validate both on benchmarks

### Sprint 4 (Weeks 7-8): ML Integration
- [ ] Integrate Material GNN
- [ ] Implement RL-MPC controller
- [ ] Add SPICE electronics

### Sprint 5 (Weeks 9-10): Validation
- [ ] NAFEMS benchmarks
- [ ] ASME V&V 20
- [ ] Real manufacturing data

### Sprint 6 (Weeks 11-12): Polish
- [ ] Documentation
- [ ] Error handling
- [ ] Performance optimization

---

## Resource Requirements

| Resource | Count | Duration |
|----------|-------|----------|
| ML Engineers | 2 | 12 weeks |
| Simulation Engineers | 1 | 8 weeks |
| DevOps | 1 | 4 weeks |
| GPU Hours | ~5000 | For FNO training |
| OpenFOAM Licenses | 10 | For data generation |

---

## Success Criteria

### FNO Models
- [ ] L2 error < 5% on validation set
- [ ] Inference time < 100ms
- [ ] Generalizes to unseen geometries

### DFM Agent
- [ ] False positive rate < 5%
- [ ] Correlation with actual cycle time > 0.8
- [ ] Tool access detection accuracy > 90%

### Material Agent
- [ ] MAE on elastic modulus < 10%
- [ ] Predicts 5+ properties
- [ ] Validates on Matbench

### Electronics Agent
- [ ] SPICE simulation integration
- [ ] Real component library
- [ ] ML surrogate 100x faster than SPICE

### Control Agent
- [ ] RL-MPC outperforms LQR
- [ ] Real-time capable (>100Hz)
- [ ] Validated on hardware

---

## Research Papers to Acquire

### Immediate Priority
1. Li et al. (2021) - FNO for Parametric PDEs [✓ ACQUIRED]
2. Kovachki et al. (2021) - Universal approximation of FNO [✓ ACQUIRED]
3. Xie & Grossman (2018) - CGCNN [✓ ACQUIRED]
4. Boothroyd et al. (2011) - DFM&A [✓ ACQUIRED]

### Medium Priority
5. Zhan et al. (2024) - AGMF-Net
6. Choudhary & DeCost (2021) - ALIGNN
7. Arroyo et al. (2022) - RL-MPC

### Low Priority
8. MatterTune (2025)
9. Amos et al. (2018) - Differentiable MPC
10. ASME V&V 20 (2006)

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| FNO training fails to converge | Medium | High | Use pre-trained weights, smaller model |
| OpenFOAM integration breaks | Medium | High | Use simpler surrogates |
| DFM validation fails | Low | Medium | Use rule-based fallback |
| CalculiX mesh automation fails | Medium | High | Implement mesh templates |
| GNN inference too slow | Low | Medium | Quantization, pruning |

---

**Last Updated:** 2026-03-02  
**Document Owner:** BRICK OS Development Team  
**Review Cycle:** Weekly during implementation
