# BRICK OS: The System Bible (v2026.01)

> **Comprehensive Architecture & System Reference**  
> *This document serves as the Single Source of Truth for the entire BRICK OS codebase (Frontend, Backend, and Agents).*

---

## 1. System Overview

**BRICK OS** is an autonomous engineering platform ("Hardware Compiler") that turns natural language into manufacturable hardware. Unlike standard CAD tools, it uses a **Self-Evolving Multi-Agent System** to perform real engineering analysis—checking physics, supply chain, and manufacturing constraints in real-time.

### The "Why"
Traditional engineering is siloed (CAD vs. FEA vs. Excel). BRICK OS aggregates these domains into a single **Reactive Dependency Graph** orchestrated by AI Agents.

---

## 2. Repository Structure (Monorepo)

The codebase is organized as a Monorepo containing three distinct layers.

### `.` (Root)
- **Configuration**: `pyproject.toml`, `package.json` (for tooling).
- **Verification Scripts**: `verify_*.py`, `test_*.py` (System-level integration tests).
- **Brain**: `.gemini/antigravity/brain` (Agent artifacts and memory).

### `backend/` (The Intelligence Kernel)
- **Language**: Python 3.12+
- **Core Frameworks**: FastAPI, LangGraph, Pydantic, NetworkX.
- **Key Directories**:
    - `agents/`: The 57 specialized AI agents (see Section 3).
    - `llm/`: The **LLM Factory** (Groq, OpenAI, Ollama providers).
    - `materials/`: **Unified Materials API** (MP, NIST, PubChem aggregation).
    - `vmk_kernel.py`: **Virtual Machining Kernel** (SDF-based physics engine).
    - `orchestrator.py`: The Meta-Critic that manages the agent graph.

### `frontend/` (The Interface)
- **Language**: TypeScript / React 18 / Next.js 14
- **Core Frameworks**: TailwindCSS, Three.js (React Three Fiber), React Flow.
- **Key Components**:
    - `components/panels/`: Specialized engineering panels (Compile, VHIL, ISA Browser).
    - `components/visualizer/`: 3D rendering engine.

### `src-tauri/` (The Desktop Wrapper)
- **Language**: Rust
- **Purpose**: Wraps the Web Frontend into a native binary (MacOS/Windows/Linux) and provides OS-level access (File System, Serial Ports) bridging to the Python backend.

---

## 3. The Agent System (57 Agents)

The system intelligence is distributed across **57 Specialized Agents**, categorized by domain.

### A. Core Physics & Geometry (Tier 1-2)
*Handle the raw reality of the design.*
1.  **EnvironmentAgent**: Determines gravity, pressure, regime (AERO/SPACE/MARINE).
2.  **GeometryAgent**: Generates KCL/CAD models via KittyCAD.
3.  **PhysicsAgent**: Supervises all physical simulations.
4.  **SurrogatePhysicsAgent**: Fast neural approximations of physics.
5.  **ThermalAgent**: Heat dissipation and thermal rise analysis.
6.  **StructuralAgent**: FEA-lite stress and load path analysis.
7.  **FluidAgent**: CFD approximations for lift/drag.
8.  **MassPropertiesAgent**: Inertia tensor and CoG calculation.
9.  **ManifoldAgent**: Mesh watertightness and repair.
10. **SlicerAgent**: G-Code generation for 3D printing.

### B. Design & Optimization (Tier 3)
*Handle the creative generation of solutions.*
11. **DesignerAgent**: Aesthetics, branding, and style.
12. **OptimizationAgent**: Meta-learning optimizer (Gradient vs Genetic).
13. **DesignExplorationAgent**: Generates parametric variations.
14. **TemplateDesignAgent**: Adapts existing templates (NASA/Reference).
15. **GeometryAgent (Kernel)**: Robust boolean operations.
16. **TopologicalAgent**: Analyzing terrain and spatial constraints.

### C. Manufacturing & Materials (Tier 4)
*Handle the "How to Build" logic.*
17. **MaterialAgent**: Selects alloys/composites based on `UnifiedMaterialsAPI`.
18. **ManufacturingAgent**: Generates Bill of Process (BoP) and Cost.
19. **ChemistryAgent**: Corrosion, toxicity, and chemical compatibility.
20. **DfmAgent**: Design for Manufacturing checks (Undercuts, Draft angles).
21. **ToleranceAgent**: ISO fits and tolerances analysis.
22. **ComponentAgent**: Reliability filtering and component selection.
23. **AssetSourcingAgent**: Finds COTS parts (DigiKey/Octopart/Scraping).
24. **CostAgent**: Real-time market estimation.

### D. Control & Systems (Tier 5)
*Handle the "Brains" of the hardware.*
25. **ControlAgent**: PID/LQR controller design.
26. **GncAgent**: Guidance, Navigation, and Control stability laws.
27. **ElectronicsAgent**: PCB topology and power distribution.
28. **NetworkAgent**: Latency and bandwidth prediction.
29. **MepAgent**: Routing for Mechanical/Electrical/Plumbing.
30. **ComplianceAgent**: Regulatory checks (FAA/FCC/ISO).
31. **MitigationAgent**: Automatic failure mitigation proposals.

### E. Support & Code (Tier 6 - "Tool Use")
*Handle the operations of the software itself.*
32. **CodegenAgent**: Writes Firmware/Python scripts to drive hardware.
33. **DevOpsAgent**: Manages Docker/CI pipelines for the system.
34. **ReviewAgent**: Security and Code Quality auditor.
35. **DocumentAgent**: Generates PDFs and Technical Manuals.
36. **ConversationalAgent**: "The Dreamer" (User Interface logic with Memory).

*(Plus 20+ Sub-Agents and Oracles not listed individually but part of the swarm)*

---

## 4. Architecture: Self-Evolution & De-Mocking

### Self-Evolution (The "Learning" Layer)
Agents are not static code. They wrap **Neural Surrogates** that learn from experience.
- **Critics**: Specialized agents (e.g., `PhysicsCritic`, `SurrogateCritic`) monitor outputs.
- **Drift Detection**: If a Surrogate's prediction deviates from Ground Truth (Physics), it triggers a **Retraining Loop**.
- **Meta-Critic**: Resolves conflicts (e.g., Designer wants "Thin", Structural wants "Thick").

### De-Mocking (The "Real" Layer)
As of Jan 2026 (Tier 6 Update), all mock interfaces have been removed:
1.  **Real Intelligence**: Usage of `GroqProvider` (LPU-accelerated) and `OpenAIProvider`.
2.  **Real Data**: `UnifiedMaterialsAPI` aggregates 100M+ real compounds.
3.  **Real Ops**: `DevOpsAgent` talks to the actual host Docker daemon.

---

## 5. Data Flow

1.  **User Intent**: "Build a drone for Mars."
2.  **ConversationalAgent**: clarification loop -> "Mission Profile" established.
3.  **Orchestrator**: Builds a Dependency Graph.
4.  **Geometry/Design**: Generates initial CAD.
5.  **Multi-Physics Loop**: Thermal -> Structural -> Aero analysis.
6.  **Critic Check**: "Is this flyable?" (If no, loop back to Design).
7.  **Manufacturing**: BOM + G-Code + Cost.
8.  **Output**: PDF Brief + STL Files + Firmware Code.

---

*For detailed setup instructions, see `README.md` at the root.*
