# BRICK OS: Comprehensive Production Implementation Guide

**Version:** 2.0  
**Date:** 2026-03-04  
**Scope:** All 76 agents + Frontend Integration  
**Document Type:** Master Implementation Guide

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Stub Agents Deep Dive](#stub-agents-deep-dive)
4. [ML Implementation Requirements](#ml-implementation-requirements)
5. [Frontend-Backend Integration](#frontend-backend-integration)
6. [Implementation Roadmap](#implementation-roadmap)
7. [Resource Requirements](#resource-requirements)
8. [Risk Assessment](#risk-assessment)

---

## Executive Summary

### Critical Finding
**Only 3 of 76 agents are production-ready.** The remaining 73 agents need significant work ranging from minor fixes to complete rewrites.

### Production-Ready Agents (3)
1. `thermal_solver_3d.py` - Validated 3D FVM (NAFEMS T1)
2. `fluid_agent.py` (Correlation mode) - Cd(Re) correlations
3. `manifold_agent.py` - Watertight mesh checking

### Stub Agents Requiring Implementation (25)
- **Pure Stubs (7):** generic, performance, doctor, asset_sourcing, pvc, remote, user
- **ML Framework Only (5):** control, diagnostic, template_design, stt, vhil
- **Partial Implementation (10):** visual_validator, network, sustainability, verification, lattice, standards, slicer, replicator, von_neumann, explainable
- **Database-Dependent (3):** standards, sustainability, asset_sourcing (partial)

### Frontend-Backend Gaps Identified
- 40+ agents lack API endpoints
- No WebSocket implementation for real-time updates
- Missing progress indicators for long operations
- Incomplete error handling

---

## Current State Analysis

### Agent Categorization

```
Total Agents: 76
├── Production Ready: 3 (4%)
├── Functional but Limited: 24 (32%)
├── Stub/Placeholder: 25 (33%)
├── Framework Only: 16 (21%)
└── Broken/Non-functional: 8 (10%)
```

### Critical Code Patterns Identified

#### Pattern 1: The "Fallback Trap"
Structural agent demonstrates this perfectly - 2,109 lines that all fallback to σ=F/A:

```python
async def _surrogate_prediction(self, ...):
    if not HAS_TORCH or self.pinn_model is None:
        return self._analytical_surrogate(...)  # FALLBACK
    
async def _rom_solution(self, ...):
    if not self.rom.is_trained:
        return await self._surrogate_prediction(...)  # FALLBACK
    
async def _full_fea(self, ...):
    if not self.fea_solver.is_available():
        return self._analytical_solution(...)  # FALLBACK
```

**Impact:** No matter what fidelity level is selected, the user gets analytical beam theory.

#### Pattern 2: Hardcoded Values

| Agent | Hardcoded Value | Line | Impact |
|-------|-----------------|------|--------|
| performance_agent.py | efficiency=0.85 | 29 | Mock benchmarking |
| electronics_agent.py | efficiency=0.5 | 45 | Mock electronics |
| topological_agent.py | score=0.85 | 89 | Mock topology |
| safety_agent.py | score=1.0 | 15 | No actual analysis |
| fluid_agent.py | confidence=0.5 | 120 | Arbitrary confidence |

#### Pattern 3: Empty Mock Arrays

```python
# asset_sourcing_agent.py:18
self.mock_assets = []  # Empty - returns nothing

# diagnostic_agent.py:21
self.surrogate = None  # Not trained

# template_design_agent.py:46
self.has_surrogate = False  # Untrained
```

---

## Stub Agents Deep Dive

### Category 1: Pure Stubs (Delete or Rewrite)

#### 1.1 generic_agent.py
**Lines:** 43  
**Action:** DELETE  
**Rationale:** Pure placeholder, no production value

**Current Code:**
```python
def run(self, params: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
    return {
        "status": "success",
        "agent_id": self.name.lower().replace(" ", "_"),
        "timestamp": time.time(),
        "logs": [f"{self.name} initialized."]
    }
```

---

#### 1.2 performance_agent.py
**Lines:** 35  
**Action:** FULL REWRITE  
**Effort:** 3 weeks

**Required Implementation:**
```python
class PerformanceAgent:
    """
    Production Performance Agent.
    
    Calculates real performance metrics:
    - Strength-to-weight ratio
    - FEA-based stress analysis  
    - Thermal efficiency
    - Aerodynamic efficiency (Cd/Cl)
    - Benchmark comparisons
    """
    
    def __init__(self):
        self.fea_solver = CalculiXInterface()
        self.material_db = MaterialDatabase()
        self.benchmarks = BenchmarkLibrary()
    
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # 1. Load mesh
        mesh = await self.load_mesh(params['mesh_path'])
        material = await self.material_db.get(params['material'])
        
        # 2. Run FEA analysis
        stress_result = await self.fea_solver.solve_stress(
            mesh, 
            params['loads'],
            material['youngs_modulus']
        )
        
        # 3. Calculate mass
        mass = self.calculate_mass(mesh, material['density'])
        
        # 4. Performance metrics
        max_stress = stress_result.max_stress
        yield_strength = material['yield_strength_mpa']
        
        # Strength-to-weight ratio (kN·m/kg)
        strength_to_weight = (yield_strength / max_stress) / mass
        
        # 5. Compare to benchmark
        benchmark = self.benchmarks.get(params['category'])
        efficiency = self.calculate_efficiency(
            strength_to_weight, 
            benchmark.strength_to_weight
        )
        
        # 6. Safety factor
        safety_factor = yield_strength / max_stress
        
        return {
            "efficiency_score": round(efficiency, 3),
            "strength_to_weight": round(strength_to_weight, 3),
            "max_stress_mpa": round(max_stress, 2),
            "safety_factor": round(safety_factor, 2),
            "mass_kg": round(mass, 4),
            "benchmark": {
                "name": benchmark.name,
                "industry_avg": benchmark.strength_to_weight,
                "percentile": self.calculate_percentile(efficiency)
            },
            "recommendations": self.generate_recommendations(
                stress_result, safety_factor, efficiency
            )
        }
```

**Dependencies:**
- `calculix` (FEA solver)
- `meshio` (mesh I/O)
- `numpy`, `scipy`

**API Endpoints:**
```python
@app.post("/api/agents/performance/analyze")
async def analyze_performance(params: PerformanceRequest):
    agent = PerformanceAgent()
    result = await agent.run(params.dict())
    return result

@app.get("/api/agents/performance/benchmarks")
async def get_benchmarks(category: Optional[str] = None):
    return {"benchmarks": BenchmarkLibrary().list(category)}
```

**Frontend Integration:**
- Component: `PerformanceDashboard.tsx`
- Charts: Gauge charts for efficiency, bar charts for comparison
- Export: PDF report generation

---

### Category 2: ML Framework Agents (Train Models)

#### 2.1 control_agent.py
**Lines:** 183  
**Current:** LQR works, RL untrained  
**Action:** Train RL-MPC policy  
**Effort:** 4 weeks

**Research Required:**
- Arroyo et al. (2022) "RL-MPC for Building Energy Management"
- Amos et al. (2018) "Differentiable MPC"
- Lin et al. (2024) "TD3-based RL-MPC"

**Implementation Plan:**

**Phase 1: Environment Setup (Week 1)**
```python
import gym
from stable_baselines3 import PPO
import torch
import torch.nn as nn

class BrickControlEnv(gym.Env):
    """
    Custom environment for control policy training.
    
    State: [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
    Action: [throttle, aileron, elevator, rudder]
    Reward: -||position - target||² - ||velocity||² - ||action||²
    """
    
    def __init__(self):
        super().__init__()
        self.dt = 0.01
        self.physics = PhysicsEngine()
        
        # State: 12D (position, velocity, orientation, angular velocity)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        
        # Action: 4D (throttle, roll, pitch, yaw commands)
        self.action_space = gym.spaces.Box(
            low=0, high=1, shape=(4,), dtype=np.float32
        )
    
    def reset(self):
        self.state = np.zeros(12)
        self.target = np.array([10, 10, 10, 0, 0, 0])  # Target position/velocity
        return self._get_observation()
    
    def step(self, action):
        # Apply physics
        self.state = self.physics.step(self.state, action, self.dt)
        
        # Calculate reward
        position_error = np.linalg.norm(self.state[:3] - self.target[:3])
        velocity_error = np.linalg.norm(self.state[3:6] - self.target[3:6])
        control_effort = np.linalg.norm(action)
        
        reward = -(position_error**2 + 0.1*velocity_error**2 + 0.01*control_effort**2)
        
        done = position_error < 0.1  # Reached target
        
        return self._get_observation(), reward, done, {}
```

**Phase 2: Training (Week 2-3)**
```python
def train_control_policy():
    """Train PPO policy for control"""
    
    # Create environment
    env = BrickControlEnv()
    env = make_vec_env(lambda: env, n_envs=8)
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log="./control_tensorboard/"
    )
    
    # Train
    model.learn(total_timesteps=1_000_000)
    
    # Save
    model.save("models/control_policy_ppo.zip")
    
    return model

if __name__ == "__main__":
    train_control_policy()
```

**Phase 3: Integration (Week 4)**
```python
class ControlAgent:
    def __init__(self):
        self.lqr = LQRController()
        self.rl_policy = self._load_rl_policy()
        self.mpc = DifferentiableMPC(horizon=10)
    
    def _load_rl_policy(self):
        try:
            from stable_baselines3 import PPO
            model = PPO.load("models/control_policy_ppo.zip")
            return model
        except:
            return None
    
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        mode = params.get("control_mode", "RL-MPC")
        state = np.array(params.get("state", []))
        target = np.array(params.get("target", []))
        
        if mode == "RL-MPC" and self.rl_policy:
            # RL-MPC Hybrid
            
            # 1. MPC baseline
            u_mpc = self.mpc.solve(state, target)
            
            # 2. RL refinement
            obs = np.concatenate([state, target])
            u_rl, _ = self.rl_policy.predict(obs, deterministic=False)
            
            # 3. Blend (MPC provides stability, RL optimizes)
            alpha = 0.7
            action = alpha * u_mpc + (1 - alpha) * u_rl
            
            return {
                "control_signal": action.tolist(),
                "method": "RL-MPC",
                "mpc_component": u_mpc.tolist(),
                "rl_component": u_rl.tolist(),
                "blend_factor": alpha
            }
        
        elif mode == "LQR":
            return self.lqr.control(state, target)
        
        else:
            return {"status": "error", "error": f"Unknown mode: {mode}"}
```

**Training Requirements:**
- GPU: NVIDIA V100 or better
- Time: ~8 hours for 1M timesteps
- Episodes: 100k+
- Convergence: Reward > -100 (position error < 10m)

**Validation:**
- Test on unseen scenarios
- Compare against LQR baseline
- Must achieve 20% better performance than LQR

---

### Category 3: Partial Implementation (Complete Features)

#### 3.1 standards_agent.py
**Lines:** 92  
**Current:** RAG with mock fallback  
**Action:** Complete standards database + RAG  
**Effort:** 3 weeks

**Required Standards Database:**
- ISO (International Organization for Standardization)
- ASTM (American Society for Testing and Materials)
- ASME (American Society of Mechanical Engineers)
- NASA Standards
- MIL-STD (Military Standards)
- IEC (International Electrotechnical Commission)

**Implementation:**

**Phase 1: Data Collection (Week 1)**
```python
class StandardsDatabaseBuilder:
    """Build vector database of engineering standards"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = SupabaseVectorStore()
    
    async def ingest_standard(self, standard_id: str, file_path: str):
        """
        Ingest a standard document into the vector database.
        
        Process:
        1. Parse PDF/document
        2. Chunk into sections
        3. Generate embeddings
        4. Store in Supabase
        """
        # Parse document
        sections = self._parse_standard(file_path)
        
        for section in sections:
            # Generate embedding
            embedding = await self.embeddings.aembed_query(section["content"])
            
            # Store with metadata
            await self.vector_store.insert({
                "standard_id": standard_id,
                "section_id": section["id"],
                "title": section["title"],
                "content": section["content"],
                "embedding": embedding,
                "page": section["page"],
                "category": section["category"]
            })
    
    def _parse_standard(self, file_path: str) -> List[Dict]:
        """Parse PDF standard into sections"""
        import PyPDF2
        
        sections = []
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                
                # Extract section headers (e.g., "4.2.1 Safety Factors")
                import re
                headers = re.findall(r'(\d+\.\d+(?:\.\d+)?)\s+(.+)', text)
                
                for section_num, title in headers:
                    sections.append({
                        "id": f"{file_path}_{section_num}",
                        "title": title,
                        "content": text,
                        "page": i + 1,
                        "category": self._categorize(title)
                    })
        
        return sections
```

**Phase 2: RAG Implementation (Week 2)**
```python
class StandardsRetriever:
    """Retrieve relevant standards using semantic search"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = SupabaseVectorStore()
    
    async def retrieve(self, query: str, filters: Dict = None, top_k: int = 5) -> List[Dict]:
        """
        Retrieve relevant standard sections.
        
        Args:
            query: Natural language query
            filters: Optional filters (e.g., {"standard_id": "ISO-9001"})
            top_k: Number of results to return
        
        Returns:
            List of relevant sections with similarity scores
        """
        # Generate query embedding
        query_embedding = await self.embeddings.aembed_query(query)
        
        # Search vector store
        results = await self.vector_store.similarity_search(
            query_embedding,
            filter=filters,
            k=top_k
        )
        
        return [
            {
                "standard_id": r["standard_id"],
                "section_id": r["section_id"],
                "title": r["title"],
                "content": r["content"][:500] + "...",  # Truncate
                "similarity": r["similarity"],
                "page": r["page"]
            }
            for r in results
        ]
```

**Phase 3: Compliance Checking (Week 3)**
```python
class StandardsAgent:
    def __init__(self):
        self.retriever = StandardsRetriever()
        self.compliance_checker = ComplianceChecker()
    
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get("query")
        design = params.get("design", {})
        
        # Retrieve relevant standards
        standards = await self.retriever.retrieve(query, top_k=10)
        
        # Check compliance for each
        compliance_results = []
        for standard in standards:
            result = await self.compliance_checker.check(design, standard)
            compliance_results.append({
                "standard": standard["standard_id"],
                "section": standard["section_id"],
                "compliant": result["compliant"],
                "requirements_met": result["met_requirements"],
                "gaps": result["gaps"],
                "recommendations": result["recommendations"]
            })
        
        # Overall compliance
        overall_compliant = all(r["compliant"] for r in compliance_results)
        
        return {
            "standards_found": len(standards),
            "compliance_results": compliance_results,
            "overall_compliant": overall_compliant,
            "critical_gaps": self._identify_critical_gaps(compliance_results),
            "action_items": self._generate_action_items(compliance_results)
        }
```

**Frontend Integration:**
```typescript
// StandardsCompliancePanel.tsx
interface StandardsCompliancePanelProps {
    designId: string;
    onComplianceUpdate: (result: ComplianceResult) => void;
}

const StandardsCompliancePanel: React.FC<StandardsCompliancePanelProps> = ({
    designId,
    onComplianceUpdate
}) => {
    const [compliance, setCompliance] = useState<ComplianceResult | null>(null);
    const [loading, setLoading] = useState(false);
    
    const checkCompliance = async () => {
        setLoading(true);
        const result = await apiClient.post('/api/agents/standards/check', {
            design_id: designId,
            query: 'safety requirements for aerospace components'
        });
        setCompliance(result);
        onComplianceUpdate(result);
        setLoading(false);
    };
    
    return (
        <div className="standards-panel">
            <Button onClick={checkCompliance} loading={loading}>
                Check Standards Compliance
            </Button>
            
            {compliance && (
                <ComplianceReport data={compliance} />
            )}
        </div>
    );
};
```

---

## ML Implementation Requirements

### Models to Train

| Model | Architecture | Training Data | Time | Hardware |
|-------|-------------|---------------|------|----------|
| Structural FNO | Fourier Neural Operator | 10k FEA sims | 2-3 days | V100 |
| Fluid FNO | Fourier Neural Operator | 10k CFD sims | 2-3 days | V100 |
| Control Policy | PPO | 1M episodes | 8 hours | RTX 4090 |
| Diagnostic BERT | BERT-base fine-tuned | 100k logs | 4 hours | RTX 4090 |
| Template Surrogate | MLP | 5k samples | 2 hours | CPU |
| Network GNN | GraphSAGE | Topology graphs | 6 hours | RTX 4090 |
| Material GNN | CGCNN | Crystal structures | 12 hours | V100 |
| Cost Predictor | XGBoost | Manufacturing data | 1 hour | CPU |

### Training Data Generation

#### FNO Training Pipeline
```python
class FNOTrainingPipeline:
    """Generate training data and train FNO models"""
    
    def __init__(self, solver_type: str = "openfoam"):
        self.solver = self._init_solver(solver_type)
        self.param_space = self._define_param_space()
    
    def generate_training_data(self, n_samples: int = 10000):
        """Generate simulation data for FNO training"""
        
        dataset = []
        for i in range(n_samples):
            # Sample parameters
            params = self.param_space.sample()
            
            # Generate geometry
            geometry = self.generate_geometry(params)
            
            # Run simulation
            result = self.solver.solve(geometry, params)
            
            # Store
            dataset.append({
                "input": self._encode_input(geometry, params),
                "output": result.field,
                "params": params
            })
            
            if i % 100 == 0:
                print(f"Generated {i}/{n_samples} samples")
        
        return dataset
    
    def train_fno(self, dataset: List[Dict]):
        """Train FNO on generated data"""
        from neural_operators import FNO2d
        
        model = FNO2d(
            modes1=12,
            modes2=12,
            width=64,
            in_channels=4,  # [x, y, Re, Ma]
            out_channels=4  # [u, v, p, nut]
        )
        
        # Training loop
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        
        for epoch in range(1000):
            total_loss = 0
            for batch in DataLoader(dataset, batch_size=32):
                optimizer.zero_grad()
                
                pred = model(batch["input"])
                loss = F.mse_loss(pred, batch["output"])
                
                # Physics-informed loss
                pde_loss = self._compute_pde_residual(pred, batch["input"])
                total_loss_batch = loss + 0.1 * pde_loss
                
                total_loss_batch.backward()
                optimizer.step()
                
                total_loss += total_loss_batch.item()
            
            scheduler.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(dataset):.6f}")
        
        # Save model
        torch.save(model.state_dict(), "models/fluid_fno.pt")
        
        return model
```

---

## Frontend-Backend Integration

### API Gap Analysis

| Agent | Has API | WebSocket | Frontend Component |
|-------|---------|-----------|-------------------|
| thermal_solver_3d | ✓ | ✗ | ✗ |
| fluid_agent | ✓ | ✗ | ✗ |
| structural_agent | ✓ | ✗ | ✗ |
| dfm_agent | ✓ | ✗ | ✗ |
| performance_agent | ✗ | ✗ | ✗ |
| control_agent | ✗ | ✗ | ✗ |
| standards_agent | ✗ | ✗ | ✗ |
| ... (40+ more) | ✗ | ✗ | ✗ |

### Required API Endpoints

```python
# Universal agent execution endpoint
@app.post("/api/agents/{agent_name}/run")
async def run_agent(
    agent_name: str,
    params: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    """
    Execute any agent by name.
    
    Args:
        agent_name: Name of agent to run
        params: Agent-specific parameters
        
    Returns:
        Agent execution result
    """
    agent = get_agent_registry().get(agent_name)
    if not agent:
        raise HTTPException(404, f"Agent {agent_name} not found")
    
    # Execute
    result = await agent.run(params)
    
    return result

# Agent status endpoint
@app.get("/api/agents/{agent_name}/status")
async def get_agent_status(agent_name: str):
    """Get agent health and capability status"""
    agent = get_agent_registry().get(agent_name)
    if not agent:
        raise HTTPException(404, f"Agent {agent_name} not found")
    
    return {
        "name": agent_name,
        "status": agent.get_status(),
        "capabilities": agent.get_capabilities(),
        "version": getattr(agent, "version", "1.0.0"),
        "last_healthy": agent.last_health_check
    }

# WebSocket for real-time updates
@app.websocket("/ws/agents/{agent_name}")
async def agent_websocket(websocket: WebSocket, agent_name: str):
    """WebSocket for real-time agent updates"""
    await websocket.accept()
    
    agent = get_agent_registry().get(agent_name)
    if not agent:
        await websocket.close(code=4004, reason="Agent not found")
        return
    
    try:
        while True:
            # Receive command
            data = await websocket.receive_json()
            
            # Execute
            result = await agent.run(data["params"])
            
            # Send progress updates
            await websocket.send_json({
                "type": "progress",
                "progress": result.get("progress", 0),
                "message": result.get("message", "")
            })
            
            # Send final result
            await websocket.send_json({
                "type": "result",
                "data": result
            })
            
    except WebSocketDisconnect:
        pass
```

### Frontend Components Required

```typescript
// AgentExecutionPanel.tsx
interface AgentExecutionPanelProps {
    agentName: string;
    parameters: ParameterSchema;
    onResult?: (result: any) => void;
    onProgress?: (progress: number) => void;
}

const AgentExecutionPanel: React.FC<AgentExecutionPanelProps> = ({
    agentName,
    parameters,
    onResult,
    onProgress
}) => {
    const [status, setStatus] = useState<'idle' | 'running' | 'completed' | 'error'>('idle');
    const [progress, setProgress] = useState(0);
    const [result, setResult] = useState<any>(null);
    const [error, setError] = useState<string | null>(null);
    
    const wsRef = useRef<WebSocket | null>(null);
    
    const runAgent = async () => {
        setStatus('running');
        setProgress(0);
        
        // Use WebSocket for real-time updates
        const ws = new WebSocket(`ws://localhost:8000/ws/agents/${agentName}`);
        wsRef.current = ws;
        
        ws.onopen = () => {
            ws.send(JSON.stringify({ params: parameters }));
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.type === 'progress') {
                setProgress(data.progress);
                onProgress?.(data.progress);
            } else if (data.type === 'result') {
                setResult(data.data);
                setStatus('completed');
                onResult?.(data.data);
                ws.close();
            }
        };
        
        ws.onerror = (err) => {
            setError('WebSocket error');
            setStatus('error');
        };
    };
    
    return (
        <Card className="agent-execution-panel">
            <CardHeader>
                <h3>{agentName}</h3>
            </CardHeader>
            <CardContent>
                {status === 'running' && (
                    <Progress value={progress} />
                )}
                {status === 'completed' && result && (
                    <ResultsViewer data={result} />
                )}
                {status === 'error' && (
                    <Alert type="error">{error}</Alert>
                )}
            </CardContent>
            <CardFooter>
                <Button 
                    onClick={runAgent} 
                    disabled={status === 'running'}
                    loading={status === 'running'}
                >
                    Run Agent
                </Button>
            </CardFooter>
        </Card>
    );
};

// ResultsViewer.tsx
interface ResultsViewerProps {
    data: any;
    visualizationType?: 'table' | 'chart' | '3d';
}

const ResultsViewer: React.FC<ResultsViewerProps> = ({
    data,
    visualizationType = 'table'
}) => {
    if (visualizationType === 'chart' && data.metrics) {
        return <MetricsChart data={data.metrics} />;
    }
    
    if (visualizationType === '3d' && data.geometry) {
        return <GeometryViewer geometry={data.geometry} />;
    }
    
    return <DataTable data={data} />;
};
```

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
**Goal:** Establish base infrastructure

- [ ] Delete generic_agent.py
- [ ] Create base agent class with common functionality
- [ ] Add health endpoints to all agents
- [ ] Implement error handling framework
- [ ] Set up ML training infrastructure
- [ ] Create API endpoint framework

**Deliverables:**
- Base agent class implemented
- All agents have /health endpoints
- API routing framework complete

---

### Phase 2: Pure Stubs (Weeks 3-10)
**Goal:** Replace stub implementations

#### Week 3-4: Performance Agent
- [ ] Implement CalculiX interface
- [ ] Add FEA stress analysis
- [ ] Create benchmark library
- [ ] Build efficiency calculations

#### Week 5: Doctor Agent
- [ ] Implement health check mechanism
- [ ] Add system metrics collection
- [ ] Create alerting system

#### Week 6: PVC Agent
- [ ] Git integration
- [ ] Version control operations
- [ ] Frontend diff viewer

#### Week 7: User Agent
- [ ] Supabase Auth integration
- [ ] User profile management
- [ ] Permission system

#### Week 8-9: Remote Agent
- [ ] WebSocket implementation
- [ ] Session management
- [ ] Real-time collaboration

#### Week 10: Asset Sourcing
- [ ] NASA API integration
- [ ] GrabCAD API integration
- [ ] Asset browser UI

---

### Phase 3: ML Framework (Weeks 11-20)
**Goal:** Train and integrate ML models

#### Week 11-14: Control Agent
- [ ] Build training environment
- [ ] Train PPO policy (8 hours GPU)
- [ ] Validate against LQR
- [ ] Integrate RL-MPC

#### Week 15-16: Diagnostic Agent
- [ ] Collect training logs
- [ ] Fine-tune BERT model
- [ ] Deploy inference

#### Week 17: Template Design
- [ ] Train surrogate model
- [ ] Quality prediction

#### Week 18: STT Agent
- [ ] Integrate Whisper
- [ ] Audio preprocessing

#### Week 19-20: vHIL Agent
- [ ] Accurate sensor models
- [ ] Physics integration

---

### Phase 4: Partial Implementation (Weeks 21-36)
**Goal:** Complete partially implemented agents

#### Week 21: Visual Validator
- [ ] Render quality assessment
- [ ] Scene composition

#### Week 22-24: Network Agent
- [ ] GNN implementation
- [ ] Topology analysis

#### Week 25-27: Sustainability
- [ ] Full LCA calculation
- [ ] Ecoinvent integration

#### Week 28-30: Verification
- [ ] Requirement parsing
- [ ] Test generation

#### Week 31-34: Lattice Synthesis
- [ ] GNoME integration
- [ ] Optimization algorithms

#### Week 35-36: Standards + Slicer
- [ ] Standards database
- [ ] Slicer integration

---

### Phase 5: Frontend Integration (Weeks 37-40)
**Goal:** Complete UI/UX integration

- [ ] Build agent execution panels
- [ ] Add real-time updates (WebSockets)
- [ ] Create results visualization
- [ ] Implement progress indicators
- [ ] Add error handling UI

---

## Resource Requirements

### Personnel

| Role | Count | Duration | Cost |
|------|-------|----------|------|
| ML Engineer | 2 | 20 weeks | $80,000 |
| Backend Engineer | 2 | 20 weeks | $80,000 |
| Frontend Engineer | 1 | 10 weeks | $25,000 |
| DevOps Engineer | 1 | 5 weeks | $12,500 |
| **Total** | **6** | | **$197,500** |

### Infrastructure

| Resource | Quantity | Duration | Cost |
|----------|----------|----------|------|
| GPU (V100) | 2 | 3 months | $4,500 |
| GPU (RTX 4090) | 2 | 3 months | $3,600 |
| Cloud Compute | - | 3 months | $2,000 |
| Storage (datasets) | 5TB | 3 months | $500 |
| **Total** | | | **$10,600** |

### APIs and Services

| Service | Monthly Cost | Annual Cost |
|---------|--------------|-------------|
| OpenAI API (embeddings) | $500 | $6,000 |
| Supabase | $25 | $300 |
| NASA API | Free | Free |
| GrabCAD API | $100 | $1,200 |
| **Total** | | **$7,500** |

### **Grand Total: ~$215,600**

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| FNO training doesn't converge | Medium | High | Start with smaller model, use transfer learning |
| OpenCASCADE integration fails | Medium | High | Use Manifold3D only, add mesh conversion |
| API rate limits (NASA/GrabCAD) | Medium | Medium | Implement caching, request rate increases |
| Frontend performance issues | Low | Medium | Optimize rendering, use Web Workers |
| Database scaling issues | Low | High | Implement read replicas, caching layer |
| ML model accuracy insufficient | Medium | High | Collect more data, try different architectures |
| Team member availability | Medium | Medium | Cross-train team, document thoroughly |

---

## Success Criteria

### By Phase

| Phase | Success Criteria |
|-------|-----------------|
| 1 | All agents have health endpoints, base class implemented |
| 2 | All 7 pure stubs replaced with functional implementations |
| 3 | All ML models trained and achieving target accuracy |
| 4 | All partial implementations completed |
| 5 | Frontend integration complete, UI/UX validated |

### Overall Metrics

| Metric | Target |
|--------|--------|
| Agents production-ready | 70+ (from 3) |
| ML surrogates working | 10+ (from 0) |
| API coverage | 100% of agents |
| Test coverage | >80% |
| Frontend integration | All agents accessible |
| Documentation | Complete for all agents |

---

## Appendices

### A. Dependencies List

**Python Packages:**
```
# Core
fastapi==0.104.0
uvicorn[standard]==0.24.0
pydantic==2.5.0

# ML/AI
torch==2.1.0
torchvision==0.16.0
stable-baselines3==2.2.0
transformers==4.35.0
neural-operators==0.2.0

# Physics/Engineering
calculix==0.1.0
fenics==2019.1.0
meshio==5.3.0
trimesh==4.0.0

# Data
supabase==2.0.0
redis==5.0.0
numpy==1.26.0
scipy==1.11.0
pandas==2.1.0

# Utilities
gitpython==3.1.0
psutil==5.9.0
aiohttp==3.9.0
python-dotenv==1.0.0
```

**External Software:**
- CalculiX (FEA solver)
- CuraEngine or PrusaSlicer (slicing)
- OpenFOAM (CFD - optional for FNO training)
- Redis (session store)
- PostgreSQL (database)

### B. API Documentation Template

Each agent API should document:
- Endpoint URL
- Request/response schemas
- Authentication requirements
- Rate limits
- Error codes
- Example usage

### C. Testing Strategy

**Unit Tests:**
- Each agent method
- API endpoints
- Utility functions

**Integration Tests:**
- Agent-to-agent communication
- Frontend-backend integration
- Database operations

**End-to-End Tests:**
- Complete design workflows
- Multi-agent orchestration
- Error recovery

---

*Document compiled: 2026-03-04*  
*Total agents analyzed: 76*  
*Total lines of code analyzed: 32,175*  
*Estimated implementation time: 40 weeks with 6 developers*
