# BRICK OS: Stub Agents Implementation Plan

**Document Type:** Production Implementation Guide  
**Agents Covered:** 25 stub/placeholder agents  
**Date:** 2026-03-04

---

## Executive Summary

This document provides detailed implementation plans for converting 25 stub/placeholder agents into production-ready components. Each agent is analyzed for:
- Current functionality gaps
- Required dependencies and APIs
- ML implementation requirements
- Frontend integration points
- Estimated effort

### Stub Agent Classification

| Category | Count | Description |
|----------|-------|-------------|
| **Pure Stubs** | 7 | No real logic, return mock data |
| **Framework Only** | 8 | Architecture exists, no trained models |
| **Partial Implementation** | 10 | Some logic, needs completion |

---

## Category 1: Pure Stubs (7 agents)

These agents have no real functionality and return mock/hardcoded data.

### 1.1 generic_agent.py
**Current State:** 43 lines, returns generic success response
**Production Goal:** Delete or repurpose as base class

**Analysis:**
- Pure placeholder that logs and returns success
- Used when agent is not implemented
- No ML, no database, no real logic

**Decision:** DELETE
- No production value
- Use proper agent factory pattern instead

**Implementation:**
```bash
# Remove file
rm backend/agents/generic_agent.py

# Update imports in any files using it
# Replace with proper agent initialization
```

**Effort:** 1 hour (cleanup only)

---

### 1.2 performance_agent.py
**Current State:** 35 lines, hardcoded efficiency=0.85
**Production Goal:** Real benchmarking system

**Current Code:**
```python
def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
    metrics = {}
    metrics["efficiency_score"] = 0.85  # Mock
    return {"status": "benchmarked", "metrics": metrics}
```

**Required Functionality:**
1. Compute actual strength-to-weight ratio
2. FEA-based stress analysis
3. Thermal efficiency calculation
4. Aerodynamic efficiency (Cd/Cl)
5. Benchmark against industry standards

**Dependencies:**
- `calculix` or `fenics` for FEA
- `scipy.optimize` for benchmark comparison
- Database access for material properties

**API Integration:**
```python
class PerformanceAgent:
    def __init__(self):
        self.fea_solver = CalculiXInterface()
        self.material_db = MaterialDatabase()
    
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # 1. Load mesh
        mesh = await self.load_mesh(params['mesh_path'])
        
        # 2. Run FEA
        stress_result = await self.fea_solver.solve_stress(mesh, params['loads'])
        
        # 3. Calculate metrics
        max_stress = stress_result.max_stress
        mass = self.calculate_mass(mesh, params['material'])
        
        # 4. Compare to benchmark
        benchmark = self.get_benchmark(params['category'])
        efficiency = self.calculate_efficiency(max_stress, mass, benchmark)
        
        return {
            "efficiency_score": efficiency,
            "max_stress_mpa": max_stress,
            "mass_kg": mass,
            "benchmark": benchmark.name,
            "comparison": self.generate_report(stress_result, benchmark)
        }
```

**Frontend Integration:**
- Endpoint: `POST /api/agents/performance/analyze`
- Display: Performance dashboard with gauge charts
- Compare against industry benchmarks

**Implementation Steps:**
1. Create CalculiX interface (Week 1)
2. Implement mesh loading (Week 1)
3. Add stress analysis pipeline (Week 2)
4. Create benchmark database (Week 2)
5. Build efficiency calculations (Week 3)
6. Frontend dashboard integration (Week 3)

**Effort:** 3 weeks
**Dependencies:** CalculiX, mesh libraries

---

### 1.3 doctor_agent.py
**Current State:** 100 lines, random health status
**Production Goal:** Real system health monitoring

**Current Issues:**
```python
# Mock health check
is_healthy = random.random() > 0.01  # Random!
```

**Required Functionality:**
1. Check agent responsiveness (ping tests)
2. Monitor resource utilization (CPU, memory, GPU)
3. Track service uptime
4. Detect agent failures
5. Alert on degraded performance

**Dependencies:**
- `psutil` (already imported)
- `prometheus_client` for metrics
- `asyncio` for concurrent health checks

**Implementation:**
```python
import psutil
import asyncio
from datetime import datetime
from typing import Dict, List
import aiohttp

class DoctorAgent:
    def __init__(self):
        self.agent_endpoints = self._load_agent_endpoints()
        self.health_history = []
        
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        agents = params.get("agent_registry", [])
        
        # Concurrent health checks
        health_tasks = [
            self._check_agent_health(agent) 
            for agent in agents
        ]
        results = await asyncio.gather(*health_tasks, return_exceptions=True)
        
        status_map = {}
        healthy_count = 0
        
        for agent, result in zip(agents, results):
            if isinstance(result, Exception):
                status_map[agent] = {"status": "ERROR", "error": str(result)}
            else:
                status_map[agent] = result
                if result["status"] == "HEALTHY":
                    healthy_count += 1
        
        # System metrics
        system_metrics = self._get_system_metrics()
        
        return {
            "system_health_score": healthy_count / len(agents) if agents else 1.0,
            "status_map": status_map,
            "system_metrics": system_metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _check_agent_health(self, agent_name: str) -> Dict[str, Any]:
        """Ping agent and check response time"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            endpoint = self.agent_endpoints.get(agent_name)
            if not endpoint:
                return {"status": "UNKNOWN", "error": "No endpoint configured"}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{endpoint}/health", 
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    response_time = asyncio.get_event_loop().time() - start_time
                    
                    if response.status == 200:
                        return {
                            "status": "HEALTHY",
                            "response_time_ms": response_time * 1000,
                            "last_check": datetime.utcnow().isoformat()
                        }
                    else:
                        return {
                            "status": "DEGRADED",
                            "http_status": response.status,
                            "response_time_ms": response_time * 1000
                        }
                        
        except asyncio.TimeoutError:
            return {"status": "TIMEOUT", "error": "Health check timed out"}
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}
    
    def _get_system_metrics(self) -> Dict[str, float]:
        """Get real system metrics"""
        process = psutil.Process()
        
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_mb": process.memory_info().rss / (1024 * 1024),
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None
        }
```

**Frontend Integration:**
- Endpoint: `GET /api/agents/health`
- WebSocket: `/ws/health` for real-time updates
- Display: Health dashboard with status indicators
- Alert system for degraded agents

**Implementation Steps:**
1. Add health endpoints to all agents (Week 1)
2. Implement ping mechanism (Week 1)
3. Add metrics collection (Week 2)
4. Create alerting system (Week 2)
5. Frontend dashboard (Week 3)

**Effort:** 3 weeks

---

### 1.4 asset_sourcing_agent.py
**Current State:** 116 lines, returns empty/mock assets
**Production Goal:** Real 3D asset search and integration

**Current Issues:**
- Empty `mock_assets = []`
- LLM intent detection but no real search
- No NASA/McMaster/GrabCAD integration

**Required APIs:**
1. NASA 3D Resources API
2. McMaster-Carr (web scraping or API)
3. GrabCAD API
4. Thingiverse API
5. Smithsonian 3D API

**Implementation:**
```python
import aiohttp
import asyncio
from typing import List, Dict, Any

class AssetSourcingAgent:
    def __init__(self):
        self.apis = {
            'nasa': NASA3DAPI(),
            'mcmaster': McMasterAPI(),
            'grabcad': GrabCADAPI(),
            'thingiverse': ThingiverseAPI()
        }
    
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get("query", "").lower()
        source = params.get("source", "").lower()
        
        # Search all sources concurrently
        search_tasks = []
        sources_to_search = [source] if source else self.apis.keys()
        
        for src in sources_to_search:
            if src in self.apis:
                search_tasks.append(self._search_source(src, query))
        
        results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Aggregate results
        all_assets = []
        for src, result in zip(sources_to_search, results):
            if not isinstance(result, Exception):
                all_assets.extend(result)
        
        # Rank by relevance
        ranked_assets = self._rank_by_relevance(all_assets, query)
        
        return {
            "assets": ranked_assets[:20],  # Top 20
            "count": len(ranked_assets),
            "sources_searched": list(sources_to_search)
        }
    
    async def _search_source(self, source: str, query: str) -> List[Dict]:
        """Search a specific source"""
        api = self.apis[source]
        return await api.search(query)

class NASA3DAPI:
    """NASA 3D Resources API client"""
    BASE_URL = "https://nasa3d.arc.nasa.gov/api"
    
    async def search(self, query: str) -> List[Dict]:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.BASE_URL}/search",
                params={"q": query, "format": "json"}
            ) as response:
                data = await response.json()
                return [
                    {
                        "id": item["id"],
                        "name": item["title"],
                        "source": "NASA 3D Resources",
                        "mesh_url": item["download_url"],
                        "thumbnail": item["thumbnail_url"],
                        "format": item.get("format", "unknown"),
                        "license": "Public Domain"
                    }
                    for item in data.get("results", [])
                ]

class McMasterAPI:
    """McMaster-Carr product search"""
    # Note: McMaster doesn't have a public API
    # May need web scraping or alternative approach
    
    async def search(self, query: str) -> List[Dict]:
        # Implementation options:
        # 1. Web scraping (fragile)
        # 2. Partner with McMaster for API access
        # 3. Use third-party service
        
        # For now, return empty - requires business development
        return []
```

**Research Required:**
- NASA 3D Resources API documentation
- GrabCAD API access (requires developer account)
- Thingiverse API (MakerBot)
- Smithsonian 3D API

**Frontend Integration:**
- Endpoint: `GET /api/components/catalog`
- Display: Asset browser with thumbnails
- Download and import functionality

**Implementation Steps:**
1. Research and sign up for APIs (Week 1-2)
2. Implement NASA API client (Week 2)
3. Implement GrabCAD client (Week 3)
4. Add search ranking (Week 3)
5. Frontend asset browser (Week 4)

**Effort:** 4 weeks
**Dependencies:** API keys, rate limits

---

### 1.5 pvc_agent.py
**Current State:** 68 lines, basic version control placeholder
**Production Goal:** Full Git-based version control

**Current Issues:**
- No actual git operations
- Returns success without doing anything

**Required Functionality:**
1. Git repository management
2. Branch creation and switching
3. Commit with metadata
4. Diff visualization
5. Merge conflict resolution
6. Version tagging

**Dependencies:**
- `gitpython` library
- File system access
- Async git operations

**Implementation:**
```python
from git import Repo, GitCommandError
from git.diff import Diff
import os
from pathlib import Path

class PVCAgent:
    """Project Version Control Agent"""
    
    def __init__(self, projects_dir: str = "projects"):
        self.projects_dir = Path(projects_dir)
        
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        operation = params.get("operation", "status")
        project_id = params.get("project_id")
        
        if not project_id:
            return {"status": "error", "error": "project_id required"}
        
        project_path = self.projects_dir / project_id
        
        operations = {
            "init": self._init_repo,
            "commit": self._commit,
            "branch": self._create_branch,
            "checkout": self._checkout,
            "diff": self._get_diff,
            "log": self._get_log,
            "status": self._get_status,
            "merge": self._merge
        }
        
        if operation not in operations:
            return {"status": "error", "error": f"Unknown operation: {operation}"}
        
        return await operations[operation](project_path, params)
    
    async def _init_repo(self, path: Path, params: Dict) -> Dict:
        """Initialize git repository"""
        try:
            repo = Repo.init(path)
            # Create initial commit
            repo.index.add(["*"])
            repo.index.commit("Initial commit by PVC Agent")
            
            return {
                "status": "initialized",
                "path": str(path),
                "default_branch": repo.active_branch.name
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _commit(self, path: Path, params: Dict) -> Dict:
        """Create commit"""
        try:
            repo = Repo(path)
            
            # Stage changes
            repo.index.add(["*"])
            
            # Create commit
            message = params.get("message", "Update")
            author = params.get("author", "BRICK Agent <agent@brick.os>")
            
            commit = repo.index.commit(message, author=author)
            
            return {
                "status": "committed",
                "commit_hash": commit.hexsha[:8],
                "message": commit.message,
                "author": str(commit.author),
                "timestamp": commit.committed_datetime.isoformat()
            }
        except GitCommandError as e:
            return {"status": "error", "error": str(e)}
```

**Frontend Integration:**
- Endpoint: `POST /api/projects/{id}/version/{operation}`
- Display: Git history visualization
- Branch management UI
- Diff viewer

**Effort:** 2 weeks

---

### 1.6 remote_agent.py
**Current State:** 66 lines, session management placeholder
**Production Goal:** Remote collaboration and session management

**Current Issues:**
- No real session persistence
- No user authentication
- No real-time collaboration

**Required Functionality:**
1. User session management
2. WebSocket-based real-time collaboration
3. Cursor tracking
4. Change synchronization
5. Conflict resolution
6. Presence indicators

**Dependencies:**
- `fastapi.WebSocket`
- `redis` for session store
- `jwt` for authentication

**Implementation:**
```python
from fastapi import WebSocket, WebSocketDisconnect
import redis
import json
from typing import Dict, Set
import asyncio

class RemoteAgent:
    """Remote collaboration agent"""
    
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.active_sessions: Dict[str, Set[WebSocket]] = {}
        
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        operation = params.get("operation")
        
        if operation == "create_session":
            return await self._create_session(params)
        elif operation == "join_session":
            return await self._join_session(params)
        elif operation == "list_sessions":
            return await self._list_sessions(params)
        
        return {"status": "error", "error": "Unknown operation"}
    
    async def _create_session(self, params: Dict) -> Dict:
        """Create new collaboration session"""
        session_id = self._generate_session_id()
        user_id = params.get("user_id")
        project_id = params.get("project_id")
        
        session_data = {
            "session_id": session_id,
            "project_id": project_id,
            "creator": user_id,
            "participants": [user_id],
            "created_at": datetime.utcnow().isoformat(),
            "status": "active"
        }
        
        # Store in Redis
        self.redis_client.setex(
            f"session:{session_id}",
            timedelta(hours=24),
            json.dumps(session_data)
        )
        
        return {
            "status": "created",
            "session_id": session_id,
            "invite_url": f"/join/{session_id}"
        }
    
    async def handle_websocket(self, websocket: WebSocket, session_id: str):
        """Handle WebSocket connection for real-time collaboration"""
        await websocket.accept()
        
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = set()
        
        self.active_sessions[session_id].add(websocket)
        
        try:
            while True:
                message = await websocket.receive_json()
                
                # Broadcast to all participants
                await self._broadcast(session_id, message, exclude=websocket)
                
        except WebSocketDisconnect:
            self.active_sessions[session_id].remove(websocket)
    
    async def _broadcast(self, session_id: str, message: Dict, exclude: WebSocket = None):
        """Broadcast message to all participants"""
        if session_id in self.active_sessions:
            for ws in self.active_sessions[session_id]:
                if ws != exclude:
                    await ws.send_json(message)
```

**Frontend Integration:**
- WebSocket: `/ws/session/{session_id}`
- Display: Multi-cursor support
- Presence indicators
- Real-time change sync

**Effort:** 4 weeks

---

### 1.7 user_agent.py
**Current State:** 87 lines, basic profile management
**Production Goal:** Full user management with Supabase Auth

**Current Issues:**
- File-based storage (not scalable)
- No authentication
- No permission system

**Required Functionality:**
1. User authentication (OAuth, email/password)
2. Profile management
3. Permission/role system
4. Team/organization support
5. Activity tracking

**Dependencies:**
- `supabase-py`
- `gotrue` for auth
- JWT handling

**Implementation:**
```python
from supabase import create_client
import os
from typing import Optional

class UserAgent:
    """User management with Supabase Auth"""
    
    def __init__(self):
        self.supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )
    
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        operation = params.get("operation")
        
        operations = {
            "signup": self._signup,
            "login": self._login,
            "logout": self._logout,
            "get_profile": self._get_profile,
            "update_profile": self._update_profile,
            "reset_password": self._reset_password
        }
        
        if operation in operations:
            return await operations[operation](params)
        
        return {"status": "error", "error": "Unknown operation"}
    
    async def _signup(self, params: Dict) -> Dict:
        """Create new user account"""
        email = params.get("email")
        password = params.get("password")
        
        try:
            result = self.supabase.auth.sign_up({
                "email": email,
                "password": password
            })
            
            # Create profile entry
            self.supabase.table("profiles").insert({
                "id": result.user.id,
                "email": email,
                "created_at": "now()"
            }).execute()
            
            return {
                "status": "success",
                "user_id": result.user.id,
                "email": result.user.email
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
```

**Frontend Integration:**
- Endpoints: Auth standard OAuth flow
- Display: Login/Signup forms
- Profile management page

**Effort:** 2 weeks

---

## Category 2: Framework Only (8 agents)

These agents have architecture but no trained ML models.

### 2.1 control_agent.py
**Current State:** 183 lines, LQR works, RL policy untrained
**Production Goal:** RL-MPC with trained policy

**Current Issues:**
- RL policy loader but no trained weights
- Mock disturbance estimation
- Falls back to LQR

**Required ML:**
1. Train PPO/TD3 policy for control
2. Implement differentiable MPC layer
3. Add online adaptation

**Research:**
- Arroyo et al. (2022) RL-MPC
- Amos et al. (2018) Differentiable MPC

**Implementation:**
```python
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO

class ControlAgent:
    def __init__(self):
        self.lqr = LQRController()
        self.rl_policy = self._load_rl_policy()
        self.mpc = DifferentiableMPC(horizon=10)
        
    def _load_rl_policy(self):
        """Load trained PPO policy"""
        try:
            model = PPO.load("models/control_policy_ppo.zip")
            return model
        except:
            return None
    
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        mode = params.get("control_mode", "RL-MPC")
        state = params.get("state")
        target = params.get("target")
        
        if mode == "RL-MPC" and self.rl_policy:
            # RL-MPC hybrid
            # 1. MPC provides baseline
            u_mpc = self.mpc.solve(state, target)
            
            # 2. RL fine-tunes
            obs = self._construct_observation(state, target)
            u_rl, _ = self.rl_policy.predict(obs, deterministic=False)
            
            # 3. Combine
            alpha = 0.8  # Blend factor
            action = alpha * u_mpc + (1 - alpha) * u_rl
            
            return {
                "control_signal": action.tolist(),
                "method": "RL-MPC",
                "mpc_component": u_mpc.tolist(),
                "rl_component": u_rl.tolist()
            }
        else:
            # Fallback to LQR
            return self.lqr.control(state, target)
```

**Training Required:**
- Environment: Hover dynamics simulation
- Algorithm: PPO or TD3
- Training time: ~8 hours on GPU
- Episodes: 1M+

**Effort:** 4 weeks (2 weeks training)

---

### 2.2 diagnostic_agent.py
**Current State:** 111 lines, regex fallback, untrained surrogate
**Production Goal:** ML-based log analysis

**Current Issues:**
- Surrogate not trained
- Regex fallback only
- No real diagnosis

**Required ML:**
1. LogBERT or similar for log analysis
2. Anomaly detection
3. Root cause classification

**Implementation:**
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class DiagnosticAgent:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "models/diagnostic_model"
        )
        self.model.eval()
        
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        logs = params.get("logs", [])
        
        # Concatenate logs
        log_text = "\n".join(logs)
        
        # Tokenize
        inputs = self.tokenizer(
            log_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probs, dim=-1)
            confidence = probs[0][prediction].item()
        
        diagnoses = ["Network", "Memory", "Configuration", "Logic", "Security"]
        diagnosis = diagnoses[prediction.item()]
        
        return {
            "diagnosis": diagnosis,
            "confidence": confidence,
            "recommended_actions": self._get_actions(diagnosis)
        }
```

**Training Required:**
- Dataset: System logs with labeled issues
- Model: Fine-tuned BERT/CodeBERT
- Training time: 4-6 hours

**Effort:** 3 weeks (1 week data collection, 1 week training)

---

### 2.3 template_design_agent.py
**Current State:** 163 lines, untrained surrogate
**Production Goal:** ML-optimized template generation

**Required ML:**
1. Train surrogate on template performance
2. Add generative models for template variation
3. Implement quality prediction

**Implementation:**
```python
import torch
import torch.nn as nn

class TemplateSurrogate(nn.Module):
    """Predicts manufacturability and performance from template params"""
    
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # [manufacturability, performance]
        )
    
    def forward(self, x):
        return torch.sigmoid(self.network(x))

class TemplateDesignAgent:
    def __init__(self):
        self.surrogate = TemplateSurrogate()
        self.surrogate.load_state_dict(
            torch.load("models/template_surrogate.pt")
        )
        self.surrogate.eval()
```

**Training Required:**
- Dataset: Template variations with performance metrics
- Training time: 2-3 hours

**Effort:** 2 weeks

---

### 2.4 stt_agent.py (Speech-to-Text)
**Current State:** 63 lines, placeholder
**Production Goal:** Production STT with Whisper

**Required:**
- OpenAI Whisper API or local model
- Audio preprocessing
- Real-time streaming

**Implementation:**
```python
import whisper
import numpy as np

class STTAgent:
    def __init__(self):
        # Load Whisper model
        self.model = whisper.load_model("base")  # or "small", "medium", "large"
    
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        audio_data = params.get("audio_data")
        
        # Transcribe
        result = self.model.transcribe(audio_data)
        
        return {
            "text": result["text"],
            "language": result.get("language", "en"),
            "segments": result.get("segments", [])
        }
```

**Dependencies:**
- `openai-whisper`
- `ffmpeg` for audio processing

**Effort:** 1 week

---

### 2.5 vhil_agent.py (Virtual Hardware-in-the-Loop)
**Current State:** 192 lines, mock sensors, basic physics
**Production Goal:** Full vHIL with accurate sensor emulation

**Current Issues:**
- Random sensor noise
- Simplified physics
- No hardware-in-the-loop capability

**Required:**
1. Accurate sensor models (IMU, GPS, LIDAR, Camera)
2. Real physics simulation
3. Hardware interface abstraction
4. Timing-accurate simulation

**Implementation:**
```python
class VhilAgent:
    def __init__(self):
        self.physics = PhysicsEngine()
        self.sensor_models = {
            "imu": IMUModel(noise_density=0.01),
            "gps": GPSModel(hdop=0.8),
            "lidar": LidarModel(range_max=100, resolution=0.25),
            "camera": CameraModel(resolution=(640, 480), fps=30)
        }
    
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        dt = params.get("dt", 0.01)
        state = params.get("state", {})
        
        # Step physics
        new_state = self.physics.step(state, dt)
        
        # Generate sensor readings
        sensor_data = {}
        for sensor_type in params.get("sensors", []):
            if sensor_type in self.sensor_models:
                sensor_data[sensor_type] = self.sensor_models[sensor_type].read(new_state)
        
        return {
            "sensor_data": sensor_data,
            "state": new_state,
            "timing_ms": dt * 1000
        }
```

**Effort:** 3 weeks

---

## Category 3: Partial Implementation (10 agents)

### 3.1 visual_validator_agent.py
**Current State:** 110 lines, basic trimesh checks
**Production Goal:** Comprehensive visual validation

**Current Functionality:**
- Watertightness check ✓
- Inverted normals check ✓
- Degenerate face detection ✓

**Missing:**
- Render quality assessment
- Scene composition analysis
- Lighting validation
- Texture resolution check

**Implementation:**
```python
import trimesh
import numpy as np
from PIL import Image

class VisualValidatorAgent:
    def __init__(self):
        self.scoring = {
            "watertightness_penalty": 0.3,
            "inverted_normals_penalty": 0.5,
            "degenerate_face_penalty": 0.1,
            "poor_lighting_penalty": 0.2,
            "low_texture_penalty": 0.15
        }
    
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Existing geometry checks
        geometry_result = self._check_geometry(params.get("mesh_path"))
        
        # New: Render quality check
        render_result = self._check_render_quality(params.get("render_path"))
        
        # New: Scene composition
        composition_result = self._check_composition(params.get("scene_metadata"))
        
        # Aggregate scores
        total_score = 1.0
        for result in [geometry_result, render_result, composition_result]:
            total_score -= result.get("penalty", 0)
        
        return {
            "is_valid": total_score > 0.7,
            "quality_score": max(0, total_score),
            "geometry": geometry_result,
            "render": render_result,
            "composition": composition_result
        }
```

**Effort:** 1 week

---

### 3.2 network_agent.py
**Current State:** 93 lines, latency formula only
**Production Goal:** Full network topology analysis

**Current:** Simple formula: `delay = base * hops + load^1.5 * factor`

**Required:**
1. Graph neural network for topology analysis
2. Real traffic simulation
3. ML-based latency prediction
4. Bandwidth estimation

**Implementation:**
```python
import networkx as nx
import torch
import torch.nn as nn

class NetworkAgent:
    def __init__(self):
        # GNN for topology analysis
        self.gnn = NetworkGNN()
        self.gnn.load_state_dict(torch.load("models/network_gnn.pt"))
    
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        topology = params.get("topology", {})
        traffic = params.get("traffic_flows", [])
        
        # Build graph
        G = self._build_graph(topology)
        
        # GNN prediction
        latency_predictions = self._predict_with_gnn(G, traffic)
        
        return {
            "network_health": self._assess_health(latency_predictions),
            "flow_predictions": latency_predictions,
            "bottlenecks": self._identify_bottlenecks(G, traffic)
        }
```

**Effort:** 3 weeks

---

### 3.3 sustainability_agent.py
**Current State:** 160 lines, database lookup only
**Production Goal:** Full LCA (Life Cycle Assessment)

**Current:** Simple carbon factor lookup

**Required:**
1. Full LCA calculation ( cradle-to-grave)
2. Material extraction impacts
3. Manufacturing energy
4. Transportation emissions
5. End-of-life scenarios

**Implementation:**
```python
class SustainabilityAgent:
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        material = params.get("material")
        mass_kg = params.get("mass_kg")
        process = params.get("process_type", "cnc_milling")
        
        # Material extraction
        extraction = await self._calculate_extraction(material, mass_kg)
        
        # Manufacturing
        manufacturing = await self._calculate_manufacturing(process, mass_kg)
        
        # Use phase (if applicable)
        use_phase = await self._calculate_use_phase(params)
        
        # End of life
        eol = await self._calculate_end_of_life(material, mass_kg, params.get("eol_scenario", "recycle"))
        
        total = extraction + manufacturing + use_phase + eol
        
        return {
            "co2_emissions_kg": round(total, 2),
            "breakdown": {
                "extraction": extraction,
                "manufacturing": manufacturing,
                "use_phase": use_phase,
                "end_of_life": eol
            },
            "rating": self._get_rating(total, mass_kg)
        }
```

**Research:**
- Ecoinvent database
- GaBi LCA software
- OpenLCA

**Effort:** 3 weeks

---

### 3.4 verification_agent.py
**Current State:** 165 lines, mock verification
**Production Goal:** Real requirement verification

**Current:** Checks if "fail" is in requirement text

**Required:**
1. Requirement parsing and understanding
2. Test case generation
3. Automated test execution
4. Traceability matrix

**Implementation:**
```python
class VerificationAgent:
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        requirements = params.get("requirements", [])
        design = params.get("design", {})
        
        results = []
        for req in requirements:
            # Parse requirement
            parsed = self._parse_requirement(req)
            
            # Generate test case
            test_case = self._generate_test(parsed, design)
            
            # Execute test
            test_result = await self._execute_test(test_case)
            
            results.append({
                "requirement": req,
                "parsed": parsed,
                "test_result": test_result,
                "passed": test_result["passed"]
            })
        
        pass_rate = sum(1 for r in results if r["passed"]) / len(results)
        
        return {
            "passed": pass_rate >= 0.95,
            "pass_rate": pass_rate,
            "results": results,
            "traceability_matrix": self._generate_matrix(results)
        }
```

**Effort:** 3 weeks

---

### 3.5 lattice_synthesis_agent.py
**Current State:** 89 lines, SDF generation only
**Production Goal:** Full lattice design with GNoME integration

**Current:** Basic TPMS SDF generation

**Required:**
1. GNoME database integration
2. Property prediction for lattices
3. Optimization algorithms
4. Manufacturing constraint checking

**Implementation:**
```python
class LatticeSynthesisAgent:
    def __init__(self):
        self.gnome_client = GNoMEClient()
        self.property_predictor = LatticePropertyPredictor()
    
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        lattice_type = params.get("type", "GYROID")
        target_properties = params.get("target_properties", {})
        
        # Query GNoME for similar structures
        similar = await self.gnome_client.query_similar(target_properties)
        
        # Optimize lattice parameters
        optimized = self._optimize_parameters(
            lattice_type, 
            target_properties,
            similar
        )
        
        # Generate geometry
        geometry = self._generate_lattice(optimized)
        
        # Predict properties
        predicted = self.property_predictor.predict(geometry)
        
        return {
            "lattice_type": lattice_type,
            "parameters": optimized,
            "geometry": geometry,
            "predicted_properties": predicted,
            "gnome_matches": similar
        }
```

**Research:**
- GNoME database access
- TPMS optimization literature

**Effort:** 4 weeks

---

### 3.6 standards_agent.py
**Current State:** 92 lines, RAG with fallback to mock
**Production Goal:** Full standards compliance checking

**Current:** Uses StandardsRetriever but falls back to mock

**Required:**
1. Complete standards database (ISO, ASTM, ASME, etc.)
2. RAG with proper embeddings
3. Compliance checking algorithms
4. Version tracking

**Implementation:**
```python
class StandardsAgent:
    def __init__(self):
        self.retriever = StandardsRetriever()
        self.compliance_checker = ComplianceChecker()
    
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get("query")
        design = params.get("design", {})
        standard_ids = params.get("standards", [])
        
        # Retrieve relevant standards
        standards = await self.retriever.retrieve(query)
        
        # Check compliance
        compliance_results = []
        for standard in standards:
            result = await self.compliance_checker.check(design, standard)
            compliance_results.append(result)
        
        return {
            "standards_found": len(standards),
            "compliance_results": compliance_results,
            "overall_compliance": all(r["compliant"] for r in compliance_results),
            "gaps": self._identify_gaps(compliance_results)
        }
```

**Effort:** 3 weeks

---

### 3.7 slicer_agent.py
**Current State:** 97 lines, basic time estimation
**Production Goal:** Full G-code generation with Cura/PrusaSlicer integration

**Current:** Simple heuristic: `time = volume * infill / (layer_height * 5)`

**Required:**
1. Real slicing engine (CuraEngine or PrusaSlicer)
2. G-code generation
3. Support structure calculation
4. Time/weight estimation
5. Cost calculation

**Implementation:**
```python
import subprocess
import tempfile
import os

class SlicerAgent:
    def __init__(self):
        self.slicer_path = "/usr/bin/cura-engine"  # or PrusaSlicer
        self.profiles_dir = "config/slicer_profiles"
    
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        mesh_path = params.get("mesh_path")
        profile = params.get("profile", "standard")
        
        # Export mesh to temp file
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            temp_mesh = f.name
        
        # Run slicer
        gcode_path = temp_mesh.replace(".stl", ".gcode")
        
        cmd = [
            self.slicer_path,
            "-i", temp_mesh,
            "-o", gcode_path,
            "-p", f"{self.profiles_dir}/{profile}.ini"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return {"status": "error", "error": result.stderr}
        
        # Parse G-code
        analysis = self._analyze_gcode(gcode_path)
        
        return {
            "status": "sliced",
            "gcode_path": gcode_path,
            "print_time_min": analysis["time_min"],
            "filament_used_g": analysis["filament_g"],
            "layer_count": analysis["layers"],
            "support_volume_cm3": analysis["support_volume"]
        }
```

**Dependencies:**
- CuraEngine or PrusaSlicer installed
- Slicing profiles

**Effort:** 2 weeks

---

### 3.8 replicator_mixin.py
**Current State:** 172 lines, basic self-replication concept
**Production Goal:** Von Neumann probe simulation

**Current:** Mock replication logic

**Required:**
1. Resource harvesting simulation
2. Manufacturing capability assessment
3. Child probe generation
4. Evolution/mutation mechanisms

**Effort:** 4 weeks (research project)

---

### 3.9 von_neumann_agent.py
**Current State:** 169 lines, self-replication simulation
**Production Goal:** Full Von Neumann probe simulation

**Current:** Mock replication

**Required:**
1. Environment modeling
2. Resource detection
3. Manufacturing planning
4. Child probe deployment

**Effort:** 4 weeks (research project)

---

### 3.10 explainable_agent.py
**Current State:** 78 lines, XAI wrapper
**Production Goal:** Full explainability framework

**Current:** Basic thought injection

**Required:**
1. LIME/SHAP integration
2. Attention visualization
3. Decision tree extraction
4. Counterfactual explanations

**Implementation:**
```python
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer

class ExplainableAgent:
    def __init__(self, agent, explainer_type="shap"):
        self.agent = agent
        self.explainer_type = explainer_type
        
        if explainer_type == "shap":
            self.explainer = shap.Explainer(agent.model)
        elif explainer_type == "lime":
            self.explainer = LimeTabularExplainer(...)
    
    async def run(self, *args, **kwargs):
        # Run agent
        result = await self.agent.run(*args, **kwargs)
        
        # Generate explanation
        if self.explainer_type == "shap":
            explanation = self.explainer(*args)
        else:
            explanation = self.explainer.explain_instance(...)
        
        return {
            "result": result,
            "explanation": explanation,
            "visualization": self._create_viz(explanation)
        }
```

**Effort:** 2 weeks

---

## Frontend-Backend Integration

### Current Gaps

| Gap | Description | Impact |
|-----|-------------|--------|
| Missing Endpoints | Many agents don't have API endpoints | Frontend can't use them |
| No WebSockets | Limited real-time updates | UI feels static |
| No Progress Indicators | Long operations show no progress | Poor UX |
| Missing Error Handling | Errors not properly surfaced | Users confused |

### Required API Endpoints

```python
# For each agent, add:

@app.post("/api/agents/{agent_name}/run")
async def run_agent(agent_name: str, params: Dict):
    agent = get_agent(agent_name)
    result = await agent.run(params)
    return result

@app.get("/api/agents/{agent_name}/status")
async def get_agent_status(agent_name: str):
    agent = get_agent(agent_name)
    return {"status": agent.get_status()}

@app.websocket("/ws/agents/{agent_name}")
async def agent_websocket(websocket: WebSocket, agent_name: str):
    # Real-time updates
    pass
```

### Frontend Components Needed

```typescript
// AgentPanel.tsx - Generic agent control panel
interface AgentPanelProps {
    agentName: string;
    parameters: ParameterSchema;
    onResult: (result: any) => void;
}

// AgentMonitor.tsx - Real-time agent status
interface AgentMonitorProps {
    agentNames: string[];
}

// ResultsViewer.tsx - Display agent results
interface ResultsViewerProps {
    results: AgentResult[];
}
```

---

## Implementation Timeline

### Phase 1: Foundation (Weeks 1-2)
- [ ] Delete generic_agent.py
- [ ] Add health endpoints to all agents
- [ ] Create base agent class with common functionality
- [ ] Implement proper error handling

### Phase 2: Pure Stubs (Weeks 3-6)
- [ ] Performance Agent (3 weeks)
- [ ] Doctor Agent (1 week)
- [ ] PVC Agent (1 week)
- [ ] User Agent (1 week)
- [ ] Remote Agent (2 weeks)

### Phase 3: ML Framework Agents (Weeks 7-12)
- [ ] Control Agent - Train RL policy (4 weeks)
- [ ] Diagnostic Agent - Train log model (3 weeks)
- [ ] Template Design Agent - Train surrogate (2 weeks)
- [ ] STT Agent - Integrate Whisper (1 week)
- [ ] vHIL Agent - Accurate sensors (3 weeks)

### Phase 4: Partial Implementation (Weeks 13-20)
- [ ] Visual Validator (1 week)
- [ ] Network Agent (3 weeks)
- [ ] Sustainability Agent (3 weeks)
- [ ] Verification Agent (3 weeks)
- [ ] Lattice Synthesis (4 weeks)
- [ ] Standards Agent (3 weeks)
- [ ] Slicer Agent (2 weeks)
- [ ] Explainable Agent (2 weeks)

### Phase 5: Research Projects (Ongoing)
- [ ] Replicator Mixin (4 weeks)
- [ ] Von Neumann Agent (4 weeks)

---

## Total Effort Estimate

| Category | Agents | Effort (weeks) |
|----------|--------|----------------|
| Pure Stubs | 7 | 14 |
| ML Framework | 5 | 13 |
| Partial Implementation | 10 | 26 |
| **TOTAL** | **22** | **53 weeks (~1 year)** |

**With 3 developers working in parallel: ~18 weeks (4.5 months)**

---

## Dependencies Summary

### APIs and Services
- NASA 3D Resources API
- GrabCAD API
- McMaster-Carr (scraping or partnership)
- Supabase Auth
- Redis (for sessions)

### ML Models to Train
- Control policy (PPO/TD3)
- Diagnostic model (BERT)
- Template surrogate (MLP)
- Network GNN
- Lattice property predictor

### Software Dependencies
- CalculiX or FEniCS (FEA)
- CuraEngine or PrusaSlicer
- OpenAI Whisper
- gitpython
- psutil
- redis-py

---

## Success Criteria

By the end of implementation:
1. All 25 stub agents are production-ready
2. All agents have API endpoints
3. Frontend integration complete
4. Documentation updated
5. Tests passing (>80% coverage)
6. Performance benchmarks defined and met
