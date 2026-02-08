# BRICK OS - Implementation Plan

> **From Analysis to Production-Ready System**
> **Duration**: 8 Weeks
> **Team Size**: 2-3 Engineers (Backend-heavy)

---

## 🗓️ Phase Overview

| Phase | Duration | Focus | Deliverable |
|-------|----------|-------|-------------|
| **Phase 1** | Week 1 | Critical Fixes | Stable backend, no silent failures |
| **Phase 2** | Week 2 | Frontend MVP | Working UI with 3D viewer |
| **Phase 3** | Week 3 | Backend Hardening | Docker, async standardization |
| **Phase 4** | Week 4 | Testing & CI/CD | Automated testing pipeline |
| **Phase 5** | Weeks 5-6 | Advanced Features | WebSockets, monitoring |
| **Phase 6** | Weeks 7-8 | Polish & Documentation | Production deployment |

---

## PHASE 1: Critical Fixes (Week 1)

### Day 1-2: Fix Silent Agent Loading (C1)

**Task**: `backend/agent_registry.py` - Fail loudly on agent load failure

**Implementation**:
```python
# backend/agent_registry.py

def _lazy_load(self, name: str) -> Any:  # Changed from Optional[Any]
    """
    Load agent by name with strict error handling.
    Raises RuntimeError on failure - never returns None.
    """
    try:
        module_path, class_name = self.AVAILABLE_AGENTS[name]
        module = importlib.import_module(module_path)
        agent_class = getattr(module, class_name)
        
        # Instantiate with XAI wrapper for observability
        instance = create_xai_wrapper(agent_class())
        self._agents[name] = instance
        
        logger.info(f"✅ Lazy loaded agent: {name}")
        return instance
        
    except (ImportError, AttributeError, KeyError) as e:
        logger.error(f"❌ Failed to load agent {name}: {e}")
        raise RuntimeError(
            f"Critical agent '{name}' failed to load. "
            f"Module: {self.AVAILABLE_AGENTS.get(name, 'UNKNOWN')}. "
            f"Error: {e}"
        ) from e
    except Exception as e:
        logger.critical(f"💥 Unexpected error loading agent {name}: {e}")
        raise RuntimeError(f"Unexpected failure loading agent {name}: {e}") from e
```

**Testing**:
```python
# tests/unit/test_agent_registry.py

def test_lazy_load_failure_raises():
    """Verify agent load failure raises, not returns None."""
    registry = GlobalAgentRegistry()
    
    with pytest.raises(RuntimeError) as exc_info:
        registry._lazy_load("NonExistentAgent")
    
    assert "failed to load" in str(exc_info.value)

def test_successful_load():
    """Verify successful agent load returns instance."""
    registry = GlobalAgentRegistry()
    agent = registry.get_agent("MaterialAgent")
    
    assert agent is not None
    assert hasattr(agent, 'run')
```

**Verification**:
```bash
python -c "from backend.agent_registry import registry; registry.initialize(); print(registry.get_agent('NonExistent'))"
# Should raise RuntimeError, not return None
```

---

### Day 2-3: Fix Async Context Violation (C2)

**Task**: Convert `GeometryAgent.run()` to async

**Implementation Steps**:

1. **Update base interface** (create if not exists):
```python
# backend/agents/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAgent(ABC):
    """Base class for all BRICK OS agents."""
    
    @abstractmethod
    async def run(self, params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute agent logic.
        
        All agents must be async to support concurrent execution
        and non-blocking I/O (LLM calls, file operations).
        """
        pass
```

2. **Update GeometryAgent**:
```python
# backend/agents/geometry_agent.py

from agents.base import BaseAgent

class GeometryAgent(BaseAgent):
    
    async def run(
        self, 
        params: Dict[str, Any], 
        intent: str, 
        environment: Dict[str, Any] = None, 
        ldp_instructions: List[Dict] = None
    ) -> Dict[str, Any]:
        """Async geometry generation."""
        
        # ... existing setup code ...
        
        # Async compile - no more asyncio.run()!
        result = await self.engine.compile(geometry_tree, format="glb")
        
        # Async validation
        manifold_agent = ManifoldAgent()
        manifold_res = await manifold_agent.run({"geometry_tree": geometry_tree})
        
        physics_validation = await validate_geometry_physics_async(
            self.physics, geometry_tree, material
        )
        
        return {
            "kcl_code": kcl_code,
            "gltf_data": result.payload,
            "validation_logs": manifold_res.get("logs", []),
            "physics_validation": physics_validation,
        }
```

3. **Update orchestrator node**:
```python
# backend/orchestrator.py (geometry_node)

async def geometry_node(state: AgentState) -> Dict[str, Any]:
    agent = registry.get_agent("GeometryAgent")
    
    # No more hasattr check needed - all agents are async
    result = await agent.run(
        params=state.get("design_parameters", {}),
        intent=state.get("user_intent", ""),
        environment=state.get("environment", {}),
        ldp_instructions=state.get("ldp_instructions", [])
    )
    
    return {
        "kcl_code": result["kcl_code"],
        "geometry_tree": result["geometry_tree"],
        "gltf_data": result["gltf_data"],
    }
```

4. **Migration script** for other agents:
```python
# scripts/migrate_agents_to_async.py

import ast
import os

def migrate_agent(filepath: str):
    """Convert sync agent to async."""
    with open(filepath, 'r') as f:
        tree = ast.parse(f.read())
    
    # Find run method, add async keyword
    # Add await to any LLM calls
    # Update return type hints
    
    # ... migration logic ...
```

**Verification**:
```bash
# Test no asyncio errors
python -c "
import asyncio
from backend.agents.geometry_agent import GeometryAgent
agent = GeometryAgent()
result = asyncio.run(agent.run({'test': True}, 'test intent'))
print('✅ Async geometry agent works')
"
```

---

### Day 3-4: Implement Redis Session Persistence (C3)

**Task**: Replace `InMemorySessionStore` with Redis

**Implementation**:

1. **Add dependency**:
```bash
pip install aioredis>=2.0
```

2. **Create Redis store**:
```python
# backend/session_store.py

import json
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import aioredis
from agents.conversational_agent import SessionStore

class RedisSessionStore(SessionStore):
    """
    Production-grade session storage with Redis.
    Supports TTL, serialization, and concurrent access.
    """
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.getenv(
            "REDIS_URL", 
            "redis://localhost:6379/0"
        )
        self._redis: Optional[aioredis.Redis] = None
        self._lock = asyncio.Lock()
    
    async def _get_redis(self) -> aioredis.Redis:
        """Lazy connection initialization."""
        if self._redis is None:
            self._redis = await aioredis.from_url(
                self.redis_url,
                encoding='utf-8',
                decode_responses=True
            )
        return self._redis
    
    async def get_discovery_state(self, session_id: str) -> Optional[Dict]:
        """Retrieve session state from Redis."""
        redis = await self._get_redis()
        key = f"brick:session:{session_id}"
        
        data = await redis.get(key)
        if data is None:
            return None
        
        # Extend TTL on access (session still active)
        await redis.expire(key, 3600)
        
        return json.loads(data)
    
    async def set_discovery_state(
        self, 
        session_id: str, 
        state: Dict, 
        ttl: int = 3600
    ):
        """Persist session state to Redis with TTL."""
        redis = await self._get_redis()
        key = f"brick:session:{session_id}"
        
        # Serialize with timestamp
        state_with_meta = {
            **state,
            "_meta": {
                "updated_at": datetime.utcnow().isoformat(),
                "ttl": ttl
            }
        }
        
        await redis.setex(
            key,
            ttl,
            json.dumps(state_with_meta, default=str)
        )
    
    async def delete_discovery_state(self, session_id: str):
        """Delete session state."""
        redis = await self._get_redis()
        await redis.delete(f"brick:session:{session_id}")
    
    async def health_check(self) -> bool:
        """Verify Redis connectivity."""
        try:
            redis = await self._get_redis()
            await redis.ping()
            return True
        except Exception:
            return False
```

3. **Update ConversationalAgent**:
```python
# backend/agents/conversational_agent.py

class ConversationalAgent(BaseAgent):
    def __init__(self):
        # Use Redis in production, fallback to in-memory for dev
        if os.getenv("REDIS_URL"):
            self.discovery = DiscoveryManager(
                session_store=RedisSessionStore()
            )
        else:
            logger.warning("Using InMemorySessionStore - data will not persist!")
            self.discovery = DiscoveryManager()
```

4. **Docker Compose update**:
```yaml
# docker-compose.yml
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    
  backend:
    build: .
    environment:
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
      
volumes:
  redis_data:
```

**Verification**:
```bash
# Start Redis
docker-compose up -d redis

# Test persistence
python -c "
import asyncio
from backend.session_store import RedisSessionStore

async def test():
    store = RedisSessionStore()
    await store.set_discovery_state('test_session', {'mission': 'mars_drone'})
    
    # Restart "server" - new store instance
    store2 = RedisSessionStore()
    state = await store2.get_discovery_state('test_session')
    assert state['mission'] == 'mars_drone'
    print('✅ Redis persistence works')

asyncio.run(test())
"
```

---

### Day 4-5: Fix Circular Import (C4)

**Task**: Break main.py ↔ orchestrator.py circular dependency

**Implementation**:

1. **Create XAI stream module**:
```python
# backend/xai_stream.py
"""
Explainable AI stream - agent thought broadcasting.
Centralized to avoid circular imports.
"""

from collections import deque
from datetime import datetime
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Global thought buffer
THOUGHT_STREAM = deque(maxlen=50)

def inject_thought(agent_name: str, thought_text: str) -> None:
    """
    Inject an agent's thought into the global stream.
    Called by agents during execution.
    """
    if not thought_text:
        return
    
    timestamp = datetime.now().isoformat()
    thought = {
        "agent": agent_name,
        "text": thought_text,
        "timestamp": timestamp
    }
    
    THOUGHT_STREAM.append(thought)
    logger.debug(f"💭 [{agent_name}] {thought_text[:100]}...")

def get_thoughts() -> list:
    """Get and clear thought buffer (destructive read)."""
    thoughts = list(THOUGHT_STREAM)
    THOUGHT_STREAM.clear()
    return thoughts

def peek_thoughts() -> list:
    """Get thoughts without clearing (non-destructive)."""
    return list(THOUGHT_STREAM)
```

2. **Update main.py**:
```python
# backend/main.py

# Remove: from orchestrator import inject_thought
# Add:
from xai_stream import inject_thought, get_thoughts

@app.get("/api/agents/thoughts")
async def get_passive_thoughts():
    """Polls for recent agent thoughts."""
    return {"thoughts": get_thoughts()}
```

3. **Update orchestrator.py**:
```python
# backend/orchestrator.py

# Remove: from main import inject_thought
# Add:
from xai_stream import inject_thought

# All nodes now use:
inject_thought("PhysicsAgent", "Calculating thermal load...")
```

4. **Verify no circular imports**:
```bash
python -c "
import backend.main
import backend.orchestrator
print('✅ No circular imports detected')
"

# Use import linter
pip install import-linter
lint-imports
```

---

### Day 5: Fix State Mutation Bug (C5)

**Task**: Immutable state updates in physics_node

**Implementation**:

```python
# backend/orchestrator.py (physics_node)

async def physics_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes physics simulation with immutable state updates.
    """
    # ... existing setup ...
    
    # Start with base flags - DO NOT MUTATE
    base_flags = state.get("validation_flags", {})
    
    # Create NEW flags dict
    flags = {
        "physics_safe": True,
        "kcl_syntax_valid": base_flags.get("kcl_syntax_valid", False),
        "geometry_manifold": base_flags.get("geometry_manifold", False),
        "manufacturing_feasible": base_flags.get("manufacturing_feasible", False),
        "reasons": []  # Fresh list
    }
    
    # ... physics calculations ...
    
    # Build reasons list immutably
    failure_reasons = []
    
    if cps_result.get("status") == "error":
        failure_reasons.append(f"CONTROL_FAILURE: {cps_result.get('error')}")
    
    if not chem_check["chemical_safe"]:
        failure_reasons.extend(chem_check["issues"])
    
    if mat_props["properties"]["is_melted"]:
        failure_reasons.append(f"MATERIAL_FAILURE: {material_name} melted at {temp}C")
    
    if struct_result["status"] == "failure":
        failure_reasons.append(f"STRUCTURAL_FAILURE: {struct_result['logs'][-1]}")
    
    if therm_result["status"] == "critical":
        failure_reasons.append(f"THERMAL_FAILURE: Overheating {therm_result['equilibrium_temp_c']}C")
    
    # Set final state
    flags["physics_safe"] = len(failure_reasons) == 0
    flags["reasons"] = failure_reasons
    
    return {
        "physics_predictions": phys_result["physics_predictions"],
        "validation_flags": flags,  # Return new dict, not mutated one
        "material_props": mat_props,
        "sub_agent_reports": {
            "electronics": elec_result,
            "thermal": therm_result,
            "structural": struct_result
        }
    }
```

---

## PHASE 2: Frontend MVP (Week 2)

### Day 6-7: Three.js 3D Viewer

**Task**: Implement functional 3D model viewer in Workspace

**Implementation**:

1. **Install additional dependencies**:
```bash
cd frontend
npm install @react-three/drei@9.100 three@0.160 @react-three/fiber@8.18
```

2. **Create ModelViewer component**:
```jsx
// frontend/src/components/viewers/ModelViewer.jsx

import React, { useRef, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, useGLTF, Grid, Box } from '@react-three/drei';
import * as THREE from 'three';

function Model({ url, physicsData }) {
  const { scene } = useGLTF(url);
  const meshRef = useRef();
  
  // Apply physics-based coloring (stress heatmap)
  useEffect(() => {
    if (physicsData && meshRef.current) {
      scene.traverse((child) => {
        if (child.isMesh) {
          // Color based on stress/temperature
          const stress = physicsData.stress || 0;
          const color = stress > 100 ? 'red' : stress > 50 ? 'yellow' : 'green';
          child.material.color.set(color);
        }
      });
    }
  }, [physicsData, scene]);
  
  return <primitive ref={meshRef} object={scene} scale={1} />;
}

function PhysicsOverlay({ data }) {
  if (!data) return null;
  
  return (
    <div className="absolute top-4 right-4 bg-black/80 text-white p-4 rounded-lg">
      <h3 className="font-bold mb-2">Physics Telemetry</h3>
      <div className="space-y-1 text-sm">
        <div>Stress: {data.stress?.toFixed(2)} MPa</div>
        <div>Temp: {data.temperature?.toFixed(1)}°C</div>
        <div>Mass: {data.mass?.toFixed(2)} kg</div>
      </div>
    </div>
  );
}

export default function ModelViewer({ modelId, physicsData }) {
  const [modelUrl, setModelUrl] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    if (modelId) {
      // Fetch GLB from backend
      fetch(`/api/geometry/model/${modelId}`)
        .then(res => res.blob())
        .then(blob => {
          const url = URL.createObjectURL(blob);
          setModelUrl(url);
          setLoading(false);
        });
    }
  }, [modelId]);
  
  if (loading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white"></div>
      </div>
    );
  }
  
  return (
    <div className="relative h-full w-full">
      <Canvas camera={{ position: [5, 5, 5], fov: 50 }}>
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 10, 5]} />
        <Grid infiniteGrid fadeDistance={50} />
        
        {modelUrl && <Model url={modelUrl} physicsData={physicsData} />}
        
        <OrbitControls enablePan enableZoom enableRotate />
      </Canvas>
      
      <PhysicsOverlay data={physicsData} />
    </div>
  );
}
```

3. **Update Workspace page**:
```jsx
// frontend/src/pages/Workspace.jsx

import ModelViewer from '../components/viewers/ModelViewer';
import apiClient from '../utils/apiClient';

export default function Workspace() {
  const { theme } = useTheme();
  const { openTabs, activeTab } = usePanel();
  const [modelData, setModelData] = useState(null);
  const [isCompiling, setIsCompiling] = useState(false);
  
  const handleCompile = async () => {
    setIsCompiling(true);
    try {
      const result = await apiClient.post('/api/orchestrator/run', {
        user_intent: activeTab?.intent || 'Generate design',
        mode: 'execute'
      });
      
      setModelData({
        modelId: result.model_id,
        physics: result.physics_predictions
      });
    } finally {
      setIsCompiling(false);
    }
  };
  
  return (
    <div className="flex h-screen" style={{ backgroundColor: theme.colors.bg.primary }}>
      <LockedSidebar
        onCompile={handleCompile}
        isCompiling={isCompiling}
        // ... other handlers
      />
      
      <div className="flex-1 flex flex-col">
        {/* Header */}
        
        <div className="flex-1 relative">
          {modelData ? (
            <ModelViewer 
              modelId={modelData.modelId} 
              physicsData={modelData.physics}
            />
          ) : (
            <div className="h-full flex items-center justify-center text-gray-500">
              <div className="text-center">
                <p className="text-xl mb-4">No Model Loaded</p>
                <button 
                  onClick={handleCompile}
                  disabled={isCompiling}
                  className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                >
                  {isCompiling ? 'Compiling...' : 'Compile Design'}
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
```

---

### Day 8-9: Compile Panel Implementation

**Task**: Connect CompilePanel to orchestration API

**Implementation**:

```jsx
// frontend/src/components/panels/CompilePanel.jsx

import React, { useState } from 'react';
import { useTheme } from '../../contexts/ThemeContext';
import { Play, RotateCcw, AlertCircle, CheckCircle } from 'lucide-react';
import apiClient from '../../utils/apiClient';

export default function CompilePanel({ width, projectId, onCompileComplete }) {
  const { theme } = useTheme();
  const [isCompiling, setIsCompiling] = useState(false);
  const [progress, setProgress] = useState([]);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  
  const runCompilation = async () => {
    setIsCompiling(true);
    setError(null);
    setProgress(['Starting compilation...']);
    
    try {
      // Start orchestration
      const response = await apiClient.post('/api/orchestrator/run', {
        project_id: projectId,
        mode: 'execute'
      });
      
      // Poll for progress (until WebSocket implemented)
      const pollInterval = setInterval(async () => {
        try {
          const thoughts = await apiClient.get('/api/agents/thoughts');
          if (thoughts.thoughts?.length > 0) {
            setProgress(prev => [...prev, ...thoughts.thoughts.map(t => t.text)]);
          }
          
          // Check if complete
          const state = await apiClient.get(`/api/state/${projectId}`);
          if (state.final_documentation) {
            clearInterval(pollInterval);
            setResult(state);
            onCompileComplete?.(state);
          }
        } catch (e) {
          console.error('Poll error:', e);
        }
      }, 1000);
      
      // Timeout after 5 minutes
      setTimeout(() => {
        clearInterval(pollInterval);
        if (!result) {
          setError('Compilation timeout');
          setIsCompiling(false);
        }
      }, 300000);
      
    } catch (err) {
      setError(err.message);
      setIsCompiling(false);
    }
  };
  
  return (
    <div 
      style={{ width, backgroundColor: theme.colors.bg.secondary }} 
      className="h-full flex flex-col p-4"
    >
      <h2 className="text-lg font-bold mb-4" style={{ color: theme.colors.text.primary }}>
        Compile Design
      </h2>
      
      {/* Compile Button */}
      <button
        onClick={runCompilation}
        disabled={isCompiling}
        className="flex items-center justify-center gap-2 p-3 rounded-lg font-medium transition-colors"
        style={{ 
          backgroundColor: isCompiling ? theme.colors.bg.tertiary : theme.colors.accent,
          color: theme.colors.text.inverse,
          opacity: isCompiling ? 0.6 : 1
        }}
      >
        {isCompiling ? (
          <>
            <RotateCcw className="animate-spin" size={18} />
            Compiling...
          </>
        ) : (
          <>
            <Play size={18} />
            Run Compilation
          </>
        )}
      </button>
      
      {/* Progress Log */}
      {progress.length > 0 && (
        <div className="mt-4 flex-1 overflow-y-auto">
          <h3 className="text-sm font-medium mb-2" style={{ color: theme.colors.text.muted }}>
            Agent Activity
          </h3>
          <div className="space-y-1">
            {progress.map((msg, i) => (
              <div 
                key={i}
                className="text-xs p-2 rounded"
                style={{ 
                  backgroundColor: theme.colors.bg.tertiary,
                  color: theme.colors.text.secondary
                }}
              >
                {msg}
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Error Display */}
      {error && (
        <div className="mt-4 p-3 rounded-lg bg-red-900/50 border border-red-500 flex items-center gap-2">
          <AlertCircle className="text-red-500" size={18} />
          <span className="text-red-200 text-sm">{error}</span>
        </div>
      )}
      
      {/* Success Display */}
      {result && (
        <div className="mt-4 p-3 rounded-lg bg-green-900/50 border border-green-500 flex items-center gap-2">
          <CheckCircle className="text-green-500" size={18} />
          <span className="text-green-200 text-sm">Compilation complete!</span>
        </div>
      )}
    </div>
  );
}
```

---

### Day 10-12: Standardize API to JSON

**Task**: Remove FormData, use JSON consistently

**Implementation**:

1. **Update RequirementsGatheringPage**:
```jsx
// frontend/src/pages/RequirementsGatheringPage.jsx

// BEFORE (FormData):
const formData = new FormData();
formData.append('message', userIntent);
formData.append('conversation_history', '[]');
const data = await apiClient.post('/chat/requirements', formData);

// AFTER (JSON):
const data = await apiClient.post('/chat/requirements', {
  message: userIntent,
  conversation_history: [],
  user_intent: userIntent,
  mode: 'requirements_gathering',
  ai_model: llmProvider,
  session_id: localSessionId
});
```

2. **Update backend endpoints**:
```python
# backend/main.py

class ChatRequirementsRequest(BaseModel):
    message: str
    conversation_history: List[Dict] = []
    user_intent: str
    mode: str = 'requirements_gathering'
    ai_model: str = 'groq'
    session_id: Optional[str] = None

@app.post("/api/chat/requirements")
async def chat_requirements(req: ChatRequirementsRequest):
    # No more manual FormData parsing
    result = await conversational_agent.process(
        message=req.message,
        session_id=req.session_id,
        llm_provider=req.ai_model
    )
    return result
```

---

## PHASE 3: Backend Hardening (Week 3)

### Day 13-15: Docker Containerization

**Task**: Full Docker setup with multi-stage build

**Implementation**:

1. **Dockerfile**:
```dockerfile
# Dockerfile

# Stage 1: Build dependencies
FROM python:3.12-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libopenmpi-dev \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Production image
FROM python:3.12-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libopenmpi3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY backend/ ./backend/
COPY projects/ ./projects/

# Make sure scripts are executable
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/health')" || exit 1

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

2. **Docker Compose**:
```yaml
# docker-compose.yml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5

  backend:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379/0
      - GROQ_API_KEY=${GROQ_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - MATERIALS_PROJECT_API_KEY=${MATERIALS_PROJECT_API_KEY}
    volumes:
      - ./projects:/app/projects
      - ./data:/app/data
    depends_on:
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:80"
    depends_on:
      - backend

volumes:
  redis_data:
```

3. **Frontend Dockerfile**:
```dockerfile
# frontend/Dockerfile

# Build stage
FROM node:20-alpine as builder

WORKDIR /app
COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

---

### Day 16-17: Async Agent Migration

**Task**: Convert remaining sync agents to async

**Migration Pattern**:
```python
# Before
class SomeAgent:
    def run(self, params):
        result = requests.get(url)  # Blocking!
        return result.json()

# After  
class SomeAgent(BaseAgent):
    async def run(self, params):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                return await resp.json()
```

**Bulk Migration Script**:
```python
# scripts/migrate_to_async.py

import ast
import os
from pathlib import Path

def migrate_agent_file(filepath: Path):
    """Convert a single agent file to async."""
    with open(filepath, 'r') as f:
        source = f.read()
    
    tree = ast.parse(source)
    
    # Check if already async
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == 'run':
            print(f"Skipping {filepath} - already async")
            return
    
    # Simple string replacement (for known patterns)
    # More sophisticated: use ast.unparse for Python 3.9+
    replacements = [
        ('def run(self,', 'async def run(self,'),
        ('agent.run(', 'await agent.run('),
        ('requests.get(', 'await aiohttp_get('),
        ('requests.post(', 'await aiohttp_post('),
    ]
    
    new_source = source
    for old, new in replacements:
        new_source = new_source.replace(old, new)
    
    # Add import
    if 'import aiohttp' not in new_source:
        new_source = 'import aiohttp\n' + new_source
    
    # Backup and write
    backup_path = filepath.with_suffix('.py.sync_backup')
    with open(backup_path, 'w') as f:
        f.write(source)
    
    with open(filepath, 'w') as f:
        f.write(new_source)
    
    print(f"Migrated {filepath}")

# Run migration
for agent_file in Path('backend/agents').glob('*_agent.py'):
    migrate_agent_file(agent_file)
```

---

### Day 18-19: AgentState Refactor

**Task**: Split god object into domain-specific states

**Implementation**:

```python
# backend/schema.py

from typing import TypedDict, Optional
from pydantic import BaseModel

# Core state - minimal, always present
class CoreState(TypedDict):
    project_id: str
    user_intent: str
    messages: List[str]
    errors: List[str]
    iteration_count: int
    execution_mode: str

# Design intent and requirements
class DesignState(TypedDict):
    constraints: Dict[str, Any]
    design_parameters: Dict[str, Any]
    design_scheme: Dict[str, Any]
    environment: Dict[str, Any]

# Geometry and CAD data
class GeometryState(TypedDict):
    kcl_code: str
    gltf_data: str
    geometry_tree: List[Dict]
    mass_properties: Dict[str, Any]

# Physics simulation results
class PhysicsState(TypedDict):
    predictions: Dict[str, float]
    thermal_analysis: Dict[str, Any]
    structural_analysis: Dict[str, Any]
    fluid_analysis: Dict[str, Any]
    sub_agent_reports: Dict[str, Any]

# Validation and quality
class ValidationState(TypedDict):
    flags: Dict[str, bool]
    verification_report: Dict[str, Any]
    quality_review_report: Dict[str, Any]

# Manufacturing and BOM
class ManufacturingState(TypedDict):
    components: Dict[str, Any]
    bom_analysis: Dict[str, Any]
    manufacturing_plan: Dict[str, Any]
    gcode: str

# New flattened AgentState
class AgentState(TypedDict):
    core: CoreState
    design: DesignState
    geometry: GeometryState
    physics: PhysicsState
    validation: ValidationState
    manufacturing: ManufacturingState
    
    # Flow control
    user_approval: Optional[str]
    approval_required: bool

# Migration helper
class StateMigrator:
    """Migrate old flat state to new nested structure."""
    
    @staticmethod
    def migrate(old_state: Dict) -> AgentState:
        return {
            "core": {
                "project_id": old_state.get("project_id", ""),
                "user_intent": old_state.get("user_intent", ""),
                "messages": old_state.get("messages", []),
                "errors": old_state.get("errors", []),
                "iteration_count": old_state.get("iteration_count", 0),
                "execution_mode": old_state.get("execution_mode", "plan"),
            },
            "design": {
                "constraints": old_state.get("constraints", {}),
                "design_parameters": old_state.get("design_parameters", {}),
                "design_scheme": old_state.get("design_scheme", {}),
                "environment": old_state.get("environment", {}),
            },
            "geometry": {
                "kcl_code": old_state.get("kcl_code", ""),
                "gltf_data": old_state.get("gltf_data", ""),
                "geometry_tree": old_state.get("geometry_tree", []),
                "mass_properties": old_state.get("mass_properties", {}),
            },
            "physics": {
                "predictions": old_state.get("physics_predictions", {}),
                "thermal_analysis": old_state.get("thermal_analysis", {}),
                "structural_analysis": old_state.get("structural_analysis", {}),
                "fluid_analysis": old_state.get("fluid_analysis", {}),
                "sub_agent_reports": old_state.get("sub_agent_reports", {}),
            },
            "validation": {
                "flags": old_state.get("validation_flags", {}),
                "verification_report": old_state.get("verification_report", {}),
                "quality_review_report": old_state.get("quality_review_report", {}),
            },
            "manufacturing": {
                "components": old_state.get("components", {}),
                "bom_analysis": old_state.get("bom_analysis", {}),
                "manufacturing_plan": old_state.get("manufacturing_plan", {}),
                "gcode": old_state.get("gcode", ""),
            },
            "user_approval": old_state.get("user_approval"),
            "approval_required": old_state.get("approval_required", False),
        }
```

---

## PHASE 4: Testing & CI/CD (Week 4)

### Day 20-22: Integration Tests

**Task**: Tests with real physics, not mocks

```python
# tests/integration/test_physics_real.py

import pytest
from backend.physics.kernel import get_physics_kernel

@pytest.fixture(scope="module")
def physics_kernel():
    """Real physics kernel - no mocks."""
    return get_physics_kernel()

def test_gravity_calculation_real(physics_kernel):
    """Verify gravity using real physics constants."""
    g = physics_kernel.get_constant("g")
    assert 9.79 < g < 9.82  # Earth gravity range

def test_stress_calculation_real(physics_kernel):
    """Verify stress calculation on aluminum beam."""
    # 100N force on 10mm x 10mm aluminum beam
    result = physics_kernel.calculate(
        domain="structures",
        equation="stress",
        force_n=100,
        area_mm2=100
    )
    
    # Stress = Force / Area = 100N / 100mm² = 1 MPa
    assert abs(result["stress_mpa"] - 1.0) < 0.01

def test_thermal_equilibrium_real(physics_kernel):
    """Verify heat transfer calculation."""
    result = physics_kernel.calculate(
        domain="thermodynamics",
        equation="equilibrium_temp",
        power_w=100,
        surface_area_m2=0.1,
        ambient_temp_c=25
    )
    
    # Higher power → higher equilibrium temp
    assert result["temp_c"] > 25
```

---

### Day 23-24: GitHub Actions CI/CD

**Implementation**:

```yaml
# .github/workflows/ci.yml

name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      
      - name: Install dependencies
        run: |
          pip install ruff mypy
          pip install -r requirements.txt
      
      - name: Lint with ruff
        run: ruff check backend/
      
      - name: Type check with mypy
        run: mypy backend/ --ignore-missing-imports

  test-backend:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:7
        ports:
          - 6379:6379
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      
      - name: Install system deps
        run: |
          sudo apt-get update
          sudo apt-get install -y libopenmpi-dev
      
      - name: Install Python dependencies
        run: |
          pip install pytest pytest-asyncio
          pip install -r requirements.txt
      
      - name: Run unit tests
        run: pytest tests/unit/ -v
        env:
          REDIS_URL: redis://localhost:6379/0
      
      - name: Run integration tests
        run: pytest tests/integration/ -v
        env:
          REDIS_URL: redis://localhost:6379/0

  test-frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Node
        uses: actions/setup-node@v4
        with:
          node-version: '20'
      
      - name: Install dependencies
        working-directory: frontend
        run: npm ci
      
      - name: Run tests
        working-directory: frontend
        run: npm test -- --coverage
      
      - name: Build
        working-directory: frontend
        run: npm run build

  build-and-push:
    needs: [lint, test-backend, test-frontend]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Login to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
      
      - name: Build and push backend
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: brickos/backend:${{ github.sha }},brickos/backend:latest
      
      - name: Build and push frontend
        uses: docker/build-push-action@v5
        with:
          context: frontend
          push: true
          tags: brickos/frontend:${{ github.sha }},brickos/frontend:latest
```

---

## PHASE 5: Advanced Features (Weeks 5-6)

### Day 25-28: WebSocket Real-time Updates

**Backend**:
```python
# backend/websocket.py

from fastapi import WebSocket
import json
import asyncio

class OrchestratorWebSocket:
    """WebSocket manager for real-time orchestration updates."""
    
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, project_id: str):
        await websocket.accept()
        self.connections[project_id] = websocket
    
    async def disconnect(self, project_id: str):
        if project_id in self.connections:
            del self.connections[project_id]
    
    async def send_progress(self, project_id: str, message: dict):
        if project_id in self.connections:
            await self.connections[project_id].send_json(message)

ws_manager = OrchestratorWebSocket()

@app.websocket("/ws/orchestrator/{project_id}")
async def orchestrator_websocket(websocket: WebSocket, project_id: str):
    await ws_manager.connect(websocket, project_id)
    try:
        while True:
            # Keep connection alive, handle pings
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except Exception:
        await ws_manager.disconnect(project_id)
```

**Frontend**:
```jsx
// frontend/src/hooks/useWebSocket.js

import { useEffect, useRef, useState } from 'react';

export function useWebSocket(projectId) {
  const ws = useRef(null);
  const [progress, setProgress] = useState([]);
  const [connected, setConnected] = useState(false);
  
  useEffect(() => {
    const wsUrl = `${import.meta.env.VITE_WS_URL}/ws/orchestrator/${projectId}`;
    ws.current = new WebSocket(wsUrl);
    
    ws.current.onopen = () => setConnected(true);
    ws.current.onclose = () => setConnected(false);
    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setProgress(prev => [...prev, data]);
    };
    
    return () => ws.current?.close();
  }, [projectId]);
  
  return { progress, connected };
}
```

---

### Day 29-32: Performance Monitoring

**Implementation**:

```python
# backend/monitoring/performance.py

import time
from functools import wraps
from typing import Callable
import logging

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Track agent execution times and bottlenecks."""
    
    def __init__(self):
        self.metrics = {}
    
    def time_agent(self, agent_name: str):
        """Decorator to time agent execution."""
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.time()
                try:
                    return await func(*args, **kwargs)
                finally:
                    duration = time.time() - start
                    self.record(agent_name, duration)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start = time.time()
                try:
                    return func(*args, **kwargs)
                finally:
                    duration = time.time() - start
                    self.record(agent_name, duration)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def record(self, agent_name: str, duration: float):
        """Record execution time."""
        if agent_name not in self.metrics:
            self.metrics[agent_name] = []
        self.metrics[agent_name].append(duration)
        
        # Alert on slow execution
        if duration > 30:
            logger.warning(f"Slow agent: {agent_name} took {duration:.1f}s")
    
    def get_stats(self) -> dict:
        """Get performance statistics."""
        stats = {}
        for agent_name, times in self.metrics.items():
            stats[agent_name] = {
                "count": len(times),
                "avg": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
                "p95": sorted(times)[int(len(times) * 0.95)] if len(times) > 20 else max(times)
            }
        return stats

monitor = PerformanceMonitor()

# Usage in agents
@monitor.time_agent("GeometryAgent")
async def run(self, params):
    # ... agent logic ...
```

---

## PHASE 6: Production Deployment (Weeks 7-8)

### Day 33-36: Environment Validation

```python
# backend/config/settings.py

from pydantic_settings import BaseSettings
from pydantic import validator
from typing import Optional

class Settings(BaseSettings):
    """Validated application settings."""
    
    # LLM Providers (at least one required)
    groq_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    kimi_api_key: Optional[str] = None
    
    # Infrastructure
    redis_url: str = "redis://localhost:6379/0"
    
    # Optional but recommended
    materials_project_api_key: Optional[str] = None
    sentry_dsn: Optional[str] = None
    
    @validator("groq_api_key", "openai_api_key", "gemini_api_key", "kimi_api_key")
    def at_least_one_llm(cls, v, values):
        """Ensure at least one LLM provider is configured."""
        # This runs for each field, so we check if any have been set
        all_keys = [
            v,
            values.get("groq_api_key"),
            values.get("openai_api_key"),
            values.get("gemini_api_key"),
            values.get("kimi_api_key")
        ]
        if not any(all_keys):
            raise ValueError("At least one LLM provider API key must be configured")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

---

### Day 37-40: Documentation & Final Polish

**Tasks**:
1. Update MASTER_ARCHITECTURE.md with new patterns
2. Create DEPLOYMENT.md guide
3. Add API documentation (auto-generated from FastAPI)
4. Create troubleshooting guide
5. Final security audit
6. Performance benchmarking

---

## 📊 Success Metrics by Phase

| Phase | Metric | Before | Target |
|-------|--------|--------|--------|
| **Phase 1** | Silent failures | 3+ | 0 |
| | Server restart data loss | 100% | 0% |
| | Circular imports | 1 | 0 |
| **Phase 2** | Working UI components | 1/15 | 15/15 |
| | 3D viewer functional | ❌ | ✅ |
| | API consistency | 60% | 100% |
| **Phase 3** | Docker deployment | ❌ | ✅ |
| | Async agents | 30% | 100% |
| | State fields per object | 40+ | <10 |
| **Phase 4** | Test coverage | 20% | 70% |
| | CI/CD pipeline | ❌ | ✅ |
| | Integration tests | 0 | 20+ |
| **Phase 5** | Real-time updates | Polling | WebSocket |
| | Performance monitoring | Basic | Full |
| **Phase 6** | Production ready | ❌ | ✅ |

---

## 🚀 Quick Start Commands

```bash
# Week 1: Critical fixes
git checkout -b phase1-critical-fixes
# ... make fixes ...
pytest tests/unit/test_agent_registry.py -v
pytest tests/unit/test_async_agents.py -v
git push origin phase1-critical-fixes

# Week 2: Frontend
cd frontend
npm install
npm run dev

# Week 3: Docker
docker-compose up --build

# Week 4: Tests
pytest tests/ -v --cov=backend --cov-report=html

# Production deploy
docker-compose -f docker-compose.prod.yml up -d
```

---

## 📝 Notes

- **Parallel Work**: Frontend (Week 2) can start while backend CI/CD (Week 4) is being set up
- **Risk Mitigation**: Each phase has verification steps - don't proceed if metrics not met
- **Rollback Plan**: Keep `main` branch stable, work in feature branches
- **Documentation**: Update docs as you go, not at the end
