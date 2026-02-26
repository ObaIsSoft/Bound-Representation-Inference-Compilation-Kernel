# ProjectOrchestrator Implementation Plan

## Overview
Replace LangGraph macro-orchestration with ISA-centric state machine while keeping LangGraph for micro-workflows where appropriate.

## Phase 1: Foundation (Files 1-4)
### 1.1 Enums and Types
- File: `backend/core/orchestrator_types.py`
- Content: Phase, PhaseStatus, GateStatus enums, dataclasses for PhaseResult, AgentTask

### 1.2 ISA Checkpoint Manager  
- File: `backend/core/isa_checkpoint.py`
- Content: Merkle-tree based checkpointing, state persistence, rollback support

### 1.3 Event System
- File: `backend/core/orchestrator_events.py`
- Content: Event bus for decoupled WebSocket broadcasting, metrics, logging

### 1.4 Parallel Agent Executor
- File: `backend/core/agent_executor.py`
- Content: Async agent execution with timeouts, retry logic, result aggregation

## Phase 2: Core Orchestrator (Files 5-6)
### 2.1 ProjectOrchestrator Core
- File: `backend/core/project_orchestrator.py`
- Content: Main orchestrator class, phase routing, human-in-the-loop gates

### 2.2 Phase Handlers
- File: `backend/core/phase_handlers.py`
- Content: Individual phase implementations (8 phases), parallel execution patterns

## Phase 3: Integration (Files 7-9)
### 3.1 API Routes
- File: `backend/controllers/orchestrator_controller.py`
- Content: HTTP endpoints for project lifecycle, approval submission

### 3.2 WebSocket Handler
- File: `backend/websocket/orchestrator_ws.py`
- Content: Real-time project status, agent progress streaming

### 3.3 Migration Adapter
- File: `backend/core/langgraph_adapter.py`
- Content: Bridge old LangGraph calls to new orchestrator during transition

## Phase 4: Testing & Migration (Files 10-11)
### 4.1 Unit Tests
- File: `backend/tests/test_orchestrator.py`
- Content: Phase execution, error handling, checkpoint/rollback

### 4.2 Integration Test
- File: `backend/tests/test_e2e_orchestrator.py`
- Content: Full project lifecycle, human-in-the-loop simulation

---

## Execution Order
1. Start with types → checkpoint → events → executor (foundation)
2. Build core orchestrator + phase handlers (heart)
3. Add API + WebSocket (interface)
4. Create adapter for gradual migration
5. Test everything

## Key Design Decisions
- ISA is the single source of truth for state
- Parallel execution by default for independent agents
- Explicit human gates with AWAITING_APPROVAL status
- Merkle hashes for deterministic checkpointing
- Event-driven WebSocket updates (not polling)
