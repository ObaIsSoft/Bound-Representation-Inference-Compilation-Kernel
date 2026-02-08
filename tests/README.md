# BRICK OS Test Suite

## Test Structure

```
tests/
├── unit/                    # Fast unit tests with mocks
│   ├── conftest.py
│   ├── test_physics.py     # Physics kernel tests (mocked)
│   ├── test_geometry.py
│   └── ...
├── integration/             # Integration tests (real dependencies)
│   ├── conftest.py         # Integration fixtures
│   ├── test_physics_kernel.py    # Real physics calculations
│   ├── test_agent_integration.py # Multi-agent communication
│   ├── test_full_pipeline.py     # E2E pipeline tests
│   └── test_simple_design.py
├── common/                  # Shared test utilities
│   └── fixtures.py
├── legacy/                  # Legacy tests (cleanup needed)
└── performance/            # Performance benchmarks
```

## Running Tests

### Unit Tests (Fast, Mocked)
```bash
cd backend
pytest ../tests/unit/ -v
```

### Integration Tests (Real Physics)
```bash
cd backend
pytest ../tests/integration/ -v -m "not slow"
```

### Specific Test Categories
```bash
# Physics only
pytest tests/integration/test_physics_kernel.py -v

# Agent communication
pytest tests/integration/test_agent_integration.py -v

# Full pipeline (slow)
pytest tests/integration/test_full_pipeline.py -v -m "slow"
```

### With Coverage
```bash
cd backend
pytest ../tests/ --cov=backend --cov-report=html
```

## Test Markers

- `integration` - Tests with real dependencies (not mocked)
- `slow` - Tests that take > 10 seconds
- `unit` - Fast unit tests
- `physics` - Tests requiring physics kernel
- `agents` - Tests requiring agent registry
- `e2e` - End-to-end pipeline tests

## Phase 4 Test Coverage

### Physics Kernel Integration Tests (`test_physics_kernel.py`)

These tests use **real physics calculations**, not mocks:

1. **Physical Constants** - Verify constants (g, c, G)
2. **Unit Conversions** - Real unit conversion math
3. **Structural Calculations** - Stress, safety factor, beam deflection
4. **Geometry Validation** - Feasible vs infeasible designs
5. **Equations of Motion** - Euler integration
6. **Materials Properties** - Real material lookups
7. **Conservation Laws** - Energy/momentum validation
8. **Multi-Fidelity Routing** - Calculation routing logic

### Agent Integration Tests (`test_agent_integration.py`)

Tests multi-agent communication:

1. **Agent Registry** - Lazy loading, error handling (C1 fix)
2. **Async Agents** - GeometryAgent async conversion (C2 fix)
3. **Session Store** - Redis/in-memory persistence (C3 fix)
4. **XAI Stream** - Thought streaming without circular imports (C4 fix)
5. **State Management** - Immutable state handling (C5 fix)
6. **Error Propagation** - Graceful failure handling

### Full Pipeline Tests (`test_full_pipeline.py`)

End-to-end orchestration:

1. **Simple Design** - Ball design through full pipeline
2. **Structural Design** - Load-bearing bracket
3. **Fluid System** - Pipe system design
4. **Physics Validation** - Infeasible design detection
5. **State Propagation** - State management through graph
6. **Performance** - Baseline performance metrics

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs:

1. **Lint** - ruff, black, isort, mypy
2. **Unit Tests** - Fast mocked tests with coverage
3. **Integration Tests** - Real physics calculations
4. **Frontend Build** - Build verification
5. **Agent Registry** - Lazy loading and error handling
6. **E2E Pipeline** - Full orchestration test
7. **Build Verification** - File structure and syntax

## Critical Fixes Verified

| Fix | Test File | Test Function |
|-----|-----------|---------------|
| C1 - Silent failures | `test_agent_integration.py` | `test_agent_not_found_raises_error` |
| C2 - Async GeometryAgent | `test_agent_integration.py` | `test_geometry_agent_async` |
| C3 - Session store | `test_agent_integration.py` | `test_session_persistence_redis` |
| C4 - Circular imports | `test_agent_integration.py` | `test_no_circular_import_error` |
| C5 - Immutable state | `test_full_pipeline.py` | `test_immutable_state_physics_node` |

## Adding New Tests

### Unit Tests (Mocked)
```python
import pytest
from unittest.mock import MagicMock, patch

def test_something_mocked():
    with patch('module.function') as mock:
        mock.return_value = {"result": 42}
        # Test logic
```

### Integration Tests (Real)
```python
import pytest

pytestmark = pytest.mark.integration

def test_something_real(physics_kernel):
    # Uses real physics kernel
    result = physics_kernel.calculate(...)
    assert result == expected_real_value
```

### Async Tests
```python
import pytest

pytestmark = pytest.mark.asyncio

async def test_async_operation():
    result = await some_async_function()
    assert result is not None
```
