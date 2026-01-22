# Specialized Critics for BRICK OS

## Overview

Five specialized critics monitor agents and oracles across BRICK OS:

| Critic | Monitors | Key Metrics | Critical Safety Checks |
|--------|----------|-------------|----------------------|
| **ChemistryCritic** | ChemistryAgent | Corrosion accuracy, Safety predictions | False negatives (predicts safe but fails) |
| **ElectronicsCritic** | ElectronicsAgent | Power balance, DRC violations | Short circuit detection |
| **MaterialCritic** | MaterialAgent | DB coverage, Strength degradation | Melting point violations |
| **ComponentCritic** | ComponentAgent | Selection quality, User acceptance | Installation failures |
| **OracleCritic** | ALL Oracles | Calculation accuracy | Conservation law violations |

## Quick Start

```python
from agents.critics import ChemistryCritic, OracleCritic

# Initialize
chem_critic = ChemistryCritic(window_size=100)
oracle_critic = OracleCritic(window_size=100)

# Observe during design cycle
chem_critic.observe(
    input_state={"materials": ["Steel"], "environment_type": "MARINE"},
    chemistry_output=chemistry_agent.run(...)
)

# Periodic analysis
report = chem_critic.analyze()
print(f"Safety Accuracy: {report['safety_accuracy']:.0%}")
print(f"Recommendations: {report['recommendations']}")

# Evolution decision
should_evolve, reason, strategy = chem_critic.should_evolve()
if should_evolve:
    print(f"Agent needs evolution: {reason}")
    print(f"Strategy: {strategy}")
```

## Demo Results

From `demo_specialized_critics.py` (30 design iterations):

**Evolution Queue:**
- ⚠️ **OracleCritic**: CRITICAL - 1 conservation law violation detected
- ⚠️ **MaterialCritic**: Database coverage at 50% - needs expansion  
- ⚠️ **ComponentCritic**: 61% over-specification rate - optimize selection

**Nominal Performance:**
- ✅ **ChemistryCritic**: 100% safety accuracy
- ✅ **ElectronicsCritic**: All checks passing

## Critical Insight: Oracle Critic is Most Important

The **OracleCritic** detected a Kirchhoff's Current Law violation:
```
I_in = 5.595A
I_out = 6.436A
```

This is a **fundamental physics violation** in the PhysicsOracle - if the oracles are wrong, everything built on them is wrong. This demonstrates why monitoring the oracles themselves is critical.

## Integration with Orchestrator

Add critics to the orchestration graph:

```python
# In orchestrator.py
from agents.critics import (
    ChemistryCritic, ElectronicsCritic, 
    MaterialCritic, ComponentCritic, OracleCritic
)

# Initialize critics
critics = {
    "chemistry": ChemistryCritic(),
    "electronics": ElectronicsCritic(),
    "material": MaterialCritic(),
    "component": ComponentCritic(),
    "oracle": OracleCritic()
}

# After each agent execution
def physics_node(state):
    result = phys_agent.run(...)
    
    # Observe with critic
    if "physics_critic" in state.get("active_critics", []):
        critics["physics"].observe(...)
    
    return result

# Periodic critic analysis (every 50 iterations)
def critic_analysis_node(state):
    reports = {}
    for name, critic in critics.items():
        if critic.total_evaluations > 50:
            reports[name] = critic.analyze()
            
            # Check evolution triggers
            should_evolve, reason, strategy = critic.should_evolve()
            if should_evolve:
                state["evolution_queue"].append({
                    "agent": name,
                    "reason": reason,
                    "strategy": strategy
                })
    
    return {"critic_reports": reports}
```

## Files Created

- `backend/agents/critics/ChemistryCritic.py` - Chemistry agent monitoring
- `backend/agents/critics/ElectronicsCritic.py` - Electronics agent monitoring  
- `backend/agents/critics/MaterialCritic.py` - Material agent monitoring
- `backend/agents/critics/ComponentCritic.py` - Component agent monitoring
- `backend/agents/critics/OracleCritic.py` - **All oracle system validation**
- `backend/agents/critics/__init__.py` - Module exports
- `demo_specialized_critics.py` - Comprehensive demonstration

## Reports

All critics export detailed JSON reports:
```bash
ls /tmp/critic_reports/
# chemistry_report.json
# electronics_report.json  
# material_report.json
# component_report.json
# oracle_report.json (most critical!)
```

## Next Steps

1. **Fix Oracle**: Address conservation law violation in PhysicsOracle
2. **Expand Material DB**: Add missing materials (currently 50% coverage)
3. **Optimize Component Selection**: Reduce over-specification from 61% to <30%
4. **Integrate into Orchestrator**: Add observation hooks to all agent nodes
5. **Create Dashboard**: UI to show critic status and evolution queue
