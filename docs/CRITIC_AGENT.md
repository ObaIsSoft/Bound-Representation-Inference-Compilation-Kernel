# Critic Agent Architecture

## Overview

The **CriticAgent** is a meta-agent designed to monitor, evaluate, and provide feedback on other agents - particularly hybrid gated agents that combine physics-based heuristics with neural network intuition.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     GATED HYBRID AGENT                          │
│                                                                 │
│  Input (mass, velocity, altitude)                              │
│         │                                                       │
│         ├──────────────┬──────────────┬──────────────┐         │
│         │              │              │              │         │
│         ▼              ▼              ▼              │         │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐        │         │
│   │ Physics │    │ Neural  │    │  Gate   │        │         │
│   │ Branch  │    │ Branch  │    │Mechanism│        │         │
│   │         │    │         │    │         │        │         │
│   │ F=m*g   │    │ Learned │    │ Sigmoid │        │         │
│   │         │    │Patterns │    │ (0-1)   │        │         │
│   └────┬────┘    └────┬────┘    └────┬────┘        │         │
│        │              │              │              │         │
│        │              │              └──────────────┤         │
│        │              │                             │         │
│        ▼              ▼                             ▼         │
│   ┌─────────────────────────────────────────────────────┐    │
│   │  GATED FUSION                                       │    │
│   │  Output = Physics*(1-gate) + Neural*gate           │    │
│   └──────────────────────────┬──────────────────────────┘    │
│                              │                                │
└──────────────────────────────┼────────────────────────────────┘
                               │
                               │ (prediction, gate_value)
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CRITIC AGENT                               │
│                                                                 │
│  ┌───────────────────────────────────────────────────────┐    │
│  │  OBSERVATION BUFFER (rolling window)                  │    │
│  │  • Predictions                                        │    │
│  │  • Ground Truth                                       │    │
│  │  • Gate Values                                        │    │
│  │  • Input States                                       │    │
│  └───────────────────────────────────────────────────────┘    │
│                              │                                 │
│                              ▼                                 │
│  ┌───────────────────────────────────────────────────────┐    │
│  │  ANALYSIS ENGINE                                      │    │
│  │  ├─ Performance Metrics                               │    │
│  │  ├─ Gate Alignment Validation                         │    │
│  │  ├─ Error Distribution Analysis                       │    │
│  │  ├─ Failure Mode Detection                            │    │
│  │  └─ Recommendation Generation                         │    │
│  └───────────────────────────────────────────────────────┘    │
│                              │                                 │
│                              ▼                                 │
│  ┌───────────────────────────────────────────────────────┐    │
│  │  DECISION LOGIC                                       │    │
│  │  • Should retrain?                                    │    │
│  │  • Training suggestions                               │    │
│  │  • Confidence scoring                                 │    │
│  └───────────────────────────────────────────────────────┘    │
│                              │                                 │
└──────────────────────────────┼─────────────────────────────────┘
                               │
                               ▼
                        CRITIC REPORT
                        • Performance: 97.23%
                        • Gate Alignment: 97.14%
                        • Recommendations
                        • Failure Modes
                        • Retrain Decision
```

## Key Components

### 1. Observation System

The critic maintains a rolling window buffer that stores:
- **Predictions**: What the agent predicted
- **Ground Truth**: What actually happened (from environment)
- **Gate Values**: The agent's meta-decision (0=physics, 1=neural)
- **Input States**: The raw sensor data that led to each decision

```python
critic.observe(
    input_state=np.array([mass, velocity, altitude]),
    prediction=agent_output,
    ground_truth=env.get_reality(),
    gate_value=gate_decision
)
```

### 2. Analysis Engine

The critic performs multi-dimensional analysis:

#### A. Performance Metrics
- **Overall Performance**: Measures prediction accuracy vs ground truth
- **Relative Error**: Normalized error accounting for scale differences
- **Error Distribution**: Breaks down errors by operational domain (low/high speed)

#### B. Gate Alignment Validation
The critic validates that the gate makes **sensible** decisions:
- At low velocities (<50): Gate should be ~0 (trust physics)
- At high velocities (>50): Gate should be ~1 (trust neural/turbulence)
- Computes alignment score: How well does the gate decision match expectations?

#### C. Failure Mode Detection
Proactively identifies issues:
- **Gate Stuck**: Gate not adapting (variance too low)
- **Gate Misaligned**: Using wrong branch for velocity regime
- **Domain-Specific Failures**: Physics failing at low speeds OR neural failing at high speeds
- **Concept Drift**: Error increasing over time
- **Numerical Instability**: Extreme predictions

#### D. Recommendation Generation
Generates actionable advice:
- When to retrain
- Which branch needs improvement
- Hyperparameter adjustments
- Data collection focus regions

### 3. Decision Logic

The critic makes autonomous decisions:

```python
should_retrain, reason = critic.should_retrain()

# Critical thresholds:
# - Performance < 50%: IMMEDIATE RETRAIN
# - Gate alignment < 50%: RETRAIN
# - Multiple failure modes (≥3): RETRAIN
# - Concept drift detected: RETRAIN
```

## What Makes This Powerful?

### 1. **Meta-Cognition**
The critic knows *when the agent knows* vs *when it doesn't know*. By monitoring the gate, it validates not just accuracy but the **reasoning** behind decisions.

### 2. **Proactive Intervention**
Rather than waiting for catastrophic failure, the critic detects subtle degradation:
- Performance drops from 97% → 88%? Flag it.
- Gate variance decreasing? Agent might be getting lazy.
- High-speed errors spiking? Neural branch needs retraining.

### 3. **Concept Drift Detection**
When the environment changes (e.g., turbulence model shifts), the critic notices:

```
BEFORE DRIFT: Performance = 97.23%
AFTER DRIFT:  Performance = 88.48%
CRITIC:       "Neural branch failing on high-speed cases (needs retraining)"
```

### 4. **Interpretability**
Unlike black-box monitoring, the critic provides **explanations**:
- Not just "error is high"
- But "high-speed errors dominate (neural branch needs more data)"
- And "focus training on velocity range [51.1, 149.5]"

### 5. **Different Timescales**
- **Agent**: Real-time predictions (milliseconds)
- **Critic**: Periodic analysis (every 100 samples, or hourly, or daily)

This separation prevents the critic from becoming a bottleneck.

## Example Output

```
CRITIC REPORT
----------------------------------------------------------------------
Overall Performance:  97.23%
Gate Alignment:       97.14%
Critic Confidence:    52.81%

📊 Error Distribution:
  • mean_error                =  21.1387
  • max_error                 =  87.6765
  • low_speed_error           =   0.0000
  • high_speed_error          =  29.3593

🎛️  Gate Statistics:
  • mean_gate                 =   0.7173
  • low_speed_gate            =   0.0851  ← Good! (trusting physics)
  • high_speed_gate           =   0.9632  ← Good! (trusting neural)

💡 Recommendations:
  • 🧠 TRAIN NEURAL: High-speed errors dominate (neural branch needs more data)

📋 Training Suggestions:
  Focus Region (velocity): [51.1, 149.5]
  Recommended new samples: 50
```

## Integration with BRICK OS

The CriticAgent can monitor your physics agents:

```python
# In your orchestrator
from agents.CriticAgent import CriticAgent

# Initialize critic for each physics domain
aero_critic = CriticAgent(window_size=500)
thermal_critic = CriticAgent(window_size=200)
stress_critic = CriticAgent(window_size=300)

# During simulation
for timestep in simulation:
    stress_pred = stress_agent.predict(design)
    stress_truth = fem_solver.solve(design)
    
    stress_critic.observe(
        input_state=design.features,
        prediction=stress_pred,
        ground_truth=stress_truth,
        gate_value=stress_agent.gate_value
    )
    
    # Periodic check
    if timestep % 100 == 0:
        report = stress_critic.analyze()
        if report.overall_performance < 0.7:
            logger.warning(f"Stress agent degrading: {report.recommendations}")
```

## Future Enhancements

### 1. **Automated Retraining Pipeline**
```
Critic detects drift → Trigger data collection → Retrain agent → A/B test → Deploy
```

### 2. **Multi-Agent Coordination**
Critic monitors interactions between agents:
- Is AeroAgent's output confusing ThermalAgent?
- Are predictions becoming inconsistent?

### 3. **Meta-Critic**
A critic that monitors the critic:
- Is the critic too conservative (flagging false positives)?
- Is the critic missing real issues (false negatives)?

### 4. **Safety Constraints**
Critic can veto unsafe agent modifications:
- "New model version predicts negative stresses - REJECTED"
- "Gate variance dropped to 0 - ROLLBACK"

### 5. **Explainable AI Integration**
Use critic reports for user-facing explanations:
- "Why did the simulation fail?" → Check critic report
- "Is this design in a known failure mode?" → Critic knows!

## Philosophical Note

The critic embodies the principle that **self-improvement requires self-awareness**. An agent that can't evaluate its own performance can't evolve. The critic provides that self-awareness loop - not just detecting when things go wrong, but understanding *why* and *how to fix it*.

This is the foundation for truly self-evolving agent systems.

---

**Files:**
- Implementation: `backend/agents/CriticAgent.py`
- Demonstration: `demo_critic_agent.py`
- Reports: `/tmp/critic_report_*.json`
