import numpy as np
from typing import Callable
import sys
sys.path.append('backend')
from agents.CriticAgent import CriticAgent

# Simplified Gated Agent (numpy-only, no TensorFlow)
class SimpleGatedAgent:
    """
    A simplified gated agent that doesn't require TensorFlow.
    Uses hand-tuned weights to demonstrate the concept.
    """
    
    def __init__(self):
        # Initialize with reasonable starting values
        # These would normally be learned by gradient descent
        self.neural_learned = True
        
    def predict(self, state):
        """
        Make a prediction using gated fusion.
        
        Args:
            state: [mass, velocity, altitude]
        Returns:
            (prediction, gate_value)
        """
        mass, velocity, altitude = state.flatten()
        
        # Branch A: Physics (F = m * g)
        physics_prediction = mass * 9.81
        
        # Branch B: Neural Intuition (learned turbulence)
        # Approximates: base_physics + turbulence for high velocities
        if velocity > 50:
            turbulence_estimate = 0.48 * (velocity ** 1.5)  # Imperfect learning
            neural_prediction = mass * 9.81 + turbulence_estimate
        else:
            neural_prediction = mass * 9.81  # Falls back to physics
        
        # Gate Mechanism: Sigmoid function of velocity
        # Low velocity -> gate ~0 (trust physics)
        # High velocity -> gate ~1 (trust neural)
        gate_value = 1.0 / (1.0 + np.exp(-0.15 * (velocity - 50)))
        
        # Gated Fusion
        final_prediction = (1 - gate_value) * physics_prediction + gate_value * neural_prediction
        
        return final_prediction, gate_value

# Environment (Ground Truth)
class NonlinearEnvironment:
    """A world where physics works, until speed gets high (Turbulence)."""
    
    def get_reality(self, state):
        mass, velocity, altitude = state.flatten()
        base_physics = mass * 9.81
        
        if velocity > 50:
            turbulence = 0.5 * (velocity ** 1.5)  # True turbulence model
            return base_physics + turbulence
        return base_physics

# --- MAIN DEMONSTRATION ---

print("=" * 70)
print("GATED AGENT + CRITIC AGENT DEMONSTRATION")
print("(Simplified numpy version - no TensorFlow required)")
print("=" * 70)

# Initialize
agent = SimpleGatedAgent()
env = NonlinearEnvironment()
critic = CriticAgent(window_size=100, error_threshold=0.1)

# --- PHASE 1: CRITIC MONITORS AGENT ---
print("\n" + "=" * 70)
print("PHASE 1: CRITIC ANALYSIS OF AGENT PERFORMANCE")
print("=" * 70)

print("\n[1] Running inference and critic observation (200 samples)...")

# Generate test data: [mass, velocity, altitude]
# Cover both low-speed (physics) and high-speed (turbulence) domains
np.random.seed(42)
X_test = np.random.uniform([10, 0, 50], [50, 150, 200], (200, 3))

for i, x in enumerate(X_test):
    x_reshaped = x.reshape(1, -1)
    
    # Agent makes prediction
    prediction, gate_value = agent.predict(x_reshaped)
    
    # Get ground truth from environment
    ground_truth = env.get_reality(x_reshaped)
    
    # Critic observes the agent's decision
    critic.observe(
        input_state=x,
        prediction=prediction,
        ground_truth=ground_truth,
        gate_value=gate_value
    )

print(f"✓ Observed {len(critic.prediction_history)} agent decisions")

# Critic generates comprehensive report
print("\n[2] Critic generating analysis report...")
report = critic.analyze()

print("\n" + "-" * 70)
print("CRITIC REPORT")
print("-" * 70)
print(f"Overall Performance:  {report.overall_performance:.2%}")
print(f"Gate Alignment:       {report.gate_alignment:.2%}")
print(f"Critic Confidence:    {report.confidence:.2%}")

print("\n📊 Error Distribution:")
for key, value in report.error_distribution.items():
    print(f"  • {key:25} = {value:8.4f}")

print("\n🎛️  Gate Statistics:")
for key, value in report.gate_statistics.items():
    print(f"  • {key:25} = {value:8.4f}")

if report.failure_modes:
    print("\n⚠️  Failure Modes Detected:")
    for mode in report.failure_modes:
        print(f"  • {mode}")
else:
    print("\n✅ No failure modes detected")

print("\n💡 Recommendations:")
for rec in report.recommendations:
    print(f"  • {rec}")

# --- PHASE 2: TEST RETRAIN DECISION ---
print("\n" + "=" * 70)
print("PHASE 2: RETRAIN DECISION")
print("=" * 70)

should_retrain, reason = critic.should_retrain()
print(f"\nShould retrain? {'YES ⚠️' if should_retrain else 'NO ✓'}")
print(f"Reason: {reason}")

if not should_retrain:
    print("\n[3] Generating training optimization suggestions...")
    suggestions = critic.generate_training_suggestions()
    print("\n📋 Training Suggestions:")
    
    if 'focus_regions' in suggestions:
        vel_range = suggestions['focus_regions']['velocity_range']
        print(f"  Focus Region (velocity): [{vel_range[0]:.1f}, {vel_range[1]:.1f}]")
        print(f"  Mean problematic velocity: {suggestions['focus_regions']['mean_problematic_velocity']:.1f}")
        print(f"  Recommended new samples: {suggestions['recommended_samples']}")
    
    if suggestions.get('hyperparameter_suggestions'):
        print(f"\n  Hyperparameter adjustments:")
        for param, adjustment in suggestions['hyperparameter_suggestions'].items():
            print(f"    - {param}: {adjustment}")

# --- PHASE 3: DETAILED TEST CASES ---
print("\n" + "=" * 70)
print("PHASE 3: DETAILED TEST CASE ANALYSIS")
print("=" * 70)

test_cases = [
    ("Low Speed (Physics Domain)", np.array([[20.0, 5.0, 100.0]])),
    ("Medium Speed (Transition)", np.array([[20.0, 45.0, 100.0]])),
    ("High Speed (Turbulence)", np.array([[20.0, 80.0, 100.0]])),
    ("Very High Speed (Extreme)", np.array([[20.0, 120.0, 100.0]])),
    ("Edge Case: Zero Velocity", np.array([[20.0, 0.0, 100.0]])),
    ("Edge Case: Threshold", np.array([[20.0, 50.0, 100.0]]))
]

print("\n{:<30} | {:>8} | {:>10} | {:>10} | {:>10}".format(
    "Test Case", "Gate", "Predicted", "Truth", "Error %"
))
print("-" * 75)

for name, test_input in test_cases:
    pred, gate_val = agent.predict(test_input)
    truth = env.get_reality(test_input)
    error_pct = abs(pred - truth) / (truth + 1e-6) * 100
    
    # Color code the gate value interpretation
    if gate_val < 0.3:
        gate_interp = "PHYS"
    elif gate_val > 0.7:
        gate_interp = "NEUR"
    else:
        gate_interp = "MIX"
    
    print("{:<30} | {:>8.3f} | {:>10.2f} | {:>10.2f} | {:>9.1f}%".format(
        name, gate_val, pred, truth, error_pct
    ))

# --- PHASE 4: SIMULATED CONCEPT DRIFT ---
print("\n" + "=" * 70)
print("PHASE 4: DETECTING CONCEPT DRIFT")
print("=" * 70)

print("\n[4] Simulating environment change (turbulence increases)...")

# Reset critic to simulate fresh monitoring
critic.reset()

# Simulate a changed environment where turbulence model changed
class DriftedEnvironment:
    """Environment where turbulence suddenly increased by 30%"""
    def get_reality(self, state):
        mass, velocity, altitude = state.flatten()
        base_physics = mass * 9.81
        
        if velocity > 50:
            # Turbulence increased! (concept drift)
            turbulence = 0.65 * (velocity ** 1.5)  # Was 0.5, now 0.65
            return base_physics + turbulence
        return base_physics

drifted_env = DriftedEnvironment()

# Agent continues operating in the changed environment
X_drift = np.random.uniform([10, 0, 50], [50, 150, 200], (100, 3))

for x in X_drift:
    x_reshaped = x.reshape(1, -1)
    prediction, gate_value = agent.predict(x_reshaped)
    ground_truth = drifted_env.get_reality(x_reshaped)
    
    critic.observe(
        input_state=x,
        prediction=prediction,
        ground_truth=ground_truth,
        gate_value=gate_value
    )

print(f"✓ Observed {len(critic.prediction_history)} decisions in drifted environment")

# Check if critic detects the drift
drift_report = critic.analyze()
print("\n[5] Critic analysis after drift:")
print(f"  Overall Performance: {drift_report.overall_performance:.2%} (was {report.overall_performance:.2%})")

should_retrain_drift, reason_drift = critic.should_retrain()
print(f"\n  Should retrain? {'YES ⚠️' if should_retrain_drift else 'NO'}")
print(f"  Reason: {reason_drift}")

if drift_report.failure_modes:
    print(f"\n  New failure modes:")
    for mode in drift_report.failure_modes:
        print(f"    • {mode}")

# --- PHASE 5: EXPORT ---
print("\n" + "=" * 70)
print("PHASE 5: EXPORTING REPORTS")
print("=" * 70)

report_path_1 = "/tmp/critic_report_nominal.json"
report_path_2 = "/tmp/critic_report_drift.json"

# Need to re-observe to regenerate first report
critic.reset()
for i, x in enumerate(X_test[:100]):
    x_reshaped = x.reshape(1, -1)
    prediction, gate_value = agent.predict(x_reshaped)
    ground_truth = env.get_reality(x_reshaped)
    critic.observe(x, prediction, ground_truth, gate_value)

critic.export_report(report_path_1)
print(f"\n✓ Nominal report exported to: {report_path_1}")

# Reset and observe drift scenario
critic.reset()
for x in X_drift:
    x_reshaped = x.reshape(1, -1)
    prediction, gate_value = agent.predict(x_reshaped)
    ground_truth = drifted_env.get_reality(x_reshaped)
    critic.observe(x, prediction, ground_truth, gate_value)

critic.export_report(report_path_2)
print(f"✓ Drift report exported to: {report_path_2}")

# --- SUMMARY ---
print("\n" + "=" * 70)
print("DEMONSTRATION COMPLETE")
print("=" * 70)
print("\n🎯 Key Insights:")
print("\n  1. THE CRITIC AS META-COGNITION:")
print("     The critic operates on a different timescale than the agent.")
print("     While the agent makes real-time predictions, the critic performs")
print("     periodic analysis to ensure decision-making remains sound.")
print("\n  2. GATE ALIGNMENT VALIDATION:")
print("     The critic doesn't just check prediction accuracy—it validates")
print("     that the gate is making SENSIBLE decisions about when to trust")
print("     physics vs neural intuition.")
print("\n  3. PROACTIVE FAILURE DETECTION:")
print("     Rather than waiting for catastrophic failure, the critic")
print("     detects subtle degradation patterns and recommends intervention.")
print("\n  4. CONCEPT DRIFT DETECTION:")
print("     When the environment changes (turbulence increased by 30%),")
print("     the critic detects the performance degradation and triggers")
print("     a retrain signal BEFORE critical failure.")
print("\n  5. FOUNDATION FOR SELF-EVOLUTION:")
print("     This critic-agent architecture is the building block for")
print("     self-evolving systems. The critic provides the feedback loop")
print("     needed for autonomous improvement.")
print("\n💡 Next Steps for Full Self-Evolution:")
print("   → Add automated retraining pipeline (critic → retrain → deploy)")
print("   → Implement A/B testing for agent versions")
print("   → Add safety constraints (critic can veto unsafe changes)")
print("   → Meta-critic to evaluate the critic itself")
print("\n" + "=" * 70)
