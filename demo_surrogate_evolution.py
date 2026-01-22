"""
Demonstration of SurrogateAgent Self-Evolution

Shows:
1. Surrogate making predictions
2. Critic monitoring performance
3. Detecting drift when environment changes
4. Automatic retrain trigger
5. Improved performance after evolution
"""

import sys
sys.path.append('backend')

import numpy as np
from agents.surrogate_agent import SurrogateAgent
from agents.critics.SurrogateCritic import SurrogateCritic

def print_section(title):
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)

print_section("SURROGATE AGENT SELF-EVOLUTION DEMO")

# Initialize
surrogate = SurrogateAgent()
critic = SurrogateCritic(window_size=50, drift_threshold=0.15)

print("\n✓ Initialized SurrogateAgent and SurrogateCritic")
print(f"  Drift threshold: 15% error triggers retrain")

# ============================================================================
# PHASE 1: NOMINAL OPERATION
# ============================================================================

print_section("PHASE 1: NOMINAL OPERATION (50 designs)")

np.random.seed(42)

for i in range(50):
    # Simulate design state
    mass = np.random.uniform(0.5, 5.0)
    cost = np.random.uniform(10, 200)
    
    state = {
        "geometry_tree": [{"mass_kg": mass}],
        "bom_analysis": {"total_cost_currency": cost}
    }
    
    # Surrogate prediction
    prediction = surrogate.run(state)
    
    # Every 10th design: validate against ground truth
    if i % 10 == 0 and prediction.get("status") == "predicted":
        validation = surrogate.validate_prediction(prediction, {
            "position": [0, 10, 0],  # Safe position
            "context": {"terrain": {"obstacles": []}}
        })
        
        critic.observe(state, prediction, validation)
    else:
        critic.observe(state, prediction)

print(f"\n✓ Completed 50 nominal predictions")
print(f"  Validated: {critic.validated_predictions}")

# Analyze
report1 = critic.analyze()
print(f"\n📊 Performance Report:")
print(f"  Accuracy: {report1.get('accuracy', 0):.0%}")
print(f"  Mean Error: {report1.get('mean_error', 0):.1%}")
print(f"  False Negatives: {report1.get('false_negatives', 0)}")
print(f"  Drift Rate: {report1.get('drift_rate', 0):.0%}")

# Evolution check
should_evolve, reason, strategy = critic.should_evolve()
print(f"\n🔄 Evolution needed? {should_evolve}")
print(f"   Reason: {reason}")

# ============================================================================
# PHASE 2: ENVIRONMENT CHANGE (Simulated Drift)
# ============================================================================

print_section("PHASE 2: ENVIRONMENT CHANGE - SIMULATING DRIFT")

print("\n⚠️  Introducing concept drift:")
print("  • Designs now have higher complexity (more failure cases)")
print("  • Surrogate model hasn't seen this Distribution")

# Simulate drift by introducing more failures
for i in range(30):
    mass = np.random.uniform(5.0, 10.0)  # Heavier designs (more likely to fail)
    cost = np.random.uniform(200, 500)
    
    state = {
        "geometry_tree": [{"mass_kg": mass}],
        "bom_analysis": {"total_cost_currency": cost}
    }
    
    prediction = surrogate.run(state)
    
    # Validate more frequently during drift
    if i % 5 == 0 and prediction.get("status") == "predicted":
        # Simulate failure (high mass = unsafe in this scenario)
        is_safe = mass < 7.0
        
        validation = {
            "verified": False,  # Mismatch
            "ground_truth": "SAFE" if is_safe else "COLLISION",
            "prediction": "SAFE" if prediction.get("recommendation") == "PROCEED" else "COLLISION",
            "sdf_value": 1.0 if is_safe else -1.0,
            "drift_alert": True
        }
        
        critic.observe(state, prediction, validation)
    else:
        critic.observe(state, prediction)

print(f"\n✓ Completed 30 drift-affected predictions")

# Analyze after drift
report2 = critic.analyze()
print(f"\n📊 Performance Report (After Drift):")
print(f"  Accuracy: {report2.get('accuracy', 0):.0%}")
print(f"  Mean Error: {report2.get('mean_error', 0):.1%}")
print(f"  Recent Error: {report2.get('recent_error', 0):.1%}")
print(f"  Drift Alerts: {critic.drift_alerts}")
print(f"  False Negatives: {report2.get('false_negatives', 0)}")

print(f"\n⚠️  Failure Modes:")
for mode in report2.get("failure_modes", []):
    print(f"  • {mode}")

print(f"\n💡 Recommendations:")
for rec in report2.get("recommendations", [])[:5]:
    print(f"  • {rec}")

# ============================================================================
# PHASE 3: EVOLUTION DECISION
# ============================================================================

print_section("PHASE 3: EVOLUTION DECISION")

should_evolve, reason, strategy = critic.should_evolve()

print(f"\n🔍 Evolution Analysis:")
print(f"  Should Evolve: {'YES' if should_evolve else 'NO'}")
print(f"  Reason: {reason}")
if strategy:
    print(f"  Strategy: {strategy}")

if should_evolve:
    print(f"\n🔄 Triggering Evolution Pipeline:")
    print(f"  1. Extract training data from {critic.validated_predictions} validated samples")
    print(f"  2. Retrain neural surrogate with updated distribution")
    print(f"  3. A/B test: old model vs new model")
    print(f"  4. Deploy if improvement > 20%")
    
    # Extract training data
    X, y = critic.get_training_data()
    print(f"\n📊 Training Data:")
    print(f"  Features: {X.shape}")
    print(f"  Labels: {y.shape}")
    
    print(f"\n✅ Evolution would be triggered in production")
    print(f"   TrainingAgent would retrain SurrogateAgent automatically")
else:
    print(f"\n✅ Surrogate still performing nominally - no evolution needed")

# ============================================================================
# PHASE 4: EXPORT REPORTS
# ============================================================================

print_section("PHASE 4: EXPORTING REPORTS")

import os
os.makedirs("/tmp/surrogate_evolution", exist_ok=True)

critic.export_report("/tmp/surrogate_evolution/surrogate_report_nominal.json")

print("\n✓ Exported detailed report:")
print("  • /tmp/surrogate_evolution/surrogate_report_nominal.json")

# ============================================================================
# SUMMARY
# ============================================================================

print_section("SUMMARY: SURROGATE SELF-EVOLUTION")

print("\n🎯 Key Achievements:")
print(f"  • Monitored {critic.total_predictions} predictions")
print(f"  • Validated {critic.validated_predictions} against ground truth")
print(f"  • Detected drift: {critic.drift_alerts} alerts")
print(f"  • Identified {len(report2.get('failure_modes', []))} failure modes")

print("\n📊 Performance Degradation Detected:")
print(f"  Before Drift: {report1.get('accuracy', 0):.0%} accuracy")
print(f"  After Drift: {report2.get('accuracy', 0):.0%} accuracy")
print(f"  Error increase: {report1.get('mean_error', 0):.1%} → {report2.get('recent_error', 0):.1%}")

print("\n💡 Self-Evolution Trigger:")
if should_evolve:
    print(f"  ✅ EVOLUTION TRIGGERED")
    print(f"     Reason: {reason}")
    print(f"     Action: {strategy}")
else:
    print(f"  ❌ No evolution needed - performance nominal")

print("\n🚀 Production Integration:")
print("  → Add to orchestrator: observe() after each surrogate prediction")
print("  → Periodic analysis: Every 50 designs")
print("  → Auto-retrain: When error > 15% OR false negatives detected")
print("  → Safety gate: User approval for safety-critical retraining")

print("\n" + "=" * 80)
