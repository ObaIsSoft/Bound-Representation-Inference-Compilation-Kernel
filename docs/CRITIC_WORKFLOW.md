# Critic Report Action Workflow

When critics generate reports, here's what to do with them:

## 1. Triage by Severity (Automated)

```python
# Priority levels based on critic output
CRITICAL = ["conservation violations", "false negatives", "melting failures", "short circuit misses"]
HIGH = ["power deficit", "safety accuracy < 70%", "installation failures"]
MEDIUM = ["db coverage < 60%", "over-specification", "error rate > 10%"]
LOW = ["diversity issues", "false positives", "conservative bias"]
```

### Current Report Triage:

| Priority | Issue | Agent | Action Required |
|----------|-------|-------|----------------|
| 🚨 **CRITICAL** | Kirchhoff's Law violated | PhysicsOracle | **Immediate fix** |
| ⚠️ **MEDIUM** | DB coverage 50% | MaterialAgent | Expand database |
| ⚠️ **MEDIUM** | 61% over-spec rate | ComponentAgent | Tune selection |

---

## 2. For CRITICAL Issues (Immediate Action)

### Example: PhysicsOracle Conservation Violation

**Problem Found:**
```json
{
  "violation": "Kirchhoff's Current Law violated: I_in=5.595A, I_out=6.434A",
  "domain": "CIRCUIT"
}
```

**Automated Response:**

```python
# 1. Alert system admin
send_alert("PhysicsOracle", "CONSERVATION_LAW_VIOLATION", severity="CRITICAL")

# 2. Disable affected oracle domain (safety measure)
oracle_registry["physics"].disable_domain("CIRCUIT")

# 3. Log detailed diagnostic
with open("oracle_failure_2026_01_18.log", "w") as f:
    f.write(json.dumps(violation_details, indent=2))

# 4. Trigger investigation workflow
investigation = {
    "task": "Debug PhysicsOracle circuit calculations",
    "files_to_check": [
        "backend/agents/physics_oracle/adapters/circuit.py",
        "backend/agents/physics_oracle/physics_oracle.py"
    ],
    "test_case": {
        "voltage": 42.9,
        "power": 240.0,
        "expected": {"current_in": 5.595, "current_out": 5.595}
    }
}
```

**Manual Fix Required:**
1. Check if floating-point rounding error or actual bug
2. Add unit test for this specific case
3. Re-enable domain after fix + validation

---

## 3. For HIGH/MEDIUM Issues (Scheduled Evolution)

### Example: Material Database Expansion

**Problem:**
```json
{
  "db_coverage": 0.5,
  "db_misses": 15,
  "top_materials": [["Aluminum 6061", 18], ["Steel", 10], ["Magnesium", 2]]
}
```

**Automated Response:**

```python
# Create work item in evolution queue
evolution_queue.add({
    "agent": "MaterialAgent",
    "priority": "MEDIUM",
    "strategy": "EXPAND_MATERIAL_DATABASE",
    "data_collection": {
        "missing_materials": log_missing_materials(),  # Track what was requested but not found
        "frequency": count_material_requests(),
        "environments": get_operating_conditions()
    },
    "estimated_effort": "2 hours",
    "auto_approve": True  # Non-safety-critical
})
```

**What Actually Happens:**

```python
# TrainingAgent picks up the task
class TrainingAgent:
    def execute_evolution(self, task):
        if task["strategy"] == "EXPAND_MATERIAL_DATABASE":
            # 1. Scrape materials data
            new_materials = self.scrape_materials_database(
                materials=task["data_collection"]["missing_materials"]
            )
            
            # 2. Add to Supabase
            for mat in new_materials:
                db.table("alloys").insert(mat)
            
            # 3. Validate
            test_result = material_agent.run("Titanium Ti-6Al-4V", 100)
            assert "Generic" not in test_result["name"]
            
            # 4. Report completion
            return {
                "status": "complete",
                "materials_added": len(new_materials),
                "new_coverage": run_coverage_test()
            }
```

---

## 4. Integration into Orchestrator

Add critic checkpoints to the design loop:

```python
# backend/orchestrator.py

async def run_orchestrator(user_intent: str, project_id: str):
    # ... existing setup ...
    
    # Initialize critics
    critics = {
        "chemistry": ChemistryCritic(),
        "electronics": ElectronicsCritic(),
        "material": MaterialCritic(),
        "component": ComponentCritic(),
        "oracle": OracleCritic()
    }
    
    # Main execution loop
    final_state = await app.ainvoke(initial_state)
    
    # POST-EXECUTION: Critic Analysis
    critic_reports = {}
    evolution_queue = []
    
    for name, critic in critics.items():
        if critic.total_evaluations > 0:
            report = critic.analyze()
            critic_reports[name] = report
            
            # Check evolution trigger
            should_evolve, reason, strategy = critic.should_evolve()
            
            # CRITICAL issues: immediate alert
            if "CRITICAL" in reason:
                logger.critical(f"{name}: {reason}")
                send_slack_alert(f"🚨 {name}: {reason}")
                
                # Disable affected system
                if name == "OracleCritic":
                    # Don't break entire system, but flag for review
                    final_state["warnings"].append({
                        "type": "ORACLE_FAILURE",
                        "message": reason,
                        "action": "Manual review required"
                    })
            
            # Queue non-critical evolution
            elif should_evolve:
                evolution_queue.append({
                    "agent": name,
                    "reason": reason,
                    "strategy": strategy,
                    "priority": classify_priority(reason),
                    "auto_approve": is_safe_to_auto_approve(name, strategy)
                })
    
    # Execute auto-approved evolutions
    for evolution in evolution_queue:
        if evolution["auto_approve"]:
            logger.info(f"Auto-executing evolution: {evolution}")
            training_agent.execute_evolution(evolution)
        else:
            # Add to user dashboard for approval
            final_state["pending_evolutions"].append(evolution)
    
    # Add reports to final state
    final_state["critic_reports"] = critic_reports
    
    return final_state
```

---

## 5. User Dashboard (Frontend)

Create a UI to show critic status:

```jsx
// frontend/src/components/critics/CriticDashboard.jsx

function CriticDashboard() {
    const [reports, setReports] = useState({});
    const [evolutionQueue, setEvolutionQueue] = useState([]);
    
    useEffect(() => {
        // Poll for critic reports
        const interval = setInterval(async () => {
            const res = await fetch('/api/critic/status');
            const data = await res.json();
            setReports(data.reports);
            setEvolutionQueue(data.evolution_queue);
        }, 5000);
        
        return () => clearInterval(interval);
    }, []);
    
    return (
        <div className="critic-dashboard">
            <h2>Agent Health Monitoring</h2>
            
            {/* Critical Alerts */}
            {evolutionQueue.filter(e => e.priority === 'CRITICAL').map(evolution => (
                <Alert severity="error" key={evolution.agent}>
                    <strong>{evolution.agent}:</strong> {evolution.reason}
                    <Button onClick={() => handleApprove(evolution)}>
                        Approve Fix
                    </Button>
                </Alert>
            ))}
            
            {/* Per-Agent Status */}
            {Object.entries(reports).map(([name, report]) => (
                <CriticCard 
                    key={name}
                    name={name}
                    confidence={report.confidence}
                    failureModes={report.failure_modes}
                    recommendations={report.recommendations}
                />
            ))}
            
            {/* Evolution Queue */}
            <EvolutionQueue items={evolutionQueue} />
        </div>
    );
}
```

---

## 6. API Endpoints

```python
# backend/main.py

@app.get("/api/critic/status")
async def get_critic_status():
    """Returns current critic reports and evolution queue."""
    reports = {}
    for name, critic in global_critics.items():
        if critic.total_evaluations > 0:
            reports[name] = critic.analyze()
    
    return {
        "reports": reports,
        "evolution_queue": evolution_queue.get_pending(),
        "last_updated": datetime.now().isoformat()
    }

@app.post("/api/critic/approve_evolution")
async def approve_evolution(request: EvolutionApprovalRequest):
    """User approves a proposed evolution."""
    evolution = evolution_queue.get(request.evolution_id)
    
    if request.approved:
        # Execute the evolution
        result = training_agent.execute_evolution(evolution)
        return {"status": "executed", "result": result}
    else:
        # User rejected - log reason
        evolution_queue.reject(request.evolution_id, request.reason)
        return {"status": "rejected"}

@app.get("/api/critic/history/{agent_name}")
async def get_evolution_history(agent_name: str):
    """Returns past evolution events for an agent."""
    return {
        "agent": agent_name,
        "evolutions": evolution_log.filter(agent=agent_name),
        "current_version": agent_registry[agent_name].version
    }
```

---

## 7. Practical Action Plan (Based on Current Reports)

### Immediate (This Week):

**1. Fix PhysicsOracle Conservation Violation** ⚠️ CRITICAL
```bash
# Investigate the bug
cd backend/agents/physics_oracle
# Check circuit calculations
# Add test case for Kirchhoff's law
pytest adapters/test_circuit.py -v
```

### Short-term (This Sprint):

**2. Expand Material Database**
```bash
# Add 50 common materials
python scripts/populate_materials_db.py --source materials_handbook.json
```

**3. Optimize Component Selection**
```python
# Tune selection algorithm to prefer closer matches
component_agent.set_selection_strategy("cost_to_spec_ratio")
```

### Long-term (Next Month):

**4. Integrate Critics into Production**
- Add observation hooks to all agent nodes
- Deploy critic dashboard to staging
- Set up Slack/email alerts for critical issues
- Implement automated evolution for non-critical agents

---

## Summary: The Complete Loop

```
1. Agent executes → 2. Critic observes → 3. Report generated
                                                    ↓
                              4. Triage (CRITICAL vs MEDIUM vs LOW)
                                                    ↓
    ┌────────────────────────────┬────────────────────────────┐
    ↓                            ↓                            ↓
CRITICAL                      HIGH/MEDIUM                    LOW
Alert admin                   Queue evolution             Log for review
Disable system                Auto-execute if safe        Monthly cleanup
Manual fix required           Update dashboard            Trend analysis
    ↓                            ↓                            ↓
5. Fix implemented → 6. Re-test → 7. Deploy → 8. Monitor again (loop)
```

**You asked: "What do we do with this report?"**

**Answer:**
1. **Right now**: Fix the PhysicsOracle conservation violation (file a bug)
2. **This week**: Expand material database (run data import script)
3. **Ongoing**: Integrate critics into orchestrator for continuous monitoring
4. **Long-term**: Build the dashboard so you can see agent health in real-time

The reports are **actionable intelligence** - they tell you exactly what's broken and how to fix it!
