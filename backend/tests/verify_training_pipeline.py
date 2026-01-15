import sys
import os
import asyncio
import csv

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import run_orchestrator
from agents.training_agent import TrainingAgent

async def verify_pipeline():
    print("--- Verifying ML Data Pipeline ---")
    
    # 1. Clean previous data
    agent = TrainingAgent()
    csv_path = agent.csv_path
    if os.path.exists(csv_path):
        os.remove(csv_path)
        # Re-init to write header
        TrainingAgent()
        
    print(f"Data Source: {csv_path}")
    
    # 2. Run Orchestration (1 Cycle)
    print("Running Simulation...")
    final_state = await run_orchestrator("Design a heavy drone for high winds", "ml_test_proj")
    
    # 3. Check CSV
    if not os.path.exists(csv_path):
        print("❌ FAIL: CSV file not created.")
        sys.exit(1)
        
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        
    print(f"Rows found: {len(rows)}")
    # Row 0 = Header, Row 1 = Data
    if len(rows) >= 2:
        header = rows[0]
        data = rows[1]
        print(f"Header: {header}")
        print(f"Data: {data}")
        
        # Verify specific columns
        # Index 5 = physics_safe, Index 6 = manufacturable
        # Index 9 = thrust_req
        try:
            thrust = float(data[9])
            print(f"Logged Thrust Requirement: {thrust} N")
            assert thrust > 0
            print("✅ PASS: Pipeline recorded valid simulation data.")
        except ValueError:
             print("❌ FAIL: Thrust column is not a valid number.")
    else:
        print("❌ FAIL: No data row generated.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(verify_pipeline())
