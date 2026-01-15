from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class CostAgent:
    """
    Cost Estimation Agent.
    Calculates BoM and manufacturing costs.
    """
    def __init__(self):
        self.name = "CostAgent"
        self.db_path = "data/materials.db"

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"{self.name} calculating cost estimate...")
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Inputs
        mass_kg = params.get("mass_kg", 5.0)
        mat_name = params.get("material_name", "Aluminum 6061") # Input param
        process = params.get("manufacturing_process", "cnc_milling") # Input param
        machining_hours = params.get("processing_time_hr", 2.0)
        
        # 1. Get Material Cost
        cursor.execute("SELECT cost_per_kg FROM alloys WHERE name = ?", (mat_name,))
        row = cursor.fetchone()
        material_cost_per_kg = 20.0 # Default fallback
        if row:
             if row["cost_per_kg"]:
                 material_cost_per_kg = row["cost_per_kg"]
        else:
             logger.warning(f"Material '{mat_name}' not found. Using default ${material_cost_per_kg}/kg")
             
        # 2. Get Machine Rate
        cursor.execute("SELECT rate_per_hr, setup_cost FROM manufacturing_rates WHERE process = ?", (process,))
        row = cursor.fetchone()
        machine_rate_hr = 120.0
        setup_cost = 0.0
        if row:
            machine_rate_hr = row["rate_per_hr"]
            setup_cost = row["setup_cost"]
        else:
            logger.warning(f"Process '{process}' not found. Using default ${machine_rate_hr}/hr")
            
        conn.close()
        
        # Calculation
        raw_mat_cost = mass_kg * material_cost_per_kg
        process_cost = (machining_hours * machine_rate_hr) + setup_cost
        total_cost = raw_mat_cost + process_cost
        
        return {
            "status": "success",
            "total_cost_usd": round(total_cost, 2),
            "breakdown": {
                "material": round(raw_mat_cost, 2),
                "processing": round(process_cost, 2),
                "setup": round(setup_cost, 2)
            },
            "logs": [
                f"Material: ${raw_mat_cost:.2f} ({mat_name}, {mass_kg}kg @ ${material_cost_per_kg}/kg)",
                f"Process: ${process_cost:.2f} ({process}, {machining_hours}hr @ ${machine_rate_hr}/hr + ${setup_cost} setup)",
                f"Total Unit Cost: ${total_cost:.2f}"
            ]
        }
