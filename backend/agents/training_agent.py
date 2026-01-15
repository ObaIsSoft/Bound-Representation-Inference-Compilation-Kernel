import csv
import os
import json
from typing import Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TrainingAgent:
    """
    The 'Scribe'.
    Passive agent that logs simulation execution data to a dataset.
    This data fuels the training of ML surrogates and optimization models.
    """
    def __init__(self, output_dir: str = "data"):
        self.name = "TrainingAgent"
        self.output_dir = output_dir
        self.csv_path = os.path.join(self.output_dir, "training_data.csv")
        
        # Ensure data directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Define Schema (Columns)
        self.columns = [
            "timestamp",
            "session_id",
            # Inputs (Features X)
            "regime",
            "material_name",
            "geometry_mass",
            # Validation (Labels Y)
            "physics_safe",
            "manufacturable",
            # Metrics (Regression Targets)
            "cost_estimate",
            "max_stress_mpa",
            "flight_thrust_req_n",
            "corrosion_score"
        ]
        
        # Initialize CSV header if new file
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.columns)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log the current state to the training dataset.
        Returns metadata about the logging operation.
        """
        logger.info(f"{self.name} logging simulation data...")
        
        # 1. Flatten State
        timestamp = datetime.now().isoformat()
        session_id = state.get("session_id", "unknown")
        
        # Inputs
        env = state.get("environment", {})
        regime = env.get("regime", "GROUND")
        
        # Try to extract material info (simplified)
        # Assuming single material for now or taking the first part's material
        material_name = "unknown"
        geo_tree = state.get("geometry", [])
        mass = 0.0
        if geo_tree and len(geo_tree) > 0:
            part0 = geo_tree[0]
            material_name = part0.get("material", {}).get("name", "unknown")
            # Sum mass
            for p in geo_tree:
                mass += p.get("mass_kg", 0.0)
        
        # Validation Flags
        flags = state.get("validation_flags", {})
        phys_safe = flags.get("physics_safe", True)
        # Mfg safe might be buried in manufacturing results or top level flags
        # Let's check results
        mfg_res = state.get("bom_analysis", {})
        # manufacturability is strictly in validation flags usually, but check BOM
        # BOM doesn't usually have 'manufacturable' flag, using default True if not found
        is_mfg = state.get("validation_flags", {}).get("manufacturing_feasible", True)
        cost = mfg_res.get("total_cost_currency", 0.0)
        
        # Physics Metrics
        # Schema defines 'physics_predictions' as top-level key
        predictions = state.get("physics_predictions", {})
        thrust_req = predictions.get("required_thrust_N", 0.0)
        
        # Other metric placeholders
        stress = 0.0 # From StructuralAgent if generic
        corrosion = 0.0 # From ChemistryAgent
        
        # 2. Construct Row
        row = [
            timestamp,
            session_id,
            regime,
            material_name,
            round(mass, 3),
            phys_safe,
            is_mfg,
            round(cost, 2),
            stress,
            thrust_req,
            corrosion
        ]
        
        # 3. Append to CSV
        try:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            return {
                "status": "success",
                "log_file": self.csv_path,
                "rows_recorded": 1
            }
        except Exception as e:
            logger.error(f"Failed to log training data: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
