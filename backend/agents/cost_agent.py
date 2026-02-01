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
        
        try:
            from models.cost_surrogate import CostSurrogate
            self.surrogate = CostSurrogate()
            self.use_surrogate = True
        except ImportError:
            self.surrogate = None
            self.use_surrogate = False

    def quick_estimate(self, params: Dict[str, Any], currency: str = "USD") -> Dict[str, Any]:
        """
        Quick cost estimate for Phase 1 feasibility check.
        Uses simplified calculations without database lookups.
        
        Args:
            params: {
                "mass_kg": float,
                "material_name": str (optional),
                "complexity": str (optional: "simple", "moderate", "complex")
            }
            currency: Target currency code (USD, EUR, GBP, JPY, CAD)
        
        Returns:
            {
                "estimated_cost": float,
                "currency": str,
                "confidence": float (0-1),
                "feasible": bool
            }
        """
        logger.info(f"{self.name} performing quick cost estimate...")
        
        mass_kg = params.get("mass_kg", 5.0)
        material = params.get("material_name", "Aluminum 6061")
        complexity = params.get("complexity", "moderate")
        
        # Simplified material cost lookup (no DB)
        material_costs = {
            "Aluminum 6061": 20.0,
            "Steel": 15.0,
            "Titanium": 150.0,
            "Carbon Fiber": 200.0,
            "PLA": 25.0,
            "ABS": 30.0
        }
        
        material_cost_per_kg = material_costs.get(material, 50.0)  # Default fallback
        
        # Complexity multiplier for processing
        complexity_multipliers = {
            "simple": 1.0,
            "moderate": 1.5,
            "complex": 2.5
        }
        
        complexity_mult = complexity_multipliers.get(complexity, 1.5)
        
        # Quick estimate (USD Base)
        raw_material_cost_usd = mass_kg * material_cost_per_kg
        processing_cost_usd = raw_material_cost_usd * complexity_mult
        total_cost_usd = raw_material_cost_usd + processing_cost_usd
        
        # Currency Conversion Rates (Approximated)
        rates = {
            "USD": 1.0,
            "EUR": 0.92,
            "GBP": 0.79,
            "JPY": 150.0,
            "CAD": 1.35
        }
        rate = rates.get(currency.upper(), 1.0)
        
        total_cost = total_cost_usd * rate
        raw_material_cost = raw_material_cost_usd * rate
        processing_cost = processing_cost_usd * rate
        
        # Feasibility check (budget threshold $100k USD)
        budget_threshold_usd = 100000.0
        feasible = total_cost_usd < budget_threshold_usd
        
        # Confidence based on whether we found the material
        confidence = 0.9 if material in material_costs else 0.6
        
        return {
            "estimated_cost": round(total_cost, 2),
            "currency": currency.upper(),
            "confidence": confidence,
            "feasible": feasible,
            "breakdown": {
                "material": round(raw_material_cost, 2),
                "processing": round(processing_cost, 2)
            },
            "logs": [
                f"Quick estimate: {currency.upper()} {total_cost:,.2f} ({material}, {mass_kg}kg)",
                f"Feasibility: {'PASS' if feasible else 'FAIL'} (threshold: $100k USD)"
            ]
        }

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
        
        # Tier 5: Market Dynamic Adjustment
        market_multiplier = 1.0
        if self.use_surrogate and self.surrogate:
            try:
                import time
                now = time.time()
                market_multiplier = self.surrogate.predict_price_multiplier(now)
                logger.info(f"Market Adjustment: {market_multiplier:.2f}x (Predicted by CostSurrogate)")
            except Exception as e:
                logger.warning(f"CostSurrogate failed: {e}")
        
        # Calculation
        raw_mat_cost = mass_kg * material_cost_per_kg * market_multiplier
        process_cost = (machining_hours * machine_rate_hr) + setup_cost
        total_cost = raw_mat_cost + process_cost
        
        return {
            "status": "success",
            "total_cost_usd": round(total_cost, 2),
            "breakdown": {
                "material": round(raw_mat_cost, 2),
                "processing": round(process_cost, 2),
                "setup": round(setup_cost, 2),
                "market_multiplier": round(market_multiplier, 3)
            },
            "logs": [
                f"Material: ${raw_mat_cost:.2f} ({mat_name}, {mass_kg}kg @ ${material_cost_per_kg}/kg)",
                f"Process: ${process_cost:.2f} ({process}, {machining_hours}hr @ ${machine_rate_hr}/hr + ${setup_cost} setup)",
                f"Total Unit Cost: ${total_cost:.2f}"
            ]
        }

    def generate_cost_artifact(self, state: Dict[str, Any], project_id: str) -> Dict[str, Any]:
        """
        Generate a structured Cost Breakdown Artifact.
        """
        # Run calculation if not present
        if "cost_estimate" not in state:
             est = self.run(state.get("cost_params", {}))
        else:
             est = state["cost_estimate"]
             
        breakdown = est.get("breakdown", {})
        total = est.get("total_cost_usd", 0)
        
        md_table = f"""### Cost Breakdown

| Category | Cost (USD) |
|----------|------------|
| Materials | ${breakdown.get('material', 0):,.2f} |
| Manufacturing | ${breakdown.get('processing', 0):,.2f} |
| Components | ${breakdown.get('components', 0):,.2f} |
| Labor | ${breakdown.get('setup', 0):,.2f} |
| **Total** | **${total:,.2f}** |
"""
        return {
            "id": f"cost-{project_id}",
            "type": "cost_breakdown",
            "title": "Project Cost Estimate",
            "content": md_table,
            "comments": [] # Interactive support
        }
