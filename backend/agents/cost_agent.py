"""
CostAgent: Cost Estimation Agent.

Uses database-driven pricing and external APIs.
No hardcoded costs - all prices come from services.
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class CostAgent:
    """
    Cost Estimation Agent with database-driven pricing.
    
    Material prices: From pricing_service (Metals-API, Yahoo Finance, or manual entry)
    Manufacturing rates: From Supabase (supplier quotes)
    Currency conversion: From currency_service (real-time APIs)
    
    No hardcoded costs - fails if pricing not available.
    """
    
    def __init__(self):
        self.name = "CostAgent"
        self._initialized = False
        
        try:
            from models.cost_surrogate import CostSurrogate
            self.surrogate = CostSurrogate()
            self.use_surrogate = True
        except ImportError:
            self.surrogate = None
            self.use_surrogate = False
    
    async def initialize(self):
        """Initialize services"""
        if self._initialized:
            return
            
        from backend.services import pricing_service
        await pricing_service.initialize()
        
        self._initialized = True
    
    async def quick_estimate(
        self, 
        params: Dict[str, Any], 
        currency: str = "USD"
    ) -> Dict[str, Any]:
        """
        Quick cost estimate using database-driven pricing.
        
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
                "feasible": bool,
                "warnings": List[str],
                "data_sources": {...}
            }
        """
        logger.info(f"{self.name} performing quick cost estimate...")
        
        await self.initialize()
        
        from backend.services import pricing_service, currency_service, supabase
        
        mass_kg = params.get("mass_kg")
        if mass_kg is None:
            return {"error": "mass_kg is required", "feasible": False}
        
        material = params.get("material_name")
        if not material:
            return {"error": "material_name is required", "feasible": False}
        
        complexity = params.get("complexity", "moderate")
        
        warnings = []
        data_sources = {}
        
        # 1. Get material cost from pricing service (APIs + database)
        material_price = None
        try:
            material_price = await pricing_service.get_material_price(material, currency)
            if material_price:
                material_cost_per_kg = material_price.price
                data_sources["material_price"] = material_price.source
                logger.info(f"Got price for {material}: ${material_cost_per_kg} {currency}/kg from {material_price.source}")
        except Exception as e:
            logger.warning(f"Pricing service error for {material}: {e}")
        
        # Fallback: Try to get from Supabase materials table
        if material_price is None:
            try:
                mat_data = await supabase.get_material(material)
                price_column = f"cost_per_kg_{currency.lower()}"
                if mat_data and mat_data.get(price_column):
                    material_cost_per_kg = float(mat_data[price_column])
                    material_price_source = mat_data.get("pricing_data_source", "database")
                    data_sources["material_price"] = f"database:{material_price_source}"
                    warnings.append(
                        f"Using cached price for {material}: {material_cost_per_kg} {currency}/kg"
                    )
                else:
                    # Return error - no price available
                    return {
                        "error": f"No price available for {material}",
                        "solution": "Options: 1) Set METALS_API_KEY (free tier at metals-api.com), 2) Install yfinance (pip install yfinance), or 3) Set price manually via /api/pricing/set-price",
                        "estimated_cost": None,
                        "confidence": 0.0,
                        "feasible": False,
                        "data_sources": {}
                    }
            except Exception as e:
                return {
                    "error": f"Material '{material}' not found in database",
                    "solution": f"Add material to database or check material name. Error: {e}",
                    "estimated_cost": None,
                    "confidence": 0.0,
                    "feasible": False
                }
        
        # 2. Get currency conversion rate if needed
        if currency.upper() != "USD":
            try:
                exchange_rate = await currency_service.get_rate("USD", currency)
                if exchange_rate:
                    data_sources["currency"] = "api"
                else:
                    warnings.append(
                        f"Currency conversion rate USD->{currency} not available. Using 1.0"
                    )
                    exchange_rate = 1.0
            except Exception as e:
                warnings.append(f"Currency service error: {e}")
                exchange_rate = 1.0
        else:
            exchange_rate = 1.0
        
        # 3. Complexity multiplier (these are relative, not absolute costs)
        complexity_multipliers = {
            "simple": 1.0,
            "moderate": 1.5,
            "complex": 2.5
        }
        complexity_mult = complexity_multipliers.get(complexity, 1.5)
        
        # 4. Calculate costs
        raw_material_cost = mass_kg * material_cost_per_kg * exchange_rate
        processing_cost = raw_material_cost * complexity_mult
        total_cost = raw_material_cost + processing_cost
        
        # 5. Feasibility check
        budget_threshold = params.get("budget_threshold")
        if budget_threshold is None:
            budget_threshold = float('inf')
        feasible = total_cost < budget_threshold
        
        # 6. Confidence based on data source
        source = data_sources.get("material_price", "unknown")
        if "yahoo_finance" in source or "metals-api" in source or "metalpriceapi" in source:
            confidence = 0.9
        elif "database" in source or "cache" in source:
            confidence = 0.7
        else:
            confidence = 0.5
        
        return {
            "estimated_cost": round(total_cost, 2),
            "currency": currency.upper(),
            "confidence": confidence,
            "feasible": feasible,
            "warnings": warnings,
            "breakdown": {
                "material": round(raw_material_cost, 2),
                "processing": round(processing_cost, 2),
                "complexity_multiplier": complexity_mult
            },
            "data_sources": data_sources,
            "logs": [
                f"Quick estimate: {currency.upper()} {total_cost:,.2f} ({material}, {mass_kg}kg)",
                f"Material cost: {material_cost_per_kg} {currency}/kg (source: {source})",
                f"Feasibility: {'PASS' if feasible else 'FAIL'} (threshold: {budget_threshold} {currency})"
            ]
        }
    
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detailed cost estimate with manufacturing rates.
        
        Uses Supabase for manufacturing rates (must be configured).
        """
        logger.info(f"{self.name} calculating detailed cost estimate...")
        
        await self.initialize()
        
        from backend.services import pricing_service, supabase
        
        # Inputs - all required
        mass_kg = params.get("mass_kg")
        if mass_kg is None:
            return {"error": "mass_kg is required", "feasible": False}
        
        mat_name = params.get("material_name")
        if not mat_name:
            return {"error": "material_name is required", "feasible": False}
        
        process = params.get("manufacturing_process")
        if not process:
            return {"error": "manufacturing_process is required", "feasible": False}
        
        region = params.get("region", "global")
        machining_hours = params.get("processing_time_hr")
        if machining_hours is None:
            return {"error": "processing_time_hr is required", "feasible": False}
        
        warnings = []
        
        # 1. Get Material Cost from pricing service
        try:
            material_price = await pricing_service.get_material_price(mat_name, "USD")
            if material_price:
                material_cost_per_kg = material_price.price
                material_source = material_price.source
            else:
                raise ValueError("Price not available from APIs")
        except Exception as e:
            # Fallback to Supabase
            try:
                mat_data = await supabase.get_material(mat_name)
                if mat_data and mat_data.get("cost_per_kg_usd"):
                    material_cost_per_kg = float(mat_data["cost_per_kg_usd"])
                    material_source = mat_data.get("pricing_data_source", "database")
                else:
                    return {
                        "error": f"No price for {mat_name}",
                        "solution": "Set price via pricing_service.set_material_price() or configure METALS_API_KEY"
                    }
            except Exception:
                return {
                    "error": f"Material '{mat_name}' not found",
                    "solution": "Check material name or add to database"
                }
        
        # 2. Get Manufacturing Rates from Supabase
        try:
            rates = await supabase.get_manufacturing_rates(process, region)
            machine_rate_hr = rates["machine_hourly_rate_usd"]
            setup_cost = rates["setup_cost_usd"]
            rate_source = rates.get("data_source", "database")
            
            if rate_source == "estimate":
                warnings.append(
                    f"Manufacturing rate for {process} is an estimate. "
                    f"Get real quote from supplier for accuracy."
                )
                
        except Exception as e:
            return {
                "error": f"Manufacturing rates not found for {process} in {region}",
                "solution": "Add rates to manufacturing_rates table with supplier quote",
                "details": str(e)
            }
        
        # 3. Market Dynamic Adjustment (if surrogate available)
        market_multiplier = 1.0
        if self.use_surrogate and self.surrogate:
            try:
                import time
                now = time.time()
                market_multiplier = self.surrogate.predict_price_multiplier(now)
                logger.info(f"Market Adjustment: {market_multiplier:.2f}x (Predicted by CostSurrogate)")
            except Exception as e:
                logger.warning(f"CostSurrogate failed: {e}")
                warnings.append("Market adjustment unavailable")
        
        # 4. Calculate Costs
        raw_mat_cost = mass_kg * material_cost_per_kg * market_multiplier
        process_cost = (machining_hours * machine_rate_hr) + setup_cost
        total_cost = raw_mat_cost + process_cost
        
        return {
            "status": "success",
            "total_cost_usd": round(total_cost, 2),
            "warnings": warnings,
            "breakdown": {
                "material": round(raw_mat_cost, 2),
                "processing": round(process_cost, 2),
                "setup": round(setup_cost, 2),
                "market_multiplier": round(market_multiplier, 3)
            },
            "data_sources": {
                "material_price": material_source,
                "manufacturing_rate": rate_source
            },
            "logs": [
                f"Material: ${raw_mat_cost:.2f} ({mat_name}, {mass_kg}kg @ ${material_cost_per_kg}/kg)",
                f"Process: ${process_cost:.2f} ({process}, {machining_hours}hr @ ${machine_rate_hr}/hr + ${setup_cost} setup)",
                f"Total Unit Cost: ${total_cost:.2f}"
            ]
        }
    
    def generate_cost_artifact(self, state: Dict[str, Any], project_id: str) -> Dict[str, Any]:
        """Generate a structured Cost Breakdown Artifact."""
        est = state.get("cost_estimate", {})
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

**Data Sources:** {est.get('data_sources', {})}
"""
        return {
            "id": f"cost-{project_id}",
            "type": "cost_breakdown",
            "title": "Project Cost Estimate",
            "content": md_table,
            "comments": []
        }
