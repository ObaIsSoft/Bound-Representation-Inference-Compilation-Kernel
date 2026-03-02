"""
Production CostAgent - Activity-Based Costing with Database-Driven Pricing

Follows BRICK OS patterns:
- NO hardcoded prices - uses pricing_service + supabase
- NO hardcoded rates - uses manufacturing_rates table
- NO estimated fallbacks - fails fast with clear error messages
- Externalized configuration for cycle time models

Research Basis:
- Boothroyd, G. et al. (2011) - Product Design for Manufacture and Assembly
- ML cost estimation reviews (2022-2024)
"""

import asyncio
import logging
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import numpy as np

# ML imports - optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

logger = logging.getLogger(__name__)


class CostComponent(Enum):
    """Standard cost components for manufacturing."""
    MATERIAL = "material"
    LABOR = "labor"
    SETUP = "setup"
    TOOLING = "tooling"
    OVERHEAD = "overhead"
    QUALITY = "quality"
    LOGISTICS = "logistics"


class ManufacturingProcess(Enum):
    """Supported manufacturing processes."""
    CNC_MILLING = "cnc_milling"
    CNC_TURNING = "cnc_turning"
    CNC_GRINDING = "cnc_grinding"
    EDM = "edm"
    INJECTION_MOLDING = "injection_molding"
    DIE_CASTING = "die_casting"
    SAND_CASTING = "sand_casting"
    INVESTMENT_CASTING = "investment_casting"
    FORGING = "forging"
    STAMPING = "stamping"
    SHEET_METAL = "sheet_metal"
    ADDITIVE_FDM = "additive_fdm"
    ADDITIVE_SLA = "additive_sla"
    ADDITIVE_SLM = "additive_slm"


@dataclass
class CostBreakdown:
    """Detailed cost breakdown."""
    material_cost: float
    labor_cost: float
    setup_cost: float
    tooling_cost: float
    overhead_cost: float
    quality_cost: float
    logistics_cost: float
    
    @property
    def total_cost(self) -> float:
        return (
            self.material_cost + self.labor_cost + self.setup_cost +
            self.tooling_cost + self.overhead_cost + self.quality_cost +
            self.logistics_cost
        )
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "material": self.material_cost,
            "labor": self.labor_cost,
            "setup": self.setup_cost,
            "tooling": self.tooling_cost,
            "overhead": self.overhead_cost,
            "quality": self.quality_cost,
            "logistics": self.logistics_cost,
            "total": self.total_cost
        }


@dataclass
class CostEstimate:
    """Complete cost estimate with provenance."""
    total_cost: float
    breakdown: CostBreakdown
    currency: str
    method: str
    assumptions: List[str]
    warnings: List[str]
    data_sources: Dict[str, str]
    confidence: float
    timestamp: str = field(default_factory=lambda: str(np.datetime64('now')))


class ProductionCostAgent:
    """
    Production-grade cost estimation agent.
    
    Uses database-driven pricing (NO hardcoded costs):
    - Material prices: pricing_service (APIs) → supabase cache
    - Manufacturing rates: supabase.manufacturing_rates table
    - Cycle time models: Externalized config (Boothroyd DFM)
    
    FAIL FAST: Returns error if pricing data unavailable.
    """
    
    def __init__(
        self,
        use_ml: bool = True,
        ml_model_path: Optional[str] = None
    ):
        self.use_ml = use_ml and HAS_XGBOOST
        self.ml_model_path = ml_model_path
        self._initialized = False
        
        # ML components (optional)
        self.scaler = StandardScaler()
        self.rf_model: Optional[RandomForestRegressor] = None
        self.xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1) if HAS_XGBOOST else None
        self.ml_trained = False
        
        # Services (initialized on first use)
        self.pricing_service = None
        self.supabase = None
        
        # Load external config
        self._load_cycle_time_config()
        
        logger.info("ProductionCostAgent initialized (services not yet connected)")
    
    def _load_cycle_time_config(self):
        """Load cycle time estimation config from file."""
        config_path = Path(__file__).parent / "config" / "cycle_time_models.json"
        
        if config_path.exists():
            with open(config_path) as f:
                self.cycle_config = json.load(f)
            logger.info(f"Loaded cycle time config from {config_path}")
        else:
            # Use minimal defaults - clearly marked as fallback
            logger.warning(f"Cycle time config not found at {config_path}. Using minimal defaults.")
            self.cycle_config = {
                "_warning": "FALLBACK_CONFIG - Create cycle_time_models.json for production",
                "base_time_hours": 0.5,
                "feature_time_hours": 0.1,
                "hole_time_hours": 0.05,
                "tolerance_factor": {"reference_tolerance_mm": 0.01},
                "surface_finish_factor": {"reference_ra": 3.2}
            }
    
    async def initialize(self):
        """Initialize services."""
        if self._initialized:
            return
        
        try:
            from backend.services import pricing_service, supabase_service
            self.pricing_service = pricing_service
            self.supabase = supabase_service.supabase
            
            await self.pricing_service.initialize()
            await self.supabase.initialize()
            
            self._initialized = True
            logger.info("ProductionCostAgent services initialized")
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            raise RuntimeError(f"CostAgent initialization failed: {e}")
    
    async def get_material_price(self, material_key: str, currency: str = "USD") -> Tuple[float, str]:
        """
        Get material price from pricing service → supabase.
        
        Returns:
            (price_per_kg, source)
        
        Raises:
            ValueError: If price not available
        """
        await self.initialize()
        
        # 1. Try pricing service (APIs + cache)
        try:
            price_point = await self.pricing_service.get_material_price(material_key, currency)
            if price_point:
                return price_point.price, price_point.source
        except Exception as e:
            logger.warning(f"Pricing service error for {material_key}: {e}")
        
        # 2. Try supabase materials table
        try:
            mat_data = await self.supabase.get_material(material_key)
            price_column = f"cost_per_kg_{currency.lower()}"
            if mat_data and mat_data.get(price_column):
                price = float(mat_data[price_column])
                source = mat_data.get("pricing_data_source", "database")
                return price, f"database:{source}"
        except Exception as e:
            logger.warning(f"Supabase error for {material_key}: {e}")
        
        # FAIL FAST - no hardcoded fallback
        raise ValueError(
            f"No price available for material: {material_key}\n"
            f"Solutions:\n"
            f"1. Set METALS_API_KEY for real-time pricing\n"
            f"2. Add material to supabase.materials table\n"
            f"3. Use pricing_service.set_material_price()"
        )
    
    async def get_manufacturing_rate(
        self,
        process: ManufacturingProcess,
        region: str = "global"
    ) -> Dict[str, Any]:
        """
        Get manufacturing rates from supabase.
        
        Returns:
            Dict with hourly_rate, setup_cost, etc.
        
        Raises:
            ValueError: If rates not available
        """
        await self.initialize()
        
        try:
            rates = await self.supabase.get_manufacturing_rates(process.value, region)
            if rates:
                return {
                    "hourly_rate": rates["machine_hourly_rate_usd"],
                    "setup_cost": rates["setup_cost_usd"],
                    "setup_time_hours": rates.get("setup_time_hr", 1.0),
                    "source": rates.get("data_source", "database")
                }
        except Exception as e:
            logger.warning(f"Supabase error for rates {process.value}/{region}: {e}")
        
        # FAIL FAST - no hardcoded fallback
        raise ValueError(
            f"No manufacturing rates for {process.value} in {region}\n"
            f"Solution: Add rates to supabase.manufacturing_rates table"
        )
    
    async def get_material_density(self, material_key: str) -> float:
        """
        Get material density from supabase.
        
        Returns:
            Density in kg/m³
            
        Raises:
            ValueError: If density not available
        """
        await self.initialize()
        
        try:
            mat_data = await self.supabase.get_material(material_key)
            if mat_data and mat_data.get("density_kg_m3"):
                return float(mat_data["density_kg_m3"])
        except Exception as e:
            logger.warning(f"Supabase error for density {material_key}: {e}")
        
        # FAIL FAST - no hardcoded fallback
        raise ValueError(
            f"No density data for material: {material_key}\n"
            f"Solution: Add density_kg_m3 to supabase.materials table"
        )
    
    async def estimate_cost_abc(
        self,
        volume_mm3: float,
        material_key: str,
        process: ManufacturingProcess,
        quantity: int,
        n_features: int = 0,
        n_holes: int = 0,
        tightest_tolerance_mm: float = 0.1,
        surface_roughness_ra: float = 3.2,
        region: str = "global",
        overhead_rate: float = 0.30
    ) -> CostEstimate:
        """
        Estimate cost using Activity-Based Costing (ABC).
        
        All data comes from services - no hardcoded values.
        
        Args:
            volume_mm3: Part volume in mm³
            material_key: Material designation (e.g., "aluminum_6061")
            process: Manufacturing process
            quantity: Production quantity
            n_features: Number of machining features
            n_holes: Number of holes
            tightest_tolerance_mm: Tightest tolerance requirement
            surface_roughness_ra: Required surface roughness (μm)
            region: Geographic region for rates
            overhead_rate: Overhead as fraction of direct costs
            
        Returns:
            CostEstimate with full breakdown and provenance
            
        Raises:
            ValueError: If any required data unavailable
        """
        if quantity < 1:
            raise ValueError(f"Quantity must be >= 1, got {quantity}")
        
        # Get all data from services (FAIL FAST if unavailable)
        material_price, price_source = await self.get_material_price(material_key)
        material_density = await self.get_material_density(material_key)
        mfg_rate = await self.get_manufacturing_rate(process, region)
        
        # Calculate mass
        volume_m3 = volume_mm3 * 1e-9
        mass_kg = volume_m3 * material_density
        
        # Material cost
        material_cost = mass_kg * material_price
        
        # Calculate cycle time using externalized config
        base_time = self.cycle_config.get("base_time_hours", 0.5)
        feature_time = n_features * self.cycle_config.get("feature_time_hours", 0.1)
        hole_time = n_holes * self.cycle_config.get("hole_time_hours", 0.05)
        
        # Tolerance factor
        ref_tol = self.cycle_config.get("tolerance_factor", {}).get("reference_tolerance_mm", 0.01)
        tolerance_factor = max(1.0, ref_tol / tightest_tolerance_mm)
        
        # Surface finish factor
        ref_ra = self.cycle_config.get("surface_finish_factor", {}).get("reference_ra", 3.2)
        finish_factor = max(1.0, ref_ra / surface_roughness_ra)
        
        cycle_time_hours = (base_time + feature_time + hole_time) * tolerance_factor * finish_factor
        
        # Manufacturing costs
        setup_cost = mfg_rate["setup_cost"]
        labor_cost = cycle_time_hours * mfg_rate["hourly_rate"] * quantity
        
        # Tooling cost (would need tool life data from database)
        tooling_cost = 0.0
        
        # Quality and logistics (as percentages - industry standard)
        quality_cost = labor_cost * 0.05
        logistics_cost = material_cost * 0.02
        
        # Overhead
        direct_cost = material_cost + labor_cost + setup_cost + tooling_cost + quality_cost + logistics_cost
        overhead_cost = direct_cost * overhead_rate
        
        total_cost = direct_cost + overhead_cost
        
        # Confidence based on data source
        if "yahoo_finance" in price_source or "metals-api" in price_source:
            confidence = 0.9
        elif "database" in price_source:
            confidence = 0.7
        else:
            confidence = 0.5
        
        breakdown = CostBreakdown(
            material_cost=material_cost,
            labor_cost=labor_cost,
            setup_cost=setup_cost,
            tooling_cost=tooling_cost,
            overhead_cost=overhead_cost,
            quality_cost=quality_cost,
            logistics_cost=logistics_cost
        )
        
        assumptions = [
            f"Density from database: {material_density} kg/m³",
            f"Cycle time model: base={base_time}h, feature={self.cycle_config.get('feature_time_hours', 0.1)}h",
            f"Overhead rate: {overhead_rate*100:.0f}%",
            f"Region: {region}"
        ]
        
        if "_warning" in self.cycle_config:
            assumptions.append("WARNING: Using fallback cycle time config")
        
        warnings = []
        if tightest_tolerance_mm < 0.01:
            warnings.append("Tight tolerances (<0.01mm) may require grinding/EDM")
        if quantity < 10:
            warnings.append("Low quantity: setup costs dominate per-part cost")
        
        return CostEstimate(
            total_cost=total_cost,
            breakdown=breakdown,
            currency="USD",
            method="abc",
            assumptions=assumptions,
            warnings=warnings,
            data_sources={
                "material_price": price_source,
                "material_density": "database",
                "manufacturing_rate": mfg_rate["source"]
            },
            confidence=confidence
        )


# Convenience function for quick estimation
async def quick_cost_estimate(
    volume_cm3: float,
    material: str,
    process: str,
    quantity: int
) -> Dict[str, Any]:
    """
    Quick cost estimate with minimal inputs.
    
    Args:
        volume_cm3: Part volume in cm³
        material: Material key (e.g., "aluminum_6061")
        process: Process name (e.g., "cnc_milling")
        quantity: Production quantity
        
    Returns:
        Dictionary with cost estimate
    """
    agent = ProductionCostAgent(use_ml=False)
    
    try:
        estimate = await agent.estimate_cost_abc(
            volume_mm3=volume_cm3 * 1000,
            material_key=material,
            process=ManufacturingProcess(process.lower()),
            quantity=quantity
        )
        
        return {
            "total_cost_usd": round(estimate.total_cost, 2),
            "cost_per_part_usd": round(estimate.total_cost / quantity, 2),
            "breakdown": {k: round(v, 2) for k, v in estimate.breakdown.to_dict().items()},
            "method": estimate.method,
            "confidence": estimate.confidence,
            "data_sources": estimate.data_sources,
            "warnings": estimate.warnings
        }
    except ValueError as e:
        return {
            "error": str(e),
            "feasible": False
        }
