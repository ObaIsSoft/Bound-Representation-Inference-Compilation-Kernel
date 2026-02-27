"""
Production tests for CostAgent and ToleranceAgent.

Tests the production implementations with mocked services.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

# ToleranceAgent tests (no external services needed)
from backend.agents.tolerance_agent_production import (
    ProductionToleranceAgent,
    ToleranceSpec,
    DistributionType,
    quick_rss_analysis,
    analyze_feature_position
)


class TestProductionToleranceAgent:
    """Test ProductionToleranceAgent - no external dependencies."""
    
    @pytest.fixture
    def agent(self):
        return ProductionToleranceAgent(default_mc_iterations=5000)
    
    def test_calculate_rss_simple(self, agent):
        """Test basic RSS calculation."""
        tolerances = [
            ToleranceSpec("A", 10.0, 0.1),
            ToleranceSpec("B", 15.0, 0.15)
        ]
        
        result = agent.calculate_rss(tolerances)
        
        assert result.nominal_stack == 25.0
        assert result.rss_tolerance > 0
        assert result.upper_limit > result.nominal_stack
        assert result.lower_limit < result.nominal_stack
        
        # RSS should be less than worst case
        worst_case_tol = sum(t.tolerance_range for t in tolerances) / 2
        assert result.rss_tolerance < worst_case_tol
    
    def test_calculate_rss_with_sensitivities(self, agent):
        """Test RSS with non-unity sensitivities."""
        tolerances = [
            ToleranceSpec("A", 10.0, 0.1),
            ToleranceSpec("B", 5.0, 0.1)
        ]
        sensitivities = [1.0, -1.0]
        
        result = agent.calculate_rss(tolerances, sensitivities=sensitivities)
        
        assert result.nominal_stack == 5.0  # 10 - 5
    
    def test_calculate_rss_with_cpk(self, agent):
        """Test RSS with Cpk calculation."""
        tolerances = [
            ToleranceSpec("A", 10.0, 0.1),
            ToleranceSpec("B", 15.0, 0.15)
        ]
        design_target = (25.0, 0.5)
        
        result = agent.calculate_rss(tolerances, design_target=design_target)
        
        assert result.cpk is not None
        assert result.cpk > 1.0  # Capable process
        assert result.percent_outside is not None
    
    def test_monte_carlo_simple(self, agent):
        """Test Monte Carlo simulation."""
        tolerances = [
            ToleranceSpec("A", 10.0, 0.1),
            ToleranceSpec("B", 15.0, 0.15)
        ]
        
        result = agent.monte_carlo_stack(tolerances)
        
        assert result.iterations == 5000
        assert result.mean > 0
        assert result.std_dev > 0
        assert "99%" in result.percentiles
        assert "1%" in result.percentiles
    
    def test_monte_carlo_different_distributions(self, agent):
        """Test Monte Carlo with different distributions."""
        tolerances = [
            ToleranceSpec("normal", 10.0, 0.1, distribution=DistributionType.NORMAL),
            ToleranceSpec("uniform", 5.0, 0.1, distribution=DistributionType.UNIFORM),
            ToleranceSpec("triangular", 3.0, 0.1, distribution=DistributionType.TRIANGULAR)
        ]
        
        result = agent.monte_carlo_stack(tolerances)
        
        assert result.mean > 0
        assert result.std_dev > 0
    
    def test_worst_case_stack(self, agent):
        """Test worst-case analysis."""
        tolerances = [
            ToleranceSpec("A", 10.0, 0.1),
            ToleranceSpec("B", 15.0, 0.15)
        ]
        
        result = agent.worst_case_stack(tolerances)
        
        assert result.nominal_stack == 25.0
        assert result.tolerance_range > 0
    
    def test_analyze_stack_comprehensive(self, agent):
        """Test comprehensive stack analysis."""
        tolerances = [
            ToleranceSpec("hole1", 10.0, 0.1),
            ToleranceSpec("hole2", 15.0, 0.15),
            ToleranceSpec("thickness", 5.0, 0.05)
        ]
        
        result = agent.analyze_stack(
            tolerances,
            stack_description="Test stack",
            design_target=(30.0, 0.5)
        )
        
        assert result.stack_description == "Test stack"
        assert result.rss is not None
        assert result.monte_carlo is not None
        assert result.worst_case is not None
        assert isinstance(result.passes_specification, bool)
    
    def test_gd_and_t_true_position(self, agent):
        """Test GD&T true position calculation."""
        result = agent.gd_and_t_true_position(
            x_tolerance=0.1,
            y_tolerance=0.1,
            bonus_tolerance=0.05,
            material_condition="MMC"
        )
        
        assert result["total_tolerance_zone"] > result["basic_tolerance_zone"]
        assert result["bonus_tolerance"] == 0.05
    
    def test_quick_rss_analysis(self):
        """Test quick RSS convenience function."""
        tolerances = [
            ("hole1", 10.0, 0.1),
            ("hole2", 15.0, 0.15)
        ]
        
        result = quick_rss_analysis(tolerances, (25.0, 0.5))
        
        assert "nominal_stack" in result
        assert "rss_tolerance" in result
        assert "passes_spec" in result
        assert "contributions" in result
    
    def test_analyze_feature_position(self):
        """Test true position analysis."""
        result = analyze_feature_position(
            x_deviation=0.05,
            y_deviation=0.03,
            position_tolerance=0.2
        )
        
        assert result["within_tolerance"] is True
        assert result["actual_position_deviation"] > 0
    
    def test_analyze_feature_position_out_of_tolerance(self):
        """Test position analysis with out-of-tolerance feature."""
        result = analyze_feature_position(
            x_deviation=0.15,
            y_deviation=0.15,
            position_tolerance=0.2
        )
        
        assert result["within_tolerance"] is False


class TestProductionCostAgent:
    """Test ProductionCostAgent with mocked services."""
    
    @pytest.fixture
    async def agent(self):
        """Create agent with mocked services."""
        from backend.agents.cost_agent_production import ProductionCostAgent
        
        agent = ProductionCostAgent(use_ml=False)
        
        # Mock services
        agent.pricing_service = AsyncMock()
        agent.supabase = AsyncMock()
        agent._initialized = True
        
        return agent
    
    @pytest.mark.asyncio
    async def test_get_material_price_success(self):
        """Test successful material price fetch."""
        from backend.agents.cost_agent_production import ProductionCostAgent
        from backend.services.pricing_service import PricePoint
        from datetime import datetime, timedelta
        
        agent = ProductionCostAgent(use_ml=False)
        agent._initialized = True
        
        # Create mock price point
        mock_price = PricePoint(
            price=3.50,
            currency="USD",
            unit="kg",
            source="metals-api",
            timestamp=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=24)
        )
        
        agent.pricing_service = AsyncMock()
        agent.pricing_service.get_material_price = AsyncMock(return_value=mock_price)
        agent.supabase = AsyncMock()
        
        price, source = await agent.get_material_price("aluminum_6061")
        
        assert price == 3.50
        assert source == "metals-api"
    
    @pytest.mark.asyncio
    async def test_get_material_price_fail_fast(self):
        """Test that missing price raises error (fail fast)."""
        from backend.agents.cost_agent_production import ProductionCostAgent
        
        agent = ProductionCostAgent(use_ml=False)
        agent._initialized = True
        
        # Mock services returning None
        agent.pricing_service = AsyncMock()
        agent.pricing_service.get_material_price = AsyncMock(return_value=None)
        agent.supabase = AsyncMock()
        agent.supabase.get_material = AsyncMock(return_value=None)
        
        with pytest.raises(ValueError, match="No price available"):
            await agent.get_material_price("unknown_material")
    
    @pytest.mark.asyncio
    async def test_get_manufacturing_rate_success(self):
        """Test successful rate fetch."""
        from backend.agents.cost_agent_production import ProductionCostAgent, ManufacturingProcess
        
        agent = ProductionCostAgent(use_ml=False)
        agent._initialized = True
        
        mock_rates = {
            "machine_hourly_rate_usd": 85.0,
            "setup_cost_usd": 150.0,
            "setup_time_hr": 2.0,
            "data_source": "supplier_quote"
        }
        
        agent.supabase = AsyncMock()
        agent.supabase.get_manufacturing_rates = AsyncMock(return_value=mock_rates)
        agent.pricing_service = AsyncMock()
        
        rates = await agent.get_manufacturing_rate(ManufacturingProcess.CNC_MILLING, "us")
        
        assert rates["hourly_rate"] == 85.0
        assert rates["setup_cost"] == 150.0
    
    @pytest.mark.asyncio
    async def test_estimate_cost_abc_success(self):
        """Test ABC cost estimation with mocked data."""
        from backend.agents.cost_agent_production import ProductionCostAgent, ManufacturingProcess
        from backend.services.pricing_service import PricePoint
        from datetime import datetime, timedelta
        
        agent = ProductionCostAgent(use_ml=False)
        agent._initialized = True
        
        # Mock pricing service
        mock_price = PricePoint(
            price=3.50,
            currency="USD",
            unit="kg",
            source="database",
            timestamp=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=24)
        )
        agent.pricing_service = AsyncMock()
        agent.pricing_service.get_material_price = AsyncMock(return_value=mock_price)
        
        # Mock supabase
        agent.supabase = AsyncMock()
        agent.supabase.get_material = AsyncMock(return_value={
            "density_kg_m3": 2700,
            "cost_per_kg_usd": 3.50
        })
        agent.supabase.get_manufacturing_rates = AsyncMock(return_value={
            "machine_hourly_rate_usd": 85.0,
            "setup_cost_usd": 150.0,
            "setup_time_hr": 2.0,
            "data_source": "database"
        })
        
        estimate = await agent.estimate_cost_abc(
            volume_mm3=100000,  # 100 cm³
            material_key="aluminum_6061",
            process=ManufacturingProcess.CNC_MILLING,
            quantity=100,
            n_features=5,
            n_holes=2
        )
        
        assert estimate.total_cost > 0
        assert estimate.breakdown.material_cost > 0
        assert estimate.breakdown.labor_cost > 0
        assert estimate.confidence > 0
        assert "material_price" in estimate.data_sources


class TestRSSCalculations:
    """Verify RSS math against manual calculations."""
    
    def test_rss_math_verified(self):
        """Verify RSS formula: RSS = sqrt(sum(sigma²)) where sigma = tol/6"""
        import math
        from backend.agents.tolerance_agent_production import ToleranceSpec, ProductionToleranceAgent
        
        agent = ProductionToleranceAgent()
        
        # Two tolerances ±0.1 (bilateral ±0.1 = range 0.2)
        # sigma = 0.2/6 = 0.03333
        # RSS sigma = sqrt(0.03333² + 0.03333²) = 0.04714
        # RSS tolerance = 3 * 0.04714 = 0.1414
        
        tols = [ToleranceSpec("A", 10, 0.1), ToleranceSpec("B", 20, 0.1)]
        result = agent.calculate_rss(tols)
        
        expected_sigma = 0.2 / 6
        expected_rss_sigma = math.sqrt(expected_sigma**2 + expected_sigma**2)
        expected_rss_tol = 3 * expected_rss_sigma
        
        assert abs(result.rss_tolerance - expected_rss_tol) < 0.001
        assert result.nominal_stack == 30.0
    
    def test_cpk_calculation_verified(self):
        """Verify Cpk formula: min((USL-μ)/3σ, (μ-LSL)/3σ)"""
        from backend.agents.tolerance_agent_production import ToleranceSpec, ProductionToleranceAgent
        
        agent = ProductionToleranceAgent()
        
        # Stack: 25 ± 0.141 (from previous test)
        # Target: 25 ± 0.5
        # USL = 25.5, LSL = 24.5
        # CPU = (25.5 - 25) / (3 * 0.04714) = 3.53
        # CPL = (25 - 24.5) / (3 * 0.04714) = 3.53
        # Cpk = 3.53
        
        tols = [ToleranceSpec("A", 10, 0.1), ToleranceSpec("B", 15, 0.1)]
        result = agent.calculate_rss(tols, design_target=(25.0, 0.5))
        
        assert result.cpk is not None
        assert result.cpk > 3.0  # Highly capable


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
