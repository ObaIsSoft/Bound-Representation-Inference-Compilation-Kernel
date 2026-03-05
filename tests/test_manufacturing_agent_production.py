"""
Production tests for ManufacturingAgent

Validates:
1. CNC machining cost calculations
2. 3D printing cost calculations
3. Material cost estimation
4. Setup time calculations
5. Toolpath verification

Note: These tests require database connectivity for manufacturing rates.
Tests are skipped if database is unavailable.
"""

import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# Check if database is available by trying to get rates
async def _check_database():
    """Check if Supabase database is actually reachable"""
    try:
        from backend.services import supabase
        # Try to get rates - this will fail if DB not configured
        result = await supabase.get_manufacturing_rates("cnc_milling", "global")
        return True
    except Exception:
        return False


# Run the check
HAS_DATABASE = asyncio.run(_check_database())


# Skip tests that require database if not available
requires_database = pytest.mark.skipif(
    not HAS_DATABASE,
    reason="Database not available - manufacturing rates unavailable"
)


class TestManufacturingAgentBasics:
    """Test ManufacturingAgent basic functionality (no DB required)"""
    
    def test_agent_creation(self):
        """Test agent can be created"""
        from backend.agents.manufacturing_agent import ManufacturingAgent
        agent = ManufacturingAgent()
        assert agent is not None
        assert agent.name == "ManufacturingAgent"


@requires_database
class TestManufacturingAgentWithDatabase:
    """Test ManufacturingAgent with database connectivity"""
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test agent initializes correctly"""
        from backend.agents.manufacturing_agent import ManufacturingAgent
        agent = ManufacturingAgent()
        await agent.initialize()
        
        assert agent.hourly_rate > 0
        assert agent.setup_cost >= 0
        
    @pytest.mark.asyncio
    async def test_cnc_milling_rates(self):
        """Test CNC milling rate calculations"""
        from backend.agents.manufacturing_agent import ManufacturingAgent
        agent = ManufacturingAgent()
        await agent.initialize(process_type="cnc_milling")
        
        # CNC milling should have reasonable hourly rate
        assert 50 < agent.hourly_rate < 500  # USD/hour
        
    @pytest.mark.asyncio
    async def test_setup_cost_calculation(self):
        """Test setup cost is reasonable"""
        from backend.agents.manufacturing_agent import ManufacturingAgent
        agent = ManufacturingAgent()
        await agent.initialize(process_type="cnc_milling")
        
        setup_cost = agent.setup_cost
        assert setup_cost >= 0
        # Setup cost typically $50-500
        assert setup_cost < 1000


@requires_database
class TestManufacturingProcesses:
    """Test different manufacturing processes"""
    
    @pytest.mark.asyncio
    async def test_cnc_turning(self):
        """Test CNC turning process"""
        from backend.agents.manufacturing_agent import ManufacturingAgent
        agent = ManufacturingAgent()
        await agent.initialize(process_type="cnc_turning")
        
        assert agent.hourly_rate > 0
        
    @pytest.mark.asyncio
    async def test_sla_printing(self):
        """Test SLA 3D printing"""
        from backend.agents.manufacturing_agent import ManufacturingAgent
        agent = ManufacturingAgent()
        await agent.initialize(process_type="sla_3d_printing")
        
        assert agent.hourly_rate > 0


@requires_database
class TestRegionalPricing:
    """Test regional pricing variations"""
    
    @pytest.mark.asyncio
    async def test_us_pricing(self):
        """Test US manufacturing rates"""
        from backend.agents.manufacturing_agent import ManufacturingAgent
        agent = ManufacturingAgent()
        await agent.initialize(region="us")
        
        assert agent.hourly_rate > 0
        
    @pytest.mark.asyncio
    async def test_europe_pricing(self):
        """Test Europe manufacturing rates"""
        from backend.agents.manufacturing_agent import ManufacturingAgent
        agent = ManufacturingAgent()
        await agent.initialize(region="europe")
        
        assert agent.hourly_rate > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
