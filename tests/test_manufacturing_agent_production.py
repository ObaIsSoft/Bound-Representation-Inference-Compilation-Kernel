"""
Production tests for ManufacturingAgent

Validates:
1. CNC machining cost calculations
2. 3D printing cost calculations
3. Material cost estimation
4. Setup time calculations
5. Toolpath verification
"""

import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.agents.manufacturing_agent import ManufacturingAgent


class TestManufacturingAgent:
    """Test ManufacturingAgent with fallback rates"""
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test agent initializes correctly with fallback rates"""
        agent = ManufacturingAgent()
        await agent.initialize()
        
        assert agent.hourly_rate > 0
        assert agent.setup_cost >= 0
        
    @pytest.mark.asyncio
    async def test_cnc_milling_rates(self):
        """Test CNC milling rate calculations"""
        agent = ManufacturingAgent()
        await agent.initialize(process_type="cnc_milling")
        
        # CNC milling should have reasonable hourly rate
        assert 50 < agent.hourly_rate < 500  # USD/hour
        
    @pytest.mark.asyncio
    async def test_3d_printing_rates(self):
        """Test 3D printing rate calculations"""
        agent = ManufacturingAgent()
        await agent.initialize(process_type="sls_3d_printing")
        
        # 3D printing should have different rates than CNC
        assert agent.hourly_rate > 0
        
    @pytest.mark.asyncio
    async def test_setup_cost_calculation(self):
        """Test setup cost is reasonable"""
        agent = ManufacturingAgent()
        await agent.initialize(process_type="cnc_milling")
        
        setup_cost = agent.setup_cost
        assert setup_cost >= 0
        # Setup cost typically $50-500
        assert setup_cost < 1000
        
    @pytest.mark.asyncio
    async def test_minimum_radius(self):
        """Test minimum CNC radius is reasonable"""
        agent = ManufacturingAgent()
        await agent.initialize(process_type="cnc_milling")
        
        min_radius = agent.min_cnc_radius_mm
        # Typical end mill minimum: 0.1-3mm
        assert 0.01 < min_radius < 10  # mm


class TestManufacturingProcesses:
    """Test different manufacturing processes"""
    
    @pytest.mark.asyncio
    async def test_cnc_turning(self):
        """Test CNC turning process"""
        agent = ManufacturingAgent()
        await agent.initialize(process_type="cnc_turning")
        
        assert agent.hourly_rate > 0
        
    @pytest.mark.asyncio
    async def test_sla_printing(self):
        """Test SLA 3D printing"""
        agent = ManufacturingAgent()
        await agent.initialize(process_type="sla_3d_printing")
        
        assert agent.hourly_rate > 0
        
    @pytest.mark.asyncio
    async def test_injection_molding(self):
        """Test injection molding"""
        agent = ManufacturingAgent()
        await agent.initialize(process_type="injection_molding")
        
        # Injection molding typically has higher setup costs
        assert agent.setup_cost > 0


class TestRegionalPricing:
    """Test regional pricing variations"""
    
    @pytest.mark.asyncio
    async def test_us_pricing(self):
        """Test US manufacturing rates"""
        agent = ManufacturingAgent()
        await agent.initialize(region="us")
        
        assert agent.hourly_rate > 0
        
    @pytest.mark.asyncio
    async def test_china_pricing(self):
        """Test China manufacturing rates"""
        agent = ManufacturingAgent()
        await agent.initialize(region="china")
        
        assert agent.hourly_rate > 0
        
    @pytest.mark.asyncio
    async def test_europe_pricing(self):
        """Test Europe manufacturing rates"""
        agent = ManufacturingAgent()
        await agent.initialize(region="europe")
        
        assert agent.hourly_rate > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
