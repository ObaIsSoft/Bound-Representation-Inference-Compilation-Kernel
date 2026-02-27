"""
Test Material Database Expansion - 50 validated materials from MIL-HDBK-5J/ASM
"""

import pytest
import json
from pathlib import Path

# Import the material agent
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "agents"))

from material_agent import ProductionMaterialAgent, DataProvenance


class TestMaterialDatabaseLoading:
    """Test that the material database loads correctly with all 50 materials"""
    
    @pytest.fixture
    def agent(self):
        """Create a fresh agent instance"""
        return ProductionMaterialAgent()
    
    def test_database_file_exists(self):
        """Verify material_database.json exists and is valid JSON"""
        db_path = Path(__file__).parent.parent / "data" / "materials" / "material_database.json"
        assert db_path.exists(), f"Database not found at {db_path}"
        
        with open(db_path) as f:
            db = json.load(f)
        
        assert "metadata" in db
        assert "materials" in db
        assert db["metadata"]["total_materials"] == 50
    
    def test_all_50_materials_loaded(self, agent):
        """Verify all 50 materials are loaded into the agent"""
        assert len(agent.materials) == 50, f"Expected 50 materials, got {len(agent.materials)}"
    
    def test_material_categories(self, agent):
        """Verify materials span expected categories"""
        categories = set(m.category for m in agent.materials.values())
        expected_categories = {
            'aluminum_wrought', 'aluminum_wrought_high_strength', 'aluminum_cast',
            'steel_low_carbon', 'steel_medium_carbon', 'steel_alloy', 
            'stainless_steel_austenitic', 'stainless_steel_martensitic', 
            'stainless_steel_precipitation',
            'tool_steel', 'titanium_alpha', 'titanium_alpha_beta',
            'nickel_superalloy', 'nickel_copper',
            'copper_wrought', 'brass', 'copper_alloy',
            'magnesium_wrought', 'cobalt_alloy',
            'refractory_metal', 'bronze', 'zinc_alloy', 'lead'
        }
        
        # Check that we have most expected categories
        assert len(categories) >= 15, f"Expected at least 15 categories, got {len(categories)}: {categories}"
    
    def test_aluminum_materials(self, agent):
        """Verify aluminum materials are present with correct data"""
        aluminum_materials = [k for k, v in agent.materials.items() 
                             if 'aluminum' in v.category]
        
        # Should have at least 12 aluminum materials
        assert len(aluminum_materials) >= 12, f"Expected at least 12 aluminum alloys, got {len(aluminum_materials)}"
        
        # Verify key alloys exist
        key_alloys = ['aluminum_6061_t6', 'aluminum_7075_t6', 'aluminum_2024_t3',
                     'aluminum_5052_h32', 'aluminum_6061_t651']
        for alloy in key_alloys:
            assert alloy in agent.materials, f"Missing key aluminum alloy: {alloy}"
    
    def test_steel_materials(self, agent):
        """Verify steel materials are present"""
        steel_materials = [k for k, v in agent.materials.items() 
                          if 'steel' in v.category or 'stainless' in v.category]
        
        # Should have at least 16 steel materials
        assert len(steel_materials) >= 16, f"Expected at least 16 steel alloys, got {len(steel_materials)}"
        
        # Verify key steels exist
        key_steels = ['steel_4140_quenched_tempered', 'steel_aisi_304', 
                     'steel_aisi_316', 'steel_1018_cold_drawn']
        for steel in key_steels:
            assert steel in agent.materials, f"Missing key steel: {steel}"
    
    def test_titanium_materials(self, agent):
        """Verify titanium materials are present"""
        ti_materials = [k for k, v in agent.materials.items() 
                       if 'titanium' in v.category]
        
        # Should have 4 titanium materials
        assert len(ti_materials) >= 4, f"Expected at least 4 titanium alloys, got {len(ti_materials)}"
        
        # Verify Ti-6Al-4V variants
        assert 'ti_6al_4v_annealed' in agent.materials
        assert 'ti_6al_4v_eli' in agent.materials
    
    def test_nickel_superalloys(self, agent):
        """Verify nickel superalloys are present"""
        ni_materials = [k for k, v in agent.materials.items() 
                       if 'nickel' in v.category]
        
        # Should have at least 5 nickel materials
        assert len(ni_materials) >= 5, f"Expected at least 5 nickel alloys, got {len(ni_materials)}"
        
        # Verify Inconel alloys
        assert 'inconel_718_solution' in agent.materials
        assert 'inconel_625' in agent.materials
    
    def test_material_property_ranges(self, agent):
        """Verify material properties are in reasonable ranges"""
        for mat_id, mat in agent.materials.items():
            # Density should be reasonable for metals
            assert 1000 <= mat.density.value <= 20000, \
                f"{mat_id}: Density {mat.density.value} out of range"
            
            # Elastic modulus should be reasonable
            assert 10 <= mat.elastic_modulus.value <= 450, \
                f"{mat_id}: E-modulus {mat.elastic_modulus.value} out of range"
            
            # Poisson's ratio should be 0.1-0.5
            assert 0.1 <= mat.poisson_ratio.value <= 0.5, \
                f"{mat_id}: Poisson ratio {mat.poisson_ratio.value} out of range"
            
            # Yield strength should be positive
            assert mat.yield_strength.value > 0, \
                f"{mat_id}: Yield strength must be positive"
            
            # Thermal conductivity should be reasonable
            assert 1 <= mat.thermal_conductivity.value <= 450, \
                f"{mat_id}: Thermal conductivity {mat.thermal_conductivity.value} out of range"
    
    def test_provenance_tracking(self, agent):
        """Verify all materials have proper provenance"""
        certified_count = 0
        unspecified_count = 0
        
        for mat_id, mat in agent.materials.items():
            # Check that density has provenance info
            assert mat.density.provenance is not None
            assert mat.density.source_reference is not None
            
            if mat.density.provenance == DataProvenance.ASTM_CERTIFIED:
                certified_count += 1
            elif mat.density.provenance == DataProvenance.UNSPECIFIED:
                unspecified_count += 1
        
        # Most materials should be certified
        assert certified_count >= 45, f"Expected at least 45 certified materials, got {certified_count}"
        assert unspecified_count == 0, f"Expected 0 unspecified materials, got {unspecified_count}"
    
    def test_source_references(self, agent):
        """Verify source references are from MIL-HDBK-5J or ASM"""
        for mat_id, mat in agent.materials.items():
            source = mat.density.source_reference or ""
            assert "MIL-HDBK-5J" in source or "ASM-V" in source, \
                f"{mat_id}: Expected MIL-HDBK-5J or ASM source, got: {source}"


class TestMaterialQueries:
    """Test material query functionality"""
    
    @pytest.fixture
    def agent(self):
        return ProductionMaterialAgent()
    
    def test_get_material_by_id(self, agent):
        """Test retrieving material by ID"""
        mat = agent.get_material("aluminum_6061_t6")
        assert mat is not None
        assert mat.name == "Aluminum 6061-T6"
        assert abs(mat.density.value - 2700) < 50
    
    def test_search_by_category(self, agent):
        """Test searching materials by category"""
        results = agent.search_materials(category="aluminum_wrought")
        assert len(results) >= 5
        
        for mat_id, mat in results.items():
            assert 'aluminum' in mat.category
    
    def test_search_by_property_range(self, agent):
        """Test searching by property range"""
        # Find high strength materials (yield > 400 MPa)
        results = agent.search_materials(
            min_yield_strength=400,
            max_yield_strength=2000
        )
        
        # Should find steels, titanium, and high-strength aluminum
        assert len(results) >= 15
        
        for mat_id, mat in results.items():
            assert mat.yield_strength.value >= 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
