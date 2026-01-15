"""
External Materials API Integration
Integrates with Materials Project, NIST, and other materials databases.
Provides access to 150,000+ materials with comprehensive properties.
"""
import requests
import json
import os
from typing import Dict, List, Optional, Any
import logging

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use system environment variables

logger = logging.getLogger(__name__)


class MaterialsProjectAPI:
    """
    Integration with Materials Project (materials.org)
    Access to 150,000+ materials with DFT-calculated properties.
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("MATERIALS_PROJECT_API_KEY")
        self.base_url = "https://api.materialsproject.org"
        self.headers = {"X-API-KEY": self.api_key} if self.api_key else {}
    
    def search_materials(self, formula: str = None, elements: List[str] = None, 
                        limit: int = 10) -> List[Dict]:
        """
        Search for materials by formula or elements.
        
        Args:
            formula: Chemical formula (e.g., "Fe2O3")
            elements: List of elements (e.g., ["Fe", "O"])
            limit: Maximum number of results
        
        Returns:
            List of material dictionaries
        """
        if not self.api_key:
            logger.warning("Materials Project API key not set. Using mock data.")
            return self._mock_search(formula, elements)
        
        endpoint = f"{self.base_url}/materials/summary"
        params = {}
        
        if formula:
            params["formula"] = formula
        if elements:
            params["elements"] = ",".join(elements)
        if limit:
            params["_limit"] = limit
        
        try:
            response = requests.get(endpoint, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except Exception as e:
            logger.error(f"Materials Project API error: {e}")
            return []
    
    def get_material(self, material_id: str) -> Optional[Dict]:
        """Get detailed material properties by ID."""
        if not self.api_key:
            return self._mock_material(material_id)
        
        endpoint = f"{self.base_url}/materials/{material_id}"
        
        try:
            response = requests.get(endpoint, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json().get("data", {})
        except Exception as e:
            logger.error(f"Materials Project API error: {e}")
            return None
    
    def _mock_search(self, formula: str = None, elements: List[str] = None) -> List[Dict]:
        """Mock data disabled."""
        logger.warning(f"Materials Project API Key missing. Skipping search for {formula}/{elements}.")
        return []
    
    def _mock_material(self, material_id: str) -> Dict:
        """Mock material data disabled."""
        return {}


class NISTChemistryAPI:
    """
    Integration with NIST Chemistry WebBook using nistchempy.
    Access to thermochemical and spectroscopic data.
    """
    
    def __init__(self):
        try:
            import nistchempy as nist
            self.nist = nist
            self.available = True
        except ImportError:
            logger.warning("nistchempy not installed. Install with: pip install nistchempy")
            self.nist = None
            self.available = False
    
    def get_compound(self, formula: str) -> Optional[Dict]:
        """
        Get compound data from NIST.
        
        Args:
            formula: Chemical formula (e.g., "H2O")
        
        Returns:
            Compound properties dictionary
        """
        if not self.available:
            return self._mock_compound(formula)
        
        try:
            # Search for compound by formula
            compound = self.nist.get_compound(formula)
            
            if compound:
                return {
                    "formula": formula,
                    "name": compound.name if hasattr(compound, 'name') else formula,
                    "molecular_weight": compound.molecular_weight if hasattr(compound, 'molecular_weight') else None,
                    "cas_number": compound.cas if hasattr(compound, 'cas') else None,
                    "inchi": compound.inchi if hasattr(compound, 'inchi') else None,
                    "thermochemistry": self._get_thermochemistry(compound),
                    "_source": "nist"
                }
        except Exception as e:
            logger.error(f"NIST API error for {formula}: {e}")
        
        return self._mock_compound(formula)
    
    def _get_thermochemistry(self, compound) -> Dict:
        """Extract thermochemical data from NIST compound."""
        thermo = {}
        
        try:
            if hasattr(compound, 'enthalpy_formation'):
                thermo['enthalpy_formation'] = compound.enthalpy_formation
            if hasattr(compound, 'entropy'):
                thermo['entropy'] = compound.entropy
            if hasattr(compound, 'heat_capacity'):
                thermo['heat_capacity'] = compound.heat_capacity
        except Exception as e:
            logger.debug(f"Error extracting thermochemistry: {e}")
        
        return thermo
    
    def search_by_name(self, name: str) -> List[Dict]:
        """Search compounds by name."""
        if not self.available:
            return []
        
        try:
            results = self.nist.search(name)
            return [
                {
                    "name": r.name if hasattr(r, 'name') else name,
                    "formula": r.formula if hasattr(r, 'formula') else None,
                    "cas_number": r.cas if hasattr(r, 'cas') else None,
                    "_source": "nist"
                }
                for r in results[:10]  # Limit to 10 results
            ]
        except Exception as e:
            logger.error(f"NIST search error: {e}")
            return []
    
    def _mock_compound(self, formula: str) -> Dict:
        """
        Mock compound data.
        REMOVED: Hardcoded data removed per configuration.
        """
        logger.warning(f"No API access and no DB match for {formula}. Hardcoded fallback disabled.")
        return {}


class MatWebAPI:
    """
    Integration with MatWeb (matweb.com)
    Access to engineering materials database.
    """
    
    def __init__(self):
        self.base_url = "http://www.matweb.com"
        # MatWeb doesn't have a public API, would need web scraping
        # Using mock data for demonstration
    
    def search_alloy(self, name: str) -> List[Dict]:
        """Search for alloys by name."""
        return self._mock_alloys(name)
    
    def _mock_alloys(self, name: str) -> List[Dict]:
        """Mock alloy data disabled."""
        return []


class UnifiedMaterialsAPI:
    """
    Unified interface to multiple materials databases.
    Automatically queries multiple sources and combines results.
    """
    
    def __init__(self, materials_project_key: str = None):
        self.mp_api = MaterialsProjectAPI(materials_project_key)
        self.nist_api = NISTChemistryAPI()
        self.matweb_api = MatWebAPI()
        
        # Cache for reducing API calls
        self.cache = {}
    
    def find_material(self, query: str, source: str = "auto") -> List[Dict]:
        """
        Find materials from multiple sources.
        
        Args:
            query: Material name, formula, or ID
            source: "auto", "materials_project", "nist", "matweb"
        
        Returns:
            List of materials from all sources
        """
        # Check cache
        cache_key = f"{source}:{query}"
        if cache_key in self.cache:
            logger.info(f"Cache hit for {cache_key}")
            return self.cache[cache_key]
        
        results = []
        
        if source in ["auto", "materials_project"]:
            mp_results = self.mp_api.search_materials(formula=query)
            results.extend([{**r, "_source": "materials_project"} for r in mp_results])
        
        if source in ["auto", "nist"]:
            nist_result = self.nist_api.get_compound(query)
            if nist_result:
                results.append({**nist_result, "_source": "nist"})
        
        if source in ["auto", "matweb"]:
            matweb_results = self.matweb_api.search_alloy(query)
            results.extend([{**r, "_source": "matweb"} for r in matweb_results])
        
        # Cache results
        self.cache[cache_key] = results
        
        return results
    
    def get_element_data(self, symbol: str) -> Optional[Dict]:
        """Get comprehensive element data."""
        # Try Materials Project first
        results = self.mp_api.search_materials(elements=[symbol], limit=1)
        if results:
            return results[0]
        return None
    
    def get_alloy_data(self, name: str) -> List[Dict]:
        """Get alloy data from all sources."""
        return self.find_material(name, source="auto")
    
    def get_compound_data(self, formula: str) -> Optional[Dict]:
        """Get compound data, preferring NIST."""
        nist_data = self.nist_api.get_compound(formula)
        if nist_data:
            return nist_data
        
        # Fall back to Materials Project
        mp_results = self.mp_api.search_materials(formula=formula, limit=1)
        if mp_results:
            return mp_results[0]
        
        return None
    
    def search_by_properties(self, min_density: float = None, max_density: float = None,
                            min_strength: float = None, elements: List[str] = None) -> List[Dict]:
        """
        Search materials by properties.
        
        Args:
            min_density: Minimum density (kg/m³)
            max_density: Maximum density (kg/m³)
            min_strength: Minimum tensile strength (Pa)
            elements: Required elements
        
        Returns:
            List of matching materials
        """
        # This would query Materials Project with property filters
        # For now, using mock implementation
        results = []
        
        if elements:
            results = self.mp_api.search_materials(elements=elements, limit=50)
        
        # Filter by properties
        if min_density or max_density:
            results = [
                r for r in results
                if (min_density is None or r.get("density", 0) >= min_density) and
                   (max_density is None or r.get("density", float('inf')) <= max_density)
            ]
        
        return results
