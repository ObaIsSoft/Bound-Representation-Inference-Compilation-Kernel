"""
External Materials API Integration
Integrates with Materials Project, NIST, AFLOW, OQMD, COD, NOMAD and other materials databases.
Provides access to 150,000+ materials with comprehensive properties.
"""
import requests
import json
import os
import re
import time
import concurrent.futures
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
from functools import wraps

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Explicitly look for .env in the parent directory (backend/) 
    # This ensures it works even when running scripts from project root
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    load_dotenv(env_path)
    # Fallback to default if explicit path fails
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use system environment variables

logger = logging.getLogger(__name__)


# Try to import new libraries
try:
    from mp_api.client import MPRester
    MP_API_AVAILABLE = True
except ImportError:
    MP_API_AVAILABLE = False
    logger.warning("mp-api not installed. Materials Project integration will be limited.")

try:
    from mendeleev import element as get_mendeleev_element
    MENDELEEV_AVAILABLE = True
except ImportError:
    MENDELEEV_AVAILABLE = False
    logger.warning("mendeleev not installed. Element properties will be limited.")


@dataclass
class APIRetryConfig:
    """Configuration for API retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    retry_on_status: tuple = (429, 500, 502, 503, 504)


def retry_with_backoff(config: APIRetryConfig = APIRetryConfig()):
    """Decorator for API calls with exponential backoff retry logic."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_retries):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.HTTPError as e:
                    last_exception = e
                    response = e.response
                    
                    # Don't retry on client errors (4xx) except rate limits (429)
                    if response.status_code < 500 and response.status_code not in config.retry_on_status:
                        raise
                    
                    # Check if we should retry
                    if response.status_code not in config.retry_on_status and attempt < config.max_retries - 1:
                        raise
                        
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                    last_exception = e
                    if attempt == config.max_retries - 1:
                        raise
                
                # Calculate delay with exponential backoff and jitter
                delay = min(
                    config.base_delay * (config.exponential_base ** attempt),
                    config.max_delay
                )
                # Add small random jitter to avoid thundering herd
                import random
                delay += random.uniform(0, 0.1 * delay)
                
                logger.warning(f"API call failed (attempt {attempt + 1}/{config.max_retries}). Retrying in {delay:.2f}s...")
                time.sleep(delay)
            
            # All retries exhausted
            raise last_exception if last_exception else Exception("All retries failed")
            
        return wrapper
    return decorator


class BaseMaterialsAPI(ABC):
    """Abstract base class for materials APIs."""
    
    @abstractmethod
    def search_materials(self, query: str, **kwargs) -> List[Dict]:
        """Search for materials."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if API is properly configured and available."""
        pass


class MaterialsProjectAPI(BaseMaterialsAPI):
    """
    Integration with Materials Project (materials.org) via official SDK (mp-api).
    Access to 150,000+ materials with DFT-calculated properties.
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("MATERIALS_PROJECT_API_KEY")
        self.retry_config = APIRetryConfig(max_retries=3, base_delay=2.0)
        
    def is_available(self) -> bool:
        return MP_API_AVAILABLE and bool(self.api_key)
    
    def _safe_get_attr(self, obj, attr: str, default=None):
        """Safely extract attribute from MP doc or dict."""
        if obj is None:
            return default
        if hasattr(obj, attr):
            return getattr(obj, attr, default)
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return default
    
    def _get_float(self, obj, attr: str) -> Optional[float]:
        """Safely get float from attribute."""
        val = self._safe_get_attr(obj, attr, None)
        if val is None:
            return None
        try:
            if hasattr(val, 'value'):
                return float(val.value)
            if isinstance(val, dict):
                return float(val.get('value', 0))
            return float(val)
        except (ValueError, TypeError):
            return None
    
    def _with_timeout(self, func, timeout_sec=5):
        """Cross-platform timeout using concurrent.futures."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(func)
            try:
                return future.result(timeout=timeout_sec)
            except concurrent.futures.TimeoutError:
                logger.warning(f"Materials Project API call timed out after {timeout_sec}s")
                return []
            except Exception as e:
                logger.error(f"Materials Project API error: {e}")
                return []

    @retry_with_backoff(APIRetryConfig(max_retries=3, base_delay=2.0))
    def search_materials(self, formula: str = None, elements: List[str] = None, 
                        limit: int = 10) -> List[Dict]:
        """
        Search for materials using MPRester with timeout.
        """
        if not self.is_available():
            logger.warning("Materials Project API key missing or SDK not installed.")
            return []
        
        def run_search():
            with MPRester(self.api_key) as mpr:
                # Prepare query args
                kwargs = {}
                if formula:
                    kwargs["formula"] = [formula]
                if elements:
                    kwargs["elements"] = [e for e in elements if len(e) <= 2]
                
                # Fields to retrieve
                fields = [
                    "material_id", "formula_pretty", "density", 
                    "energy_above_hull", "structure", 
                    "bulk_modulus", "shear_modulus", "symmetry" 
                ]
                
                # Use the 'summary' endpoint which centralizes key properties
                docs = mpr.materials.summary.search(
                    **kwargs, 
                    fields=fields
                )
                
                # Convert MPDocs to dictionaries
                results = []
                for doc in docs:
                    res = {
                        "material_id": str(self._safe_get_attr(doc, "material_id")),
                        "formula_pretty": str(self._safe_get_attr(doc, "formula_pretty")),
                        "density": self._get_float(doc, "density"),
                        "energy_above_hull": self._get_float(doc, "energy_above_hull"),
                        "elasticity": {
                            "G_VRH": self._get_float(doc, "shear_modulus"),
                            "K_VRH": self._get_float(doc, "bulk_modulus")
                        },
                        "symmetry": self._safe_get_attr(doc, "symmetry"),
                        "_source": "materials_project"
                    }
                    results.append(res)
                
                results.sort(key=lambda x: x["energy_above_hull"] if x["energy_above_hull"] is not None else float('inf'))
                return results[:limit]

        return self._with_timeout(run_search, timeout_sec=8)
    
    @retry_with_backoff(APIRetryConfig(max_retries=3, base_delay=2.0))
    def get_material(self, material_id: str) -> Optional[Dict]:
        """Get detailed material properties by ID."""
        if not self.is_available():
            return None
        
        try:
            with MPRester(self.api_key) as mpr:
                 docs = mpr.materials.summary.search(material_ids=[material_id])
                 if docs:
                     d = docs[0]
                     return {
                         "material_id": str(d.material_id),
                         "formula_pretty": str(d.formula_pretty),
                         "density": float(d.density) if d.density else None,
                         "energy_above_hull": float(d.energy_above_hull) if d.energy_above_hull is not None else None,
                         "_source": "materials_project"
                     }
        except Exception as e:
            logger.error(f"Materials Project API error: {e}")
            return None


class AFLOWAPI(BaseMaterialsAPI):
    """
    Integration with AFLOW (Automatic Flow for Materials Discovery).
    Access to 3M+ compounds with DFT calculations.
    """
    
    def __init__(self):
        self.base_url = "http://aflow.org/API/aflowlib/v1.0"
        self.api_key = os.getenv("AFLOW_API_KEY")  # Optional for some endpoints
        self.retry_config = APIRetryConfig(max_retries=3, base_delay=1.0)
        
    def is_available(self) -> bool:
        # AFLOW has open endpoints but key increases rate limits
        return True
    
    @retry_with_backoff(APIRetryConfig(max_retries=3, base_delay=1.0))
    def search_materials(self, formula: str = None, elements: List[str] = None, 
                        limit: int = 10) -> List[Dict]:
        """Search AFLOW database."""
        try:
            params = {
                "catalog": "ICSD",  # Inorganic Crystal Structure Database
                "format": "json",
                "limit": limit
            }
            
            if formula:
                params["compound"] = formula
            elif elements:
                # AFLOW uses element list
                params["species"] = ",".join(elements)
            
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            response = requests.get(
                f"{self.base_url}/search",
                params=params,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            for entry in data.get("entries", []):
                props = entry.get("properties", {})
                results.append({
                    "aflow_id": entry.get("auid"),
                    "formula": props.get("compound"),
                    "spacegroup": props.get("spacegroup_relax"),
                    "energy_atom": props.get("energy_atom"),
                    "density": props.get("density"),
                    "band_gap": props.get("Egap"),
                    "_source": "aflow"
                })
            
            return results
            
        except Exception as e:
            logger.error(f"AFLOW API error: {e}")
            return []
    
    def get_material(self, aflow_id: str) -> Optional[Dict]:
        """Get specific material by AFLOW ID."""
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
                
            response = requests.get(
                f"{self.base_url}/entry/{aflow_id}",
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            props = data.get("properties", {})
            return {
                "aflow_id": aflow_id,
                "formula": props.get("compound"),
                "density": props.get("density"),
                "band_gap": props.get("Egap"),
                "bulk_modulus": props.get("bulk_modulus_vrh"),
                "shear_modulus": props.get("shear_modulus_vrh"),
                "_source": "aflow"
            }
        except Exception as e:
            logger.error(f"AFLOW API error: {e}")
            return None


class OQMDAPI(BaseMaterialsAPI):
    """
    Integration with OQMD (Open Quantum Materials Database).
    Access to 1M+ DFT calculations from Northwestern University.
    """
    
    def __init__(self):
        self.base_url = "http://oqmd.org/oqmdapi"
        self.retry_config = APIRetryConfig(max_retries=3, base_delay=1.0)
        
    def is_available(self) -> bool:
        return True  # Open API
    
    @retry_with_backoff(APIRetryConfig(max_retries=3, base_delay=1.0))
    def search_materials(self, formula: str = None, elements: List[str] = None,
                        limit: int = 10) -> List[Dict]:
        """Search OQMD database."""
        try:
            params = {
                "format": "json",
                "limit": limit
            }
            
            if formula:
                params["composition"] = formula
            elif elements:
                params["element"] = ",".join(elements)
            
            response = requests.get(
                f"{self.base_url}/formationenergy",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            for entry in data.get("data", []):
                calc = entry.get("calculation", {})
                results.append({
                    "entry_id": entry.get("entry_id"),
                    "formula": entry.get("composition"),
                    "band_gap": calc.get("band_gap"),
                    "energy_above_hull": entry.get("formationenergy"),
                    "spacegroup": entry.get("spacegroup"),
                    "_source": "oqmd"
                })
            
            return results
            
        except Exception as e:
            logger.error(f"OQMD API error: {e}")
            return []


class CODAPI(BaseMaterialsAPI):
    """
    Integration with COD (Crystallography Open Database).
    Access to 500K+ experimental crystal structures.
    """
    
    def __init__(self):
        self.base_url = "http://www.crystallography.net/cod/result"
        self.retry_config = APIRetryConfig(max_retries=3, base_delay=1.0)
        
    def is_available(self) -> bool:
        return True
    
    @retry_with_backoff(APIRetryConfig(max_retries=3, base_delay=1.0))
    def search_materials(self, formula: str = None, elements: List[str] = None,
                        limit: int = 10) -> List[Dict]:
        """Search COD for experimental structures."""
        try:
            params = {
                "format": "json",
                "limit": limit
            }
            
            if formula:
                params["formula"] = formula
            elif elements:
                params["el1"] = elements[0] if len(elements) > 0 else None
                if len(elements) > 1:
                    params["el2"] = elements[1]
                if len(elements) > 2:
                    params["el3"] = elements[2]
            
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for entry in data:
                results.append({
                    "cod_id": entry.get("file"),
                    "formula": entry.get("formula"),
                    "spacegroup": entry.get("sg"),
                    "a": entry.get("a"),
                    "b": entry.get("b"),
                    "c": entry.get("c"),
                    "alpha": entry.get("alpha"),
                    "beta": entry.get("beta"),
                    "gamma": entry.get("gamma"),
                    "_source": "cod"
                })
            
            return results
            
        except Exception as e:
            logger.error(f"COD API error: {e}")
            return []


class NOMADAPI(BaseMaterialsAPI):
    """
    Integration with NOMAD (Novel Materials Discovery).
    FAIR data infrastructure for materials science.
    """
    
    def __init__(self):
        self.base_url = "http://nomad-lab.eu/prod/rae/api"
        self.retry_config = APIRetryConfig(max_retries=3, base_delay=1.0)
        
    def is_available(self) -> bool:
        return True
    
    @retry_with_backoff(APIRetryConfig(max_retries=3, base_delay=1.0))
    def search_materials(self, formula: str = None, elements: List[str] = None,
                        limit: int = 10) -> List[Dict]:
        """Search NOMAD archive."""
        try:
            query = {}
            if formula:
                query["formula"] = formula
            elif elements:
                query["elements"] = elements
            
            payload = {
                "query": query,
                "pagination": {"page_size": limit},
                "required": {
                    "include": ["formula", "energy", "band_gap"]
                }
            }
            
            response = requests.post(
                f"{self.base_url}/entries/query",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            for entry in data.get("data", []):
                archive = entry.get("archive", {})
                results.append({
                    "nomad_id": entry.get("entry_id"),
                    "formula": archive.get("formula"),
                    "energy": archive.get("energy"),
                    "band_gap": archive.get("band_gap"),
                    "_source": "nomad"
                })
            
            return results
            
        except Exception as e:
            logger.error(f"NOMAD API error: {e}")
            return []


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
        """Get compound data from NIST."""
        if not self.available:
            return None
        
        try:
            result = self.nist.run_search(formula, 'formula')
            
            if result and result.success:
                 result.load_found_compounds()
                 
            if result and result.success and result.compounds:
                compound = result.compounds[0]
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
        
        return None
    
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


class MatWebAPI:
    """
    Integration with MatWeb (matweb.com)
    Access to engineering materials database.
    """
    
    def __init__(self):
        self.base_url = "http://www.matweb.com"
    
    def search_alloy(self, name: str) -> List[Dict]:
        """Search for alloys by name."""
        logger.warning("MatWeb requires web scraping. Implementation pending.")
        return []


class RSCApi:
    """
    Integration with Royal Society of Chemistry (RSC) API.
    Primary use: ChemSpider / Compounds data.
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("RSC_API_KEY")
        self.base_url = "https://api.rsc.org/compounds/v1"
        self.headers = {"apikey": self.api_key} if self.api_key else {}
        
    def search_compound(self, name: str) -> List[Dict]:
        """Search for compounds by name using RSC API."""
        if not self.api_key:
            logger.warning("RSC API Key missing.")
            return []
            
        try:
            # 1. Search to get Record ID
            search_endpoint = f"{self.base_url}/filter/name"
            payload = {"name": name}
            
            resp = requests.post(search_endpoint, json=payload, headers=self.headers, timeout=10)
            if resp.status_code == 401:
                logger.error("RSC API Unauthorized. Check Key.")
                return []
            resp.raise_for_status()
            
            query_id = resp.json().get("queryId")
            
            # 2. Fetch Results using queryId
            results_endpoint = f"{self.base_url}/filter/{query_id}/results"
            results_resp = requests.get(results_endpoint, headers=self.headers, timeout=10)
            results_resp.raise_for_status()
            
            record_ids = results_resp.json().get("results", [])
            if not record_ids:
                return []
                
            # 3. Get Details for first result
            top_id = record_ids[0]
            details_endpoint = f"{self.base_url}/records/{top_id}/details"
            details_resp = requests.get(details_endpoint, headers=self.headers, params={"fields": "CommonName,Formula,MolecularWeight"}, timeout=10)
            details_resp.raise_for_status()
            
            data = details_resp.json()
            
            return [{
                "name": data.get("commonName", name),
                "formula": data.get("formula"),
                "molecular_weight": data.get("molecularWeight"),
                "record_id": top_id,
                "_source": "rsc"
            }]
            
        except Exception as e:
            logger.error(f"RSC API Error: {e}")
            return []


class PubChemAPI:
    """
    Integration with NCBI PubChem PUG REST API.
    Access to 100M+ compounds.
    """
    
    def __init__(self):
        self.base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        
    def search_compound(self, name: str) -> List[Dict]:
        """Search for compound by name and retrieve properties."""
        try:
            props = "MolecularFormula,MolecularWeight,CanonicalSMILES,InChIKey,Title"
            endpoint = f"{self.base_url}/compound/name/{name}/property/{props}/JSON"
            
            response = requests.get(endpoint, timeout=10)
            if response.status_code == 404:
                return []
            response.raise_for_status()
            
            data = response.json()
            properties = data.get("PropertyTable", {}).get("Properties", [])
            results = []
            
            for p in properties:
                results.append({
                    "name": p.get("Title", name),
                    "formula": p.get("MolecularFormula"),
                    "molecular_weight": float(p.get("MolecularWeight", 0)) if p.get("MolecularWeight") else None,
                    "smiles": p.get("CanonicalSMILES"),
                    "inchi_key": p.get("InChIKey"),
                    "cid": p.get("CID"),
                    "_source": "pubchem"
                })
                
            return results
            
        except Exception as e:
            logger.error(f"PubChem API Error: {e}")
            return []


class ThermoLibrary:
    """
    Integration with Caleb Bell's 'thermo' library.
    Provides comprehensive chemical engineering properties.
    """
    def __init__(self):
        self.available = False
        self._cache = {}
        try:
            import thermo
            self.thermo = thermo
            self.available = True
            logger.info("Thermo library loaded successfully.")
        except ImportError:
            logger.warning("Thermo library not installed.")
    
    def _with_timeout(self, func, timeout_sec=5):
        """Cross-platform timeout using concurrent.futures."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(func)
            try:
                return future.result(timeout=timeout_sec)
            except concurrent.futures.TimeoutError:
                return None
    
    def get_chemical(self, name: str, temperature: float = 298.15):
        """Get a Chemical object with caching and cross-platform timeout."""
        if not self.available:
            return None
        
        cache_key = (name.lower(), temperature)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        def create_chemical():
            return self.thermo.Chemical(name, T=temperature, P=101325)
        
        try:
            c = self._with_timeout(create_chemical, timeout_sec=5)
            if c is not None:
                self._cache[cache_key] = c
            return c
        except Exception as e:
            logger.debug(f"Could not create Chemical for '{name}': {e}")
            return None
            
    def search(self, name: str) -> List[Dict]:
        """Search for chemical by name."""
        if not self.available:
            return []
        
        c = self.get_chemical(name, 298.15)
        if c and c.rho:
            return [{
                "name": c.name or name,  
                "formula": c.formula,
                "density": c.rho,
                "molecular_weight": c.MW,
                "_source": "thermo",
                "_chemical_name": name
            }]
        return []


class UnifiedMaterialsAPI:
    """
    Unified interface to multiple materials databases.
    Automatically queries multiple sources and combines results.
    """
    
    def __init__(self, materials_project_key: str = None):
        self._thermo_lib_instance = None
        self.mp_api = MaterialsProjectAPI(materials_project_key)
        self.aflow_api = AFLOWAPI()
        self.oqmd_api = OQMDAPI()
        self.cod_api = CODAPI()
        self.nomad_api = NOMADAPI()
        self.nist_api = NISTChemistryAPI()
        self.matweb_api = MatWebAPI()
        self.rsc_api = RSCApi()
        self.pubchem_api = PubChemAPI()
        
        # Pymatgen (Local Library) Integration
        try:
            from pymatgen.core import Element
            self.pymatgen_element = Element
            self.pymatgen_available = True
        except ImportError:
            logger.warning("Pymatgen not available.")
            self.pymatgen_available = False
        
        self.cache = {}
        
    @property
    def thermo_lib(self):
        """Lazy load ThermoLibrary only when needed."""
        if self._thermo_lib_instance is None:
            self._thermo_lib_instance = ThermoLibrary()
        return self._thermo_lib_instance
    
    def _validate_unit_conversion(self, value: float, from_unit: str, to_unit: str) -> float:
        """Validate and perform unit conversions with bounds checking."""
        if value is None or value < 0:
            raise ValueError(f"Invalid value for conversion: {value}")
        
        # Density: g/cm³ -> kg/m³ (multiply by 1000)
        if from_unit == "g/cm3" and to_unit == "kg/m3":
            result = value * 1000.0
            if result > 50000:  # Osmium is ~22500 kg/m³, anything higher is suspicious
                logger.warning(f"Unusually high density: {result} kg/m³")
            return result
        
        # Pressure: GPa -> Pa
        elif from_unit == "GPa" and to_unit == "Pa":
            return value * 1e9
        
        # Energy: eV/atom -> keep as is but validate
        elif from_unit == "eV/atom":
            if abs(value) > 20:  # Sanity check for formation energies
                logger.warning(f"Unusual energy value: {value} eV/atom")
            return value
        
        return value
    
    def find_material(self, query: str, source: str = "auto") -> List[Dict]:
        """Find materials from multiple sources with improved routing."""
        cache_key = f"{source}:{query}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        results = []
        
        # Heuristic: Is the query a chemical formula?
        formula_pattern = re.compile(r"^([A-Z][a-z]?\\d*(\\.\\d+)?)+$")
        is_formula = bool(formula_pattern.match(query))
        
        # Check Thermo Library (for names, not formulas)
        should_check_thermo_first = (source in ["auto", "thermo"]) and (not is_formula)
        
        if should_check_thermo_first:
             logger.info(f"Searching thermo for '{query}' (appears to be chemical/polymer name)")
             thermo_results = self.thermo_lib.search(query)
             results.extend(thermo_results)
        
        # Common Name -> Formula Mapping (expanded)
        NAME_TO_FORMULA = {
            "titanium": "Ti",
            "copper": "Cu", 
            "gold": "Au", 
            "silver": "Ag",
            "silicon": "Si", 
            "iron": "Fe", 
            "carbon": "C",
            "oxygen": "O",
            "aluminum": "Al",
            "aluminium": "Al",
            "nickel": "Ni",
            "chromium": "Cr"
        }
        
        query_lower = query.lower()
        search_query = NAME_TO_FORMULA.get(query_lower, query)
        
        # Query all computational databases in parallel for formulas
        search_tasks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            if source in ["auto", "materials_project"]:
                search_tasks.append(executor.submit(self.mp_api.search_materials, formula=search_query))
                if search_query != query:
                    search_tasks.append(executor.submit(self.mp_api.search_materials, formula=query))
            
            if source in ["auto", "aflow"] and is_formula:
                search_tasks.append(executor.submit(self.aflow_api.search_materials, formula=search_query))
                
            if source in ["auto", "oqmd"] and is_formula:
                search_tasks.append(executor.submit(self.oqmd_api.search_materials, formula=search_query))
                
            if source in ["auto", "cod"] and is_formula:
                search_tasks.append(executor.submit(self.cod_api.search_materials, formula=search_query))
                
            if source in ["auto", "nist"]:
                search_tasks.append(executor.submit(self.nist_api.get_compound, search_query))
                
            if source in ["auto", "pubchem"]:
                search_tasks.append(executor.submit(self.pubchem_api.search_compound, query))
                
            # Wait for all tasks with a collective timeout
            done, not_done = concurrent.futures.wait(search_tasks, timeout=8.0)
            
            for future in done:
                try:
                    res = future.result()
                    if isinstance(res, list):
                        results.extend(res)
                    elif isinstance(res, dict):
                        results.append(res)
                except Exception as e:
                    logger.debug(f"Async search task failed: {e}")
            
            # Cancel anything still running
            for future in not_done:
                future.cancel()
        
        # Fallback: If formula but no results, try Thermo
        if is_formula and not results and source in ["auto", "thermo"]:
             logger.info(f"Formula '{query}' not found in DBs, trying Thermo...")
             thermo_results = self.thermo_lib.search(query)
             results.extend(thermo_results)
            
        self.cache[cache_key] = results
        return results

    def get_element_data(self, symbol: str) -> Optional[Dict]:
        """Get comprehensive element data."""
        results = self.mp_api.search_materials(elements=[symbol], limit=1)
        if results:
            return results[0]
        return None
    
    def get_alloy_data(self, name: str) -> List[Dict]:
        """
        Get alloy data. Note: Generic 'steel' requests are rejected.
        Users must specify grade (e.g., 'AISI 304', '316L').
        """
        # Reject generic steel queries
        if name.lower() in ["steel", "stainless steel", "carbon steel"]:
            logger.error(f"Generic alloy name '{name}' is ambiguous. Please specify grade (e.g., 'AISI 304', '316L').")
            raise ValueError(
                f"'{name}' is too generic. Specify an alloy grade like 'AISI 304', '316L', '17-4PH', etc. "
                "Or query the base element 'Fe' for elemental iron properties."
            )
        
        return self.find_material(name, source="auto")
    
    def get_compound_data(self, formula: str) -> Optional[Dict]:
        """Get compound data, preferring NIST."""
        nist_data = self.nist_api.get_compound(formula)
        if nist_data:
            return nist_data
        
        mp_results = self.mp_api.search_materials(formula=formula, limit=1)
        if mp_results:
            return mp_results[0]
        
        return None
    
    def search_by_properties(self, min_density: float = None, max_density: float = None,
                            min_strength: float = None, elements: List[str] = None) -> List[Dict]:
        """Search materials by properties."""
        results = []
        
        if elements:
            results = self.mp_api.search_materials(elements=elements, limit=50)
        
        if min_density or max_density:
            results = [
                r for r in results
                if (min_density is None or r.get("density", 0) >= min_density) and
                   (max_density is None or r.get("density", float('inf')) <= max_density)
            ]
        
        return results

    def _get_from_pymatgen(self, material: str, property_name: str) -> Optional[float]:
        """Try to get basic atomic properties from Pymatgen local library."""
        if not self.pymatgen_available:
            return None
            
        NAME_TO_FORMULA = {
            "titanium": "Ti",  
            "copper": "Cu", "gold": "Au", "silver": "Ag", 
            "silicon": "Si", "iron": "Fe", "carbon": "C", 
            "oxygen": "O", "aluminum": "Al", "nickel": "Ni"
        }
        
        symbol = NAME_TO_FORMULA.get(material.lower(), material)
        
        try:
            el = self.pymatgen_element(symbol)
            
            if property_name == "density":
                return float(el.density_of_solid)
            elif property_name == "youngs_modulus":
                val = getattr(el, "youngs_modulus", None)
                if val:
                    return float(val * 1e9)
            elif property_name == "thermal_conductivity":
                val = getattr(el, "thermal_conductivity", None)
                if val:
                    return float(val)
                if MENDELEEV_AVAILABLE:
                    try:
                        men_el = get_mendeleev_element(symbol)
                        if men_el.thermal_conductivity:
                            return float(men_el.thermal_conductivity)
                    except: 
                        pass
            elif property_name == "specific_heat":
                if MENDELEEV_AVAILABLE:
                    try:
                        men_el = get_mendeleev_element(symbol)
                        val = getattr(men_el, "specific_heat", None)
                        if val:
                            return float(val * 1000.0)
                    except: 
                        pass
                
                molar_heat = getattr(el, "molar_heat_capacity", None)
                mass = getattr(el, "atomic_mass", None)
                if molar_heat and mass:
                    return float((molar_heat / mass) * 1000.0)
            elif property_name == "energy_above_hull":
                return 0.0
                 
        except Exception:
            pass
            
        return None

    def get_property(
        self,
        material: str,
        property_name: str,
        temperature: float = 293
    ) -> float:
        """
        Get a specific physical property for a material.
        Normalizes units to SI (kg, m, s, Pa, K).
        """
        
        # Try Local Library (Pymatgen) for basic elemental properties
        local_val = self._get_from_pymatgen(material, property_name)
        if local_val is not None:
             logger.info(f"Found {property_name} for {material} in Pymatgen")
             return local_val

        # Search for material data in Remote APIs
        candidates = self.find_material(material)
            
        if not candidates:
             raise ValueError(f"Material '{material}' not found in database.")

        valid_values = []
        
        for i, data in enumerate(candidates):
            source = data.get("_source", "unknown")
            value = None
            found = False
            
            try:
                if property_name == "density":
                    if source == "materials_project":
                        raw = data.get("density")
                        if raw and raw > 0:
                            # MP density is in g/cm³, validate and convert
                            value = self._validate_unit_conversion(raw, "g/cm3", "kg/m3")
                            found = True
                    elif source == "thermo":
                        chem_name = data.get("_chemical_name") or data.get("name")
                        if chem_name and self.thermo_lib.available:
                            c = self.thermo_lib.get_chemical(chem_name, temperature)
                            if c and c.rho:
                                value = c.rho
                                found = True
                    elif source == "aflow":
                        raw = data.get("density")
                        if raw:
                            value = self._validate_unit_conversion(raw, "g/cm3", "kg/m3")
                            found = True
                         
                elif property_name == "youngs_modulus":
                    if source == "materials_project":
                        elasticity = data.get("elasticity")
                        if elasticity:
                            g_mod = elasticity.get("G_VRH") 
                            k_mod = elasticity.get("K_VRH") 
                            if g_mod and k_mod:
                                e_val = (9 * k_mod * g_mod) / (3 * k_mod + g_mod)
                                value = self._validate_unit_conversion(e_val, "GPa", "Pa")
                                found = True
                    elif source == "aflow":
                        # AFLOW provides VRH averages
                        g_mod = data.get("shear_modulus")
                        k_mod = data.get("bulk_modulus")
                        if g_mod and k_mod:
                            e_val = (9 * k_mod * g_mod) / (3 * k_mod + g_mod)
                            value = self._validate_unit_conversion(e_val, "GPa", "Pa")
                            found = True
                
                elif property_name == "specific_heat":
                    if source == "thermo":
                        chem_name = data.get("_chemical_name") or data.get("name")
                        if chem_name and self.thermo_lib.available:
                            c = self.thermo_lib.get_chemical(chem_name, temperature)
                            if c and c.Cp:
                                value = c.Cp  # J/kg/K
                                found = True
                
                elif property_name == "thermal_conductivity":
                    if source == "thermo":
                        chem_name = data.get("_chemical_name") or data.get("name")
                        if chem_name and self.thermo_lib.available:
                            c = self.thermo_lib.get_chemical(chem_name, temperature)
                            if c and c.k:
                                value = c.k  # W/m/K
                                found = True
                                
                elif property_name == "energy_above_hull":
                    if source == "materials_project":
                        val = data.get("energy_above_hull")
                        if val is None and "thermo" in data:
                             val = data["thermo"].get("energy_above_hull")
                        if val is not None: 
                            value = self._validate_unit_conversion(float(val), "eV/atom", "eV/atom")
                            found = True
                    elif source == "oqmd":
                        val = data.get("energy_above_hull")
                        if val is not None:
                            value = float(val)
                            found = True

                if found and value is not None:
                     if property_name == "energy_above_hull" or value > 0:
                        logger.info(f"Candidate {i} ({source}) has {property_name}: {value}")
                        valid_values.append(value)
                         
            except Exception as e:
                logger.debug(f"Error extracting {property_name} from {source}: {e}")
                continue
        
        # Fallback to Thermo for thermal properties
        if (not valid_values) or (property_name in ["specific_heat", "thermal_conductivity"]):
             if self.thermo_lib.available:
                  try:
                       c = self.thermo_lib.get_chemical(material, temperature)
                       if c:
                           val = None
                           if property_name == "specific_heat" and c.Cp:
                               val = c.Cp
                           elif property_name == "thermal_conductivity" and c.k:
                               val = c.k
                           elif property_name == "density" and c.rho:
                               val = c.rho
                               
                           if val is not None:
                               if not any(abs(v - val) < 1e-6 for v in valid_values):
                                    logger.info(f"Fallback: Found {property_name} for {material} in Thermo: {val}")
                                    valid_values.append(val)
                  except Exception:
                       pass
        
        if valid_values:
            import statistics
            if property_name == "energy_above_hull":
                best_val = min(valid_values)
                logger.info(f"Stability (E_hull) for '{material}': Min {best_val} eV/atom")
                return best_val
            else:
                median_val = statistics.median(valid_values)
                logger.info(f"Property '{property_name}' for '{material}': Found {len(valid_values)} values. Median: {median_val}")
                return median_val
        
        msg = f"Property '{property_name}' not found for '{material}' in any of {len(candidates)} sources."
        logger.warning(msg)
        raise ValueError(msg)
