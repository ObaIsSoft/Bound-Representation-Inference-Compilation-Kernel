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


class MaterialsProjectAPI:
    """
    Integration with Materials Project (materials.org) via official SDK (mp-api).
    Access to 150,000+ materials with DFT-calculated properties.
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("MATERIALS_PROJECT_API_KEY")
        if not MP_API_AVAILABLE:
            self.api_key = None # Disable if SDK not present
        
    
    def search_materials(self, formula: str = None, elements: List[str] = None, 
                        limit: int = 10) -> List[Dict]:
        """
        Search for materials using MPRester.
        """
        if not self.api_key or not MP_API_AVAILABLE:
            logger.warning("Materials Project API key missing or SDK not installed.")
            return []
        
        try:
            with MPRester(self.api_key) as mpr:
                # Prepare query args
                kwargs = {}
                if formula:
                    kwargs["formula"] = [formula]
                if elements:
                    # mp-api expects list of elements
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
                # Limit manually since search might return more
                for doc in docs: # Process all, then sort and limit
                    # Helper to safely get attribute from doc object
                    def get_val(obj, attr):
                        return getattr(obj, attr, None)
                        
                    res = {
                        "material_id": str(get_val(doc, "material_id")),
                        "formula_pretty": str(get_val(doc, "formula_pretty")),
                        "density": float(get_val(doc, "density")) if get_val(doc, "density") else None,
                        "energy_above_hull": float(get_val(doc, "energy_above_hull")) if get_val(doc, "energy_above_hull") is not None else None,
                        # Flatten elasticity
                        "elasticity": {
                            "G_VRH": float(get_val(doc, "shear_modulus")) if get_val(doc, "shear_modulus") else None,
                            "K_VRH": float(get_val(doc, "bulk_modulus")) if get_val(doc, "bulk_modulus") else None
                        },
                        "_source": "materials_project"
                    }
                            
                    results.append(res)
                
                # Sort by stability (E_hull low -> high) locally
                results.sort(key=lambda x: x["energy_above_hull"] if x["energy_above_hull"] is not None else float('inf'))
                
                return results[:limit]

        except Exception as e:
            logger.error(f"Materials Project API error: {e}")
            return []
    
    def get_material(self, material_id: str) -> Optional[Dict]:
        """Get detailed material properties by ID."""
        if not self.api_key or not MP_API_AVAILABLE:
            return None
        
        try:
            with MPRester(self.api_key) as mpr:
                 docs = mpr.materials.summary.search(material_ids=[material_id])
                 if docs:
                     # Reuse the search parsing logic or manual map
                     # For now, just return valid dict
                     d = docs[0]
                     return {
                         "material_id": str(d.material_id),
                         "formula_pretty": str(d.formula_pretty),
                         "density": float(d.density),
                         "energy_above_hull": float(d.energy_above_hull) if d.energy_above_hull is not None else None,
                         "_source": "materials_project"
                     }
        except Exception as e:
            logger.error(f"Materials Project API error: {e}")
            return None

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
            # Use run_search correctly (signature: query, type)
            result = self.nist.run_search(formula, 'formula')
            
            # Load data if needed
            if result and result.success:
                 result.load_found_compounds()
                 
            if result and result.success and result.compounds:
                print(f"[DEBUG] NIST Search Success: {len(result.compounds)} matches")
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
            else:
                 print(f"[DEBUG] NIST Search Failed/Empty. Result: {result}, Success: {getattr(result, 'success', 'UNK')}, Cmp: {getattr(result, 'compounds', 'UNK')}")
        except Exception as e:
            logger.error(f"NIST API error for {formula}: {e}")
            import traceback
            traceback.print_exc()
        
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
        """
        Search for compounds by name using RSC API.
        """
        if not self.api_key:
            logger.warning("RSC API Key missing.")
            return []
            
        try:
            # 1. Search to get Record ID
            search_endpoint = f"{self.base_url}/filter/name"
            payload = {"name": name}
            
            # Post request for search ID
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
                
            # 3. Get Details for first result (to save bandwidth)
            # Only fetching basic details for now
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
        """
        Search for compound by name and retrieve properties.
        Properties: Formula, MolecularWeight, SMILES, InChIKey.
        """
        try:
            # URL: /compound/name/{name}/property/{props}/JSON
            props = "MolecularFormula,MolecularWeight,CanonicalSMILES,InChIKey,Title"
            endpoint = f"{self.base_url}/compound/name/{name}/property/{props}/JSON"
            
            response = requests.get(endpoint, timeout=10)
            if response.status_code == 404:
                return []
            response.raise_for_status()
            
            data = response.json()
            # Parse Response
            # Structure: {'PropertyTable': {'Properties': [{...}]}}
            
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

class ThermoLibrary:
    """
    Integration with Caleb Bell's 'thermo' library.
    Provides comprehensive chemical engineering properties (density, heat capacity, etc.).
    """
    def __init__(self):
        self.available = False
        self._cache = {}  # Cache Chemical objects to avoid slow re-instantiation
        try:
            import thermo
            self.thermo = thermo
            self.available = True
            logger.info("Thermo library loaded successfully.")
        except ImportError:
            logger.warning("Thermo library not installed.")
    
    def get_chemical(self, name: str, temperature: float = 298.15) -> 'Chemical':
        """Get a Chemical object with caching. Returns None if not found."""
        if not self.available:
            return None
        
        cache_key = (name.lower(), temperature)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            # Try to instantiate the Chemical
            # This can be slow on first call
            c = self.thermo.Chemical(name, T=temperature, P=101325)
            self._cache[cache_key] = c
            return c
        except Exception as e:
            logger.debug(f"Could not create Chemical for '{name}': {e}")
            return None
            
    def search(self, name: str) -> List[Dict]:
        """Search for chemical by name. Returns basic info at STP."""
        if not self.available:
            return []
        
        c = self.get_chemical(name, 298.15)
        if c and c.rho:  # Check if we got valid data
            return [{
                "name": c.name or name,  
                "formula": c.formula,
                "density": c.rho, # kg/m3 by default in thermo
                "molecular_weight": c.MW, # g/mol
                "_source": "thermo",
                "_chemical_name": name  # Store original name for re-lookup
            }]
            
        return []


class UnifiedMaterialsAPI:
    """
    Unified interface to multiple materials databases.
    Automatically queries multiple sources and combines results.
    """
    
    
    def __init__(self, materials_project_key: str = None):
        # self.thermo_lib = ThermoLibrary() # Moved to lazy property
        self._thermo_lib_instance = None
        self.mp_api = MaterialsProjectAPI(materials_project_key)
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
        
        # Cache for reducing API calls
        self.cache = {}
        
    @property
    def thermo_lib(self):
        """Lazy load ThermoLibrary only when needed."""
        if self._thermo_lib_instance is None:
            # logger.info("Initializing ThermoLibrary (lazy load)...")
            self._thermo_lib_instance = ThermoLibrary()
        return self._thermo_lib_instance
        
    def find_material(self, query: str, source: str = "auto") -> List[Dict]:
        """Find materials from multiple sources."""
        # Cache check
        cache_key = f"{source}:{query}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        results = []
        
        import re
        
        # Heuristic: Is the query a chemical formula?
        # Matches strings like "Fe2O3", "Al", "H2O", "GaAs", "Li7La3Zr2O12"
        # Does NOT match "Water", "Polyethylene" (lowercase letters not valid in element symbols unless 2nd char)
        formula_pattern = re.compile(r"^([A-Z][a-z]?\d*(\.\d+)?)+$")
        is_formula = bool(formula_pattern.match(query))
        
        # 1. Check Thermo Library 
        # Only if it's a NAME (not a formula) OR we explicitly asked for thermo
        # This prevents "Fe2O3" from triggering massive Thermo DB load
        should_check_thermo_first = (source in ["auto", "thermo"]) and (not is_formula)
        
        if should_check_thermo_first:
             logger.info(f"Searching thermo for '{query}' (appears to be chemical/polymer name)")
             thermo_results = self.thermo_lib.search(query)
             results.extend(thermo_results)
              
        # 2. Check Pymatgen/MP (Metals/Semiconductors) via API
        # Always check for formulas, or if specifically requested
        
        # Common Name -> Formula Mapping
        NAME_TO_FORMULA = {
            "aluminum": "Al", "aluminium": "Al", "titanium": "Ti",
            "steel": "Fe", "iron": "Fe", "copper": "Cu", "gold": "Au",
            "silver": "Ag", "silicon": "Si",
        }
        query_lower = query.lower()
        search_query = NAME_TO_FORMULA.get(query_lower, query)
        
        if source in ["auto", "materials_project"]:
            mp_results = self.mp_api.search_materials(formula=search_query)
            if not mp_results and search_query != query:
                 mp_results = self.mp_api.search_materials(formula=query)
            results.extend([{**r, "_source": "materials_project"} for r in mp_results])
            
        if source in ["auto", "nist"]:
            nist_result = self.nist_api.get_compound(search_query)
            if nist_result:
                results.append({**nist_result, "_source": "nist"})
                
        if source in ["auto", "pubchem"]:
            pc_results = self.pubchem_api.search_compound(query)
            results.extend(pc_results)
            
        # RSC API disabled - consistently returns 400 errors
        # if source in ["auto", "rsc"]:
        #     rsc_results = self.rsc_api.search_compound(query)
        #     results.extend(rsc_results)
        
        # Fallback: If it IS a formula but MP found nothing, try Thermo now
        if is_formula and not results and source in ["auto", "thermo"]:
             logger.info(f"Formula '{query}' not found in MP, trying Thermo as fallback...")
             thermo_results = self.thermo_lib.search(query)
             results.extend(thermo_results)
            
        self.cache[cache_key] = results
        return results

    # ... get_property update separate ...
    
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

    def _get_from_pymatgen(self, material: str, property_name: str) -> Optional[float]:
        """Try to get basic atomic properties from Pymatgen local library."""
        if not self.pymatgen_available:
            return None
            
        # Map common names to symbols
        NAME_TO_FORMULA = {
            "aluminum": "Al", "aluminium": "Al", "titanium": "Ti", 
            "copper": "Cu", "gold": "Au", "silver": "Ag", 
            "silicon": "Si", "iron": "Fe", "carbon": "C", 
            "oxygen": "O", "steel": "Fe" # Approx
        }
        
        symbol = NAME_TO_FORMULA.get(material.lower(), material)
        
        try:
            # Only reliable for elements
            el = self.pymatgen_element(symbol)
            
            if property_name == "density":
                # density_of_solid in kg/m^3 (pymatgen uses SI mostly, but let's be careful)
                # Verified: Pymatgen .density_of_solid returns kg/m^3
                return float(el.density_of_solid)
                
            elif property_name == "youngs_modulus":
                # .youngs_modulus returns GPa
                val = getattr(el, "youngs_modulus", None)
                if val:
                    return float(val * 1e9) # Convert to Pa
                    
            elif property_name == "yield_strength":
                 # Elements don't have defined yield strength, it's a microstructural property
                 pass
                 
            elif property_name == "thermal_conductivity":
                val = getattr(el, "thermal_conductivity", None)
                if val:
                    return float(val)
                
                # Try Mendeleev
                if MENDELEEV_AVAILABLE:
                    try:
                        men_el = get_mendeleev_element(symbol)
                        if men_el.thermal_conductivity:
                            return float(men_el.thermal_conductivity)
                    except: pass

            elif property_name == "specific_heat":
                # Pymatgen doesn't have direct specific heat
                # Try Mendeleev first for Cp (Specific Heat Capacity J/g K -> J/kg K * 1000)
                # Actually mendeleev 'specific_heat_capacity' is usually J/(g K) or similar? 
                # Docs say specific_heat is J / (g K) @ 20C.
                if MENDELEEV_AVAILABLE:
                    try:
                        men_el = get_mendeleev_element(symbol)
                        val = getattr(men_el, "specific_heat", None) # J / (g K)
                        if val:
                            return float(val * 1000.0) # Convert to J / (kg K)
                    except: pass
                
                # Fallback to Pymatgen Molar Heat Capacity
                molar_heat = getattr(el, "molar_heat_capacity", None)
                mass = getattr(el, "atomic_mass", None)
                if molar_heat and mass:
                    return float((molar_heat / mass) * 1000.0) 
            
            elif property_name == "energy_above_hull":
                 # Pure elements in standard state is 0
                 return 0.0
                 
        except Exception:
            # Not a valid element
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
        
        # 0. Try Local Library (Pymatgen) for basic elemental properties
        local_val = self._get_from_pymatgen(material, property_name)
        if local_val is not None:
             # Pymatgen data is usually room temperature (STP)
             logger.info(f"Found {property_name} for {material} in Pymatgen")
             return local_val
        
        # 0.5 Check if we already have thermo data from find_material()
        # This avoids slow Chemical() instantiation here
        # Temperature-dependent lookups will be handled in candidates iteration below
        
        # Skip direct thermo lookup here to avoid hanging
        # Instead, rely on find_material() to have already populated candidates


        # 1. Search for material data in Remote APIs
        candidates = self.find_material(material)
        if not candidates:
             if "steel" in material.lower():
                 candidates = self.find_material("Fe")
             else:
                 raise ValueError(f"Material '{material}' not found in database.")
            
        if not candidates:
             raise ValueError(f"Material '{material}' not found in database.")

        # Iterate through ALL candidates to collect property values
        valid_values = []
        
        for i, data in enumerate(candidates):
            source = data.get("_source", "unknown")
            value = None
            found = False
            
            try:
                # --- DENSITY ---
                if property_name == "density":
                    if source == "materials_project":
                        raw = data.get("density")
                        if raw and raw > 0:
                            value = raw * 1000.0 # g/cm3 -> kg/m3
                            found = True
                    elif source == "thermo":
                        # Re-calculate at the requested temperature
                        chem_name = data.get("_chemical_name") or data.get("name")
                        if chem_name and self.thermo_lib.available:
                            c = self.thermo_lib.get_chemical(chem_name, temperature)
                            if c and c.rho:
                                value = c.rho
                                found = True
                    elif source == "pubchem":
                        # PubChem sometimes provides density in property list if we expanded retrieval
                        pass
                    elif source == "nist":
                        # NIST main strength is Isobaric/Saturation properties (fluid density)
                        # Implemented in future expansion
                        pass
                    
                # --- YIELD STRENGTH ---
                elif property_name == "yield_strength":
                    pass 
                         
                # --- YOUNGS MODULUS ---
                elif property_name == "youngs_modulus":
                    if source == "materials_project":
                        elasticity = data.get("elasticity")
                        if elasticity:
                            g_mod = elasticity.get("G_VRH") 
                            k_mod = elasticity.get("K_VRH") 
                            if g_mod and k_mod:
                                e_val = (9 * k_mod * g_mod) / (3 * k_mod + g_mod)
                                value = e_val * 1e9 
                                found = True
                
                # --- SPECIFIC HEAT ---
                elif property_name == "specific_heat":
                    if source == "thermo":
                        chem_name = data.get("_chemical_name") or data.get("name")
                        if chem_name and self.thermo_lib.available:
                            c = self.thermo_lib.get_chemical(chem_name, temperature)
                            if c and c.Cp:
                                value = c.Cp  # J/kg/K
                                found = True
                
                # --- THERMAL CONDUCTIVITY ---
                elif property_name == "thermal_conductivity":
                    if source == "thermo":
                        chem_name = data.get("_chemical_name") or data.get("name")
                        if chem_name and self.thermo_lib.available:
                            c = self.thermo_lib.get_chemical(chem_name, temperature)
                            if c and c.k:
                                value = c.k  # W/m/K
                                found = True
                                
                # --- THERMAL & STABILITY ---
                elif property_name == "energy_above_hull":
                    if source == "materials_project":
                        # MP v2 usually puts this in 'thermo' dict, but sometimes top-level
                        val = data.get("energy_above_hull")
                        
                        if val is None and "thermo" in data:
                             val = data["thermo"].get("energy_above_hull")
                        
                        if val is not None: 
                            value = float(val)
                            found = True
                            
                elif property_name == "thermal_conductivity":
                     # MP 'thermo' dict usually has formation energy, not conductivity
                     # Conductivity needs separate API query or ML model (matminer)
                     pass

                if found and value is not None:
                     # Filter non-physical values if needed (except e_hull which can be 0)
                     if property_name == "energy_above_hull" or value > 0:
                        logger.info(f"Candidate {i} ({source}) has {property_name}: {value}")
                        valid_values.append(value)
                         
            except Exception as e:
                continue
        
        # Fallback: If no values found (or for specific thermal props), try Thermo library directly
        # This handles cases where find_material skipped thermo (for elements/formulas) but MP failed to provide property
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
                               # Avoid duplicates if we already found it via candidate search
                               if not any(abs(v - val) < 1e-6 for v in valid_values):
                                    logger.info(f"Fallback: Found {property_name} for {material} in Thermo: {val}")
                                    valid_values.append(val)
                  except Exception:
                       pass
        
        if valid_values:
            import statistics
            if property_name == "energy_above_hull":
                # For stability, we want the MINIMUM energy above hull (ground state)
                # aggregation should be min, not median
                best_val = min(valid_values)
                logger.info(f"Stability (E_hull) for '{material}': Min {best_val} eV/atom")
                return best_val
            else:
                median_val = statistics.median(valid_values)
                logger.info(f"Property '{property_name}' for '{material}': Found {len(valid_values)} values. Median: {median_val}")
                return median_val
        
        # If we exhausted all candidates and didn't find the property
        msg = f"Property '{property_name}' not found for '{material}' in any of {len(candidates)} sources."
        logger.warning(msg)
        raise ValueError(msg)
