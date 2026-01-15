
import sys
import os
import logging

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from materials.materials_db import MaterialsDatabase
from agents.chemistry_agent import ChemistryAgent
from agents.manufacturing_agent import ManufacturingAgent
from agents.materials_oracle.adapters.polymers_adapter import PolymersAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MaterialsVerification")

def verify():
    print("--- Starting Materials System Verification (Cloud Only) ---")
    
    # 1. MaterialsDatabase
    print("\n[1] Testing MaterialsDatabase...")
    db = MaterialsDatabase()
    
    # Monomers
    monomers = db.get_monomers()
    if monomers and "ETHYLENE" in monomers:
        print(f"✅ Found Monomer: ETHYLENE ({monomers['ETHYLENE']})")
    else:
        print("⚠️ 'ETHYLENE' not found in monomers. (Supabase empty or not connected?)")
        
    # Ballistic Threats
    threats = db.get_ballistic_threats()
    if threats and "50_BMG" in threats:
        print(f"✅ Found Threat: 50_BMG")
    else:
        print("⚠️ '50_BMG' not found!")
        
    # Element Lookup
    elem = db.get_element("Fe")
    if elem:
        print(f"✅ Found Element Fe via {elem.get('_source', 'Unknown')}")
    else:
        print("⚠️ Element 'Fe' not found (Expected if API key missing).")

    # 2. Agents
    print("\n[2] Testing Agents...")
    
    # Chemistry
    try:
        chem_agent = ChemistryAgent()
        # Test step (Kinetics lookup)
        # Note: If kinetics params missing in DB, uses defaults.
        res = chem_agent.step({"integrity": 1.0}, {"material_type": "Aluminum", "ph": 4.0}, dt=1.0)
        print(f"✅ ChemistryAgent Step Result: {res['metrics']}")
    except Exception as e:
        print(f"❌ ChemistryAgent Failed: {e}")

    # Manufacturing
    try:
        man_agent = ManufacturingAgent()
        # Test generic aluminum lookup
        bom = man_agent.run([{"type": "box", "params": {"width": 10, "length": 10, "height": 1}}], "Aluminum")
        print(f"✅ ManufacturingAgent BOM: {len(bom['components'])} items.")
    except Exception as e:
        print(f"❌ ManufacturingAgent Failed: {e}")

    # Polymers Adapter
    try:
        poly_adapter = PolymersAdapter()
        # Should rely on DB or gracefully handle missing data (using safety .get defaults in adapter logic IF adapter was refactored safely, otherwise might fail)
        synth = poly_adapter._synthesize({"monomer": "PROPYLENE", "chain_length": 500})
        # PROPYLENE not in my seed list, so it tests fallback/default logic or failure
        # Let's test ETHYLENE if possible
        synth_eth = poly_adapter._synthesize({"monomer": "ETHYLENE", "chain_length": 500})
        
        print(f"✅ Polymer Synthesis (Ethylene): {synth_eth['properties']['classification']}")
    except Exception as e:
        print(f"❌ PolymersAdapter Failed: {e}")

if __name__ == "__main__":
    verify()
