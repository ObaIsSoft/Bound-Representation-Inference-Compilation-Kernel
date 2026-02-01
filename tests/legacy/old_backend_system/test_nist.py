
import sys
import os
import logging
sys.path.insert(0, 'backend')
from materials.materials_api import NISTChemistryAPI

logging.basicConfig(level=logging.ERROR)
print('--- TESTING NIST API ---')
api = NISTChemistryAPI()

if not api.available:
    print("NIST API (nistchempy) is NOT installed/available.")
    sys.exit(0)

compounds = ["CO2", "H2O", "Methane"]
for c in compounds:
    data = api.get_compound(c)
    print(f"Query: {c}")
    if data:
        print(f"  Name: {data.get('name')}")
        print(f"  Thermo: {data.get('thermochemistry')}")
    else:
        print("  No data found.")
