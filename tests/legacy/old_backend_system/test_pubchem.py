
import sys
import os
import logging
sys.path.insert(0, 'backend')
from materials.materials_api import PubChemAPI

logging.basicConfig(level=logging.ERROR)
print('--- TESTING PUBCHEM API ---')
api = PubChemAPI()

queries = ["Aspirin", "Benzene", "Caffeine"]
for q in queries:
    results = api.search_compound(q)
    print(f"Query: {q}")
    if results:
        r = results[0]
        print(f"  Name: {r.get('name')}")
        print(f"  Formula: {r.get('formula')}")
        print(f"  MW: {r.get('molecular_weight')}")
        print(f"  CID: {r.get('cid')}")
    else:
        print("  No results found.")
