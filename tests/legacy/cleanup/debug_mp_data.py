import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from materials.materials_api import UnifiedMaterialsAPI

api = UnifiedMaterialsAPI()
print("Searching for Fe2O3...")
candidates = api.find_material("Fe2O3", source="materials_project")
print(f"Found {len(candidates)} candidates.")

if candidates:
    print("First candidate keys:", candidates[0].keys())
    print("First candidate Data:", json.dumps(candidates[0], default=str, indent=2))
else:
    print("No candidates found.")
