import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from materials.materials_api import UnifiedMaterialsAPI

# Disable lazy load logs
import logging
logging.getLogger("materials.materials_api").setLevel(logging.WARNING)

api = UnifiedMaterialsAPI()
candidates = api.find_material("Fe2O3", source="materials_project")

if candidates:
    c = candidates[0]
    # Create simplified dict without structure/sites
    simple = {k:v for k,v in c.items() if k not in ['structure', 'sites', 'xyz', 'species']}
    print(json.dumps(simple, indent=2))
else:
    print("No candidates.")
