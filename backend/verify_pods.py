import os
import sys
import logging
import asyncio
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Add backend to path for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from core.hierarchical_resolver import ModularISA
from pod_manager import PodManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_modular_pod_system_phase24_dehardcoded():
    # 1. Initialize PodManager
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "projects"))
    pm = PodManager(project_root)
    
    # 2. Create a Pod Instance
    prop_pod = ModularISA(name="Generic Radial Test")
    
    # 3. Discovery & Hydration
    logger.info("--- Step 1: De-hardcoded Discovery ---")
    pm.discover_and_hydrate(prop_pod, "propeller_pod")
    
    print(f"Assembly Pattern: {prop_pod.assembly_pattern}")
    assert prop_pod.assembly_pattern == "RADIAL"
    
    # 4. Trigger Merge (Pattern-based snapping)
    logger.info("--- Step 2: Merging (Pattern-based Snapping) ---")
    pm.merge_pod(prop_pod)
    
    # Verify Snapping for 3 'rotators' (360/3 = 120, 240)
    # The first component p001 is the hub (no rotation).
    # p002, p003, p004 are the rotators.
    p002 = next(c for c in prop_pod.linked_components if c["id"] == "p002")
    p003 = next(c for c in prop_pod.linked_components if c["id"] == "p003")
    p004 = next(c for c in prop_pod.linked_components if c["id"] == "p004")
    
    print(f"P002 (Index 0) Rotation: {p002['transform']['rotate']}")
    print(f"P003 (Index 1) Rotation: {p003['transform']['rotate']}")
    print(f"P004 (Index 2) Rotation: {p004['transform']['rotate']}")
    
    assert p002['transform']['rotate'][2] == 0.0
    assert p003['transform']['rotate'][2] == 120.0
    assert p004['transform']['rotate'][2] == 240.0

    logger.info("De-hardcoded Pattern Snapping (Phase 24) Verification COMPLETED successfully.")

if __name__ == "__main__":
    asyncio.run(test_modular_pod_system_phase24_dehardcoded())
