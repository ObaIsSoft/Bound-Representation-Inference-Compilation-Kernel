
import logging
import sys
import os
import numpy as np

sys.path.append(os.path.abspath("."))

from backend.agents.visual_validator_agent import VisualValidatorAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VisualVerify")

def create_temp_stl(filename, broken=False):
    """Create a simple ASCII STL. If broken, omit a face."""
    # Simple Tetrahedron
    p1 = "0 0 0"
    p2 = "1 0 0"
    p3 = "0 1 0"
    p4 = "0 0 1"
    
    # Faces: 123, 124, 234, 134 (Normal convention ignored for brevity, just want topology)
    faces = [
        (p1, p3, p2),
        (p1, p2, p4),
        (p2, p3, p4),
        (p1, p4, p3)
    ]
    
    if broken:
        faces.pop() # Create a hole
        
    with open(filename, "w") as f:
        f.write("solid test\n")
        for a, b, c in faces:
            f.write(f"facet normal 0 0 0\nouter loop\nvertex {a}\nvertex {b}\nvertex {c}\nendloop\nendfacet\n")
        f.write("endsolid test\n")

def verify_visual_validator():
    logger.info("--- Starting Tier 5 Verification (Visual) ---")
    
    agent = VisualValidatorAgent()
    
    # 1. Valid Mesh
    create_temp_stl("temp_valid.stl", broken=False)
    logger.info("Testing Valid Mesh...")
    res_valid = agent.run({"mesh_path": "temp_valid.stl"})
    
    if res_valid["is_valid"]:
        logger.info("SUCCESS: Valid mesh passes.")
    else:
        logger.error(f"FAILURE: Valid mesh flagged: {res_valid['artifacts_detected']}")
        
    # 2. Broken Mesh
    create_temp_stl("temp_broken.stl", broken=True)
    logger.info("Testing Broken Mesh...")
    res_broken = agent.run({"mesh_path": "temp_broken.stl"})
    
    if not res_broken["is_valid"]:
        logger.info(f"SUCCESS: Broken mesh flagged: {res_broken['artifacts_detected']}")
    else:
        logger.error("FAILURE: Broken mesh NOT flagged.")

    # Cleanup
    if os.path.exists("temp_valid.stl"): os.remove("temp_valid.stl")
    if os.path.exists("temp_broken.stl"): os.remove("temp_broken.stl")

if __name__ == "__main__":
    verify_visual_validator()
