import sys
import os
import logging
import random
import copy

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agents.evolution import GeometryGenome, EvolutionaryMutator
from agents.generative.latent_agent import LatentSpaceAgent

# Configure logging
logging.basicConfig(level=logging.INFO)

def run_latent_verification():
    print("=== Latent Space Agent Verification ===")
    
    agent = LatentSpaceAgent()
    
    # 1. Test Parametric Interpolation (Same Topology)
    print("\n--- Test 1: Parametric Morph ---")
    g_a = GeometryGenome() # Default Cube (1.0)
    # Modify A
    g_a.graph.nodes[g_a.root_id]['data'].params['width'].value = 1.0
    
    g_b = g_a.clone()
    g_b.graph.nodes[g_b.root_id]['data'].params['width'].value = 10.0 # Big Cube
    
    # Interpolate
    morph_sequence = agent.interpolate(g_a, g_b, steps=4)
    for i, frame in enumerate(morph_sequence):
        # Check width of root node
        nodes = frame['nodes']
        root = nodes[0]
        width = root['params']['width']['value']
        print(f"Step {i}: Root Width = {width:.2f}")
        
    expected_mid = 5.5
    actual_mid = morph_sequence[2]['nodes'][0]['params']['width']['value'] # Step 2 is 50%
    if abs(actual_mid - expected_mid) < 0.1:
        print("[SUCCESS] Parametric Interpolation accurate.")
    else:
        print(f"[FAILED] Expected {expected_mid}, got {actual_mid}")

    # 2. Test Manifold Learning (Dimensionality Reduction)
    print("\n--- Test 2: Manifold Learning (PCA) ---")
    # Generate population
    population = []
    for _ in range(10):
        g = GeometryGenome()
        # Mutate randomly to create variance
        for _ in range(3):
            EvolutionaryMutator._add_random_primitive(g)
        population.append(g)
        
    # Fit
    agent.learn_manifold(population)
    
    # Encode
    z = agent.encode(population[0])
    print(f"Encoded Vector (Z): {z}")
    
    if len(z) == 3:
        print("[SUCCESS] Encoding produced 3D vector.")
    else:
        print("[FAILED] Encoding dimension mismatch.")
        
    # Decode (Retrieval)
    recon = agent.decode(z)
    if recon:
        print(f"Decoded (Retrieved) Node Count: {len(recon['nodes'])}")
        print("[SUCCESS] Decoding retrieved a valid design.")
    else:
        print("[FAILED] Decoding returned None.")

if __name__ == "__main__":
    run_latent_verification()
