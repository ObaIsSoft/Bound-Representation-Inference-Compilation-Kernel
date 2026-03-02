#!/usr/bin/env python3
"""
Generate Training Data for FNO

Runs OpenFOAM simulations across parameter space to create
training data for Fourier Neural Operator.

Usage:
    python generate_fno_training_data.py --samples 100 --output ./training_data

Requirements:
    - OpenFOAM installed (openfoam2406)
    - ProductionFluidAgent from backend.agents
"""

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.agents.fluid_agent_production import (
    ProductionFluidAgent, FlowConditions, GeometryConfig, FidelityLevel
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_openfoam_simulation(
    agent: ProductionFluidAgent,
    geometry: GeometryConfig,
    conditions: FlowConditions,
    case_id: int,
    output_base_dir: Path
) -> Dict[str, Any]:
    """
    Run single OpenFOAM simulation and save results.
    
    Returns:
        Metadata dict with paths and parameters
    """
    case_name = f"case_{case_id:04d}_{geometry.shape_type}_Re{conditions.reynolds_number(geometry.length):.0f}"
    case_dir = output_base_dir / case_name
    
    logger.info(f"Running {case_name}...")
    
    try:
        # Run OpenFOAM with save_case=True
        result = agent.analyze(
            geometry,
            conditions,
            fidelity=FidelityLevel.RANS,
            save_openfoam_case=True
        )
        
        # Get case directory from result
        if result.openfoam_case_dir:
            # Copy to permanent location
            import shutil
            if case_dir.exists():
                shutil.rmtree(case_dir)
            shutil.copytree(result.openfoam_case_dir, case_dir)
            
            # Clean up temp
            shutil.rmtree(result.openfoam_case_dir)
        
        metadata = {
            'case_id': case_id,
            'case_name': case_name,
            'shape_type': geometry.shape_type,
            'length': geometry.length,
            'velocity': conditions.velocity,
            'temperature': conditions.temperature,
            'density': conditions.density,
            'reynolds': result.reynolds_number,
            'mach': result.mach_number,
            'cd': result.drag_coefficient,
            'cl': result.lift_coefficient,
            'case_dir': str(case_dir),
            'status': 'success'
        }
        
        logger.info(f"  Cd={result.drag_coefficient:.4f}, Re={result.reynolds_number:.2e}")
        
        return metadata
        
    except Exception as e:
        logger.error(f"Failed {case_name}: {e}")
        return {
            'case_id': case_id,
            'case_name': case_name,
            'status': 'failed',
            'error': str(e)
        }


def generate_parameter_space(n_samples: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Generate random parameter space for simulations."""
    np.random.seed(seed)
    
    samples = []
    shapes = ["cylinder", "box", "sphere"]
    
    for i in range(n_samples):
        # Random shape
        shape = np.random.choice(shapes)
        
        # Log-uniform length (0.01m to 10m)
        length = 10 ** np.random.uniform(-2, 1)
        
        # Log-uniform velocity (1 to 100 m/s)
        velocity = 10 ** np.random.uniform(0, 2)
        
        # Temperature (250K to 350K)
        temperature = np.random.uniform(250, 350)
        
        samples.append({
            'id': i,
            'shape_type': shape,
            'length': length,
            'width': length * np.random.uniform(0.5, 1.5),
            'height': length * np.random.uniform(0.5, 1.5),
            'velocity': velocity,
            'temperature': temperature,
            'density': 1.225,  # Air at sea level
        })
    
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Generate FNO training data from OpenFOAM"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of simulations to run (default: 100)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./fno_training_data",
        help="Output directory (default: ./fno_training_data)"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel jobs (default: 1)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating {args.samples} OpenFOAM simulations")
    logger.info(f"Output directory: {output_dir}")
    
    # Check OpenFOAM
    agent = ProductionFluidAgent()
    if not agent.openfoam_available:
        logger.error("OpenFOAM not available! Install with:")
        logger.error("  brew install openfoam@2406")
        sys.exit(1)
    
    # Generate parameter space
    params = generate_parameter_space(args.samples)
    
    # Save parameter space
    with open(output_dir / "parameter_space.json", "w") as f:
        json.dump(params, f, indent=2)
    
    logger.info(f"Parameter space saved to {output_dir / 'parameter_space.json'}")
    
    # Run simulations
    metadata_list = []
    
    for i, p in enumerate(params):
        geometry = GeometryConfig(
            shape_type=p['shape_type'],
            length=p['length'],
            width=p['width'],
            height=p['height']
        )
        conditions = FlowConditions(
            velocity=p['velocity'],
            temperature=p['temperature'],
            density=p['density']
        )
        
        metadata = run_openfoam_simulation(
            agent, geometry, conditions, i, output_dir
        )
        metadata_list.append(metadata)
    
    # Save metadata
    with open(output_dir / "simulation_metadata.json", "w") as f:
        json.dump(metadata_list, f, indent=2)
    
    # Summary
    successful = sum(1 for m in metadata_list if m['status'] == 'success')
    logger.info(f"\nCompleted: {successful}/{args.samples} simulations successful")
    logger.info(f"Results saved to: {output_dir}")
    
    # Next steps
    logger.info("\nNext steps:")
    logger.info("1. Extract flow fields from OpenFOAM cases")
    logger.info("2. Train FNO: python -m backend.agents.fno_fluid")
    logger.info("3. Validate against test cases")


if __name__ == "__main__":
    main()
