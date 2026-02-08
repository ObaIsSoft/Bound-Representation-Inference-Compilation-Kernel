#!/usr/bin/env python3
"""
Seed Critic Thresholds

Imports critic thresholds into Supabase.
This script can be run independently or as part of setup.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Default critic thresholds
# These should match the SQL seed data in 001_critic_thresholds.sql
CRITIC_THRESHOLDS = [
    {
        "critic_name": "ControlCritic",
        "vehicle_type": "drone_small",
        "thresholds": {
            "max_thrust_n": 100.0,
            "max_torque_nm": 10.0,
            "max_velocity_ms": 20.0,
            "max_position_m": 500.0,
            "control_effort_threshold": 50.0,
            "energy_increase_limit": 1.1
        }
    },
    {
        "critic_name": "ControlCritic",
        "vehicle_type": "drone_medium",
        "thresholds": {
            "max_thrust_n": 500.0,
            "max_torque_nm": 50.0,
            "max_velocity_ms": 35.0,
            "max_position_m": 1000.0,
            "control_effort_threshold": 75.0,
            "energy_increase_limit": 1.1
        }
    },
    {
        "critic_name": "ControlCritic",
        "vehicle_type": "drone_large",
        "thresholds": {
            "max_thrust_n": 1000.0,
            "max_torque_nm": 100.0,
            "max_velocity_ms": 50.0,
            "max_position_m": 2000.0,
            "control_effort_threshold": 100.0,
            "energy_increase_limit": 1.1
        }
    },
    {
        "critic_name": "MaterialCritic",
        "vehicle_type": "default",
        "thresholds": {
            "high_temp_threshold_c": 150,
            "degradation_rate_threshold": 0.5,
            "mass_error_threshold_pct": 10,
            "db_coverage_threshold": 0.7,
            "material_diversity_min": 3
        }
    },
    {
        "critic_name": "ElectronicsCritic",
        "vehicle_type": "default",
        "thresholds": {
            "power_deficit_threshold": 0.3,
            "short_detection_min_rate": 0.8,
            "over_conservative_margin_w": 1000,
            "false_alarm_threshold": 5,
            "scale_issue_threshold": 5
        }
    },
    {
        "critic_name": "SurrogateCritic",
        "vehicle_type": "default",
        "thresholds": {
            "drift_threshold": 0.15,
            "min_accuracy": 0.7,
            "min_gate_alignment": 0.7,
            "low_speed_threshold_ms": 10,
            "high_speed_threshold_ms": 50,
            "max_false_positive_rate": 0.3
        }
    },
    {
        "critic_name": "GeometryCritic",
        "vehicle_type": "default",
        "thresholds": {
            "max_failure_rate": 0.2,
            "performance_target_seconds": 2.0,
            "min_sdf_resolution": 32,
            "max_sdf_resolution": 256
        }
    }
]


async def seed_critic_thresholds(supabase_service=None) -> bool:
    """
    Seed critic thresholds into the database.
    
    Args:
        supabase_service: Optional Supabase service instance
        
    Returns:
        True if successful
    """
    if supabase_service is None:
        # Import the service
        try:
            from backend.services import supabase
            supabase_service = supabase
        except ImportError:
            logger.error("Cannot import supabase service. Run from project root.")
            return False
    
    try:
        # Check if already seeded
        existing = await supabase_service.client.table("critic_thresholds")\
            .select("id")\
            .limit(1)\
            .execute()
        
        if existing.data:
            logger.info(f"Found {len(existing.data)} existing thresholds")
            logger.info("Skipping seed - thresholds already exist")
            logger.info("Use --force to overwrite")
            return True
        
        # Insert thresholds
        for threshold in CRITIC_THRESHOLDS:
            try:
                result = await supabase_service.client.table("critic_thresholds")\
                    .insert({
                        "critic_name": threshold["critic_name"],
                        "vehicle_type": threshold["vehicle_type"],
                        "thresholds": threshold["thresholds"],
                        "created_at": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow().isoformat()
                    })\
                    .execute()
                
                logger.info(
                    f"✓ Seeded {threshold['critic_name']} / {threshold['vehicle_type']}"
                )
                
            except Exception as e:
                # Likely already exists (unique constraint)
                logger.debug(f"Skipping {threshold['critic_name']}: {e}")
                continue
        
        logger.info(f"✓ Seeded {len(CRITIC_THRESHOLDS)} critic thresholds")
        return True
        
    except Exception as e:
        logger.error(f"Failed to seed thresholds: {e}")
        return False


async def export_existing_thresholds(supabase_service, output_file: str) -> bool:
    """
    Export existing thresholds from database to file.
    
    Args:
        supabase_service: Supabase service instance
        output_file: Path to output JSON file
        
    Returns:
        True if successful
    """
    try:
        result = await supabase_service.client.table("critic_thresholds")\
            .select("*")\
            .execute()
        
        thresholds = result.data
        
        # Clean up for export
        for t in thresholds:
            t.pop("id", None)
            t.pop("created_at", None)
        
        with open(output_file, 'w') as f:
            json.dump(thresholds, f, indent=2)
        
        logger.info(f"✓ Exported {len(thresholds)} thresholds to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to export thresholds: {e}")
        return False


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Seed critic thresholds")
    parser.add_argument(
        "--export",
        metavar="FILE",
        help="Export existing thresholds to JSON file"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-seed even if data exists"
    )
    
    args = parser.parse_args()
    
    # Import and initialize
    import sys
    sys.path.insert(0, '/Users/obafemi/Documents/dev/brick')
    
    from backend.services import supabase
    
    async def run():
        await supabase.connect()
        
        if args.export:
            await export_existing_thresholds(supabase, args.export)
        else:
            await seed_critic_thresholds(supabase)
    
    asyncio.run(run())


if __name__ == "__main__":
    main()
