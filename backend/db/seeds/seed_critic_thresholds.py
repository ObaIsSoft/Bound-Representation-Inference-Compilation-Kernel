#!/usr/bin/env python3
"""
Seed Critic Thresholds

This script inserts verified critic thresholds into the database.
Edit the VERIFIED_THRESHOLDS list below with your validated values.

Requirements for each threshold:
    - Engineering analysis or physical testing
    - Verified by named engineer
    - Documented verification method

Usage:
    1. Edit VERIFIED_THRESHOLDS below
    2. Run: python backend/db/seeds/seed_critic_thresholds.py
"""

import asyncio
import json
import logging
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# EDIT THIS SECTION WITH YOUR VERIFIED THRESHOLDS
# ============================================================
#
# Example structure:
# VERIFIED_THRESHOLDS = [
#     {
#         "critic_name": "ControlCritic",
#         "vehicle_type": "drone_small",
#         "thresholds": {
#             "max_thrust_n": 100.0,
#             "max_torque_nm": 10.0,
#             "max_velocity_ms": 20.0,
#             "max_position_m": 500.0,
#             "control_effort_threshold": 50.0,
#             "energy_increase_limit": 1.1
#         },
#         "verified_by": "Jane Engineer",
#         "verification_method": "simulation"  # or "testing", "analysis"
#     },
# ]
#
# ============================================================

VERIFIED_THRESHOLDS = [
    # Add your verified thresholds here
    # {
    #     "critic_name": "ControlCritic",
    #     "vehicle_type": "drone_small",
    #     "thresholds": {
    #         "max_thrust_n": 100.0,
    #         "max_torque_nm": 10.0,
    #         "max_velocity_ms": 20.0,
    #         "max_position_m": 500.0,
    #         "control_effort_threshold": 50.0,
    #         "energy_increase_limit": 1.1
    #     },
    #     "verified_by": "Your Name",
    #     "verification_method": "simulation"
    # },
]


async def seed_critic_thresholds() -> bool:
    """Seed verified critic thresholds into the database."""
    
    try:
        from backend.services import supabase
        await supabase.connect()
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {e}")
        logger.error("Check your .env file has SUPABASE_URL and SUPABASE_SERVICE_KEY")
        return False
    
    if not VERIFIED_THRESHOLDS:
        logger.warning("=" * 60)
        logger.warning("NO VERIFIED THRESHOLDS CONFIGURED")
        logger.warning("=" * 60)
        logger.warning("")
        logger.warning("The VERIFIED_THRESHOLDS list is empty.")
        logger.warning("Edit this file and add your verified thresholds.")
        logger.warning("")
        logger.warning("Example:")
        logger.warning(json.dumps({
            "critic_name": "ControlCritic",
            "vehicle_type": "drone_small",
            "thresholds": {
                "max_thrust_n": 100.0,
                "max_torque_nm": 10.0
            },
            "verified_by": "Jane Engineer",
            "verification_method": "simulation"
        }, indent=2))
        return False
    
    inserted_count = 0
    
    for threshold in VERIFIED_THRESHOLDS:
        # Validate required fields
        required = ["critic_name", "vehicle_type", "thresholds", 
                   "verified_by", "verification_method"]
        missing = [f for f in required if f not in threshold]
        if missing:
            logger.error(f"Skipping entry - missing fields: {missing}")
            continue
        
        try:
            result = await supabase.client.table("critic_thresholds")\
                .upsert({
                    "critic_name": threshold["critic_name"],
                    "vehicle_type": threshold["vehicle_type"],
                    "thresholds": threshold["thresholds"],
                    "verified_by": threshold["verified_by"],
                    "verified_at": datetime.utcnow().isoformat(),
                    "verification_method": threshold["verification_method"],
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                }, on_conflict="critic_name,vehicle_type")\
                .execute()
            
            logger.info(
                f"✓ Seeded {threshold['critic_name']} / {threshold['vehicle_type']} "
                f"(verified by {threshold['verified_by']})"
            )
            inserted_count += 1
            
        except Exception as e:
            logger.error(f"Failed to insert {threshold['critic_name']}: {e}")
            continue
    
    logger.info(f"\n✓ Inserted/updated {inserted_count} verified critic thresholds")
    return inserted_count > 0


def main():
    """CLI entry point"""
    success = asyncio.run(seed_critic_thresholds())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
