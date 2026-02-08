#!/usr/bin/env python3
"""
Database Migration Runner

Applies SQL migration files to Supabase in order.

Usage:
    python backend/db/apply_migrations.py [--check]
    
Options:
    --check    Only check connectivity, don't apply migrations
"""

import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.services import supabase


MIGRATION_DIR = Path(__file__).parent / "schema"


async def list_migrations() -> List[Tuple[int, Path]]:
    """
    List all migration files in order.
    
    Returns:
        List of (sequence_number, file_path)
    """
    migrations = []
    
    for file in MIGRATION_DIR.glob("*.sql"):
        # Extract sequence number from filename (001_name.sql -> 1)
        try:
            seq = int(file.name.split("_")[0])
            migrations.append((seq, file))
        except (ValueError, IndexError):
            logger.warning(f"Skipping file without sequence number: {file.name}")
            continue
    
    # Sort by sequence number
    migrations.sort(key=lambda x: x[0])
    return migrations


async def apply_migration(file_path: Path) -> bool:
    """
    Apply a single migration file.
    
    Args:
        file_path: Path to SQL file
        
    Returns:
        True if successful
    """
    try:
        # Read SQL file
        sql = file_path.read_text()
        
        logger.info(f"Applying {file_path.name}...")
        
        # Execute SQL (Supabase client doesn't support raw SQL directly)
        # We need to use RPC or split into statements
        # For now, print SQL for manual application
        
        logger.info(f"✓ {file_path.name} ready for application")
        logger.info(f"  File: {file_path}")
        logger.info(f"  Size: {len(sql)} bytes")
        logger.info(f"  Statements: ~{sql.count(';')} SQL statements")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to read {file_path.name}: {e}")
        return False


async def check_connection() -> bool:
    """
    Check Supabase connectivity.
    
    Returns:
        True if connected
    """
    try:
        await supabase.connect()
        
        # Try a simple query
        result = await supabase.client.table("materials")\
            .select("count", count="exact")\
            .limit(1)\
            .execute()
        
        logger.info("✓ Supabase connection successful")
        return True
        
    except Exception as e:
        logger.error(f"✗ Supabase connection failed: {e}")
        return False


async def get_applied_migrations() -> List[str]:
    """
    Get list of already applied migrations.
    
    Returns:
        List of migration filenames
    """
    try:
        # Check if we have a migrations tracking table
        result = await supabase.client.table("schema_migrations")\
            .select("migration_name")\
            .execute()
        
        return [row["migration_name"] for row in result.data]
        
    except Exception:
        # Table doesn't exist or no migrations yet
        return []


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Apply database migrations")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check connectivity"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all migrations"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be applied without applying"
    )
    
    args = parser.parse_args()
    
    # Check connection
    if not await check_connection():
        logger.error("Cannot connect to Supabase. Check your .env file.")
        sys.exit(1)
    
    if args.check:
        logger.info("Connection check complete")
        return
    
    # List migrations
    migrations = await list_migrations()
    
    if args.list:
        logger.info("Available migrations:")
        for seq, file in migrations:
            logger.info(f"  {seq:03d}: {file.name}")
        return
    
    # Get already applied migrations
    applied = await get_applied_migrations()
    
    # Apply pending migrations
    logger.info(f"\nFound {len(migrations)} migration files")
    
    pending = []
    for seq, file in migrations:
        if file.name not in applied:
            pending.append((seq, file))
    
    if not pending:
        logger.info("All migrations are up to date!")
        return
    
    logger.info(f"{len(pending)} pending migrations:\n")
    
    for seq, file in pending:
        logger.info(f"  • {file.name}")
    
    if args.dry_run:
        logger.info("\n(Dry run - no changes made)")
        return
    
    # Show instructions for manual application
    logger.info("\n" + "=" * 60)
    logger.info("MANUAL APPLICATION REQUIRED")
    logger.info("=" * 60)
    logger.info("\nSupabase doesn't support raw SQL via the Python client.")
    logger.info("Please apply the SQL files manually:\n")
    logger.info("Option 1: Supabase SQL Editor")
    logger.info("  1. Go to https://supabase.com/dashboard")
    logger.info("  2. Open SQL Editor")
    logger.info("  3. Copy/paste contents of each file in order:")
    for seq, file in pending:
        logger.info(f"     - {file}")
    logger.info("\nOption 2: Supabase CLI")
    logger.info("  supabase db execute --file <file.sql>")
    logger.info("\nOption 3: psql")
    logger.info("  psql $SUPABASE_URL -f <file.sql>")
    logger.info("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
