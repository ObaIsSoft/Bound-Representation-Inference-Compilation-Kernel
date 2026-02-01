
import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class SupabaseClient:
    """
    Wrapper for Supabase interaction.
    Handles connection and basic CRUD operations.
    """
    def __init__(self):
        # Load env if local
        try:
            from dotenv import load_dotenv
            load_dotenv() # Loads .env from current or parent dirs
            # Specifically point to backend/.env if running from root
            load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
        except ImportError:
            pass
            
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
        self.client = None
        self.enabled = False

        if self.url and self.key:
            try:
                from supabase import create_client, Client, ClientOptions
                self.client: Client = create_client(
                    self.url, 
                    self.key,
                    options=ClientOptions(
                        postgrest_client_timeout=10, 
                        storage_client_timeout=10
                    )
                )
                self.enabled = True
                logger.info("Supabase Client Initialized")
            except ImportError:
                logger.error("Supabase Python SDK not found. Install with 'pip install supabase'.")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase: {e}")
        else:
            logger.warning("Supabase URL or Key missing in environment.")

    def fetch_table(self, table_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch all rows from a table.
        """
        if not self.enabled:
            return []
            
        try:
            response = self.client.table(table_name).select("*").limit(limit).execute()
            return response.data
        except Exception as e:
            logger.error(f"Supabase Fetch Error ({table_name}): {e}")
            return []

    def upsert_data(self, table_name: str, data: List[Dict[str, Any]], primary_key: str = "id"):
        """
        Upsert data into a table.
        """
        if not self.enabled:
            return
            
        try:
            # Upsert logic depends on exact Supabase capabilities, assuming standard upsert
            response = self.client.table(table_name).upsert(data).execute()
            logger.info(f"Synced {len(data)} rows to {table_name}")
            return response
        except Exception as e:
            logger.error(f"Supabase Sync Error ({table_name}): {e}")
