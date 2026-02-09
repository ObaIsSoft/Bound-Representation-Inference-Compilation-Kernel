"""
Standards Sync Service

Coordinates fetching standards from multiple sources and syncing to database.
- Fetches from NIST, NASA, web sources
- Parses purchased PDFs
- Validates data quality
- Syncs to Supabase standards_reference table
- Tracks sync history
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from .standards_fetcher import StandardsFetcher, FetchResult
from .connectors.nist_connector import NISTConnector
from .connectors.nasa_connector import NASAConnector
from .connectors.web_scraper import StandardsWebScraper
from .parsers.pdf_parser import PDFStandardParser

logger = logging.getLogger(__name__)


class StandardsSync:
    """
    Syncs standards from multiple sources to database.
    
    Usage:
        sync = StandardsSync()
        
        # Sync specific standard
        result = await sync.sync_standard("nist", "FIPS", "140-3")
        
        # Sync all from a source
        results = await sync.sync_all_from_source("nist")
        
        # Parse and sync PDF
        result = await sync.parse_and_sync_pdf("/path/to/ASTM_A36.pdf", "astm")
    """
    
    def __init__(self):
        self.connectors: Dict[str, StandardsFetcher] = {
            "nist": NISTConnector(),
            "nasa": NASAConnector(),
            "web": StandardsWebScraper()
        }
        self.pdf_parser = PDFStandardParser()
        
    async def sync_standard(
        self,
        source: str,
        standard_type: str,
        standard_number: str,
        revision: Optional[str] = None,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Sync a single standard from source to database.
        
        Args:
            source: Connector name (nist, nasa, web)
            standard_type: Type of standard
            standard_number: Standard number
            revision: Specific revision
            validate: Validate data before syncing
            
        Returns:
            Sync result with status
        """
        connector = self.connectors.get(source)
        if not connector:
            return {
                "success": False,
                "error": f"Unknown source: {source}",
                "standard": f"{standard_type}-{standard_number}"
            }
        
        try:
            # Fetch from source
            logger.info(f"Fetching {standard_type}-{standard_number} from {source}")
            result = await connector.fetch_standard(
                standard_type, standard_number, revision
            )
            
            if not result.success:
                return {
                    "success": False,
                    "error": result.errors,
                    "standard": f"{standard_type}-{standard_number}",
                    "source": source
                }
            
            # Validate if requested
            if validate:
                result = await self._validate_standard(result)
                if result.validation_status == "invalid":
                    return {
                        "success": False,
                        "error": result.errors,
                        "standard": f"{standard_type}-{standard_number}",
                        "validation": "failed"
                    }
            
            # Store in database
            db_result = await self._store_standard(result)
            
            return {
                "success": True,
                "standard": f"{standard_type}-{standard_number}",
                "revision": result.revision,
                "source": source,
                "stored": db_result,
                "validation": result.validation_status
            }
            
        except Exception as e:
            logger.error(f"Sync error: {e}")
            return {
                "success": False,
                "error": str(e),
                "standard": f"{standard_type}-{standard_number}",
                "source": source
            }
    
    async def sync_all_from_source(
        self,
        source: str,
        standard_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Sync all available standards from a source.
        
        Args:
            source: Connector name
            standard_type: Filter by type
            limit: Max standards to sync
            
        Returns:
            List of sync results
        """
        connector = self.connectors.get(source)
        if not connector:
            return [{"success": False, "error": f"Unknown source: {source}"}]
        
        # Get available standards from source
        available = await connector.list_available(standard_type)
        
        results = []
        for std in available[:limit]:
            # For each available standard, try to sync
            std_type = std.get("type", "unknown")
            
            # Skip if it's just metadata (no specific standard number)
            if std_type in ["FIPS", "SP 800", "NASA-STD"]:
                # These are series, would need specific numbers
                results.append({
                    "success": False,
                    "standard": std_type,
                    "error": "Series listing only - need specific standard number",
                    "info": std
                })
                continue
            
            # Try to sync
            result = await self.sync_standard(source, std_type, "")
            results.append(result)
        
        return results
    
    async def parse_and_sync_pdf(
        self,
        pdf_path: str,
        standard_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Parse a purchased standards PDF and sync to database.
        
        Args:
            pdf_path: Path to PDF file
            standard_type: Hint for parser (iso, astm, etc.)
            
        Returns:
            Sync result
        """
        try:
            # Parse PDF
            logger.info(f"Parsing PDF: {pdf_path}")
            parsed = self.pdf_parser.parse_pdf(pdf_path, standard_type)
            
            # Convert to FetchResult
            result = FetchResult(
                success=True,
                standard_type=parsed.standard_type.lower(),
                standard_number=parsed.standard_number,
                revision=parsed.revision,
                data={
                    "title": parsed.title,
                    "abstract": parsed.abstract,
                    "scope": parsed.scope,
                    "tables": parsed.tables,
                    "sections": parsed.sections,
                    "parsed_from_pdf": True,
                    "pdf_path": pdf_path
                },
                source="PDF_Parser",
                fetched_at=datetime.now()
            )
            
            # Validate
            result = await self._validate_standard(result)
            
            # Store
            db_result = await self._store_standard(result)
            
            return {
                "success": True,
                "standard": f"{parsed.standard_type}-{parsed.standard_number}",
                "revision": parsed.revision,
                "title": parsed.title,
                "tables_extracted": len(parsed.tables),
                "stored": db_result,
                "warnings": parsed.warnings
            }
            
        except Exception as e:
            logger.error(f"PDF parse error: {e}")
            return {
                "success": False,
                "error": str(e),
                "pdf_path": pdf_path
            }
    
    async def search_and_sync(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search across sources and sync found standards.
        
        Args:
            query: Search query
            sources: List of sources to search (default: all)
            limit: Max results per source
            
        Returns:
            List of sync results
        """
        sources = sources or list(self.connectors.keys())
        results = []
        
        for source_name in sources:
            connector = self.connectors[source_name]
            
            try:
                # Search
                search_results = await connector.search_standards(query, limit=limit)
                
                # Try to sync each found standard
                for found in search_results:
                    std_type = found.get("type", "unknown")
                    std_number = found.get("number", "")
                    
                    if std_number:
                        sync_result = await self.sync_standard(
                            source_name, std_type, std_number
                        )
                        results.append(sync_result)
                    else:
                        results.append({
                            "success": False,
                            "standard": f"{std_type}-{std_number}",
                            "error": "No standard number found",
                            "source": source_name,
                            "search_result": found
                        })
                        
            except Exception as e:
                logger.error(f"Search error for {source_name}: {e}")
                results.append({
                    "success": False,
                    "source": source_name,
                    "error": str(e)
                })
        
        return results
    
    async def _validate_standard(self, result: FetchResult) -> FetchResult:
        """Validate standard data"""
        # Define validation rules by standard type
        rules = {
            "iso": {
                "required_fields": ["title"],
                "numeric_ranges": {}
            },
            "astm": {
                "required_fields": ["title"],
                "numeric_ranges": {}
            },
            "nist": {
                "required_fields": ["title"],
                "numeric_ranges": {}
            },
            "nasa": {
                "required_fields": ["title"],
                "numeric_ranges": {}
            }
        }
        
        std_type = result.standard_type.lower()
        validation_rules = rules.get(std_type, {})
        
        return await result.validate_data(result, validation_rules)
    
    async def _store_standard(self, result: FetchResult) -> bool:
        """Store standard in database"""
        try:
            from backend.services import supabase
            
            # Prepare data for database
            db_data = {
                "standard_type": result.standard_type,
                "standard_key": result.standard_number,
                "standard_org": self._get_organization(result.source),
                "standard_number": result.standard_number,
                "standard_revision": result.revision,
                "data": result.data,
                "description": result.data.get("abstract") or result.data.get("scope") or "",
                "fetched_at": result.fetched_at.isoformat(),
                "source_url": result.url,
                "validation_status": result.validation_status
            }
            
            # Upsert to database
            # Use raw Supabase client for the upsert
            await supabase.initialize()
            
            # Check if exists
            existing = supabase.client.table("standards_reference")\
                .select("id")\
                .eq("standard_type", result.standard_type)\
                .eq("standard_key", result.standard_number)\
                .execute()
            
            if existing.data:
                # Update
                supabase.client.table("standards_reference")\
                    .update(db_data)\
                    .eq("id", existing.data[0]["id"])\
                    .execute()
                logger.info(f"Updated {result.standard_type}-{result.standard_number}")
            else:
                # Insert
                supabase.client.table("standards_reference")\
                    .insert(db_data)\
                    .execute()
                logger.info(f"Inserted {result.standard_type}-{result.standard_number}")
            
            return True
            
        except Exception as e:
            logger.error(f"Database store error: {e}")
            return False
    
    def _get_organization(self, source: str) -> str:
        """Map source to organization"""
        mapping = {
            "NIST": "NIST",
            "NASA": "NASA",
            "ISO.org": "ISO",
            "ASTM.org": "ASTM",
            "ANSI": "ANSI",
            "PDF_Parser": "Parsed"
        }
        return mapping.get(source, source)
    
    async def get_sync_status(self) -> Dict[str, Any]:
        """Get sync status summary"""
        try:
            from backend.services import supabase
            await supabase.initialize()
            
            # Count by source
            result = supabase.client.table("standards_reference")\
                .select("standard_org", count="exact")\
                .execute()
            
            total = len(result.data) if result.data else 0
            
            # Count by type
            by_type = {}
            for row in result.data:
                org = row.get("standard_org", "unknown")
                by_type[org] = by_type.get(org, 0) + 1
            
            return {
                "total_standards": total,
                "by_organization": by_type,
                "sources_available": list(self.connectors.keys()),
                "pdf_parser_available": True
            }
            
        except Exception as e:
            logger.error(f"Status error: {e}")
            return {
                "error": str(e),
                "sources_available": list(self.connectors.keys())
            }
