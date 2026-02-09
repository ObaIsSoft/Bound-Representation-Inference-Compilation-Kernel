"""
Base class for standards fetchers.

Defines the interface for fetching standards from various sources.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class FetchResult:
    """Result of a standards fetch operation"""
    success: bool
    standard_type: str  # e.g., "iso", "astm", "nec"
    standard_number: str  # e.g., "286-1", "A36", "310.16"
    revision: Optional[str]  # e.g., "2010", "2023"
    data: Dict[str, Any]  # The actual standard data
    source: str  # Where it came from
    fetched_at: datetime
    url: Optional[str] = None  # Source URL
    pdf_url: Optional[str] = None  # PDF link if available
    validation_status: str = "unvalidated"  # unvalidated, valid, invalid
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class StandardsFetcher:
    """
    Base class for fetching engineering standards.
    
    All connectors must implement:
    - fetch_standard(): Fetch a specific standard
    - search_standards(): Search for standards
    - list_available(): List what this source can provide
    """
    
    def __init__(self, source_name: str):
        self.source_name = source_name
        self._cache: Dict[str, FetchResult] = {}
        self._cache_ttl_hours = 24
        
    async def fetch_standard(
        self,
        standard_type: str,
        standard_number: str,
        revision: Optional[str] = None
    ) -> FetchResult:
        """
        Fetch a specific standard.
        
        Args:
            standard_type: Type of standard (iso, astm, nec, etc.)
            standard_number: Standard number (286-1, A36, etc.)
            revision: Specific revision (2010, 2023, etc.)
            
        Returns:
            FetchResult with data or error
        """
        raise NotImplementedError("Subclasses must implement fetch_standard()")
    
    async def search_standards(
        self,
        query: str,
        standard_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for standards matching query.
        
        Args:
            query: Search terms
            standard_type: Filter by type
            limit: Max results
            
        Returns:
            List of matching standards metadata
        """
        raise NotImplementedError("Subclasses must implement search_standards()")
    
    async def list_available(
        self,
        standard_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List available standards from this source.
        
        Returns:
            List of available standards with metadata
        """
        raise NotImplementedError("Subclasses must implement list_available()")
    
    def _get_cache_key(
        self,
        standard_type: str,
        standard_number: str,
        revision: Optional[str] = None
    ) -> str:
        """Generate cache key for a standard"""
        rev_str = f"_{revision}" if revision else ""
        return f"{self.source_name}:{standard_type}:{standard_number}{rev_str}"
    
    def _get_cached(
        self,
        standard_type: str,
        standard_number: str,
        revision: Optional[str] = None
    ) -> Optional[FetchResult]:
        """Get cached result if not expired"""
        key = self._get_cache_key(standard_type, standard_number, revision)
        cached = self._cache.get(key)
        
        if cached:
            age_hours = (datetime.now() - cached.fetched_at).total_seconds() / 3600
            if age_hours < self._cache_ttl_hours:
                logger.debug(f"Cache hit for {key}")
                return cached
            else:
                logger.debug(f"Cache expired for {key}")
                del self._cache[key]
        
        return None
    
    def _set_cached(self, result: FetchResult):
        """Cache a fetch result"""
        key = self._get_cache_key(
            result.standard_type,
            result.standard_number,
            result.revision
        )
        self._cache[key] = result
        logger.debug(f"Cached {key}")
    
    async def validate_data(
        self,
        result: FetchResult,
        validation_rules: Dict[str, Any]
    ) -> FetchResult:
        """
        Validate fetched data against rules.
        
        Args:
            result: FetchResult to validate
            validation_rules: Rules for validation
            
        Returns:
            FetchResult with updated validation_status
        """
        errors = []
        
        # Check required fields
        required_fields = validation_rules.get("required_fields", [])
        for field in required_fields:
            if field not in result.data or result.data[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Check numeric ranges
        ranges = validation_rules.get("numeric_ranges", {})
        for field, (min_val, max_val) in ranges.items():
            if field in result.data:
                val = result.data[field]
                if val is not None and (val < min_val or val > max_val):
                    errors.append(
                        f"Field {field}={val} out of range [{min_val}, {max_val}]"
                    )
        
        # Update result
        if errors:
            result.validation_status = "invalid"
            result.errors.extend(errors)
        else:
            result.validation_status = "valid"
        
        return result
