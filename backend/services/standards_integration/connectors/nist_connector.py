"""
NIST Standards Connector

Fetches NIST standards information.
Many NIST standards are freely available as PDFs.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from ..standards_fetcher import StandardsFetcher, FetchResult

logger = logging.getLogger(__name__)


class NISTConnector(StandardsFetcher):
    """
    Connector for NIST standards.
    
    NIST provides many standards as free PDFs at:
    https://nvlpubs.nist.gov/
    
    Known working standards:
    - FIPS 140-3, FIPS 197, FIPS 180-4, FIPS 186-5
    - SP 800 series (various)
    """
    
    NIST_PUBS_URL = "https://nvlpubs.nist.gov/nistpubs"
    
    # Known working standards database
    KNOWN_STANDARDS = {
        "FIPS": {
            "140-3": {"title": "Security Requirements for Cryptographic Modules", "url": "FIPS/NIST.FIPS.140-3.pdf"},
            "197": {"title": "Advanced Encryption Standard (AES)", "url": "FIPS/NIST.FIPS.197.pdf"},
            "180-4": {"title": "Secure Hash Standard (SHA)", "url": "FIPS/NIST.FIPS.180-4.pdf"},
            "186-5": {"title": "Digital Signature Standard (DSS)", "url": "FIPS/NIST.FIPS.186-5.pdf"},
        },
        "SP": {
            "800-53": {"title": "Security and Privacy Controls", "url": "SpecialPublication/NIST.SP.800-53r5.pdf"},
            "800-53a": {"title": "Assessing Security and Privacy Controls", "url": "SpecialPublication/NIST.SP.800-53Ar5.pdf"},
            "800-171": {"title": "Protecting Controlled Unclassified Information", "url": "SpecialPublication/NIST.SP.800-171r2.pdf"},
            "800-30": {"title": "Guide for Conducting Risk Assessments", "url": "SpecialPublication/NIST.SP.800-30r1.pdf"},
            "800-37": {"title": "Risk Management Framework", "url": "SpecialPublication/NIST.SP.800-37r2.pdf"},
            "800-207": {"title": "Zero Trust Architecture", "url": "SpecialPublication/NIST.SP.800-207.pdf"},
        },
        "NISTIR": {
            "7628": {"title": "Guidelines for Smart Grid Cybersecurity", "url": "IR/NIST.IR.7628r1.pdf"},
            "8259": {"title": "Core Cybersecurity Feature Baseline", "url": "IR/NIST.IR.8259.pdf"},
        }
    }
    
    def __init__(self):
        super().__init__("NIST")
        self.client = None
        
    async def _get_client(self):
        if self.client is None and HAS_HTTPX:
            self.client = httpx.AsyncClient(timeout=30.0)
        return self.client
    
    async def fetch_standard(
        self,
        standard_type: str,
        standard_number: str,
        revision: Optional[str] = None
    ) -> FetchResult:
        """
        Fetch NIST standard.
        
        Args:
            standard_type: fips, sp800, sp, nistir
            standard_number: Standard number
            revision: Revision (r1, r2, r5, etc.)
        """
        cached = self._get_cached(standard_type, standard_number, revision)
        if cached:
            return cached
        
        # Map type to our database
        type_map = {
            "fips": "FIPS",
            "sp800": "SP",
            "sp": "SP",
            "nistir": "NISTIR",
            "ir": "NISTIR"
        }
        
        db_type = type_map.get(standard_type.lower(), standard_type.upper())
        
        # Check if we know this standard
        if db_type in self.KNOWN_STANDARDS:
            std_db = self.KNOWN_STANDARDS[db_type]
            
            # Try exact match first
            if standard_number in std_db:
                std_info = std_db[standard_number]
                pdf_url = f"{self.NIST_PUBS_URL}/{std_info['url']}"
                
                # Verify PDF exists
                if await self._check_pdf_exists(pdf_url):
                    result = FetchResult(
                        success=True,
                        standard_type=standard_type,
                        standard_number=standard_number,
                        revision=revision,
                        data={
                            "title": std_info["title"],
                            "organization": "NIST",
                            "series": db_type,
                            "number": standard_number,
                            "pdf_url": pdf_url,
                            "pdf_available": True,
                            "access": "Free download"
                        },
                        source="NIST",
                        fetched_at=datetime.now(),
                        url=pdf_url,
                        pdf_url=pdf_url
                    )
                    self._set_cached(result)
                    return result
        
        # Not found in known database - try to construct URL
        return await self._try_construct_url(standard_type, standard_number, revision)
    
    async def _check_pdf_exists(self, url: str) -> bool:
        """Check if PDF URL exists"""
        if not HAS_HTTPX:
            return False
        
        try:
            client = await self._get_client()
            response = await client.head(url, follow_redirects=True)
            return response.status_code == 200
        except:
            return False
    
    async def _try_construct_url(
        self,
        standard_type: str,
        standard_number: str,
        revision: Optional[str]
    ) -> FetchResult:
        """Try to construct NIST URL from pattern"""
        
        # Map to NIST series
        series_map = {
            "fips": ("FIPS", "FIPS"),
            "sp800": ("SpecialPublication", "SP"),
            "sp": ("SpecialPublication", "SP"),
            "nistir": ("IR", "IR"),
        }
        
        series_folder, series_code = series_map.get(
            standard_type.lower(), 
            (standard_type.upper(), standard_type.upper())
        )
        
        # Try various URL patterns
        urls_to_try = [
            f"{self.NIST_PUBS_URL}/{series_folder}/NIST.{series_code}.{standard_number}.pdf",
        ]
        
        if revision:
            urls_to_try.insert(0, f"{self.NIST_PUBS_URL}/{series_folder}/NIST.{series_code}.{standard_number}{revision}.pdf")
            urls_to_try.insert(1, f"{self.NIST_PUBS_URL}/{series_folder}/NIST.{series_code}.{standard_number}.{revision}.pdf")
        
        for url in urls_to_try:
            if await self._check_pdf_exists(url):
                result = FetchResult(
                    success=True,
                    standard_type=standard_type,
                    standard_number=standard_number,
                    revision=revision,
                    data={
                        "title": f"NIST {series_code} {standard_number}",
                        "pdf_url": url,
                        "pdf_available": True,
                        "note": "URL constructed from pattern - verify content"
                    },
                    source="NIST",
                    fetched_at=datetime.now(),
                    url=url,
                    pdf_url=url
                )
                self._set_cached(result)
                return result
        
        # Not found
        return FetchResult(
            success=False,
            standard_type=standard_type,
            standard_number=standard_number,
            revision=revision,
            data={},
            source="NIST",
            fetched_at=datetime.now(),
            errors=[f"Standard not found in NIST database or URL not accessible"]
        )
    
    async def search_standards(
        self,
        query: str,
        standard_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search known NIST standards"""
        results = []
        query_lower = query.lower()
        
        for series, standards in self.KNOWN_STANDARDS.items():
            if standard_type and series.lower() != standard_type.lower():
                continue
            
            for number, info in standards.items():
                if (query_lower in series.lower() or
                    query_lower in number.lower() or
                    query_lower in info["title"].lower()):
                    
                    results.append({
                        "type": series,
                        "number": number,
                        "title": info["title"],
                        "url": f"{self.NIST_PUBS_URL}/{info['url']}"
                    })
        
        return results[:limit]
    
    async def list_available(
        self,
        standard_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List NIST standards series"""
        series = [
            {
                "type": "FIPS",
                "name": "Federal Information Processing Standards",
                "description": "Standards for computer systems",
                "count": len(self.KNOWN_STANDARDS.get("FIPS", {})),
                "url": "https://csrc.nist.gov/publications/fips",
                "access": "Free PDF downloads"
            },
            {
                "type": "SP",
                "name": "Special Publications",
                "description": "Security and IT guidelines",
                "count": len(self.KNOWN_STANDARDS.get("SP", {})),
                "url": "https://csrc.nist.gov/publications/sp800",
                "access": "Free PDF downloads"
            },
            {
                "type": "NISTIR",
                "name": "NIST Interagency/Internal Reports",
                "description": "Research publications",
                "count": len(self.KNOWN_STANDARDS.get("NISTIR", {})),
                "url": "https://www.nist.gov/publications",
                "access": "Free PDF downloads"
            }
        ]
        
        if standard_type:
            return [s for s in series if s["type"].lower() == standard_type.lower()]
        return series
