"""
NASA Standards Connector

Fetches NASA standards from NASA Technical Reports Server (NTRS).

NASA provides standards through:
- NASA Technical Standards: https://standards.nasa.gov
- NTRS: https://ntrs.nasa.gov/
- Many NASA standards are public domain
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


class NASAConnector(StandardsFetcher):
    """
    Connector for NASA standards.
    
    NASA standards are available through NTRS (NASA Technical Reports Server).
    Most NASA standards are public domain and freely accessible.
    """
    
    # NASA Technical Standards Website
    NASA_STANDARDS_URL = "https://standards.nasa.gov"
    
    # NTRS API
    NTRS_API_URL = "https://ntrs.nasa.gov/api"
    
    # Direct standards PDF URL pattern
    NASA_STD_PDF_URL = "https://standards.nasa.gov/standard/{standard_id}"
    
    def __init__(self):
        super().__init__("NASA")
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
        Fetch NASA standard.
        
        Args:
            standard_type: NASA-STD, MSFC-STD, etc.
            standard_number: Standard number (e.g., "5005")
            revision: Revision letter (e.g., "A")
        """
        cached = self._get_cached(standard_type, standard_number, revision)
        if cached:
            return cached
        
        if not HAS_HTTPX:
            return FetchResult(
                success=False,
                standard_type=standard_type,
                standard_number=standard_number,
                revision=revision,
                data={},
                source="NASA",
                fetched_at=datetime.now(),
                errors=["httpx not installed"]
            )
        
        try:
            client = await self._get_client()
            
            # Construct standard ID
            if revision:
                std_id = f"{standard_type}-{standard_number}{revision}"
            else:
                std_id = f"{standard_type}-{standard_number}"
            
            # Try to get from NTRS
            # NASA standards are in NTRS with specific search
            search_url = f"{self.NTRS_API_URL}/search"
            params = {
                "q": std_id,
                "category": "STI",
                "center": "NASA",
                "sort": "-publicationDate",
                "limit": 5
            }
            
            response = await client.get(search_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                if results:
                    # Found in NTRS
                    doc = results[0]
                    parsed = self._parse_ntrs_document(doc)
                    
                    # Construct PDF URL
                    pdf_url = None
                    if doc.get("downloads"):
                        downloads = doc.get("downloads", [])
                        for dl in downloads:
                            if dl.get("fileExtension") == "pdf":
                                pdf_url = dl.get("downloadLink")
                                break
                    
                    result = FetchResult(
                        success=True,
                        standard_type=standard_type,
                        standard_number=standard_number,
                        revision=revision or parsed.get("revision"),
                        data=parsed,
                        source="NASA_NTRS",
                        fetched_at=datetime.now(),
                        url=parsed.get("url"),
                        pdf_url=pdf_url
                    )
                else:
                    # Not in NTRS - check NASA Standards website
                    result = await self._fetch_from_standards_website(
                        standard_type, standard_number, revision
                    )
            else:
                # NTRS failed, try standards website
                result = await self._fetch_from_standards_website(
                    standard_type, standard_number, revision
                )
            
            self._set_cached(result)
            return result
            
        except Exception as e:
            logger.error(f"NASA fetch error: {e}")
            return FetchResult(
                success=False,
                standard_type=standard_type,
                standard_number=standard_number,
                revision=revision,
                data={},
                source="NASA",
                fetched_at=datetime.now(),
                errors=[str(e)]
            )
    
    async def _fetch_from_standards_website(
        self,
        standard_type: str,
        standard_number: str,
        revision: Optional[str] = None
    ) -> FetchResult:
        """Try to fetch from NASA Standards website"""
        client = await self._get_client()
        
        # Construct standard ID
        if revision:
            std_id = f"{standard_type}-{standard_number}{revision}"
        else:
            std_id = f"{standard_type}-{standard_number}"
        
        # NASA standards PDF URL pattern
        pdf_url = f"https://standards.nasa.gov/file/{std_id}.pdf"
        
        # Check if PDF exists
        response = await client.head(pdf_url, follow_redirects=True)
        
        if response.status_code == 200:
            return FetchResult(
                success=True,
                standard_type=standard_type,
                standard_number=standard_number,
                revision=revision,
                data={
                    "standard_id": std_id,
                    "pdf_url": pdf_url,
                    "pdf_available": True,
                    "note": "PDF available - metadata from standards website"
                },
                source="NASA_Standards",
                fetched_at=datetime.now(),
                url=f"https://standards.nasa.gov/standard/{std_id}",
                pdf_url=pdf_url
            )
        else:
            return FetchResult(
                success=False,
                standard_type=standard_type,
                standard_number=standard_number,
                revision=revision,
                data={},
                source="NASA",
                fetched_at=datetime.now(),
                errors=[f"Standard {std_id} not found in NTRS or standards website"]
            )
    
    def _parse_ntrs_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Parse NTRS document"""
        return {
            "title": doc.get("title"),
            "abstract": doc.get("abstract"),
            "publication_date": doc.get("publicationDate"),
            "revision": self._extract_revision(doc.get("title", "")),
            "authors": [a.get("name") for a in doc.get("authors", []) if isinstance(a, dict)],
            "subjects": doc.get("subjects", []),
            "center": doc.get("center", {}).get("name") if isinstance(doc.get("center"), dict) else None,
            "document_id": doc.get("id"),
            "download_count": doc.get("downloadCount"),
            "citation_count": doc.get("citationCount"),
            "url": f"https://ntrs.nasa.gov/citations/{doc.get('id')}",
            "source": "NASA_NTRS"
        }
    
    def _extract_revision(self, title: str) -> Optional[str]:
        """Extract revision from title"""
        import re
        match = re.search(r'-(\d+)([A-Z])', title)
        if match:
            return match.group(2)
        return None
    
    async def search_standards(
        self,
        query: str,
        standard_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search NASA standards"""
        if not HAS_HTTPX:
            return []
        
        try:
            client = await self._get_client()
            
            # Add "standard" to query for better results
            search_query = f"{query} standard"
            
            params = {
                "q": search_query,
                "category": "STI",
                "sort": "-publicationDate",
                "limit": limit
            }
            
            response = await client.get(f"{self.NTRS_API_URL}/search", params=params)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for doc in data.get("results", []):
                    # Check if it looks like a standard
                    title = doc.get("title", "")
                    if any(x in title for x in ["STD", "Standard", "specification"]):
                        std_type = self._detect_standard_type(title)
                        
                        results.append({
                            "type": std_type,
                            "number": self._extract_number(title),
                            "title": title,
                            "year": doc.get("publicationDate", "")[:4] if doc.get("publicationDate") else None,
                            "center": doc.get("center", {}).get("code") if isinstance(doc.get("center"), dict) else None,
                            "abstract": doc.get("abstract", "")[:200] + "..." if doc.get("abstract") else "",
                            "url": f"https://ntrs.nasa.gov/citations/{doc.get('id')}"
                        })
                
                return results
            
            return []
            
        except Exception as e:
            logger.error(f"NASA search error: {e}")
            return []
    
    def _detect_standard_type(self, title: str) -> str:
        """Detect NASA standard type"""
        if "NASA-STD" in title:
            return "NASA-STD"
        elif "MSFC-STD" in title:
            return "MSFC-STD"
        elif "GSFC-STD" in title:
            return "GSFC-STD"
        elif "JSC-STD" in title:
            return "JSC-STD"
        elif "NESC" in title:
            return "NESC"
        else:
            return "NASA-OTHER"
    
    def _extract_number(self, title: str) -> str:
        """Extract standard number from title"""
        import re
        match = re.search(r'(STD|TM|TP|CR)-(\d+[A-Z]?)', title)
        if match:
            return match.group(2)
        return "unknown"
    
    async def list_available(
        self,
        standard_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List NASA standards series"""
        series = [
            {
                "type": "NASA-STD",
                "name": "NASA Technical Standards",
                "description": "Agency-wide technical standards",
                "examples": ["NASA-STD-5005 (Structural)", "NASA-STD-5018 (Fasteners)"],
                "url": "https://standards.nasa.gov/nasa-technical-standards",
                "access": "PDF downloads (public domain)"
            },
            {
                "type": "MSFC-STD",
                "name": "Marshall Space Flight Center Standards",
                "description": "MSFC-specific standards",
                "examples": ["MSFC-STD-486 (Fastener Torque)"],
                "url": "https://standards.nasa.gov/center-standards",
                "access": "PDF downloads (public domain)"
            },
            {
                "type": "GSFC-STD",
                "name": "Goddard Space Flight Center Standards",
                "description": "GSFC-specific standards",
                "examples": ["GSFC-STD-7000 (Systems Engineering)"],
                "url": "https://standards.nasa.gov/center-standards",
                "access": "PDF downloads (public domain)"
            },
            {
                "type": "NESC",
                "name": "NASA Engineering and Safety Center",
                "description": "Technical assessments",
                "examples": ["NESC-RP-19-01470 (Battery Safety)"],
                "url": "https://www.nasa.gov/nesc",
                "access": "PDF downloads (public domain)"
            }
        ]
        
        if standard_type:
            return [s for s in series if s["type"].lower() == standard_type.lower()]
        return series
