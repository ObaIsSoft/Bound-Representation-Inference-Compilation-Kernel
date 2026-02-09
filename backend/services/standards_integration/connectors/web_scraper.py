"""
Standards Web Scraper

Scrapes publicly available standards from websites:
- ANSI Webstore (public listings)
- ISO.org (public listings)
- ASTM.org (public listings)
- Engineering Toolbox (reference data)
- Electronics Tutorials (for EE standards)

Note: This scrapes METADATA only. Full standards require purchase.
For full standards, use official APIs or purchase PDFs.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

from ..standards_fetcher import StandardsFetcher, FetchResult

logger = logging.getLogger(__name__)


class StandardsWebScraper(StandardsFetcher):
    """
    Web scraper for publicly available standards metadata.
    
    This collects:
    - Standard titles and descriptions
    - Abstracts and scope
    - Revision history
    - Purchase links
    
    Does NOT scrape full standard content (copyright protected).
    """
    
    SOURCES = {
        "iso": {
            "name": "ISO Standards",
            "url": "https://www.iso.org/standard",
            "search_url": "https://www.iso.org/search.html"
        },
        "astm": {
            "name": "ASTM Standards",
            "url": "https://www.astm.org/standards",
            "search_url": "https://www.astm.org/search/fullsite-search.html"
        },
        "ansi": {
            "name": "ANSI Standards",
            "url": "https://webstore.ansi.org",
            "search_url": "https://webstore.ansi.org/find-standard"
        },
        "engineering_toolbox": {
            "name": "Engineering Toolbox",
            "url": "https://www.engineeringtoolbox.com",
            "base": "Reference data and equations"
        }
    }
    
    def __init__(self):
        super().__init__("WebScraper")
        self.client = None
        
    async def _get_client(self):
        if self.client is None and HAS_HTTPX:
            self.client = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; BRICK-OS/1.0; Standards Research)"
                }
            )
        return self.client
    
    async def fetch_standard(
        self,
        standard_type: str,
        standard_number: str,
        revision: Optional[str] = None
    ) -> FetchResult:
        """
        Scrape standard metadata from web.
        
        Args:
            standard_type: iso, astm, ansi, etc.
            standard_number: Standard number
            revision: Year/revision
            
        Returns:
            FetchResult with public metadata
        """
        cached = self._get_cached(standard_type, standard_number, revision)
        if cached:
            return cached
        
        if not HAS_HTTPX or not HAS_BS4:
            return FetchResult(
                success=False,
                standard_type=standard_type,
                standard_number=standard_number,
                revision=revision,
                data={},
                source="WebScraper",
                fetched_at=datetime.now(),
                errors=["Missing dependencies. Run: pip install httpx beautifulsoup4"]
            )
        
        # Route to appropriate scraper
        if standard_type.lower() in ["iso", "iec"]:
            result = await self._scrape_iso(standard_number, revision)
        elif standard_type.lower() == "astm":
            result = await self._scrape_astm(standard_number, revision)
        elif standard_type.lower() == "ansi":
            result = await self._scrape_ansi(standard_number, revision)
        else:
            result = FetchResult(
                success=False,
                standard_type=standard_type,
                standard_number=standard_number,
                revision=revision,
                data={},
                source="WebScraper",
                fetched_at=datetime.now(),
                errors=[f"Scraper not implemented for {standard_type}"]
            )
        
        self._set_cached(result)
        return result
    
    async def _scrape_iso(
        self,
        number: str,
        revision: Optional[str] = None
    ) -> FetchResult:
        """Scrape ISO standard metadata"""
        try:
            client = await self._get_client()
            
            # ISO format: ISO 286-1:2010
            if revision:
                iso_id = f"{number}:{revision}"
            else:
                iso_id = number
            
            # Search ISO website
            search_url = f"https://www.iso.org/search.html"
            params = {"q": iso_id, "type": "standard"}
            
            response = await client.get(search_url, params=params)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract metadata from search results
                # ISO site structure changes, so be defensive
                data = {
                    "standard_id": iso_id,
                    "organization": "ISO",
                    "url": f"https://www.iso.org/standard/{number}.html",
                    "purchase_url": f"https://www.iso.org/obp/ui/#iso:std:iso:{number}",
                    "publicly_available": False,  # ISO standards require purchase
                    "abstract_available": True
                }
                
                # Try to find title
                title_tag = soup.find('h2') or soup.find('h1')
                if title_tag:
                    data["title"] = title_tag.get_text(strip=True)
                
                # Try to find abstract
                abstract_div = soup.find('div', class_='abstract') or soup.find('div', {'id': 'abstract'})
                if abstract_div:
                    data["abstract"] = abstract_div.get_text(strip=True)[:500]
                
                return FetchResult(
                    success=True,
                    standard_type="iso",
                    standard_number=number,
                    revision=revision,
                    data=data,
                    source="ISO.org",
                    fetched_at=datetime.now(),
                    url=data.get("url")
                )
            else:
                return FetchResult(
                    success=False,
                    standard_type="iso",
                    standard_number=number,
                    revision=revision,
                    data={},
                    source="ISO.org",
                    fetched_at=datetime.now(),
                    errors=[f"HTTP {response.status_code}"]
                )
                
        except Exception as e:
            logger.error(f"ISO scrape error: {e}")
            return FetchResult(
                success=False,
                standard_type="iso",
                standard_number=number,
                revision=revision,
                data={},
                source="ISO.org",
                fetched_at=datetime.now(),
                errors=[str(e)]
            )
    
    async def _scrape_astm(
        self,
        number: str,
        revision: Optional[str] = None
    ) -> FetchResult:
        """Scrape ASTM standard metadata"""
        try:
            client = await self._get_client()
            
            # ASTM format: A36/A36M-19
            astm_id = f"{number}-{revision}" if revision else number
            
            url = f"https://www.astm.org/standards/{astm_id}.htm"
            
            response = await client.get(url)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                data = {
                    "standard_id": astm_id,
                    "organization": "ASTM",
                    "url": url,
                    "purchase_url": url,
                    "publicly_available": False
                }
                
                # Try to extract title and scope
                title = soup.find('h1') or soup.find('h2')
                if title:
                    data["title"] = title.get_text(strip=True)
                
                # Look for scope section
                scope = soup.find('div', text=lambda t: t and 'scope' in t.lower())
                if scope:
                    data["scope"] = scope.get_text(strip=True)[:500]
                
                return FetchResult(
                    success=True,
                    standard_type="astm",
                    standard_number=number,
                    revision=revision,
                    data=data,
                    source="ASTM.org",
                    fetched_at=datetime.now(),
                    url=url
                )
            else:
                return FetchResult(
                    success=False,
                    standard_type="astm",
                    standard_number=number,
                    revision=revision,
                    data={},
                    source="ASTM.org",
                    fetched_at=datetime.now(),
                    errors=[f"HTTP {response.status_code}"]
                )
                
        except Exception as e:
            logger.error(f"ASTM scrape error: {e}")
            return FetchResult(
                success=False,
                standard_type="astm",
                standard_number=number,
                revision=revision,
                data={},
                source="ASTM.org",
                fetched_at=datetime.now(),
                errors=[str(e)]
            )
    
    async def _scrape_ansi(
        self,
        number: str,
        revision: Optional[str] = None
    ) -> FetchResult:
        """Scrape ANSI standard metadata from webstore"""
        # ANSI webstore requires JavaScript for search
        # Return minimal metadata
        return FetchResult(
            success=True,
            standard_type="ansi",
            standard_number=number,
            revision=revision,
            data={
                "standard_id": f"ANSI {number}",
                "organization": "ANSI",
                "url": f"https://webstore.ansi.org/standards/find-standard?query={number}",
                "purchase_url": f"https://webstore.ansi.org",
                "note": "Full metadata requires webstore API or purchase"
            },
            source="ANSI",
            fetched_at=datetime.now(),
            url=f"https://webstore.ansi.org"
        )
    
    async def search_standards(
        self,
        query: str,
        standard_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for standards across web sources"""
        results = []
        
        if not HAS_HTTPX or not HAS_BS4:
            return results
        
        # Search ISO
        if not standard_type or standard_type.lower() == "iso":
            try:
                client = await self._get_client()
                search_url = "https://www.iso.org/search.html"
                response = await client.get(search_url, params={"q": query})
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    # Parse results (defensive parsing)
                    for item in soup.find_all('div', class_='item')[:limit]:
                        title = item.find('h2') or item.find('h3')
                        if title:
                            results.append({
                                "type": "ISO",
                                "title": title.get_text(strip=True),
                                "url": item.find('a', href=True)['href'] if item.find('a') else None,
                                "source": "ISO.org"
                            })
            except Exception as e:
                logger.debug(f"ISO search error: {e}")
        
        return results[:limit]
    
    async def list_available(
        self,
        standard_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List scrapable sources"""
        sources = [
            {
                "type": "ISO",
                "name": "International Organization for Standardization",
                "description": "International standards for industry and business",
                "scrapable": "metadata only",
                "url": "https://www.iso.org",
                "purchase_required": True
            },
            {
                "type": "ASTM",
                "name": "ASTM International",
                "description": "Standards for materials, products, systems, and services",
                "scrapable": "metadata only",
                "url": "https://www.astm.org",
                "purchase_required": True
            },
            {
                "type": "ANSI",
                "name": "American National Standards Institute",
                "description": "U.S. standards and conformity assessment",
                "scrapable": "listings only",
                "url": "https://webstore.ansi.org",
                "purchase_required": True
            },
            {
                "type": "IEC",
                "name": "International Electrotechnical Commission",
                "description": "International standards for electrical and electronic technologies",
                "scrapable": "metadata only",
                "url": "https://www.iec.ch",
                "purchase_required": True
            }
        ]
        
        if standard_type:
            return [s for s in sources if s["type"].lower() == standard_type.lower()]
        return sources
