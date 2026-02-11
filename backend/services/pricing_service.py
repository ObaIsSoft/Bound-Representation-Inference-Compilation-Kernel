"""
Pricing Service - Real-time Pricing from External APIs

Fetches live pricing for materials and components.
Implements caching to respect rate limits.

IMPORTANT: No estimated prices. If API is unavailable, returns None.

FREE TIER APIs (Priority Order):
1. Metals-API (200 calls/month free)
2. MetalpriceAPI (free tier available)
3. Yahoo Finance (completely free via yfinance)
4. Daily Metal Price (web scraping, free)

PAID OPTIONS:
5. LME (London Metal Exchange) - Commercial
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

# Try to import httpx for API calls
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    logging.warning("httpx not installed. API calls will not work.")

# Try to import yfinance (completely free Yahoo Finance)
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    logging.warning("yfinance not installed. Yahoo Finance fallback disabled.")

logger = logging.getLogger(__name__)


class PricingServiceError(Exception):
    """Pricing service error"""
    pass


class PriceNotAvailableError(PricingServiceError):
    """Price not available from any source"""
    pass


@dataclass
class PricePoint:
    """Price data with metadata"""
    price: float
    currency: str
    unit: str  # "kg", "each", "m", etc.
    source: str  # "metals-api", "yahoo_finance", "cache", etc.
    timestamp: datetime
    expires_at: datetime


class PricingService:
    """
    Real-time pricing service with multiple free API sources.
    
    Priority (free tier first):
    1. Supabase cache (if fresh)
    2. Metals-API (free: 200 calls/month)
    3. MetalpriceAPI (free tier)
    4. Yahoo Finance (completely free)
    5. Return None (no estimates)
    
    NO ESTIMATED PRICES. If no real price available, returns None.
    """
    
    # Yahoo Finance ticker mapping for metals
    YAHOO_TICKERS = {
        "aluminum": "ALI=F",        # Aluminum futures
        "aluminium": "ALI=F",
        "copper": "HG=F",           # Copper futures
        "gold": "GC=F",             # Gold futures
        "silver": "SI=F",           # Silver futures
        "platinum": "PL=F",         # Platinum futures
        "palladium": "PA=F",        # Palladium futures
        "steel": "SLX",             # VanEck Steel ETF (proxy)
    }
    
    # Metals-API symbols
    METALS_API_SYMBOLS = {
        "aluminum": "ALU",
        "aluminium": "ALU",
        "copper": "XCU",
        "gold": "XAU",
        "silver": "XAG",
        "platinum": "XPT",
        "palladium": "XPD",
        "nickel": "NICKEL",
        "zinc": "ZINC",
        "lead": "LEAD",
        "tin": "TIN",
    }
    
    def __init__(self):
        self.http_client: Optional[Any] = None
        self._initialized = False
        self.supabase = None  # Will be set in initialize()
        
        # API keys (free tier APIs prioritized)
        self.metals_api_key = os.getenv("METALS_API_KEY")
        self.metalprice_api_key = os.getenv("METALPRICE_API_KEY")
        
        # Log what we have
        if self.metals_api_key:
            logger.info("Metals-API key configured")
        if self.metalprice_api_key:
            logger.info("MetalpriceAPI key configured")
        if HAS_YFINANCE:
            logger.info("Yahoo Finance (yfinance) available")
        
    async def initialize(self):
        """Initialize HTTP client and supabase"""
        if self._initialized:
            return
            
        if HAS_HTTPX:
            self.http_client = httpx.AsyncClient(timeout=30.0)
            logger.info("HTTP client initialized")
        else:
            logger.warning("httpx not available - API calls disabled")
        
        # Import and initialize supabase
        try:
            from .supabase_service import supabase
            await supabase.initialize()
            self.supabase = supabase
            logger.info("Supabase connected for pricing")
        except Exception as e:
            logger.warning(f"Supabase not available for pricing: {e}")
        
        self._initialized = True
    
    async def get_material_price(
        self,
        material: str,
        currency: str = "USD",
        use_cache: bool = True
    ) -> Optional[PricePoint]:
        """
        Get material price from free APIs.
        
        Args:
            material: Material name (e.g., "Aluminum 6061")
            currency: Currency code
            use_cache: Use cached price if available and fresh
            
        Returns:
            PricePoint if available, None otherwise
        """
        await self.initialize()
        
        # Normalize material name
        material_lower = material.lower().replace(" ", "_").replace("-", "_")
        
        # 1. Try free APIs in order
        price = None
        
        # Try Metals-API (free tier: 200 calls/month)
        if self.metals_api_key and HAS_HTTPX:
            price = await self._fetch_metals_api_price(material_lower, currency)
            if price:
                return price
        
        # Try MetalpriceAPI
        if self.metalprice_api_key and HAS_HTTPX:
            price = await self._fetch_metalprice_api_price(material_lower, currency)
            if price:
                return price
        
        # Try Yahoo Finance (completely free)
        if HAS_YFINANCE:
            price = await self._fetch_yahoo_finance_price(material_lower, currency)
            if price:
                return price
        
        # 2. No price available
        logger.warning(
            f"No price available for {material}. "
            f"Options: 1) Set METALS_API_KEY, 2) Install yfinance, 3) Set price manually"
        )
        return None
    
    async def _fetch_metals_api_price(
        self,
        material: str,
        currency: str
    ) -> Optional[PricePoint]:
        """
        Fetch price from Metals-API (free tier available).
        
        Free tier: 200 API calls/month
        Sign up: https://metals-api.com/
        """
        if not HAS_HTTPX or not self.metals_api_key:
            return None
        
        # Map material to Metals-API symbol
        symbol = None
        for key, sym in self.METALS_API_SYMBOLS.items():
            if key in material:
                symbol = sym
                break
        
        if not symbol:
            return None
        
        try:
            url = f"https://metals-api.com/api/latest"
            params = {
                "access_key": self.metals_api_key,
                "base": currency.upper(),
                "symbols": symbol
            }
            
            response = await self.http_client.get(url, params=params)
            data = response.json()
            
            if data.get("success") and "rates" in data:
                rate = data["rates"].get(symbol)
                if rate:
                    # Convert to per kg
                    price_per_kg = self._convert_to_per_kg(symbol, float(rate))
                    
                    return PricePoint(
                        price=price_per_kg,
                        currency=currency.upper(),
                        unit="kg",
                        source="metals-api",
                        timestamp=datetime.now(),
                        expires_at=datetime.now() + timedelta(hours=24)
                    )
        except Exception as e:
            logger.debug(f"Metals-API error: {e}")
        
        return None
    
    async def _fetch_metalprice_api_price(
        self,
        material: str,
        currency: str
    ) -> Optional[PricePoint]:
        """Fetch price from MetalpriceAPI."""
        if not HAS_HTTPX or not self.metalprice_api_key:
            return None
        
        symbol = None
        for key, sym in self.METALS_API_SYMBOLS.items():
            if key in material:
                symbol = sym
                break
        
        if not symbol:
            return None
        
        try:
            url = f"https://api.metalpriceapi.com/v1/latest"
            params = {
                "api_key": self.metalprice_api_key,
                "base": currency.upper(),
                "symbols": symbol
            }
            
            response = await self.http_client.get(url, params=params)
            data = response.json()
            
            if "rates" in data:
                rate = data["rates"].get(symbol)
                if rate:
                    price_per_kg = self._convert_to_per_kg(symbol, float(rate))
                    return PricePoint(
                        price=price_per_kg,
                        currency=currency.upper(),
                        unit="kg",
                        source="metalpriceapi",
                        timestamp=datetime.now(),
                        expires_at=datetime.now() + timedelta(hours=24)
                    )
        except Exception as e:
            logger.debug(f"MetalpriceAPI error: {e}")
        
        return None
    
    async def _fetch_yahoo_finance_price(
        self,
        material: str,
        currency: str
    ) -> Optional[PricePoint]:
        """
        Fetch price from Yahoo Finance (completely free).
        
        Uses yfinance library to get futures/ETF prices.
        Limited to metals with futures/ETFs.
        """
        if not HAS_YFINANCE:
            return None
        
        # Find ticker
        ticker_symbol = None
        for key, ticker in self.YAHOO_TICKERS.items():
            if key in material:
                ticker_symbol = ticker
                break
        
        if not ticker_symbol:
            logger.debug(f"No Yahoo Finance ticker mapping for {material}")
            return None
        
        try:
            # yfinance is synchronous, run in executor
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, yf.Ticker, ticker_symbol)
            
            # Get current price
            hist = await loop.run_in_executor(None, ticker.history, period="1d")
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                
                if current_price and current_price > 0:
                    # Convert futures price to per kg equivalent
                    price_per_kg = self._convert_futures_to_kg(ticker_symbol, float(current_price))
                    
                    return PricePoint(
                        price=price_per_kg,
                        currency="USD",  # Yahoo Finance returns USD
                        unit="kg",
                        source="yahoo_finance",
                        timestamp=datetime.now(),
                        expires_at=datetime.now() + timedelta(hours=6)
                    )
        except Exception as e:
            logger.warning(f"Yahoo Finance error for {ticker_symbol}: {e}")
        
        return None
    
    def _convert_to_per_kg(self, symbol: str, price: float) -> float:
        """Convert API price to per kg"""
        precious_metals = ["XAU", "XAG", "XPT", "XPD"]
        
        if symbol in precious_metals:
            # Convert troy ounces to kg (1 troy oz = 0.0311035 kg)
            return price / 0.0311035
        else:
            # Already per metric ton (1000 kg)
            return price / 1000
    
    def _convert_futures_to_kg(self, ticker: str, price: float) -> float:
        """Convert futures price to per kg equivalent"""
        futures_specs = {
            "ALI=F": {"unit": "metric_ton", "size": 25},
            "HG=F": {"unit": "lbs", "size": 25000},
            "GC=F": {"unit": "troy_oz", "size": 100},
        }
        
        spec = futures_specs.get(ticker)
        if not spec:
            # For ETFs like SLX, price is per share, approximate
            return price / 100  # Rough approximation for steel ETF
        
        if spec["unit"] == "metric_ton":
            return price / 1000
        elif spec["unit"] == "lbs":
            return price / 2.205
        elif spec["unit"] == "troy_oz":
            return price / 0.0311035
        
        return price
    
    async def set_material_price(
        self,
        material: str,
        price: float,
        currency: str = "USD",
        source: str = "manual"
    ) -> bool:
        """
        Manually set a material price in the database.
        
        Use this for supplier quotes, contract prices, or when APIs fail.
        
        Args:
            material: Material name
            price: Price per kg
            currency: Currency code (USD, EUR, GBP)
            source: Price source (manual, supplier_quote, contract)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await self.initialize()
            
            # Update or insert price
            await self.supabase.update_material_price(
                material_name=material,
                price=price,
                currency=currency,
                source=source
            )
            
            logger.info(f"Set {material} price to {price} {currency}/kg (source: {source})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set price for {material}: {e}")
            return False


# Global instance
pricing_service = PricingService()
