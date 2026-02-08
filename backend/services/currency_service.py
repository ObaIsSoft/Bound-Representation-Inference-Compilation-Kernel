"""
Currency Service

Provides real-time currency conversion.
Falls back to stored rates if API unavailable.

Supported APIs:
- OpenExchangeRates (free tier available)
- CurrencyLayer (free tier available)
- ExchangeRate-API (free tier available)
"""

import os
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

logger = logging.getLogger(__name__)


@dataclass
class ExchangeRate:
    """Exchange rate data"""
    base_currency: str
    target_currency: str
    rate: float
    timestamp: datetime
    source: str


class CurrencyService:
    """
    Currency conversion service.
    
    Priority:
    1. Real-time API (if available and fresh)
    2. Cached rates from database
    3. Return None (fail fast - no hardcoded rates!)
    """
    
    # Supported currencies
    SUPPORTED_CURRENCIES = {
        "USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY",
        "SEK", "NZD", "MXN", "SGD", "HKD", "NOK", "KRW", "INR"
    }
    
    def __init__(self):
        self.http_client: Optional[Any] = None
        self._initialized = False
        
        # API keys
        self.openexchange_app_id = os.getenv("OPENEXCHANGERATES_APP_ID")
        self.currencylayer_key = os.getenv("CURRENCYLAYER_API_KEY")
        self.exchangerate_key = os.getenv("EXCHANGERATE_API_KEY")
        
        # Cache duration
        self.cache_duration = timedelta(hours=1)
        
    async def initialize(self):
        """Initialize HTTP client"""
        if self._initialized:
            return
            
        if HAS_HTTPX:
            self.http_client = httpx.AsyncClient(timeout=30.0)
        
        self._initialized = True
    
    async def convert(
        self,
        amount: float,
        from_currency: str,
        to_currency: str
    ) -> Optional[float]:
        """
        Convert amount between currencies.
        
        Args:
            amount: Amount to convert
            from_currency: Source currency code (USD, EUR, etc.)
            to_currency: Target currency code
            
        Returns:
            Converted amount or None if rate unavailable
        """
        await self.initialize()
        
        # Normalize currency codes
        from_currency = from_currency.upper()
        to_currency = to_currency.upper()
        
        # Same currency
        if from_currency == to_currency:
            return amount
        
        # Validate currencies
        if from_currency not in self.SUPPORTED_CURRENCIES:
            logger.warning(f"Unsupported currency: {from_currency}")
            return None
        if to_currency not in self.SUPPORTED_CURRENCIES:
            logger.warning(f"Unsupported currency: {to_currency}")
            return None
        
        # Get exchange rate
        rate = await self.get_rate(from_currency, to_currency)
        
        if rate is None:
            return None
        
        return amount * rate
    
    async def get_rate(
        self,
        from_currency: str,
        to_currency: str
    ) -> Optional[float]:
        """
        Get exchange rate between two currencies.
        
        Args:
            from_currency: Source currency code
            to_currency: Target currency code
            
        Returns:
            Exchange rate or None
        """
        await self.initialize()
        
        from_currency = from_currency.upper()
        to_currency = to_currency.upper()
        
        if from_currency == to_currency:
            return 1.0
        
        # Try real-time APIs in order of preference
        rate = await self._get_rate_from_apis(from_currency, to_currency)
        
        if rate:
            return rate
        
        # Try database cache
        rate = await self._get_rate_from_cache(from_currency, to_currency)
        
        if rate:
            return rate
        
        logger.warning(f"No exchange rate available for {from_currency}/{to_currency}")
        return None
    
    async def get_all_rates(self, base_currency: str = "USD") -> Optional[Dict[str, float]]:
        """
        Get all exchange rates for a base currency.
        
        Args:
            base_currency: Base currency code
            
        Returns:
            Dictionary of currency -> rate or None
        """
        await self.initialize()
        
        base_currency = base_currency.upper()
        
        # Try ExchangeRate-API (free tier supports this)
        if self.exchangerate_key and HAS_HTTPX:
            try:
                url = f"https://v6.exchangerate-api.com/v6/{self.exchangerate_key}/latest/{base_currency}"
                
                response = await self.http_client.get(url)
                data = response.json()
                
                if data.get("result") == "success":
                    rates = data.get("conversion_rates", {})
                    # Store in cache
                    await self._cache_rates(base_currency, rates)
                    return rates
                    
            except Exception as e:
                logger.error(f"ExchangeRate-API error: {e}")
        
        # Try OpenExchangeRates
        if self.openexchange_app_id and HAS_HTTPX:
            try:
                url = f"https://openexchangerates.org/api/latest.json?app_id={self.openexchange_app_id}&base={base_currency}"
                
                response = await self.http_client.get(url)
                data = response.json()
                
                if "rates" in data:
                    rates = data["rates"]
                    await self._cache_rates(base_currency, rates)
                    return rates
                    
            except Exception as e:
                logger.error(f"OpenExchangeRates error: {e}")
        
        # Try database cache
        return await self._get_all_rates_from_cache(base_currency)
    
    async def _get_rate_from_apis(
        self,
        from_currency: str,
        to_currency: str
    ) -> Optional[float]:
        """Get rate from real-time APIs"""
        
        # Try ExchangeRate-API first (free tier available)
        if self.exchangerate_key and HAS_HTTPX:
            try:
                url = f"https://v6.exchangerate-api.com/v6/{self.exchangerate_key}/pair/{from_currency}/{to_currency}"
                
                response = await self.http_client.get(url)
                data = response.json()
                
                if data.get("result") == "success":
                    rate = data.get("conversion_rate")
                    if rate:
                        await self._cache_rate(from_currency, to_currency, rate, "exchangerate-api")
                        return rate
                        
            except Exception as e:
                logger.debug(f"ExchangeRate-API error: {e}")
        
        # Try CurrencyLayer
        if self.currencylayer_key and HAS_HTTPX:
            try:
                url = f"https://api.currencylayer.com/live?access_key={self.currencylayer_key}&source={from_currency}&currencies={to_currency}"
                
                response = await self.http_client.get(url)
                data = response.json()
                
                if data.get("success"):
                    rates = data.get("quotes", {})
                    key = f"{from_currency}{to_currency}"
                    if key in rates:
                        rate = rates[key]
                        await self._cache_rate(from_currency, to_currency, rate, "currencylayer")
                        return rate
                        
            except Exception as e:
                logger.debug(f"CurrencyLayer error: {e}")
        
        return None
    
    async def _get_rate_from_cache(
        self,
        from_currency: str,
        to_currency: str
    ) -> Optional[float]:
        """Get rate from database cache"""
        try:
            # Import here to avoid circular dependency
            from .supabase_service import supabase
            
            # Check cache
            result = await supabase.client.table("pricing_cache")\
                .select("*")\
                .eq("category", "currency_rate")\
                .eq("item_key", f"{from_currency}_{to_currency}")\
                .execute()
            
            if result.data:
                cached = result.data[0]
                expires_at = datetime.fromisoformat(cached["expires_at"].replace('Z', '+00:00'))
                
                if datetime.now() < expires_at:
                    price_data = cached.get("price_data", {})
                    return price_data.get("rate")
                
        except Exception as e:
            logger.debug(f"Cache read error: {e}")
        
        return None
    
    async def _get_all_rates_from_cache(
        self,
        base_currency: str
    ) -> Optional[Dict[str, float]]:
        """Get all rates for base currency from cache"""
        try:
            from .supabase_service import supabase
            
            result = await supabase.client.table("pricing_cache")\
                .select("*")\
                .eq("category", "currency_rate")\
                .like("item_key", f"{base_currency}_%")\
                .execute()
            
            rates = {}
            for row in result.data:
                expires_at = datetime.fromisoformat(row["expires_at"].replace('Z', '+00:00'))
                if datetime.now() < expires_at:
                    # Extract target currency from key (USD_EUR -> EUR)
                    target = row["item_key"].split("_")[1]
                    price_data = row.get("price_data", {})
                    rates[target] = price_data.get("rate")
            
            return rates if rates else None
            
        except Exception as e:
            logger.debug(f"Cache read error: {e}")
        
        return None
    
    async def _cache_rate(
        self,
        from_currency: str,
        to_currency: str,
        rate: float,
        source: str
    ):
        """Cache a single rate"""
        try:
            from .supabase_service import supabase
            
            now = datetime.now()
            expires = now + self.cache_duration
            
            await supabase.client.table("pricing_cache")\
                .upsert({
                    "category": "currency_rate",
                    "item_key": f"{from_currency}_{to_currency}",
                    "price_data": {"rate": rate, "source": source},
                    "currency": from_currency,
                    "source": source,
                    "cached_at": now.isoformat(),
                    "expires_at": expires.isoformat()
                }, on_conflict="category,item_key,currency")\
                .execute()
                
        except Exception as e:
            logger.debug(f"Cache write error: {e}")
    
    async def _cache_rates(self, base_currency: str, rates: Dict[str, float]):
        """Cache multiple rates"""
        try:
            from .supabase_service import supabase
            
            now = datetime.now()
            expires = now + self.cache_duration
            
            for target, rate in rates.items():
                if target == base_currency:
                    continue
                    
                await supabase.client.table("pricing_cache")\
                    .upsert({
                        "category": "currency_rate",
                        "item_key": f"{base_currency}_{target}",
                        "price_data": {"rate": rate},
                        "currency": base_currency,
                        "source": "api",
                        "cached_at": now.isoformat(),
                        "expires_at": expires.isoformat()
                    }, on_conflict="category,item_key,currency")\
                    .execute()
                    
        except Exception as e:
            logger.debug(f"Bulk cache write error: {e}")


# Global instance
currency_service = CurrencyService()
