# Free Pricing APIs for BRICK OS

This document lists free and open-source alternatives to paid pricing APIs.

---

## 🆓 Free Tier APIs (Recommended)

### 1. Metals-API (Best Free Option)

**Website:** https://metals-api.com/

**Free Tier:**
- 200 API calls/month
- Real-time rates
- 16 currency pairs
- Base metals + precious metals

**Supported Metals:**
- Aluminum (ALU)
- Copper (XCU)
- Gold (XAU)
- Silver (XAG)
- Platinum (XPT)
- Palladium (XPD)
- Nickel, Zinc, Lead, Tin

**Setup:**
```bash
# Sign up at metals-api.com
export METALS_API_KEY=your_api_key
```

**Pricing:**
- Free: 200 calls/month
- Paid: Starting at $9.99/month for 10,000 calls

---

### 2. MetalpriceAPI

**Website:** https://metalpriceapi.com/

**Free Tier:**
- Available (limits vary)
- Live and historical rates
- 150+ currencies

**Setup:**
```bash
# Sign up at metalpriceapi.com
export METALPRICE_API_KEY=your_api_key
```

---

### 3. Yahoo Finance (Completely Free)

**Library:** yfinance (Python)

**Cost:** FREE - No API key needed, unlimited requests

**Supported Metals:**
- Aluminum (ALI=F futures)
- Copper (HG=F futures)
- Gold (GC=F futures)
- Silver (SI=F futures)
- Platinum (PL=F futures)
- Palladium (PA=F futures)
- Steel (SLX ETF - proxy)

**Setup:**
```bash
pip install yfinance
```

**No environment variable needed!**

**Note:** Uses futures/ETF prices as proxies. Good for relative pricing, may need adjustment for spot prices.

---

### 4. Daily Metal Price (Web Scraping)

**Website:** https://www.dailymetalprice.com/

**Cost:** FREE - Web scraping (no API)

**Data:**
- 23 base metals
- Historical prices back to 2000
- Daily updates
- Charts and tables

**Note:** Not implemented yet. Can be added as fallback scraping source.

---

## 💰 Paid APIs (If You Need Higher Limits)

### London Metal Exchange (LME)

**Website:** https://www.lme.com/

**Cost:** Commercial licensing (expensive)

**Best for:** Professional trading, official settlement prices

---

## 🔧 Configuration Priority

The `pricing_service.py` tries APIs in this order:

1. **Supabase Cache** (if fresh < 24 hours)
2. **Metals-API** (if `METALS_API_KEY` set)
3. **MetalpriceAPI** (if `METALPRICE_API_KEY` set)
4. **Yahoo Finance** (always tried, completely free)
5. **Return None** (fail fast - no estimates)

---

## 📊 Comparison

| API | Cost | Limits | Metals | Key Required | Best For |
|-----|------|--------|--------|--------------|----------|
| Metals-API | Free | 200/mo | Base + Precious | Yes | Most users |
| MetalpriceAPI | Free tier | Varies | Base + Precious | Yes | Backup |
| Yahoo Finance | Free | Unlimited | Major metals | No | Development |
| LME | Paid | Commercial | All | Yes | Enterprise |

---

## 🚀 Quick Start

### Recommended Setup (Free)

1. **Sign up for Metals-API:**
   ```bash
   # Visit https://metals-api.com/ and create free account
   ```

2. **Set environment variable:**
   ```bash
   export METALS_API_KEY=your_api_key_here
   ```

3. **Install yfinance (optional backup):**
   ```bash
   pip install yfinance
   ```

4. **Test:**
   ```python
   import asyncio
   from backend.services import pricing_service
   
   async def test():
       price = await pricing_service.get_material_price("Aluminum 6061-T6")
       print(f"Price: ${price.price}/kg from {price.source}")
   
   asyncio.run(test())
   ```

---

## 📝 Environment Variables

Add to your `backend/.env`:

```bash
# Free tier (recommended)
METALS_API_KEY=your_metals_api_key_here
METALPRICE_API_KEY=your_metalprice_api_key_here

# Paid (optional)
LME_API_KEY=your_lme_key_here  # Only if you need official LME prices
```

---

## ⚠️ Important Notes

1. **Yahoo Finance** prices are futures/ETF based, not direct spot prices
2. **Metals-API free tier** resets monthly (200 calls)
3. **Always cache** - prices are cached for 24 hours to reduce API calls
4. **Manual fallback** - If all APIs fail, you can always set prices manually

---

## 🔄 Rate Limiting Strategy

The service implements:

- **Caching:** 24-hour cache for all prices
- **Failover:** Tries multiple free sources before giving up
- **Graceful degradation:** Returns None (not estimates) if all fail

Example usage pattern:
```python
# First call - hits API
price = await pricing_service.get_material_price("Aluminum")
# Price cached for 24 hours

# Second call - uses cache (no API hit)
price = await pricing_service.get_material_price("Aluminum")
```

With 200 calls/month from Metals-API + unlimited Yahoo Finance, you can support:
- 6 unique materials with daily updates
- Or 200 material checks per month (with cache hits)
