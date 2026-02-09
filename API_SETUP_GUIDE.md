# Cost Estimation API Setup Guide

## ✅ Completed Changes

### 1. Backend Requirements Updated

**File:** `backend/requirements.txt`

Added dependencies:
```
# Database & API Client
supabase>=2.0.0
httpx>=0.24.0
python-dotenv>=1.0.0

# Free Pricing APIs
yfinance>=0.2.0

# Web scraping (optional)
beautifulsoup4>=4.12.0
lxml>=4.9.0
```

### 2. Pricing Service Updated

**File:** `backend/services/pricing_service.py`

Features:
- ✅ **Metals-API** support (200 free calls/month)
- ✅ **MetalpriceAPI** support (free tier)
- ✅ **Yahoo Finance** support (completely free via yfinance)
- ✅ Proper error handling (no fake prices)
- ✅ Automatic fallback between APIs

### 3. CostAgent Migrated

**File:** `backend/agents/cost_agent.py`

Changes:
- Uses `pricing_service.get_material_price()` instead of hardcoded dict
- Returns error with helpful message when prices unavailable
- Supports currency conversion via `currency_service`
- Falls back to database cache when APIs unavailable

### 4. API Endpoints Added

**File:** `backend/main.py`

New endpoints:

#### POST `/api/cost/estimate`
Request:
```json
{
  "mass_kg": 5.0,
  "material_name": "Aluminum 6061-T6",
  "complexity": "moderate",
  "currency": "USD",
  "budget_threshold": 10000
}
```

Response (success):
```json
{
  "success": true,
  "estimate": {
    "estimated_cost": 350.00,
    "currency": "USD",
    "confidence": 0.9,
    "feasible": true,
    "data_sources": {
      "material_price": "yahoo_finance"
    }
  }
}
```

Response (no price available):
```json
{
  "success": false,
  "error": "No price available for Aluminum 6061-T6",
  "solution": "Configure METALS_API_KEY or set price manually...",
  "setup_guide": {
    "option_1": "Set METALS_API_KEY (free tier: https://metals-api.com/)",
    "option_2": "Install yfinance: pip install yfinance (completely free)",
    "option_3": "POST to /api/pricing/set-price to set manual prices"
  }
}
```

#### POST `/api/pricing/set-price`
Request:
```json
{
  "material": "Aluminum 6061-T6",
  "price": 3.50,
  "currency": "USD",
  "source": "supplier_quote"
}
```

#### GET `/api/pricing/check`
Returns status of pricing APIs:
```json
{
  "apis": {
    "metals_api": {"configured": true, "status": "ready"},
    "yahoo_finance": {"installed": true, "working": true, "status": "ready"}
  }
}
```

---

## 🚀 Setup Instructions

### Step 1: Install Dependencies

```bash
cd /Users/obafemi/Documents/dev/brick/backend
pip install -r requirements.txt
```

Or just the essentials:
```bash
pip install yfinance httpx supabase python-dotenv
```

### Step 2: Configure Environment

Edit `backend/.env`:

```bash
# Option 1: Metals-API (recommended, 200 free calls/month)
# Sign up at https://metals-api.com/
METALS_API_KEY=your_key_here

# Option 2: Yahoo Finance (completely free, no key needed!)
# Already works if yfinance is installed

# Option 3: Manual pricing (no APIs)
# Don't set any keys - use /api/pricing/set-price endpoint
```

### Step 3: Test the API

Start the server:
```bash
cd /Users/obafemi/Documents/dev/brick/backend
python main.py
```

Test in browser/curl:
```bash
# Check pricing API status
curl http://localhost:8000/api/pricing/check

# Get cost estimate
curl -X POST http://localhost:8000/api/cost/estimate \
  -H "Content-Type: application/json" \
  -d '{"mass_kg": 5, "material_name": "Aluminum 6061-T6"}'

# Set manual price (if no APIs)
curl -X POST http://localhost:8000/api/pricing/set-price \
  -H "Content-Type: application/json" \
  -d '{"material": "Aluminum 6061-T6", "price": 3.50, "currency": "USD"}'
```

---

## 📊 API Priority Order

When you request a price, the service tries these in order:

1. **Metals-API** (if `METALS_API_KEY` set)
2. **MetalpriceAPI** (if `METALPRICE_API_KEY` set)
3. **Yahoo Finance** (always tried, completely free)
4. **Return None** (fail fast - no fake data)

---

## 🆓 Free Options Summary

| Option | Cost | Setup | Best For |
|--------|------|-------|----------|
| **Yahoo Finance** | FREE | `pip install yfinance` | Most users |
| **Metals-API** | 200 calls/mo | Sign up + API key | Higher volume |
| **Manual Entry** | FREE | POST to /api/pricing/set-price | Custom suppliers |

---

## 🔗 Frontend Integration

Example JavaScript/TypeScript:

```typescript
// Check pricing status
const status = await fetch('/api/pricing/check').then(r => r.json());

// Get cost estimate
const estimate = await fetch('/api/cost/estimate', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    mass_kg: 5.0,
    material_name: 'Aluminum 6061-T6',
    complexity: 'moderate',
    currency: 'USD'
  })
}).then(r => r.json());

if (estimate.success) {
  console.log(`Cost: $${estimate.estimate.estimated_cost}`);
} else {
  console.error(estimate.error);
  console.log(estimate.setup_guide);
}
```

---

## ✅ Verification Checklist

- [ ] `pip install yfinance httpx supabase python-dotenv`
- [ ] Backend server starts without errors
- [ ] `GET /api/pricing/check` returns API status
- [ ] `POST /api/cost/estimate` returns price or helpful error
- [ ] (Optional) Set `METALS_API_KEY` for higher rate limits
