# BRICK OS - Session Context

## Current Implementation Status

**Phase**: Week 1 - Safety Critical Agents Migration  
**Status**: ✅ Service Layer Foundation COMPLETE  
**Next**: Apply SQL Migrations → Migrate Agents

---

## ✅ Completed This Session (2026-02-08)

### 1. Service Layer (6 Services, ~2,300 lines)

| Service | File | Lines | Purpose |
|---------|------|-------|---------|
| SupabaseService | `supabase_service.py` | ~300 | Centralized DB client |
| PricingService | `pricing_service.py` | ~380 | LME/DigiKey/Climatiq integration |
| StandardsService | `standards_service.py` | ~180 | ISO/AWG/ASME standards |
| ComponentCatalogService | `component_catalog_service.py` | ~380 | Nexar/Mouser/Octopart |
| AssetSourcingService | `asset_sourcing_service.py` | ~390 | NASA 3D/Sketchfab/CGTrader |
| CurrencyService | `currency_service.py` | ~350 | Exchange rates |

### 2. Database Schema (4 SQL Files, ~2,100 lines)

| Schema | Tables | Records | Purpose |
|--------|--------|---------|---------|
| `001_critic_thresholds.sql` | critic_thresholds | 7 | Vehicle configs |
| `002_manufacturing_rates.sql` | manufacturing_rates | 10 | Regional costs |
| `003_materials_extended.sql` | materials | 12 | Material properties |
| `004_standards_reference.sql` | standards_reference | 20+ | ISO/AWG/ASME |

### 3. Environment Variables (Added 40+ new keys)

**`.env` updated with:**
- **Pricing APIs**: LME, Fastmarkets, OpenExchangeRates, CurrencyLayer
- **Component APIs**: DigiKey, Mouser, Octopart
- **Asset APIs**: NASA 3D, Sketchfab, CGTrader, Thingiverse, GrabCAD
- **Sustainability**: Climatiq, Carbon Interface
- **Manufacturing**: Xometry, Protolabs, Hubs, Fictiv

### 4. Seed & Migration Scripts

- `seed_critic_thresholds.py` - Seed initial thresholds
- `apply_migrations.py` - Migration runner (with manual instructions)

### 5. Documentation

- `DATA_SOURCES.md` - Complete data source reference
- Updated `task.md` with progress
- Updated `SESSION_CONTEXT.md` (this file)

---

## 📁 Files Created/Modified

```
backend/
├── services/
│   ├── __init__.py                    [UPDATED] Export all 6 services
│   ├── supabase_service.py            [EXISTING]
│   ├── pricing_service.py             [NEW]
│   ├── standards_service.py           [NEW]
│   ├── component_catalog_service.py   [NEW]
│   ├── asset_sourcing_service.py      [NEW]
│   └── currency_service.py            [NEW]
├── db/
│   ├── schema/
│   │   ├── 001_critic_thresholds.sql  [NEW]
│   │   ├── 002_manufacturing_rates.sql [NEW]
│   │   ├── 003_materials_extended.sql [NEW]
│   │   └── 004_standards_reference.sql [NEW]
│   ├── seeds/
│   │   └── seed_critic_thresholds.py  [NEW]
│   ├── apply_migrations.py            [NEW]
│   └── DATA_SOURCES.md                [NEW]
├── .env                               [UPDATED] 40+ new API keys
├── task.md                            [UPDATED] Progress tracking
└── SESSION_CONTEXT.md                 [UPDATED] This file
```

---

## 🔧 Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    AGENTS (ControlCritic, etc.)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   SERVICE LAYER                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │  Supabase   │ │  Pricing    │ │  Standards  │           │
│  │  Service    │ │  Service    │ │  Service    │           │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │  Component  │ │   Asset     │ │  Currency   │           │
│  │  Catalog    │ │  Sourcing   │ │  Service    │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────────────────────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │  Supabase   │   │ External    │   │   Cache     │
    │  PostgreSQL │   │   APIs      │   │   (Redis)   │
    └─────────────┘   └─────────────┘   └─────────────┘
```

---

## 🎯 Critical Path - Next Steps

### Step 1: Apply SQL Migrations (REQUIRED)

```bash
# Check connectivity
python backend/db/apply_migrations.py --check

# List pending migrations
python backend/db/apply_migrations.py --list

# Show instructions for manual application
python backend/db/apply_migrations.py
```

**Then manually apply via Supabase SQL Editor:**
1. Go to https://supabase.com/dashboard
2. Open SQL Editor
3. Copy/paste each file in order:
   - `001_critic_thresholds.sql`
   - `002_manufacturing_rates.sql`
   - `003_materials_extended.sql`
   - `004_standards_reference.sql`

### Step 2: Seed Initial Data

```bash
python backend/db/seeds/seed_critic_thresholds.py
```

### Step 3: Migrate ControlCritic (SAFETY CRITICAL! ⚠️)

**Current (DANGEROUS):**
```python
self.MAX_THRUST = 1000.0  # Hardcoded!
self.MAX_TORQUE = 100.0   # Hardcoded!
```

**Target:**
```python
from backend.services import supabase

async def load_vehicle_limits(self, vehicle_type: str):
    limits = await supabase.get_critic_thresholds(
        critic_name="ControlCritic",
        vehicle_type=vehicle_type
    )
    self.max_thrust = limits["max_thrust_n"]
    self.max_torque = limits["max_torque_nm"]
```

### Step 4: Migrate CostAgent

Replace hardcoded material costs with:
```python
from backend.services import pricing_service

price = await pricing_service.get_material_price("Aluminum 6061")
```

### Step 5: Migrate SafetyAgent

Replace hardcoded stress limits with:
```python
from backend.services import supabase

material = await supabase.get_material("Steel 4140")
yield_strength = material["yield_strength_mpa"]
```

---

## 🚨 Safety Critical Warning

**ControlCritic has HARDCODED limits that are DANGEROUS:**
- `MAX_THRUST = 1000N` (fixed for all vehicles!)
- `MAX_TORQUE = 100Nm` (fixed for all vehicles!)

A small drone with these limits could damage itself or cause injury.

**These MUST be migrated to database-driven values before production use!**

---

## 📊 Data Coverage

### ✅ In Database (Ready to Use)

| Data | Records | Source |
|------|---------|--------|
| Material properties | 12 | ASM Handbook |
| Manufacturing rates | 10 | Regional research |
| Critic thresholds | 7 | Safety analysis |
| ISO 286 fits | 4 | ISO standard |
| AWG ampacity | 8 | NEC/NASA |
| Safety factors | 5 | Industry standards |

### 🔄 API Integrated (Needs API Keys)

| Service | API | Free Tier |
|---------|-----|-----------|
| LME Metals | 🔄 | Contact LME |
| DigiKey | Nexar | ✅ Free |
| Currency | ExchangeRate-API | 1,500 req/mo |
| Climatiq | 🔄 | Free tier |

### ⚠️ Missing Data Sources

| Data | Needed For | Status |
|------|------------|--------|
| Plastic pricing | CostAgent | Not configured |
| McMaster-Carr | ComponentAgent | No API |
| PCB pricing | CostAgent | Not configured |
| LCA databases | Sustainability | Commercial |

---

## 📝 API Keys to Obtain (Optional)

The system works WITHOUT these - it just won't have real-time pricing.

**High Value:**
- `EXCHANGERATE_API_KEY` - Free tier, 1,500 requests/month
- `CLIMATIQ_API_KEY` - Carbon footprint calculations

**Medium Value:**
- `LME_API_KEY` - Real metal prices (commercial)
- `SKETCHFAB_API_KEY` - 3D model search

**Lower Priority:**
- `MOUSER_API_KEY` - Alternative component source
- `XOMETRY_API_KEY` - Instant manufacturing quotes

---

## 🔍 Verification Checklist

- [ ] SQL migrations applied to Supabase
- [ ] Seed data loaded
- [ ] ControlCritic migrated
- [ ] CostAgent migrated
- [ ] SafetyAgent migrated
- [ ] Tests updated
- [ ] Documentation updated
