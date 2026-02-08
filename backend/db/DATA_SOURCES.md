# BRICK OS - Data Sources Reference

Complete reference for all data sources used in the BRICK OS backend.

---

## 📊 Database vs External APIs

### Legend
- ✅ **In Database** - Data seeded in Supabase
- 🔄 **API Integration** - Fetched from external API
- ⚠️ **Missing** - Needs data source or implementation

---

## 1. Materials Data

### Physical Properties
| Property | Source | Status | Notes |
|----------|--------|--------|-------|
| Density | ASM Handbook | ✅ In DB | `materials.density_kg_m3` |
| Yield Strength | ASM Handbook | ✅ In DB | `materials.yield_strength_mpa` |
| Ultimate Strength | ASM Handbook | ✅ In DB | `materials.ultimate_strength_mpa` |
| Elastic Modulus | ASM Handbook | ✅ In DB | `materials.elastic_modulus_gpa` |
| Thermal Conductivity | ASM Handbook | ✅ In DB | `materials.thermal_conductivity_w_mk` |
| Max Temperature | Material Datasheets | ✅ In DB | `materials.max_temp_c` |

### Pricing Data
| Data | Source | Status | API Key Required |
|------|--------|--------|------------------|
| Aluminum Prices | LME (London Metal Exchange) | 🔄 API | `LME_API_KEY` |
| Copper Prices | LME | 🔄 API | `LME_API_KEY` |
| Steel Prices | Fastmarkets | 🔄 API | `FASTMARKETS_API_KEY` |
| Plastic Prices | PlasticsNews/ICIS | ⚠️ Missing | Not configured |
| Titanium Prices | LME/Traders | 🔄 API | `LME_API_KEY` |

### Carbon Footprint
| Data | Source | Status | API Key Required |
|------|--------|--------|------------------|
| Material Carbon Factors | Database (ecoinvent data) | ✅ In DB | None |
| Real-time Carbon | Climatiq API | 🔄 API | `CLIMATIQ_API_KEY` |
| LCA Data | OpenLCA | ⚠️ Optional | Self-hosted |

### Materials Currently in Database (003_materials_extended.sql)

| Material | Density (kg/m³) | Yield (MPa) | Cost ($/kg) | Carbon (kg CO2/kg) |
|----------|-----------------|-------------|-------------|-------------------|
| Aluminum 6061-T6 | 2,700 | 276 | 3.50 | 12.7 |
| Aluminum 7075-T6 | 2,810 | 503 | 5.50 | 13.5 |
| Steel A36 | 7,850 | 250 | 0.80 | 1.9 |
| Steel 4140 | 7,850 | 655 | 1.50 | 2.1 |
| Stainless 304 | 8,000 | 215 | 4.00 | 2.8 |
| Titanium Ti-6Al-4V | 4,430 | 880 | 35.00 | 45.0 |
| PLA (3D Printing) | 1,250 | 60 | 3.00 | 3.4 |
| ABS (3D Printing) | 1,050 | 40 | 2.50 | 3.8 |
| Nylon 12 | 1,020 | 45 | 8.00 | 8.5 |
| PETG | 1,270 | 30 | 3.50 | 3.9 |
| Carbon Fiber | 1,600 | 1,500 | 45.00 | 55.0 |
| GFRP (Fiberglass) | 1,850 | 350 | 8.00 | 4.5 |

---

## 2. Manufacturing Rates

### Process Costs by Region
| Process | Region | Machine Rate | Setup Cost | Status |
|---------|--------|--------------|------------|--------|
| CNC Milling | Global | $75/hr | $150 | ✅ In DB |
| CNC Milling | US | $85/hr | $200 | ✅ In DB |
| CNC Milling | EU | $80/hr | $180 | ✅ In DB |
| CNC Milling | Asia | $45/hr | $100 | ✅ In DB |
| FDM Printing | Global | $25/hr | $25 | ✅ In DB |
| FDM Printing | US | $30/hr | $30 | ✅ In DB |
| SLA Printing | Global | $45/hr | $50 | ✅ In DB |
| SLS Printing | Global | $65/hr | $100 | ✅ In DB |

### Real-Time Quoting APIs
| Service | Data | Status | API Key Required |
|---------|------|--------|------------------|
| Xometry | Instant quotes | 🔄 API | `XOMETRY_API_KEY` |
| Protolabs | DFM + quotes | 🔄 API | `PROTOLABS_API_KEY` |
| Hubs.com | Manufacturing network | 🔄 API | `HUBS_API_KEY` |
| Fictiv | Platform quotes | 🔄 API | `FICTIV_API_KEY` |

---

## 3. Critic Thresholds

### ControlCritic (Safety Critical)
| Vehicle Type | Max Thrust (N) | Max Torque (Nm) | Max Velocity (m/s) | Status |
|--------------|----------------|-----------------|-------------------|--------|
| drone_small | 100 | 10 | 20 | ✅ In DB |
| drone_medium | 500 | 50 | 35 | ✅ In DB |
| drone_large | 1000 | 100 | 50 | ✅ In DB |

### Other Critics
| Critic | Thresholds | Status |
|--------|------------|--------|
| MaterialCritic | High temp, degradation, mass error | ✅ In DB |
| ElectronicsCritic | Power deficit, short detection | ✅ In DB |
| SurrogateCritic | Drift, accuracy, gate alignment | ✅ In DB |
| GeometryCritic | Failure rate, performance target | ✅ In DB |

---

## 4. Engineering Standards

### ISO 286 Fit Classes
| Fit Class | Type | Status |
|-----------|------|--------|
| H7/g6 | Clearance | ✅ In DB |
| H7/k6 | Transition | ✅ In DB |
| H7/p6 | Interference | ✅ In DB |
| H7/h6 | Sliding | ✅ In DB |

### AWG Wire Ampacity (60°C insulation)
| AWG | Diameter (mm) | Ampacity (A) | Status |
|-----|---------------|--------------|--------|
| 10 | 2.588 | 30 | ✅ In DB |
| 12 | 2.052 | 20 | ✅ In DB |
| 14 | 1.628 | 15 | ✅ In DB |
| 16 | 1.291 | 10 | ✅ In DB |
| 18 | 1.024 | 7 | ✅ In DB |
| 20 | 0.812 | 5 | ✅ In DB |
| 22 | 0.644 | 3 | ✅ In DB |
| 24 | 0.511 | 2.1 | ✅ In DB |

### Safety Factors
| Application | Safety Factor | Status |
|-------------|---------------|--------|
| Aerospace | 1.5 | ✅ In DB |
| Automotive | 2.0 | ✅ In DB |
| Consumer | 2.5 | ✅ In DB |
| Medical | 3.0 | ✅ In DB |
| Industrial | 3.0 | ✅ In DB |

### Manufacturing Constraints
| Process | Constraint | Value | Status |
|---------|------------|-------|--------|
| CNC Milling | Min wall thickness | 1.5 mm | ✅ In DB |
| FDM Printing | Min wall thickness | 0.8 mm | ✅ In DB |
| SLS Printing | Min wall thickness | 0.8 mm | ✅ In DB |
| SLA Printing | Min wall thickness | 0.5 mm | ✅ In DB |

---

## 5. Electronic Components

### Component Catalog APIs
| Supplier | Search | Inventory | Pricing | Status |
|----------|--------|-----------|---------|--------|
| DigiKey (Nexar) | 🔄 API | 🔄 API | 🔄 API | `NEXAR_API_KEY` configured |
| Mouser | 🔄 API | 🔄 API | 🔄 API | `MOUSER_API_KEY` needed |
| Octopart | 🔄 API | 🔄 API | 🔄 API | `OCTOPART_API_KEY` needed |

### Data Available
| Data Type | Source | Status |
|-----------|--------|--------|
| MPN | Suppliers | 🔄 API |
| Datasheets | Suppliers | 🔄 API |
| CAD Models | Suppliers | 🔄 API |
| Pricing Tiers | Suppliers | 🔄 API |
| Stock Levels | Suppliers | 🔄 API |
| Lead Times | Suppliers | 🔄 API |

---

## 6. 3D Assets / Models

### Asset Sources
| Source | Type | License | Status | API Key |
|--------|------|---------|--------|---------|
| NASA 3D | Spacecraft, instruments | NASA Open Data | 🔄 API | Free |
| Sketchfab | All categories | CC + Commercial | 🔄 API | `SKETCHFAB_API_KEY` |
| CGTrader | Engineering models | Commercial | 🔄 API | `CGTRADER_API_KEY` |
| Thingiverse | 3D printable | CC | 🔄 API | `THINGIVERSE_CLIENT_ID` |
| GrabCAD | Engineering CAD | Various | 🔄 API | `GRABCAD_API_KEY` |

---

## 7. Currency Exchange Rates

### API Sources
| Provider | Free Tier | Status | API Key |
|----------|-----------|--------|---------|
| ExchangeRate-API | 1,500 req/month | 🔄 API | `EXCHANGERATE_API_KEY` |
| OpenExchangeRates | 1,000 req/month | 🔄 API | `OPENEXCHANGERATES_APP_ID` |
| CurrencyLayer | 250 req/month | 🔄 API | `CURRENCYLAYER_API_KEY` |

### Supported Currencies
USD, EUR, GBP, JPY, CAD, AUD, CHF, CNY, SEK, NZD, MXN, SGD, HKD, NOK, KRW, INR

---

## 8. Missing Data Sources

### High Priority
| Data | Needed For | Potential Sources |
|------|------------|-------------------|
| Plastic pricing | CostAgent | PlasticsNews, ICIS, PolymerUpdate |
| Fastener catalogs | ComponentAgent | McMaster-Carr (scraping), Bossard |
| PCB pricing | CostAgent | PCBShopper, Seeed Studio, JLCPCB |

### Medium Priority
| Data | Needed For | Potential Sources |
|------|------------|-------------------|
| Terrain elevation | TopologicalAgent | Google Elevation, OpenTopoData |
| Network simulation | NetworkAgent | NS-3, OMNeT++ |
| Circuit simulation | ElectronicsAgent | NGSpice, PySpice |

### Low Priority
| Data | Needed For | Potential Sources |
|------|------------|-------------------|
| LCA databases | SustainabilityAgent | ecoinvent, GaBi |
| Weather data | TopologicalAgent | OpenWeatherMap |

---

## API Signup Links

### Pricing & Materials
- LME: https://www.lme.com/Trading/Analytics/Reports
- Fastmarkets: https://www.fastmarkets.com/contact-us/
- OpenExchangeRates: https://openexchangerates.org/signup/free

### Components
- Nexar (DigiKey): https://portal.nexar.com/sign-up
- Mouser: https://www.mouser.com/api-hub/
- Octopart: https://octopart.com/api/home

### 3D Assets
- NASA 3D: Free, no signup required (https://nasa3d.arc.nasa.gov/)
- Sketchfab: https://sketchfab.com/developers/
- CGTrader: https://www.cgtrader.com/developers

### Sustainability
- Climatiq: https://www.climatiq.io/pricing

### Manufacturing
- Xometry: https://www.xometry.com/api/
- Hubs: https://www.hubs.com/api/

---

## Environment Variables Summary

```bash
# Required for basic operation
SUPABASE_URL=
SUPABASE_SERVICE_KEY=

# For live pricing (optional - system works without)
LME_API_KEY=
FASTMARKETS_API_KEY=
OPENEXCHANGERATES_APP_ID=

# For component sourcing (optional)
NEXAR_API_KEY=        # Already configured
MOUSER_API_KEY=
OCTOPART_API_KEY=

# For 3D assets (optional)
SKETCHFAB_API_KEY=
CGTRADER_API_KEY=

# For carbon calculations (optional)
CLIMATIQ_API_KEY=

# For manufacturing quotes (optional)
XOMETRY_API_KEY=
HUBS_API_KEY=
```

---

## Data Freshness Strategy

| Data Type | Cache Duration | Source Priority |
|-----------|----------------|-----------------|
| Metal Prices | 24 hours | API → Database |
| Component Prices | 6 hours | API → Database |
| Currency Rates | 1 hour | API → Database |
| Carbon Factors | 30 days | Database |
| Material Properties | Static | Database |
| Standards | Static | Database |
