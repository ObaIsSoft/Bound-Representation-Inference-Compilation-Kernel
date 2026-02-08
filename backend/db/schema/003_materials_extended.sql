-- Materials Table (Extended)
-- Stores material properties with real-time pricing and sourcing
--
-- ⚠️ WARNING ON PRICING: All cost_per_kg values are ESTIMATES.
--    Real prices vary by supplier, volume, and market conditions.
--    For production use, integrate with LME API (metals) and supplier APIs.
--    See DATA_VERIFICATION_AUDIT.md for details.
--
-- ✅ VERIFIED: Physical properties are from ASM Handbook and ASTM standards.
-- ⚠️ ESTIMATED: Pricing data is approximate and should not be used for production quotes.

CREATE TABLE IF NOT EXISTS materials (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL UNIQUE,
    
    -- Physical properties
    density_kg_m3 DECIMAL(10,2),
    yield_strength_mpa DECIMAL(10,2),
    ultimate_strength_mpa DECIMAL(10,2),
    elastic_modulus_gpa DECIMAL(10,2),
    max_temp_c DECIMAL(8,2),
    min_temp_c DECIMAL(8,2),
    thermal_conductivity_w_mk DECIMAL(8,4),
    
    -- Electrical properties
    conductivity_s_m DECIMAL(12,4),
    resistivity_ohm_m DECIMAL(12,12),
    dielectric_constant DECIMAL(8,4),
    
    -- Pricing (cached from external APIs)
    -- ⚠️ These are estimates! Use pricing_service for real-time quotes.
    cost_per_kg_usd DECIMAL(10,4),
    cost_per_kg_eur DECIMAL(10,4),
    cost_per_kg_gbp DECIMAL(10,4),
    currency_last_updated TIMESTAMP WITH TIME ZONE,
    
    -- Carbon footprint
    -- Source: ecoinvent, World Steel Association, or estimated
    carbon_footprint_kg_co2_per_kg DECIMAL(10,4),
    carbon_data_source TEXT, -- 'ecoinvent', 'worldsteel', 'estimated'
    
    -- Certification
    recyclable BOOLEAN DEFAULT FALSE,
    rohs_compliant BOOLEAN DEFAULT TRUE,
    aerospace_approved BOOLEAN DEFAULT FALSE,
    medical_grade BOOLEAN DEFAULT FALSE,
    
    -- Data quality tracking
    property_data_source TEXT, -- 'asm_handbook', 'astm_standard', 'datasheet', 'estimated'
    property_verified_at TIMESTAMP WITH TIME ZONE,
    pricing_data_source TEXT, -- 'lme', 'fastmarkets', 'estimate'
    pricing_confidence TEXT DEFAULT 'low', -- 'low', 'medium', 'high'
    
    -- Metadata
    common_uses TEXT[],
    typical_suppliers TEXT[],
    last_verified_at TIMESTAMP WITH TIME ZONE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_materials_name ON materials(name);
CREATE INDEX IF NOT EXISTS idx_materials_strength ON materials(yield_strength_mpa);
CREATE INDEX IF NOT EXISTS idx_materials_cost ON materials(cost_per_kg_usd);

-- Trigger
CREATE TRIGGER update_materials_updated_at
    BEFORE UPDATE ON materials
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Insert common materials
-- 
-- DATA SOURCE LEGEND:
-- ✅ ASM Handbook Vol. 1 & 2 - Verified physical properties
-- ✅ ASTM Standards - Verified specifications
-- ⚠️ Pricing - ESTIMATED (see notes on each material)
-- ⚠️ Carbon - Mixed (ecoinvent where available, estimated otherwise)

INSERT INTO materials (
    name, density_kg_m3, yield_strength_mpa, ultimate_strength_mpa,
    elastic_modulus_gpa, max_temp_c, thermal_conductivity_w_mk,
    cost_per_kg_usd, carbon_footprint_kg_co2_per_kg,
    common_uses, property_data_source, pricing_data_source, pricing_confidence
) VALUES
-- ============================================
-- ALUMINUM ALLOYS
-- Properties: ✅ Verified from ASM Handbook Vol. 2
-- Pricing: ⚠️ Estimated (LME aluminum: ~$2.50/kg base + processing)
-- ============================================
('Aluminum 6061-T6', 2700, 276, 310, 68.9, 170, 167, 
 3.50, 12.7,
 ARRAY['aerospace', 'automotive', 'cycling', 'marine'],
 'asm_handbook', 'estimate', 'low'),

('Aluminum 7075-T6', 2810, 503, 572, 71.7, 120, 130,
 5.50, 13.5,
 ARRAY['aerospace', 'high_performance_sports'],
 'asm_handbook', 'estimate', 'low'),

-- ============================================
-- STEELS
-- Properties: ✅ Verified from ASM Handbook Vol. 1 & ASTM standards
-- Pricing: ⚠️ Estimated (LME steel: ~$0.60/kg base + processing)
-- ============================================
('Steel A36', 7850, 250, 400, 200, 540, 52,
 0.80, 1.9,
 ARRAY['structural', 'construction', 'automotive'],
 'astm_standard', 'estimate', 'low'),

('Steel 4140', 7850, 655, 850, 205, 400, 42.6,
 1.50, 2.1,
 ARRAY['axles', 'shafts', 'gears', 'fasteners'],
 'asm_handbook', 'estimate', 'low'),

('Stainless Steel 304', 8000, 215, 505, 193, 800, 16.2,
 4.00, 2.8,
 ARRAY['food_processing', 'medical', 'marine', 'architecture'],
 'astm_standard', 'estimate', 'low'),

-- ============================================
-- TITANIUM
-- Properties: ✅ Verified from ASM Handbook Vol. 2
-- Pricing: ⚠️ Highly variable ($25-80/kg). Titanium sponge price + alloy premium.
-- ============================================
('Titanium Ti-6Al-4V', 4430, 880, 950, 114, 315, 6.7,
 35.00, 45.0,
 ARRAY['aerospace', 'medical_implants', 'high_performance'],
 'asm_handbook', 'estimate', 'low'),

-- ============================================
-- 3D PRINTING FILAMENTS
-- Properties: ⚠️ Estimated - VARY SIGNIFICANTLY by print orientation and settings!
--   XY (in-plane): Higher strength
--   Z (vertical): Lower strength (layer adhesion)
-- Pricing: ⚠️ Consumer filament prices (1kg spools)
-- ============================================
('PLA (3D Printing)', 1250, 60, 70, 3.5, 55, 0.13,
 3.00, 3.4,
 ARRAY['prototyping', 'low_temp_applications', 'biodegradable'],
 'datasheet', 'estimate', 'low'),

('ABS (3D Printing)', 1050, 40, 45, 2.3, 85, 0.13,
 2.50, 3.8,
 ARRAY['prototyping', 'consumer_electronics', 'automotive_interiors'],
 'datasheet', 'estimate', 'low'),

('Nylon 12', 1020, 45, 50, 1.7, 150, 0.25,
 8.00, 8.5,
 ARRAY['gears', 'bearings', 'automotive', 'aerospace'],
 'datasheet', 'estimate', 'low'),

('PETG', 1270, 30, 50, 2.0, 75, 0.2,
 3.50, 3.9,
 ARRAY['food_safe', 'prototyping', 'mechanical_parts'],
 'datasheet', 'estimate', 'low'),

-- ============================================
-- COMPOSITES
-- Properties: ⚠️ Highly dependent on fiber volume fraction and layup!
--   Carbon Fiber: 50-70% fiber volume typical
--   GFRP: 30-50% fiber volume typical
-- Pricing: ⚠️ Highly variable by fiber type, resin, and volume
-- ============================================
('Carbon Fiber (Standard Modulus)', 1600, 1500, 1600, 230, 150, 5.0,
 45.00, 55.0,
 ARRAY['aerospace', 'automotive', 'sports_equipment'],
 'estimated', 'estimate', 'low'),

('GFRP (Fiberglass)', 1850, 350, 450, 25, 120, 0.3,
 8.00, 4.5,
 ARRAY['marine', 'automotive', 'construction', 'wind_energy'],
 'estimated', 'estimate', 'low')

ON CONFLICT (name) DO UPDATE SET
    density_kg_m3 = EXCLUDED.density_kg_m3,
    yield_strength_mpa = EXCLUDED.yield_strength_mpa,
    ultimate_strength_mpa = EXCLUDED.ultimate_strength_mpa,
    elastic_modulus_gpa = EXCLUDED.elastic_modulus_gpa,
    max_temp_c = EXCLUDED.max_temp_c,
    thermal_conductivity_w_mk = EXCLUDED.thermal_conductivity_w_mk,
    cost_per_kg_usd = EXCLUDED.cost_per_kg_usd,
    carbon_footprint_kg_co2_per_kg = EXCLUDED.carbon_footprint_kg_co2_per_kg,
    common_uses = EXCLUDED.common_uses,
    property_data_source = EXCLUDED.property_data_source,
    pricing_data_source = EXCLUDED.pricing_data_source,
    pricing_confidence = EXCLUDED.pricing_confidence,
    updated_at = NOW();

-- Create view for material comparison
CREATE OR REPLACE VIEW material_comparison AS
SELECT 
    name,
    density_kg_m3,
    yield_strength_mpa,
    ultimate_strength_mpa,
    elastic_modulus_gpa,
    cost_per_kg_usd,
    carbon_footprint_kg_co2_per_kg,
    property_data_source,
    pricing_confidence,
    -- Calculate specific strength (strength / density)
    (yield_strength_mpa * 1000000) / density_kg_m3 as specific_strength_pa_m3_kg,
    -- Calculate specific stiffness (modulus / density)
    (elastic_modulus_gpa * 1000000000) / density_kg_m3 as specific_stiffness_pa_m3_kg,
    -- Warning for estimated data
    CASE 
        WHEN pricing_confidence = 'low' THEN '⚠️ Pricing is estimated - get real quotes'
        WHEN property_data_source = 'estimated' THEN '⚠️ Properties are estimates'
        ELSE '✅ Data verified'
    END as data_quality_warning
FROM materials;
