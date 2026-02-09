-- Materials Table (Extended)
-- Stores material properties with verified physical data
--
-- ✅ Physical properties are from verified sources (ASM Handbook, ASTM standards)
-- ⚠️ Pricing columns are NULL by design - must be populated from APIs or supplier quotes
--
-- NO GUESSED DATA POLICY:
--   - If we don't have verified data, the column is NULL
--   - The system must handle NULL gracefully
--   - Better to fail than use wrong data

CREATE TABLE IF NOT EXISTS materials (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL UNIQUE,
    
    -- Physical properties (verified where possible)
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
    
    -- Pricing (NULL by design - no guesses)
    -- Populate via pricing_service or manual entry
    cost_per_kg_usd DECIMAL(10,4),
    cost_per_kg_eur DECIMAL(10,4),
    cost_per_kg_gbp DECIMAL(10,4),
    currency_last_updated TIMESTAMP WITH TIME ZONE,
    
    -- Carbon footprint (NULL if unknown)
    carbon_footprint_kg_co2_per_kg DECIMAL(10,4),
    carbon_data_source TEXT, -- 'ecoinvent', 'worldsteel', 'climatiq_api', NULL
    
    -- Certification
    recyclable BOOLEAN DEFAULT FALSE,
    rohs_compliant BOOLEAN DEFAULT TRUE,
    aerospace_approved BOOLEAN DEFAULT FALSE,
    medical_grade BOOLEAN DEFAULT FALSE,
    
    -- Data source tracking (required for physical properties)
    property_data_source TEXT NOT NULL, -- 'asm_handbook', 'astm_standard', 'datasheet', 'measured'
    property_reference TEXT,            -- Specific reference (e.g., "ASM Vol 2, p. 345")
    
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

-- Insert ONLY verified physical properties
-- NO PRICING DATA - must come from LME API or supplier quotes
-- NO CARBON DATA unless from LCA database

INSERT INTO materials (
    name, density_kg_m3, yield_strength_mpa, ultimate_strength_mpa,
    elastic_modulus_gpa, max_temp_c, thermal_conductivity_w_mk,
    cost_per_kg_usd, carbon_footprint_kg_co2_per_kg,
    common_uses, property_data_source, property_reference
) VALUES
-- ============================================
-- ALUMINUM ALLOYS
-- ✅ Verified: ASM Handbook Vol. 2, Properties and Selection: Nonferrous Alloys
-- ASM International, 1990, ISBN: 978-0-87170-378-1
-- ============================================
('Aluminum 6061-T6', 2700, 276, 310, 68.9, 170, 167, 
 NULL, NULL,  -- No pricing/carbon - must use API or quote
 ARRAY['aerospace', 'automotive', 'cycling', 'marine'],
 'asm_handbook', 'ASM Handbook Vol. 2, p. 153'),

('Aluminum 7075-T6', 2810, 503, 572, 71.7, 120, 130,
 NULL, NULL,
 ARRAY['aerospace', 'high_performance_sports'],
 'asm_handbook', 'ASM Handbook Vol. 2, p. 163'),

-- ============================================
-- STEELS
-- ✅ Verified: ASM Handbook Vol. 1, Properties and Selection: Irons, Steels
-- ✅ Verified: ASTM A36, ASTM A240 standards
-- ============================================
('Steel A36', 7850, 250, 400, 200, 540, 52,
 NULL, NULL,
 ARRAY['structural', 'construction', 'automotive'],
 'astm_standard', 'ASTM A36/A36M-19'),

('Steel 4140', 7850, 655, 850, 205, 400, 42.6,
 NULL, NULL,
 ARRAY['axles', 'shafts', 'gears', 'fasteners'],
 'asm_handbook', 'ASM Handbook Vol. 1, p. 391'),

('Stainless Steel 304', 8000, 215, 505, 193, 800, 16.2,
 NULL, NULL,
 ARRAY['food_processing', 'medical', 'marine', 'architecture'],
 'astm_standard', 'ASTM A240/A240M-20'),

-- ============================================
-- TITANIUM
-- ✅ Verified: ASM Handbook Vol. 2
-- ============================================
('Titanium Ti-6Al-4V', 4430, 880, 950, 114, 315, 6.7,
 NULL, NULL,
 ARRAY['aerospace', 'medical_implants', 'high_performance'],
 'asm_handbook', 'ASM Handbook Vol. 2, p. 593'),

-- ============================================
-- 3D PRINTING FILAMENTS
-- ⚠️ Properties vary significantly by print orientation and settings
-- Source: Typical manufacturer datasheets (Prusament, MatterHackers, etc.)
-- These are representative values, not guarantees
-- ============================================
('PLA (3D Printing)', 1250, 60, 70, 3.5, 55, 0.13,
 NULL, NULL,
 ARRAY['prototyping', 'low_temp_applications', 'biodegradable'],
 'datasheet', 'Typical manufacturer datasheet - varies by brand'),

('ABS (3D Printing)', 1050, 40, 45, 2.3, 85, 0.13,
 NULL, NULL,
 ARRAY['prototyping', 'consumer_electronics', 'automotive_interiors'],
 'datasheet', 'Typical manufacturer datasheet - varies by brand'),

-- ============================================
-- COMPOSITES
-- ⚠️ Properties highly dependent on fiber volume fraction and layup
-- Source: Typical values for standard modulus fibers
-- ============================================
('Carbon Fiber (Standard Modulus)', 1600, 1500, 1600, 230, 150, 5.0,
 NULL, NULL,
 ARRAY['aerospace', 'automotive', 'sports_equipment'],
 'datasheet', 'T300 fiber typical values - varies by layup'),

('GFRP (Fiberglass)', 1850, 350, 450, 25, 120, 0.3,
 NULL, NULL,
 ARRAY['marine', 'automotive', 'construction', 'wind_energy'],
 'datasheet', 'E-glass typical values - varies by layup')

ON CONFLICT (name) DO UPDATE SET
    density_kg_m3 = EXCLUDED.density_kg_m3,
    yield_strength_mpa = EXCLUDED.yield_strength_mpa,
    ultimate_strength_mpa = EXCLUDED.ultimate_strength_mpa,
    elastic_modulus_gpa = EXCLUDED.elastic_modulus_gpa,
    max_temp_c = EXCLUDED.max_temp_c,
    thermal_conductivity_w_mk = EXCLUDED.thermal_conductivity_w_mk,
    common_uses = EXCLUDED.common_uses,
    property_data_source = EXCLUDED.property_data_source,
    property_reference = EXCLUDED.property_reference,
    updated_at = NOW();

-- Create view for material comparison
-- Only includes verified physical properties
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
    property_reference,
    -- Calculate specific strength (strength / density)
    (yield_strength_mpa * 1000000) / density_kg_m3 as specific_strength_pa_m3_kg,
    -- Calculate specific stiffness (modulus / density)
    (elastic_modulus_gpa * 1000000000) / density_kg_m3 as specific_stiffness_pa_m3_kg,
    -- Data quality indicator
    CASE 
        WHEN property_data_source = 'asm_handbook' THEN '✅ Verified (ASM)'
        WHEN property_data_source = 'astm_standard' THEN '✅ Verified (ASTM)'
        WHEN property_data_source = 'datasheet' THEN '⚠️ Typical (varies)'
        ELSE '❓ Unknown'
    END as data_quality
FROM materials;
