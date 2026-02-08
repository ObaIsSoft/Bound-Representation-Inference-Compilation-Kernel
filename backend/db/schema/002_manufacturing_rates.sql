-- Manufacturing Rates Table
-- Stores regional manufacturing costs and process constraints
--
-- ⚠️ WARNING: Default rates are ESTIMATES only.
--    Real rates vary significantly by supplier, equipment, and volume.
--    For accurate quotes, integrate with Xometry/Protolabs APIs.
--    See DATA_VERIFICATION_AUDIT.md for details.

CREATE TABLE IF NOT EXISTS manufacturing_rates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    process_type TEXT NOT NULL,
    region TEXT DEFAULT 'global',
    
    -- Cost structure
    machine_hourly_rate_usd DECIMAL(10,2),
    setup_cost_usd DECIMAL(10,2),
    material_waste_pct DECIMAL(5,2) DEFAULT 5.0,
    
    -- Process constraints
    min_wall_thickness_mm DECIMAL(8,4),
    max_aspect_ratio DECIMAL(8,2),
    tolerance_mm DECIMAL(8,4),
    max_part_size_mm DECIMAL(10,2),
    
    -- Material compatibility
    material_compatibility TEXT[],
    
    -- Time estimates
    setup_time_minutes INTEGER DEFAULT 30,
    post_processing_time_minutes INTEGER DEFAULT 0,
    
    -- Data quality tracking
    data_source TEXT, -- 'xometry_api', 'protolabs_api', 'estimate', 'industry_survey'
    confidence_level TEXT DEFAULT 'low', -- 'low', 'medium', 'high'
    
    -- Metadata
    notes TEXT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Unique constraint per process/region
    UNIQUE(process_type, region)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_manufacturing_rates_process 
ON manufacturing_rates(process_type);

CREATE INDEX IF NOT EXISTS idx_manufacturing_rates_region 
ON manufacturing_rates(region);

-- Trigger for updated_at
CREATE TRIGGER update_manufacturing_rates_updated_at
    BEFORE UPDATE ON manufacturing_rates
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Insert CNC Milling rates by region
-- ⚠️ ESTIMATED VALUES - Real rates vary 5-10x by supplier!
INSERT INTO manufacturing_rates (
    process_type, region, machine_hourly_rate_usd, setup_cost_usd,
    min_wall_thickness_mm, max_aspect_ratio, tolerance_mm,
    material_compatibility, data_source, confidence_level, notes
) VALUES
-- Global defaults
('cnc_milling', 'global', 75.00, 150.00, 1.5, 10.0, 0.01,
 ARRAY['aluminum', 'steel', 'titanium', 'brass', 'plastic'],
 'industry_estimate', 'low',
 'Reference only. Get real quotes from Xometry/Protolabs/Hubs'),

-- US rates
('cnc_milling', 'us', 85.00, 200.00, 1.5, 10.0, 0.005,
 ARRAY['aluminum', 'steel', 'titanium', 'brass', 'plastic'],
 'industry_estimate', 'low',
 'US labor costs vary by region. Silicon Valley: $120+/hr, Midwest: $60-80/hr'),

-- EU rates
('cnc_milling', 'eu', 80.00, 180.00, 1.5, 10.0, 0.005,
 ARRAY['aluminum', 'steel', 'titanium', 'brass', 'plastic'],
 'industry_estimate', 'low',
 'EU rates vary by country. Germany: €80-120/hr, Eastern EU: €40-60/hr'),

-- Asia rates
('cnc_milling', 'asia', 45.00, 100.00, 1.5, 10.0, 0.01,
 ARRAY['aluminum', 'steel', 'brass', 'plastic'],
 'industry_estimate', 'low',
 'China: $20-40/hr (job shops), $60-80/hr (precision). Quality varies.')

ON CONFLICT (process_type, region) DO UPDATE SET
    machine_hourly_rate_usd = EXCLUDED.machine_hourly_rate_usd,
    setup_cost_usd = EXCLUDED.setup_cost_usd,
    min_wall_thickness_mm = EXCLUDED.min_wall_thickness_mm,
    max_aspect_ratio = EXCLUDED.max_aspect_ratio,
    tolerance_mm = EXCLUDED.tolerance_mm,
    material_compatibility = EXCLUDED.material_compatibility,
    data_source = EXCLUDED.data_source,
    confidence_level = EXCLUDED.confidence_level,
    notes = EXCLUDED.notes,
    updated_at = NOW();

-- Insert 3D Printing rates (FDM)
-- ⚠️ ESTIMATED VALUES - Desktop vs industrial printers vary 10x!
INSERT INTO manufacturing_rates (
    process_type, region, machine_hourly_rate_usd, setup_cost_usd,
    min_wall_thickness_mm, max_aspect_ratio, tolerance_mm,
    material_compatibility, data_source, confidence_level, notes
) VALUES
('fdm_printing', 'global', 25.00, 25.00, 0.8, 10.0, 0.2,
 ARRAY['pla', 'abs', 'petg', 'nylon', 'tpu'],
 'industry_estimate', 'low',
 'Desktop printers: $5-15/hr. Industrial (Stratasys): $50-100/hr'),

('fdm_printing', 'us', 30.00, 30.00, 0.8, 10.0, 0.2,
 ARRAY['pla', 'abs', 'petg', 'nylon', 'tpu'],
 'industry_estimate', 'low',
 'US services: $10-50/hr depending on quality/printer type'),

('fdm_printing', 'eu', 28.00, 25.00, 0.8, 10.0, 0.2,
 ARRAY['pla', 'abs', 'petg', 'nylon', 'tpu'],
 'industry_estimate', 'low',
 'EU services: €8-45/hr')

ON CONFLICT (process_type, region) DO UPDATE SET
    machine_hourly_rate_usd = EXCLUDED.machine_hourly_rate_usd,
    setup_cost_usd = EXCLUDED.setup_cost_usd,
    min_wall_thickness_mm = EXCLUDED.min_wall_thickness_mm,
    max_aspect_ratio = EXCLUDED.max_aspect_ratio,
    tolerance_mm = EXCLUDED.tolerance_mm,
    material_compatibility = EXCLUDED.material_compatibility,
    data_source = EXCLUDED.data_source,
    confidence_level = EXCLUDED.confidence_level,
    notes = EXCLUDED.notes,
    updated_at = NOW();

-- Insert SLA Printing rates
-- ⚠️ ESTIMATED VALUES - Desktop vs industrial vary significantly
INSERT INTO manufacturing_rates (
    process_type, region, machine_hourly_rate_usd, setup_cost_usd,
    min_wall_thickness_mm, max_aspect_ratio, tolerance_mm,
    material_compatibility, data_source, confidence_level, notes
) VALUES
('sla_printing', 'global', 45.00, 50.00, 0.5, 8.0, 0.05,
 ARRAY['standard_resin', 'tough_resin', 'castable_resin', 'dental_resin'],
 'industry_estimate', 'low',
 'Desktop (Elegoo/Anycubic): $15-25/hr. Industrial (Formlabs): $75-150/hr'),

('sla_printing', 'us', 55.00, 60.00, 0.5, 8.0, 0.05,
 ARRAY['standard_resin', 'tough_resin', 'castable_resin', 'dental_resin'],
 'industry_estimate', 'low',
 'US services: $30-100/hr depending on resin quality')

ON CONFLICT (process_type, region) DO UPDATE SET
    machine_hourly_rate_usd = EXCLUDED.machine_hourly_rate_usd,
    setup_cost_usd = EXCLUDED.setup_cost_usd,
    min_wall_thickness_mm = EXCLUDED.min_wall_thickness_mm,
    max_aspect_ratio = EXCLUDED.max_aspect_ratio,
    tolerance_mm = EXCLUDED.tolerance_mm,
    material_compatibility = EXCLUDED.material_compatibility,
    data_source = EXCLUDED.data_source,
    confidence_level = EXCLUDED.confidence_level,
    notes = EXCLUDED.notes,
    updated_at = NOW();

-- Insert SLS Printing rates (Nylon)
-- ⚠️ ESTIMATED VALUES - Industrial only, no desktop SLS
INSERT INTO manufacturing_rates (
    process_type, region, machine_hourly_rate_usd, setup_cost_usd,
    min_wall_thickness_mm, max_aspect_ratio, tolerance_mm,
    material_compatibility, data_source, confidence_level, notes
) VALUES
('sls_printing', 'global', 65.00, 100.00, 0.8, 6.0, 0.1,
 ARRAY['nylon_12', 'nylon_11', 'tpu'],
 'industry_estimate', 'low',
 'Industrial SLS (EOS, 3D Systems): $60-150/hr. Bureau services: $40-80/hr')

ON CONFLICT (process_type, region) DO UPDATE SET
    machine_hourly_rate_usd = EXCLUDED.machine_hourly_rate_usd,
    setup_cost_usd = EXCLUDED.setup_cost_usd,
    min_wall_thickness_mm = EXCLUDED.min_wall_thickness_mm,
    max_aspect_ratio = EXCLUDED.max_aspect_ratio,
    tolerance_mm = EXCLUDED.tolerance_mm,
    material_compatibility = EXCLUDED.material_compatibility,
    data_source = EXCLUDED.data_source,
    confidence_level = EXCLUDED.confidence_level,
    notes = EXCLUDED.notes,
    updated_at = NOW();
