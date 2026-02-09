-- Manufacturing Rates Table
-- Stores regional manufacturing costs and process constraints
--
-- ⚠️ IMPORTANT: This table is EMPTY by design.
--    No estimated rates are provided.
--    Users must either:
--      1. Get quotes from real suppliers (Xometry, Protolabs, etc.)
--      2. Insert their own verified rates
--
-- The system will FAIL if manufacturing rates are requested but not configured.
-- This is intentional - better to fail than use wrong data.

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
    
    -- Data source tracking (required)
    data_source TEXT NOT NULL, -- 'xometry_api', 'protolabs_api', 'supplier_quote', 'internal'
    supplier_name TEXT,        -- Name of supplier who provided the quote
    quote_reference TEXT,      -- Quote number or reference
    quote_date TIMESTAMP WITH TIME ZONE,
    
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

-- NO DEFAULT DATA INSERTED
-- All rates must come from real suppliers or verified sources
--
-- Example insertion:
-- INSERT INTO manufacturing_rates (
--     process_type, region, machine_hourly_rate_usd, setup_cost_usd,
--     data_source, supplier_name, quote_reference, quote_date
-- ) VALUES (
--     'cnc_milling', 'us', 85.00, 200.00,
--     'supplier_quote', 'Xometry', 'Q-12345', '2026-02-08'
-- );
