-- Standards Reference Table
-- Stores engineering standards data (ISO, ASME, ASTM, etc.)

CREATE TABLE IF NOT EXISTS standards_reference (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Standard identification
    standard_type TEXT NOT NULL,  -- "iso_fit", "awg_ampacity", "safety_factor", etc.
    standard_key TEXT NOT NULL,   -- Specific key within type (e.g., "H7/g6")
    
    -- Standard organization reference
    standard_org TEXT,            -- "ISO", "ASME", "ASTM", "IEC", etc.
    standard_number TEXT,         -- Standard number (e.g., "286-1", "B18.2.1")
    
    -- Data payload
    data JSONB NOT NULL,
    
    -- Description
    description TEXT,
    notes TEXT,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Unique constraint
    UNIQUE(standard_type, standard_key)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_standards_type ON standards_reference(standard_type);
CREATE INDEX IF NOT EXISTS idx_standards_org ON standards_reference(standard_org);

-- Trigger
CREATE TRIGGER update_standards_updated_at
    BEFORE UPDATE ON standards_reference
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ISO 286 Fit Classes
INSERT INTO standards_reference (standard_type, standard_key, standard_org, standard_number, data, description) VALUES
('iso_fit', 'H7/g6', 'ISO', '286-1', 
 '{"fit_type": "clearance", "hole_deviation": "H7", "shaft_deviation": "g6", 
   "fundamental_deviation": -0.005, "tolerance_grade": 0.012, 
   "max_clear": 0.041, "min_clear": 0.005}',
 'Loose running fit for wide commercial tolerances'),

('iso_fit', 'H7/k6', 'ISO', '286-1',
 '{"fit_type": "transition", "hole_deviation": "H7", "shaft_deviation": "k6",
   "fundamental_deviation": 0.002, "tolerance_grade": 0.012,
   "max_interference": 0.023, "max_clearance": 0.019}',
 'Transition fit for accurate location where disassembly is infrequent'),

('iso_fit', 'H7/p6', 'ISO', '286-1',
 '{"fit_type": "interference", "hole_deviation": "H7", "shaft_deviation": "p6",
   "fundamental_deviation": 0.026, "tolerance_grade": 0.012,
   "min_interference": 0.014, "max_interference": 0.05}',
 'Interference fit for permanent/semi-permanent assemblies'),

('iso_fit', 'H7/h6', 'ISO', '286-1',
 '{"fit_type": "sliding", "hole_deviation": "H7", "shaft_deviation": "h6",
   "fundamental_deviation": 0, "tolerance_grade": 0.009,
   "max_clearance": 0.025, "min_clearance": 0}',
 'Sliding fit for precise guiding of shafts and sliding parts')

ON CONFLICT (standard_type, standard_key) DO UPDATE SET
    data = EXCLUDED.data,
    description = EXCLUDED.description,
    updated_at = NOW();

-- AWG Wire Ampacity (at 60°C insulation rating)
INSERT INTO standards_reference (standard_type, standard_key, standard_org, standard_number, data, description) VALUES
('awg_ampacity', '10', 'NEC', '310.16',
 '{"diameter_mm": 2.588, "area_mm2": 5.26, "ampacity_a": 30, "resistance_ohm_per_m": 0.00328}',
 'AWG 10 copper wire ampacity'),

('awg_ampacity', '12', 'NEC', '310.16',
 '{"diameter_mm": 2.052, "area_mm2": 3.31, "ampacity_a": 20, "resistance_ohm_per_m": 0.00521}',
 'AWG 12 copper wire ampacity'),

('awg_ampacity', '14', 'NEC', '310.16',
 '{"diameter_mm": 1.628, "area_mm2": 2.08, "ampacity_a": 15, "resistance_ohm_per_m": 0.00829}',
 'AWG 14 copper wire ampacity'),

('awg_ampacity', '16', 'NEC', '310.16',
 '{"diameter_mm": 1.291, "area_mm2": 1.31, "ampacity_a": 10, "resistance_ohm_per_m": 0.0132}',
 'AWG 16 copper wire ampacity'),

('awg_ampacity', '18', 'NEC', '310.16',
 '{"diameter_mm": 1.024, "area_mm2": 0.823, "ampacity_a": 7, "resistance_ohm_per_m": 0.021}',
 'AWG 18 copper wire ampacity'),

('awg_ampacity', '20', 'NEC', '310.16',
 '{"diameter_mm": 0.812, "area_mm2": 0.518, "ampacity_a": 5, "resistance_ohm_per_m": 0.0333}',
 'AWG 20 copper wire ampacity'),

('awg_ampacity', '22', 'NEC', '310.16',
 '{"diameter_mm": 0.644, "area_mm2": 0.326, "ampacity_a": 3, "resistance_ohm_per_m": 0.053}',
 'AWG 22 copper wire ampacity'),

('awg_ampacity', '24', 'NEC', '310.16',
 '{"diameter_mm": 0.511, "area_mm2": 0.205, "ampacity_a": 2.1, "resistance_ohm_per_m": 0.084}',
 'AWG 24 copper wire ampacity - common for signal wiring')

ON CONFLICT (standard_type, standard_key) DO UPDATE SET
    data = EXCLUDED.data,
    description = EXCLUDED.description,
    updated_at = NOW();

-- Safety Factors by Application
INSERT INTO standards_reference (standard_type, standard_key, standard_org, standard_number, data, description) VALUES
('safety_factor', 'aerospace', 'MIL', 'STD-882',
 '{"safety_factor": 1.5, "catastrophic_factor": 4.0, "critical_factor": 2.0}',
 'Aerospace safety factors - high reliability required'),

('safety_factor', 'automotive', 'ISO', '26262',
 '{"safety_factor": 2.0, "critical_system_factor": 4.0}',
 'Automotive safety factors per ISO 26262'),

('safety_factor', 'consumer', 'ISO', '12100',
 '{"safety_factor": 2.5, "moving_parts_factor": 3.0}',
 'Consumer product safety factors'),

('safety_factor', 'medical', 'IEC', '62304',
 '{"safety_factor": 3.0, "life_support_factor": 10.0}',
 'Medical device safety factors'),

('safety_factor', 'industrial', 'OSHA', '1910',
 '{"safety_factor": 3.0, "lifting_factor": 5.0}',
 'Industrial equipment safety factors')

ON CONFLICT (standard_type, standard_key) DO UPDATE SET
    data = EXCLUDED.data,
    description = EXCLUDED.description,
    updated_at = NOW();

-- Manufacturing Constraints
INSERT INTO standards_reference (standard_type, standard_key, standard_org, data, description) VALUES
('manufacturing_constraint', 'cnc_milling_min_wall_thickness', 'ISO',
 '{"min_wall_thickness_mm": 1.5, "recommended_mm": 2.0}',
 'Minimum wall thickness for CNC milled parts'),

('manufacturing_constraint', 'fdm_printing_min_wall_thickness', 'ISO',
 '{"min_wall_thickness_mm": 0.8, "recommended_mm": 1.2}',
 'Minimum wall thickness for FDM 3D printed parts'),

('manufacturing_constraint', 'sls_printing_min_wall_thickness', 'ISO',
 '{"min_wall_thickness_mm": 0.8, "recommended_mm": 1.0}',
 'Minimum wall thickness for SLS 3D printed parts'),

('manufacturing_constraint', 'sla_printing_min_wall_thickness', 'ISO',
 '{"min_wall_thickness_mm": 0.5, "recommended_mm": 0.8}',
 'Minimum wall thickness for SLA 3D printed parts')

ON CONFLICT (standard_type, standard_key) DO UPDATE SET
    data = EXCLUDED.data,
    description = EXCLUDED.description,
    updated_at = NOW();
