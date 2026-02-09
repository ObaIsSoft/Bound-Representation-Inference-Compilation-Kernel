-- Critic Thresholds Table
-- Stores configurable thresholds for all critic agents
-- 
-- ⚠️ IMPORTANT: This table is EMPTY by design.
--    All thresholds must be populated by domain experts before use.
--    No fictional or guessed values are allowed.
--
-- To add thresholds:
--   1. Perform engineering analysis for your specific application
--   2. Validate with physical testing or simulation
--   3. Insert verified values via migration or admin interface

CREATE TABLE IF NOT EXISTS critic_thresholds (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    critic_name TEXT NOT NULL,
    vehicle_type TEXT DEFAULT 'default',
    thresholds JSONB NOT NULL,
    version INTEGER DEFAULT 1,
    -- Verification tracking
    verified_by TEXT,           -- Name of engineer who verified
    verified_at TIMESTAMP WITH TIME ZONE,
    verification_method TEXT,   -- 'testing', 'simulation', 'analysis', 'expert_review'
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Unique constraint per critic/vehicle combination
    UNIQUE(critic_name, vehicle_type)
);

-- Index for fast lookup
CREATE INDEX IF NOT EXISTS idx_critic_thresholds_lookup 
ON critic_thresholds(critic_name, vehicle_type);

-- Trigger to auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_critic_thresholds_updated_at
    BEFORE UPDATE ON critic_thresholds
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- NO DEFAULT DATA INSERTED
-- All thresholds must be explicitly configured by users
-- Example of how to insert verified thresholds:
--
-- INSERT INTO critic_thresholds (
--     critic_name, vehicle_type, thresholds, 
--     verified_by, verified_at, verification_method
-- ) VALUES (
--     'ControlCritic', 
--     'drone_small',
--     '{"max_thrust_n": 100.0, "max_torque_nm": 10.0}'::jsonb,
--     'Jane Engineer',
--     NOW(),
--     'simulation'
-- );
