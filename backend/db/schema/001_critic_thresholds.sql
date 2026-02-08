-- Critic Thresholds Table
-- Stores configurable thresholds for all critic agents
-- Enables vehicle-specific and deployment-specific configuration
-- 
-- ⚠️ WARNING: Default thresholds are PLACEHOLDERS for development only.
--    Production systems require domain expert validation.
--    See DATA_VERIFICATION_AUDIT.md for details.

CREATE TABLE IF NOT EXISTS critic_thresholds (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    critic_name TEXT NOT NULL,
    vehicle_type TEXT DEFAULT 'default',
    thresholds JSONB NOT NULL,
    version INTEGER DEFAULT 1,
    -- Data quality tracking
    verification_status TEXT DEFAULT 'unverified',
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

-- Insert default thresholds for ControlCritic
-- ⚠️ PLACEHOLDER VALUES - Must be validated for actual hardware!
INSERT INTO critic_thresholds (critic_name, vehicle_type, thresholds, verification_status) VALUES
('ControlCritic', 'drone_small', '{
    "max_thrust_n": 100.0,
    "max_torque_nm": 10.0,
    "max_velocity_ms": 20.0,
    "max_position_m": 500.0,
    "control_effort_threshold": 50.0,
    "energy_increase_limit": 1.1,
    "_warning": "PLACEHOLDER VALUES - Requires vehicle dynamics analysis",
    "_last_updated": "2026-02-08"
}'::jsonb, 'unverified'),
('ControlCritic', 'drone_medium', '{
    "max_thrust_n": 500.0,
    "max_torque_nm": 50.0,
    "max_velocity_ms": 35.0,
    "max_position_m": 1000.0,
    "control_effort_threshold": 75.0,
    "energy_increase_limit": 1.1,
    "_warning": "PLACEHOLDER VALUES - Requires vehicle dynamics analysis",
    "_last_updated": "2026-02-08"
}'::jsonb, 'unverified'),
('ControlCritic', 'drone_large', '{
    "max_thrust_n": 1000.0,
    "max_torque_nm": 100.0,
    "max_velocity_ms": 50.0,
    "max_position_m": 2000.0,
    "control_effort_threshold": 100.0,
    "energy_increase_limit": 1.1,
    "_warning": "PLACEHOLDER VALUES - Requires vehicle dynamics analysis",
    "_last_updated": "2026-02-08"
}'::jsonb, 'unverified')
ON CONFLICT (critic_name, vehicle_type) DO UPDATE SET
    thresholds = EXCLUDED.thresholds,
    version = critic_thresholds.version + 1,
    updated_at = NOW();

-- Insert default thresholds for MaterialCritic
INSERT INTO critic_thresholds (critic_name, vehicle_type, thresholds, verification_status) VALUES
('MaterialCritic', 'default', '{
    "high_temp_threshold_c": 150,
    "degradation_rate_threshold": 0.5,
    "mass_error_threshold_pct": 10,
    "db_coverage_threshold": 0.7,
    "material_diversity_min": 3,
    "_warning": "PLACEHOLDER VALUES - Should be material-specific",
    "_last_updated": "2026-02-08"
}'::jsonb, 'unverified')
ON CONFLICT (critic_name, vehicle_type) DO UPDATE SET
    thresholds = EXCLUDED.thresholds,
    version = critic_thresholds.version + 1,
    updated_at = NOW();

-- Insert default thresholds for ElectronicsCritic
INSERT INTO critic_thresholds (critic_name, vehicle_type, thresholds, verification_status) VALUES
('ElectronicsCritic', 'default', '{
    "power_deficit_threshold": 0.3,
    "short_detection_min_rate": 0.8,
    "over_conservative_margin_w": 1000,
    "false_alarm_threshold": 5,
    "scale_issue_threshold": 5,
    "_warning": "PLACEHOLDER VALUES - Requires circuit analysis",
    "_last_updated": "2026-02-08"
}'::jsonb, 'unverified')
ON CONFLICT (critic_name, vehicle_type) DO UPDATE SET
    thresholds = EXCLUDED.thresholds,
    version = critic_thresholds.version + 1,
    updated_at = NOW();

-- Insert default thresholds for SurrogateCritic
INSERT INTO critic_thresholds (critic_name, vehicle_type, thresholds, verification_status) VALUES
('SurrogateCritic', 'default', '{
    "drift_threshold": 0.15,
    "min_accuracy": 0.7,
    "min_gate_alignment": 0.7,
    "low_speed_threshold_ms": 10,
    "high_speed_threshold_ms": 50,
    "max_false_positive_rate": 0.3,
    "_warning": "PLACEHOLDER VALUES - Requires ML model validation",
    "_last_updated": "2026-02-08"
}'::jsonb, 'unverified')
ON CONFLICT (critic_name, vehicle_type) DO UPDATE SET
    thresholds = EXCLUDED.thresholds,
    version = critic_thresholds.version + 1,
    updated_at = NOW();

-- Insert default thresholds for GeometryCritic
INSERT INTO critic_thresholds (critic_name, vehicle_type, thresholds, verification_status) VALUES
('GeometryCritic', 'default', '{
    "max_failure_rate": 0.2,
    "performance_target_seconds": 2.0,
    "min_sdf_resolution": 32,
    "max_sdf_resolution": 256,
    "_warning": "PLACEHOLDER VALUES - Requires performance benchmarking",
    "_last_updated": "2026-02-08"
}'::jsonb, 'unverified')
ON CONFLICT (critic_name, vehicle_type) DO UPDATE SET
    thresholds = EXCLUDED.thresholds,
    version = critic_thresholds.version + 1,
    updated_at = NOW();
