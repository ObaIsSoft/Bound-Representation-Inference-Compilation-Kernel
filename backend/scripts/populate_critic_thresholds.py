"""
Populate critic_thresholds table in Supabase with comprehensive thresholds.

Run this script to ensure all critics have proper threshold configuration.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

from supabase import create_client
import json

# Comprehensive threshold definitions for all critics
CRITIC_THRESHOLDS = {
    # PhysicsCritic - Physics validation thresholds
    "PhysicsCritic": {
        "default": {
            "window_size": 100,
            "error_threshold": 0.1,
            "gate_alignment_threshold": 0.8,
            "conservation_tolerance": 0.01,  # 1% tolerance for conservation laws
            "max_velocity_m_s": 1000.0,
            "max_acceleration_m_s2": 100.0,
            "description": "Default physics validation thresholds"
        },
        "drone": {
            "window_size": 100,
            "error_threshold": 0.05,  # Stricter for drones
            "gate_alignment_threshold": 0.85,
            "conservation_tolerance": 0.005,
            "max_velocity_m_s": 50.0,
            "max_acceleration_m_s2": 20.0,
            "description": "Drone-specific physics thresholds"
        },
        "aircraft": {
            "window_size": 200,
            "error_threshold": 0.02,  # Very strict for aircraft
            "gate_alignment_threshold": 0.9,
            "conservation_tolerance": 0.001,
            "max_velocity_m_s": 500.0,
            "max_acceleration_m_s2": 50.0,
            "description": "Aircraft physics thresholds"
        }
    },
    
    # MaterialCritic - Material property validation
    "MaterialCritic": {
        "default": {
            "window_size": 100,
            "high_temp_threshold_c": 150,
            "db_coverage_min": 0.7,
            "db_coverage_warning": 0.8,
            "db_coverage_critical": 0.6,
            "degradation_threshold": 0.5,
            "degradation_critical": 0.6,
            "mass_error_threshold_pct": 10,
            "mass_error_warning_pct": 5,
            "mass_error_critical_pct": 15,
            "high_temp_ratio_threshold": 0.3,
            "min_unique_materials": 3,
            "strength_factor_warning": 0.8,
            "description": "Default material validation thresholds"
        },
        "aerospace": {
            "window_size": 200,
            "high_temp_threshold_c": 100,  # Lower threshold for aerospace
            "db_coverage_min": 0.9,  # Higher coverage required
            "db_coverage_warning": 0.95,
            "db_coverage_critical": 0.8,
            "degradation_threshold": 0.3,
            "degradation_critical": 0.4,
            "mass_error_threshold_pct": 5,  # Stricter mass accuracy
            "mass_error_warning_pct": 2,
            "mass_error_critical_pct": 10,
            "high_temp_ratio_threshold": 0.2,
            "min_unique_materials": 5,
            "strength_factor_warning": 0.9,
            "description": "Aerospace material standards"
        }
    },
    
    # GeometryCritic - Geometry validation
    "GeometryCritic": {
        "default": {
            "window_size": 100,
            "failure_rate_threshold": 0.2,
            "avg_time_threshold_ms": 1000,
            "sdf_resolution": 64,
            "min_edge_length_mm": 0.1,
            "max_aspect_ratio": 100,
            "description": "Default geometry validation"
        },
        "high_precision": {
            "window_size": 100,
            "failure_rate_threshold": 0.05,
            "avg_time_threshold_ms": 5000,  # Allow longer for precision
            "sdf_resolution": 128,
            "min_edge_length_mm": 0.01,
            "max_aspect_ratio": 50,
            "description": "High precision geometry"
        }
    },
    
    # ComponentCritic - Component selection validation
    "ComponentCritic": {
        "default": {
            "window_size": 100,
            "zero_result_rate": 0.1,
            "zero_result_threshold": 0.05,
            "over_spec_rate": 0.2,
            "over_spec_threshold": 0.1,
            "installation_success": 0.95,
            "user_acceptance": 0.8,
            "diversity_threshold": 0.6,
            "catalog_coverage_min": 0.7,
            "price_accuracy_pct": 15,
            "availability_check": True,
            "description": "Default component selection thresholds"
        }
    },
    
    # ElectronicsCritic - Electronics validation
    "ElectronicsCritic": {
        "default": {
            "window_size": 100,
            "deficit_rate": 0.05,
            "deficit_threshold": 0.02,
            "avg_margin": 0.2,
            "short_detection_rate": 0.95,
            "false_alarms": 0.05,
            "thermal_margin_c": 20,
            "emi_compliance": True,
            "voltage_tolerance_pct": 5,
            "current_margin_pct": 20,
            "description": "Default electronics validation"
        },
        "medical": {
            "window_size": 200,
            "deficit_rate": 0.01,  # Very strict for medical
            "deficit_threshold": 0.005,
            "avg_margin": 0.3,
            "short_detection_rate": 0.999,
            "false_alarms": 0.01,
            "thermal_margin_c": 30,
            "emi_compliance": True,
            "voltage_tolerance_pct": 2,
            "current_margin_pct": 30,
            "description": "Medical device electronics standards"
        }
    },
    
    # ChemistryCritic - Chemical compatibility validation
    "ChemistryCritic": {
        "default": {
            "window_size": 100,
            "rejection_rate": 0.1,
            "rejection_threshold": 0.05,
            "safety_accuracy": 0.95,
            "environment_analysis": True,
            "material_bias_threshold": 0.3,
            "corrosion_check": True,
            "reaction_threshold": 0.01,
            "description": "Default chemistry validation"
        }
    },
    
    # DesignCritic - Design quality validation
    "DesignCritic": {
        "default": {
            "window_size": 100,
            "diversity_score": 0.7,
            "acceptance_rate": 0.8,
            "max_entropy": 2.0,
            "creativity_threshold": 0.6,
            "description": "Default design quality thresholds"
        }
    },
    
    # OracleCritic - Physics oracle validation
    "OracleCritic": {
        "default": {
            "window_size": 100,
            "conservation_tolerance": 0.01,
            "energy_error_rate": 0.02,
            "current_error_rate": 0.01,
            "mass_error_rate": 0.005,
            "nuclear_error_rate": 0.001,
            "orbital_tolerance_m": 1000,
            "min_calls_for_stats": 10,
            "cache_hit_rate": 0.8,
            "verification_rate": 0.95,
            "description": "Default oracle validation"
        }
    },
    
    # SurrogateCritic - ML surrogate validation
    "SurrogateCritic": {
        "default": {
            "window_size": 100,
            "drift_threshold": 0.1,
            "drift_rate": 0.05,
            "accuracy_threshold": 0.9,
            "validation_coverage": 0.8,
            "gate_alignment": 0.85,
            "false_positive_rate": 0.05,
            "false_negative_rate": 0.02,
            "confidence_threshold": 0.8,
            "retrain_trigger": 0.15,
            "min_samples": 100,
            "max_age_days": 30,
            "description": "Default surrogate validation"
        }
    },
    
    # TopologicalCritic - Topology validation
    "TopologicalCritic": {
        "default": {
            "window_size": 100,
            "failure_rate": 0.15,
            "failure_threshold": 0.1,
            "slope_penalty": 0.6,
            "roughness_penalty": 0.4,
            "traversability_threshold": 0.7,
            "description": "Default topology validation"
        }
    },
    
    # ControlCritic - Control system validation
    "ControlCritic": {
        "default": {
            "window_size": 100,
            "settling_time_threshold_s": 5.0,
            "overshoot_threshold_pct": 10.0,
            "steady_state_error_pct": 2.0,
            "rise_time_threshold_s": 2.0,
            "peak_time_threshold_s": 3.0,
            "stability_margin_db": 6.0,
            "phase_margin_deg": 30.0,
            "gain_margin_db": 6.0,
            "description": "Default control system thresholds"
        },
        "drone_racing": {
            "window_size": 100,
            "settling_time_threshold_s": 0.5,  # Very fast for racing
            "overshoot_threshold_pct": 5.0,
            "steady_state_error_pct": 1.0,
            "rise_time_threshold_s": 0.1,
            "peak_time_threshold_s": 0.2,
            "stability_margin_db": 3.0,
            "phase_margin_deg": 20.0,
            "gain_margin_db": 3.0,
            "description": "Drone racing - aggressive performance"
        },
        "drone_delivery": {
            "window_size": 100,
            "settling_time_threshold_s": 2.0,  # Moderate for delivery
            "overshoot_threshold_pct": 15.0,
            "steady_state_error_pct": 5.0,
            "rise_time_threshold_s": 1.0,
            "peak_time_threshold_s": 1.5,
            "stability_margin_db": 10.0,
            "phase_margin_deg": 45.0,
            "gain_margin_db": 10.0,
            "description": "Drone delivery - stability focused"
        },
        "aircraft_small": {
            "window_size": 200,
            "settling_time_threshold_s": 10.0,
            "overshoot_threshold_pct": 2.0,
            "steady_state_error_pct": 0.5,
            "rise_time_threshold_s": 5.0,
            "peak_time_threshold_s": 8.0,
            "stability_margin_db": 12.0,
            "phase_margin_deg": 60.0,
            "gain_margin_db": 12.0,
            "description": "Small aircraft - conservative stability"
        }
    },
    
    # FluidCritic - Fluid dynamics validation
    "FluidCritic": {
        "default": {
            "window_size": 100,
            "cd_min": 0.01,
            "cd_max": 2.0,
            "cl_min": -2.0,
            "cl_max": 5.0,
            "drag_error_tolerance": 0.1,
            "pressure_ratio_min": 0.1,
            "pressure_ratio_max": 10.0,
            "velocity_tolerance": 0.05,
            "mass_conservation_tolerance": 0.01,
            "description": "Default fluid dynamics thresholds"
        }
    }
}


def populate_thresholds():
    """Populate Supabase with all critic thresholds."""
    client = create_client(
        os.getenv('SUPABASE_URL'),
        os.getenv('SUPABASE_SERVICE_KEY')
    )
    
    print("=== POPULATING CRITIC THRESHOLDS ===\n")
    
    total_records = 0
    
    for critic_name, vehicle_types in CRITIC_THRESHOLDS.items():
        for vehicle_type, thresholds in vehicle_types.items():
            try:
                # Check if record exists
                existing = client.table('critic_thresholds')\
                    .select('id')\
                    .eq('critic_name', critic_name)\
                    .eq('vehicle_type', vehicle_type)\
                    .execute()
                
                record = {
                    'critic_name': critic_name,
                    'vehicle_type': vehicle_type,
                    'thresholds': thresholds
                }
                
                if existing.data:
                    # Update existing
                    record_id = existing.data[0]['id']
                    client.table('critic_thresholds')\
                        .update(record)\
                        .eq('id', record_id)\
                        .execute()
                    print(f"  ✓ Updated {critic_name}/{vehicle_type}")
                else:
                    # Insert new
                    client.table('critic_thresholds')\
                        .insert(record)\
                        .execute()
                    print(f"  ✓ Inserted {critic_name}/{vehicle_type}")
                
                total_records += 1
                
            except Exception as e:
                print(f"  ✗ Error for {critic_name}/{vehicle_type}: {e}")
    
    print(f"\n=== COMPLETE ===")
    print(f"Total threshold records: {total_records}")
    
    # Verify
    print("\n=== VERIFICATION ===")
    try:
        result = client.table('critic_thresholds')\
            .select('critic_name, vehicle_type')\
            .execute()
        print(f"Records in database: {len(result.data)}")
        
        # Group by critic
        by_critic = {}
        for row in result.data:
            name = row['critic_name']
            if name not in by_critic:
                by_critic[name] = []
            by_critic[name].append(row['vehicle_type'])
        
        print("\nCritics configured:")
        for name, vehicles in sorted(by_critic.items()):
            print(f"  {name}: {', '.join(sorted(vehicles))}")
            
    except Exception as e:
        print(f"Verification error: {e}")


if __name__ == "__main__":
    populate_thresholds()
