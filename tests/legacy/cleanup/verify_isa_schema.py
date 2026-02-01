import sys
import os
import json
from pydantic import ValidationError

# Ensure backend acts as a package
sys.path.append(os.path.join(os.getcwd()))

try:
    from backend.schemas.isa_schema import ISAHierarchy, ISAPod, ISAParameter, ISAConstraint, ISAConstraintType

    def verify_schema():
        print("Verifying ISA Schema...")
        
        # 1. Create Parameters
        thrust_p = ISAParameter(magnitude=5000.0, unit="N", name="Thrust", description="Engine output")
        mass_p = ISAParameter(magnitude=1500.0, unit="kg", name="Dry Mass")
        
        # 2. Create Constraints
        c1 = ISAConstraint(
            id="c_thrust_min",
            min_value=4000.0,
            type=ISAConstraintType.GREATER_THAN,
            status="VALID"
        )
        
        # 3. Create Pod
        propulsion_pod = ISAPod(
            id="pod_prop",
            name="Propulsion System",
            description="Main engines and fuel",
            parameters={"thrust": thrust_p, "mass": mass_p},
            constraints=[c1]
        )
        
        # 4. Create Hierarchy
        isa = ISAHierarchy(
            project_id="verify_proj",
            revision=1,
            environment="EARTH_AERO",
            pods=[propulsion_pod]
        )
        
        # 5. Serialize
        json_output = isa.json()
        print("Schema Verification Successful!")
        print(json.dumps(json.loads(json_output), indent=2))

    if __name__ == "__main__":
        verify_schema()

except ImportError as e:
    print(f"Import Error: {e}")
except ValidationError as e:
    print(f"Validation Error: {e}")
except Exception as e:
    print(f"Error: {e}")
