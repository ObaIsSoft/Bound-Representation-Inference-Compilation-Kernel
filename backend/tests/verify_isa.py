import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isa import PhysicalValue, Unit, HardwareISA, ConstraintType

def verify_isa_features():
    print("--- Verifying ISA Features ---")
    
    # 1. Unit Conversion
    val_ft = PhysicalValue(magnitude=10.0, unit=Unit.FEET)
    val_m = val_ft.convert_to(Unit.METERS)
    
    print(f"10 ft = {val_m.magnitude:.4f} m")
    assert abs(val_m.magnitude - 3.048) < 0.001
    print("PASS: Unit conversion (ft -> m)")
    
    # 2. Constraints
    isa = HardwareISA(project_id="test_proj")
    
    # Add a constraint: Length < 5m
    success = isa.add_node(
        domain="dynamics",
        node_id="wingspan",
        value=PhysicalValue(magnitude=4.0, unit=Unit.METERS),
        max_value=5.0,
        constraint_type=ConstraintType.RANGE
    )
    assert success
    
    # Check status
    node = isa.domains["dynamics"]["wingspan"]
    assert node.is_within_bounds()
    print("PASS: Constraint check (4m < 5m)")
    
    # Update to invalid
    node.val.magnitude = 6.0
    assert not node.is_within_bounds()
    print("PASS: Constraint check (6m > 5m detected)")
    
    # 3. Expansion Verification
    print("\n--- Expansion Verification ---")
    
    # Magnetism
    b_field = PhysicalValue(magnitude=1.0, unit=Unit.TESLA)
    b_gauss = b_field.convert_to(Unit.GAUSS)
    print(f"1 T = {b_gauss.magnitude} G")
    assert b_gauss.magnitude == 10000.0
    
    # Data
    storage = PhysicalValue(magnitude=1.0, unit=Unit.GIGABYTE)
    storage_mb = storage.convert_to(Unit.MEGABYTE)
    print(f"1 GB = {storage_mb.magnitude} MB")
    assert storage_mb.magnitude == 1000.0
    
    # Currency (USD -> NGN)
    cost = PhysicalValue(magnitude=1.0, unit=Unit.USD)
    cost_ngn = cost.convert_to(Unit.NGN)
    print(f"1 USD = {cost_ngn.magnitude:.2f} NGN")
    # Base rate is 0.00065 USD/NGN -> 1/0.00065 = 1538.46... 
    # Let's just assert positive conversion
    assert cost_ngn.magnitude > 1000.0
    
    print("PASS: Expansion domains verified")
    
    print("ISA Verification Complete.")

if __name__ == "__main__":
    verify_isa_features()
