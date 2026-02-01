from typing import Dict, Any, List
import uuid

from isa import HardwareISA, ConstraintNode
from schemas.isa_schema import (
    ISAHierarchy, ISAPod, ISAParameter, ISAConstraint, ISAConstraintType, ISAUnitDimension
)

class HandshakeController:
    """
    Controller for handling ISA Handshake and Serialization logic.
    Decouples API routes from internal data processing.
    """
    
    @staticmethod
    def serialize_isa(isa: HardwareISA) -> ISAHierarchy:
        """
        Converts internal HardwareISA model to public ISAHierarchy DTO.
        """
        pods = []
        
        # 1. Map Domains to Pods
        for domain_name, constraints_map in isa.domains.items():
            pod_params: Dict[str, ISAParameter] = {}
            pod_constraints: List[ISAConstraint] = []
            
            for const_id, node in constraints_map.items():
                # Map PhysicalValue to ISAParameter
                # Use description or ID as display name if not explicitly set
                display_name = node.description if node.description else const_id.replace("_", " ").title()
                
                pod_params[const_id] = ISAParameter(
                    magnitude=node.val.magnitude,
                    unit=node.val.unit.value,
                    locked=node.val.locked,
                    tolerance=node.val.tolerance,
                    name=display_name,
                    description=node.description,
                    significant_figures=node.val.significant_figures
                )
                
                # Map ConstraintNode to ISAConstraint
                # Ensure Enum values are extracted safely
                constraint_type_val = node.constraint_type.value if hasattr(node.constraint_type, 'value') else node.constraint_type
                priority_val = node.priority.value if hasattr(node.priority, 'value') else node.priority
                
                pod_constraints.append(ISAConstraint(
                    id=node.id,
                    min_value=node.min_value,
                    max_value=node.max_value,
                    type=constraint_type_val,
                    status=node.status,
                    dependencies=node.dependencies,
                    priority=priority_val
                ))
            
            # Create Pod
            pods.append(ISAPod(
                id=domain_name,
                name=domain_name.capitalize().replace("_", " "),
                description=f"Constraints and parameters for {domain_name} domain",
                parameters=pod_params,
                constraints=pod_constraints
            ))
            
        # 2. Construct Hierarchy
        return ISAHierarchy(
            project_id=isa.project_id,
            revision=isa.revision,
            environment=isa.environment_kernel,
            pods=pods
        )

    @staticmethod
    def process_handshake(client_version: str, capabilities: List[str]) -> Dict[str, Any]:
        """
        Process the handshake request.
        """
        # Instantiate a template ISA to get current server version/revision
        isa_template = HardwareISA(project_id="template")
        server_revision = isa_template.revision
        server_version_str = f"1.0.{server_revision}"
        
        # Logic: Simple Major.Minor compatibility
        # In a real system, use semver library
        is_compatible = client_version.startswith("1.0")
        
        response = {
            "status": "connected" if is_compatible else "incompatible",
            "server_version": server_version_str,
            "compatible": is_compatible,
            "session_id": f"sess_{uuid.uuid4().hex[:8]}",
            "isa_hierarchy": None
        }
        
        if is_compatible:
            # Include the full ISA tree in the handshake response for capable clients
            # This saves an extra round-trip
            hierarchy = HandshakeController.serialize_isa(isa_template)
            # Use json() serialization from Pydantic model
            response["isa_hierarchy"] = hierarchy.dict()
            
        return response
