from enum import Enum

class InteractionMode(str, Enum):
    DISCOVERY = "discovery"
    REQUIREMENTS = "requirements_gathering"
    PLANNING = "plan"
    CODING = "code"
    EXECUTION = "execute"

class OrchestratorMode(str, Enum):
    PLAN = "plan"
    EXECUTE = "execute"

class ValidationStatus(str, Enum):
    VALID = "valid"
    NEEDS_OPTIMIZATION = "needs_optimization"
    FORENSIC = "forensic_node" # For orchestrator mapping
    FAILED = "failed" # For gate internal logic
    PASSED = "passed"

class FeasibilityStatus(str, Enum):
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"

class ApprovalStatus(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    PLAN_ONLY = "plan_only"

class FluidNeeded(str, Enum):
    RUN = "run_fluid"
    SKIP = "skip_fluid"

class ManufacturingType(str, Enum):
    PRINT_3D = "3d_print"
    ASSEMBLY = "assembly"
    STANDARD = "standard"

class LatticeNeeded(str, Enum):
    YES = "lattice"
    NO = "no_lattice"
