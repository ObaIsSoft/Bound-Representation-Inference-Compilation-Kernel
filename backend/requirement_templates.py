"""
Modular Requirement Templates for Design Projects

Flexible, extensible template system that supports diverse project types
beyond the standard categories. Templates can be dynamically loaded and
customized based on project needs.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class RequirementPriority(Enum):
    """Priority levels for requirements"""
    CRITICAL = "critical"  # Must have before planning
    IMPORTANT = "important"  # Should have, can proceed without
    OPTIONAL = "optional"  # Nice to have, can skip

@dataclass
class RequirementField:
    """Single requirement field definition"""
    key: str
    question: str
    priority: RequirementPriority
    data_type: str  # 'number', 'string', 'choice', 'range', 'boolean'
    unit: Optional[str] = None
    choices: Optional[List[str]] = None
    validation: Optional[Dict[str, Any]] = None
    help_text: Optional[str] = None
    dependencies: Optional[List[str]] = None  # Other fields this depends on

class RequirementTemplate:
    """Base template for project requirements"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.fields: List[RequirementField] = []
        self.tags: List[str] = []
    
    def add_field(self, field: RequirementField):
        """Add a requirement field"""
        self.fields.append(field)
        return self
    
    def get_critical_fields(self) -> List[RequirementField]:
        """Get only critical requirements"""
        return [f for f in self.fields if f.priority == RequirementPriority.CRITICAL]
    
    def get_missing_requirements(self, gathered: Dict[str, Any]) -> List[RequirementField]:
        """Determine which requirements are still missing"""
        missing = []
        for field in self.fields:
            if field.key not in gathered:
                # Check dependencies
                if field.dependencies:
                    deps_met = all(dep in gathered for dep in field.dependencies)
                    if not deps_met:
                        continue  # Skip if dependencies not met
                missing.append(field)
        return missing
    
    def is_ready_for_planning(self, gathered: Dict[str, Any]) -> bool:
        """Check if all critical requirements are gathered"""
        critical = self.get_critical_fields()
        return all(field.key in gathered for field in critical)


# ============================================================================
# MODULAR TEMPLATE DEFINITIONS
# ============================================================================

def create_aerial_template() -> RequirementTemplate:
    """Template for aerial vehicles (drones, aircraft, etc.)"""
    template = RequirementTemplate(
        name="aerial_vehicle",
        description="Flying vehicles including drones, aircraft, and UAVs"
    )
    template.tags = ["aerial", "flight", "aviation"]
    
    template.add_field(RequirementField(
        key="payload_mass",
        question="What is the target payload mass?",
        priority=RequirementPriority.CRITICAL,
        data_type="number",
        unit="kg",
        validation={"min": 0, "max": 10000}
    ))
    
    template.add_field(RequirementField(
        key="flight_time",
        question="What is the required flight time/endurance?",
        priority=RequirementPriority.CRITICAL,
        data_type="number",
        unit="minutes",
        validation={"min": 1, "max": 1440}
    ))
    
    template.add_field(RequirementField(
        key="environment",
        question="Indoor or outdoor operation?",
        priority=RequirementPriority.CRITICAL,
        data_type="choice",
        choices=["indoor", "outdoor", "both"]
    ))
    
    template.add_field(RequirementField(
        key="max_wind_speed",
        question="Maximum wind speed it should handle?",
        priority=RequirementPriority.IMPORTANT,
        data_type="number",
        unit="m/s",
        dependencies=["environment"]
    ))
    
    template.add_field(RequirementField(
        key="altitude_range",
        question="Operating altitude range?",
        priority=RequirementPriority.IMPORTANT,
        data_type="range",
        unit="meters",
        help_text="Format: min-max (e.g., 0-1000)"
    ))
    
    template.add_field(RequirementField(
        key="autonomy_level",
        question="Autonomous, manual, or hybrid control?",
        priority=RequirementPriority.CRITICAL,
        data_type="choice",
        choices=["fully_autonomous", "semi_autonomous", "manual", "hybrid"]
    ))
    
    return template


def create_ground_template() -> RequirementTemplate:
    """Template for ground vehicles"""
    template = RequirementTemplate(
        name="ground_vehicle",
        description="Land-based vehicles including rovers, cars, robots"
    )
    template.tags = ["ground", "terrestrial", "rover"]
    
    template.add_field(RequirementField(
        key="terrain_type",
        question="What type of terrain will it operate on?",
        priority=RequirementPriority.CRITICAL,
        data_type="choice",
        choices=["paved", "off_road", "mixed", "extreme"]
    ))
    
    template.add_field(RequirementField(
        key="max_speed",
        question="Maximum speed requirement?",
        priority=RequirementPriority.IMPORTANT,
        data_type="number",
        unit="m/s"
    ))
    
    template.add_field(RequirementField(
        key="load_capacity",
        question="Load carrying capacity?",
        priority=RequirementPriority.CRITICAL,
        data_type="number",
        unit="kg"
    ))
    
    template.add_field(RequirementField(
        key="wheel_config",
        question="Wheel configuration?",
        priority=RequirementPriority.IMPORTANT,
        data_type="choice",
        choices=["2-wheel", "4-wheel", "6-wheel", "tracked", "legged"]
    ))
    
    return template


def create_marine_template() -> RequirementTemplate:
    """Template for marine vehicles"""
    template = RequirementTemplate(
        name="marine_vehicle",
        description="Water-based vehicles including boats, submarines, ROVs"
    )
    template.tags = ["marine", "underwater", "surface"]
    
    template.add_field(RequirementField(
        key="operation_mode",
        question="Surface or underwater operation?",
        priority=RequirementPriority.CRITICAL,
        data_type="choice",
        choices=["surface", "underwater", "both"]
    ))
    
    template.add_field(RequirementField(
        key="max_depth",
        question="Maximum operating depth?",
        priority=RequirementPriority.CRITICAL,
        data_type="number",
        unit="meters",
        dependencies=["operation_mode"]
    ))
    
    template.add_field(RequirementField(
        key="propulsion_type",
        question="Propulsion system type?",
        priority=RequirementPriority.IMPORTANT,
        data_type="choice",
        choices=["propeller", "jet", "hybrid", "sail"]
    ))
    
    return template


def create_parachute_template() -> RequirementTemplate:
    """Template for parachutes and recovery systems"""
    template = RequirementTemplate(
        name="parachute",
        description="Parachutes and recovery systems"
    )
    template.tags = ["aerospace", "recovery", "descent"]
    
    template.add_field(RequirementField(
        key="payload_mass",
        question="What is the target payload mass?",
        priority=RequirementPriority.CRITICAL,
        data_type="number",
        unit="kg"
    ))
    
    template.add_field(RequirementField(
        key="descent_rate",
        question="What is the target descent rate?",
        priority=RequirementPriority.CRITICAL,
        data_type="number",
        unit="m/s"
    ))
    
    template.add_field(RequirementField(
        key="environment",
        question="Environment (outdoor/indoor/space)?",
        priority=RequirementPriority.CRITICAL,
        data_type="choice",
        choices=["outdoor", "indoor", "space"]
    ))
    
    template.add_field(RequirementField(
        key="deployment_type",
        question="Deployment method (manual/automatic)?",
        priority=RequirementPriority.IMPORTANT,
        data_type="choice",
        choices=["manual", "automatic", "static_line"]
    ))
    
    return template


def create_robotics_template() -> RequirementTemplate:
    """Template for robotic systems"""
    template = RequirementTemplate(
        name="robotics",
        description="Robotic manipulators, arms, and automated systems"
    )
    template.tags = ["robotics", "manipulator", "automation"]
    
    template.add_field(RequirementField(
        key="degrees_of_freedom",
        question="How many degrees of freedom (DOF)?",
        priority=RequirementPriority.CRITICAL,
        data_type="number",
        validation={"min": 1, "max": 12}
    ))
    
    template.add_field(RequirementField(
        key="reach",
        question="Required reach/workspace?",
        priority=RequirementPriority.CRITICAL,
        data_type="number",
        unit="meters"
    ))
    
    template.add_field(RequirementField(
        key="precision",
        question="Required positioning precision?",
        priority=RequirementPriority.IMPORTANT,
        data_type="number",
        unit="mm"
    ))
    
    template.add_field(RequirementField(
        key="end_effector",
        question="Type of end effector needed?",
        priority=RequirementPriority.IMPORTANT,
        data_type="choice",
        choices=["gripper", "suction", "magnetic", "custom", "none"]
    ))
    
    return template


def create_custom_template(name: str, description: str, fields: List[Dict]) -> RequirementTemplate:
    """Create a custom template from configuration"""
    template = RequirementTemplate(name=name, description=description)
    
    for field_config in fields:
        field = RequirementField(
            key=field_config["key"],
            question=field_config["question"],
            priority=RequirementPriority(field_config.get("priority", "important")),
            data_type=field_config["data_type"],
            unit=field_config.get("unit"),
            choices=field_config.get("choices"),
            validation=field_config.get("validation"),
            help_text=field_config.get("help_text"),
            dependencies=field_config.get("dependencies")
        )
        template.add_field(field)
    
    return template


class TemplateRegistry:
    """Registry of all available requirement templates"""
    
    def __init__(self):
        self.templates: Dict[str, RequirementTemplate] = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load built-in templates"""
        self.register(create_aerial_template())
        self.register(create_ground_template())
        self.register(create_marine_template())
        self.register(create_robotics_template())
        self.register(create_parachute_template())
    
    def register(self, template: RequirementTemplate):
        """Register a template"""
        self.templates[template.name] = template
    
    def get(self, name: str) -> Optional[RequirementTemplate]:
        """Get template by name"""
        return self.templates.get(name)
    
    def find_by_tags(self, tags: List[str]) -> List[RequirementTemplate]:
        """Find templates matching tags"""
        return [
            t for t in self.templates.values()
            if any(tag in t.tags for tag in tags)
        ]
    
    def list_all(self) -> List[str]:
        """List all template names"""
        return list(self.templates.keys())


# Global template registry
template_registry = TemplateRegistry()
