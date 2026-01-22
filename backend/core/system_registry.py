from core.hierarchical_resolver import ModularISA, HierarchicalResolver

# Singleton State
_root_isa = ModularISA(name="Sesame_V1", constraints={"mass_budget": 50.0})

# Initial "Sesame" Robot Hierarchy (Demo)
_leg1 = ModularISA(name="Leg_1", parent_id=_root_isa.id, constraints={"length": 0.5})
_leg2 = ModularISA(name="Leg_2", parent_id=_root_isa.id, constraints={"length": 0.5})

_root_isa.sub_pods["legs"] = ModularISA(name="Legs_Group", parent_id=_root_isa.id)
_root_isa.sub_pods["legs"].sub_pods["front_left"] = _leg1
_root_isa.sub_pods["legs"].sub_pods["front_right"] = _leg2

# The Global Resolver
SYSTEM_RESOLVER = HierarchicalResolver(_root_isa)

def get_system_resolver():
    return SYSTEM_RESOLVER
