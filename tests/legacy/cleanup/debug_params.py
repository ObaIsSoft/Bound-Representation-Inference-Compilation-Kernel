"""
Debug parameter parsing
"""

import sys
sys.path.insert(0, '/Users/obafemi/Documents/dev/brick/backend')

from agents.openscad_parser import OpenSCADParser

scad_code = """
module fuselage(height=20, radius=2) {
    cylinder(h=height, r=radius, $fn=16);
}

union() {
    fuselage(height=20, radius=2);
}
"""

parser = OpenSCADParser()
ast_nodes = parser.parse(scad_code)

print("Modules:", parser.modules)
print()

# Get the union node
union_node = ast_nodes[0]
print(f"Union children: {len(union_node.children)}")

# Get the fuselage call
fuselage_call = union_node.children[0]
print(f"Fuselage call params: {fuselage_call.params}")
print(f"Fuselage call code: {fuselage_call.code}")
print()

# Check module definition
module_def = parser.modules['fuselage']
print(f"Module params: {module_def['params']}")
print(f"Module body: {module_def['body']}")
