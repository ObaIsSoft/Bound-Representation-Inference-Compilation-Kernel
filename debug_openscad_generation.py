"""
Debug OpenSCAD compilation - print actual generated code
"""

import sys
sys.path.insert(0, '/Users/obafemi/Documents/dev/brick/backend')

from agents.openscad_parser import OpenSCADParser
from agents.openscad_agent import OpenSCADAgent

# Complex assembly
scad_code = """
// Wing module
module wing(length=10, width=1, thickness=0.5) {
    cube([length, width, thickness]);
}

// Fuselage module
module fuselage(height=20, radius=2) {
    cylinder(h=height, r=radius, $fn=16);
}

union() {
    fuselage(height=20, radius=2);
    translate([0, -5, 10]) rotate([0, 90, 0]) wing(length=12, width=1, thickness=0.5);
}
"""

print("=" * 60)
print("PARSING CODE")
print("=" * 60)

parser = OpenSCADParser()
ast_nodes = parser.parse(scad_code)

print(f"Modules found: {list(parser.modules.keys())}")
print(f"Root nodes: {len(ast_nodes)}")

# Flatten
all_nodes = parser.flatten_ast(ast_nodes)
compilable = [n for n in all_nodes if n.node_type.value in ['primitive', 'module']]

print(f"Compilable nodes: {len(compilable)}")
print()

# Generate code for each node
agent = OpenSCADAgent()

for idx, node in enumerate(compilable):
    print("=" * 60)
    print(f"NODE {idx}: {node.name} (type={node.node_type.value}, depth={node.depth})")
    print("=" * 60)
    
    generated_code = agent._generate_scad_for_node(node)
    
    print("Generated OpenSCAD code:")
    print("-" * 60)
    print(generated_code)
    print("-" * 60)
    print()
