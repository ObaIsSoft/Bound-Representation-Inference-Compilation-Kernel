"""
Debug parser - see what's being extracted
"""

import sys
sys.path.insert(0, '/Users/obafemi/Documents/dev/brick/backend')

from agents.openscad_parser import OpenSCADParser

scad_code = """
module simple_wing() {
    cube([1000, 200, 50]);
}

module simple_fuselage() {
    cylinder(h=2000, r=200);
}

union() {
    simple_fuselage();
    simple_wing();
}
"""

parser = OpenSCADParser()
print("Parsing...")
ast_nodes = parser.parse(scad_code)

print(f"\nModules found: {list(parser.modules.keys())}")
print(f"Root nodes: {len(ast_nodes)}")

for i, node in enumerate(ast_nodes):
    print(f"\nRoot node {i}:")
    print(f"  Type: {node.node_type}")
    print(f"  Name: {node.name}")
    print(f"  Children: {len(node.children)}")
    print(f"  Code preview: {node.code[:100]}...")
    
    for j, child in enumerate(node.children):
        print(f"  Child {j}: {child.node_type} - {child.name}")

# Flatten
all_nodes = parser.flatten_ast(ast_nodes)
print(f"\nTotal nodes (flattened): {len(all_nodes)}")

compilable = [n for n in all_nodes if n.node_type.value in ['primitive', 'module']]
print(f"Compilable nodes: {len(compilable)}")

for i, node in enumerate(compilable):
    print(f"  {i}: {node.name} (type={node.node_type.value})")
