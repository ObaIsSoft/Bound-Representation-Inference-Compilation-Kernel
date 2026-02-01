"""
Debug union block parsing
"""

import sys
sys.path.insert(0, '/Users/obafemi/Documents/dev/brick/backend')

from agents.openscad_parser import OpenSCADParser

scad_code = """
module wing(length=10, width=1) {
    cube([length, width, 50]);
}

union() {
    wing(length=1000, width=200);
}
"""

parser = OpenSCADParser()
ast_nodes = parser.parse(scad_code)

print(f"Modules: {list(parser.modules.keys())}")
print(f"Root nodes: {len(ast_nodes)}")

for i, node in enumerate(ast_nodes):
    print(f"\nRoot node {i}:")
    print(f"  Type: {node.node_type}")
    print(f"  Name: {node.name}")
    print(f"  Children: {len(node.children)}")
    print(f"  Code: {node.code[:100]}...")
    
    for j, child in enumerate(node.children):
        print(f"  Child {j}: {child.node_type} - {child.name}")
        print(f"    Params: {child.params}")
        print(f"    Children: {len(child.children)}")
