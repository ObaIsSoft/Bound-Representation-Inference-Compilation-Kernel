"""
Test parameter substitution in modules
"""

import sys
sys.path.insert(0, '/Users/obafemi/Documents/dev/brick/backend')

from agents.openscad_parser import OpenSCADParser
from agents.openscad_agent import OpenSCADAgent

# Test with parameters
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

all_nodes = parser.flatten_ast(ast_nodes)
print(f"All nodes: {len(all_nodes)}")

compilable = [n for n in all_nodes if n.node_type.value in ['primitive', 'module']]
print(f"Compilable nodes: {len(compilable)}")

if not compilable:
    print("ERROR: No compilable nodes found!")
    sys.exit(1)

agent = OpenSCADAgent()

print("=" * 60)
print("TESTING PARAMETER SUBSTITUTION")
print("=" * 60)

for i, node in enumerate(compilable):
    print(f"\nNode {i}: {node.name} (type={node.node_type.value})")
    print(f"Params: {node.params}")
    
    try:
        generated_code = agent._generate_scad_for_node(node)
        print(f"Generated code:")
        print(generated_code)
    except Exception as e:
        print(f"ERROR generating code: {e}")
        import traceback
        traceback.print_exc()
    print("-" * 60)
