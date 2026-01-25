"""
Test module code generation with transforms
"""

import sys
sys.path.insert(0, '/Users/obafemi/Documents/dev/brick/backend')

from agents.openscad_parser import OpenSCADParser
from agents.openscad_agent import OpenSCADAgent

# Test with transforms
scad_code = """
module wing() {
    cube([1000, 200, 50]);
}

module fuselage() {
    cylinder(h=2000, r=200);
}

union() {
    fuselage();
    translate([0, 0, 1000]) wing();
    translate([0, 0, -1000]) rotate([0, 180, 0]) wing();
}
"""

parser = OpenSCADParser()
ast_nodes = parser.parse(scad_code)

all_nodes = parser.flatten_ast(ast_nodes)
compilable = [n for n in all_nodes if n.node_type.value in ['primitive', 'module']]

agent = OpenSCADAgent()

print("=" * 60)
print("TESTING MODULE CODE GENERATION WITH TRANSFORMS")
print("=" * 60)
print(f"\nFound {len(compilable)} compilable nodes\n")

for i, node in enumerate(compilable):
    print(f"Node {i}: {node.name} (type={node.node_type.value}, depth={node.depth})")
    
    try:
        generated_code = agent._generate_scad_for_node(node)
        print(f"Generated code:")
        print(generated_code)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    print("-" * 60)
