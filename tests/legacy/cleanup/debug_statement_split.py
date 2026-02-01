"""
Debug statement splitting
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

# Extract main body
parser._extract_modules(scad_code)
parser._extract_variables(scad_code)
main_body = parser._extract_main_body(scad_code)

print("Main body:")
print(repr(main_body))
print()

# Split into statements
statements = parser._split_statements(main_body)
print(f"Statements: {len(statements)}")
for i, stmt in enumerate(statements):
    print(f"\nStatement {i}:")
    print(repr(stmt))
