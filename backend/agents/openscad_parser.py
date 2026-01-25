"""
OpenSCAD AST Parser
Parses OpenSCAD code into an Abstract Syntax Tree for parallel compilation.
Handles all OpenSCAD features: modules, primitives, transforms, booleans, conditionals, loops.
"""

from typing import List, Dict, Any, Set, Tuple
import re
from dataclasses import dataclass, field
from enum import Enum


class NodeType(Enum):
    """Types of nodes in the OpenSCAD AST"""
    MODULE = "module"
    PRIMITIVE = "primitive"
    TRANSFORM = "transform"
    BOOLEAN = "boolean"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    VARIABLE = "variable"
    FUNCTION_CALL = "function_call"


@dataclass
class ASTNode:
    """A node in the OpenSCAD Abstract Syntax Tree"""
    node_type: NodeType
    name: str
    code: str
    params: Dict[str, Any] = field(default_factory=dict)
    header: str = "" # Reconstructable header (e.g. "translate([0,0,0])")
    children: List['ASTNode'] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)
    depth: int = 0
    line_number: int = 0


class OpenSCADParser:
    """
    Robust OpenSCAD parser that handles all language features.
    Builds a dependency graph for parallel execution.
    """
    
    def __init__(self):
        self.modules = {}
        self.variables = {}
        self.functions = {}
        
    def parse(self, scad_code: str) -> List[ASTNode]:
        """
        Parse OpenSCAD code into an AST.
        
        Args:
            scad_code: OpenSCAD source code
            
        Returns:
            List of root AST nodes (operations to execute)
        """
        # Step 1: Extract module definitions
        self._extract_modules(scad_code)
        
        # Step 2: Extract variable assignments
        self._extract_variables(scad_code)
        
        # Step 3: Parse main body (everything outside module definitions)
        main_body = self._extract_main_body(scad_code)
        
        # Step 4: Build AST from main body
        ast_nodes = self._parse_statements(main_body)
        
        # Step 5: Resolve dependencies and calculate depth
        self._resolve_dependencies(ast_nodes)
        
        return ast_nodes
    
    def _extract_modules(self, code: str):
        """Extract all module definitions using brace counting"""
        # We need to parse robustly.
        # 1. Remove comments to avoid confusing braces inside comments
        clean_code = self._remove_comments(code)
        
        # 2. Find "module" keywords
        # We can't just split by "module" because it might be in strings.
        # But for SCAD, "module" keyword is top level.
        
        # Iterating through the code to find 'module name(params) {'
        # Then capturing the body until matching '}'
        
        cursor = 0
        length = len(clean_code)
        
        while cursor < length:
            # Find next 'module' keyword
            match = re.search(r'\bmodule\s+(\w+)\s*\(', clean_code[cursor:])
            if not match:
                break
                
            module_start_idx = cursor + match.start()
            name = match.group(1)
            
            # Find the opening brace of the body
            # We need to skip the parameters (...) part
            # Parameters might have parentheses inside? (vectors)
            
            # Start searching for body start '{' after the module name declaration
            # robustly handle params
            params_start = clean_code.find('(', module_start_idx)
            if params_start == -1: break # Should not happen if regex matched
            
            # Find matching closing paren for params
            # We need to handle nested Parens? SCAD params are usually simple but can have default values with calls?
            # Yes: module foo(a = sin(x))
            
            p_depth = 0
            params_end = -1
            for k in range(params_start, length):
                if clean_code[k] == '(':
                    p_depth += 1
                elif clean_code[k] == ')':
                    p_depth -= 1
                    if p_depth == 0:
                        params_end = k
                        break
            
            if params_end == -1: break # Malformed
            
            # Extract params string
            params_str = clean_code[params_start+1 : params_end]
            
            # Now search for opening brace '{'
            body_start = clean_code.find('{', params_end)
            if body_start == -1: break
            
            # Check if there is only whitespace between params end and body start
            # If there is code, maybe it's not a module definition?
            # SCAD: module foo() { ... }
            # module foo() variable = 1; { ... } ? No.
            
            # Find module body end
            b_depth = 0
            body_end = -1
            
            for k in range(body_start, length):
                if clean_code[k] == '{':
                    b_depth += 1
                elif clean_code[k] == '}':
                    b_depth -= 1
                    if b_depth == 0:
                        body_end = k
                        break
            
            if body_end == -1: break # Unclosed brace?
            
            body_content = clean_code[body_start+1 : body_end]
            full_definition = clean_code[module_start_idx : body_end+1]
            
            self.modules[name] = {
                'params': self._parse_params(params_str),
                'body': body_content,
                'code': full_definition
            }
            
            # Advance cursor
            cursor = body_end + 1
    
    def _remove_comments(self, code: str) -> str:
        """Remove C-style comments from code"""
        # Remove // comments
        code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
        # Remove /* */ comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        return code

    def _extract_variables(self, code: str):
        """Extract all variable assignments"""
        # Remove comments first
        clean_code = self._remove_comments(code)
        
        # Remove all module bodies to ensure we only get top-level variables
        # (and don't extract local module variables as globals)
        for module_data in self.modules.values():
            # Use simple replace (might be slow for huge files but robust)
            # Make sure to handle potential whitespace variations if 'code' here is cleaned?
            # self.modules contains 'code' extracted from raw source.
            # clean_code has comments removed. 
            # So mod['code'] (raw) won't match clean_code exactly if it had comments.
            # Better strategy: Extract variables from MAIN BODY logic?
            pass
            
        # Strategy: We assume modules are already extracted.
        # We can mask them out.
        # But wait, self.modules[name]['code'] includes comments?
        # Let's rely on the fact that _remove_comments preserves structure mostly.
        
        # ACTUALLY: Just stripping comments helps regex.
        # To avoid module internals, we can rely on indentation? No.
        # We can implement a "strip modules" helper that respects braces.
        
        # Simplified approach: iterating regex matches is fast.
        # We just need to check if the match.start() is inside a module definition range.
        
        # Helper to map module spans
        module_spans = []
        for mod in self.modules.values():
            # Find usage of this module code in clean_code? 
            # We don't have spans.
            # Let's try removing modules from clean_code using regex again?
            pass

        # Robust fix: 
        # 1. Remove comments.
        # 2. Blank out module definitions {} blocks.
        # 3. Parse variables.
        
        # Regex to match module { ... } recursively? Hard.
        # But we already extracted modules!
        # self.modules matches in ORIGINAL code.
        
        # Let's use the 'main_body' extraction logic but for variables.
        # Create a temp string with modules removed.
        # Since we can't easily map raw modules to clean code, let's work on raw code?
        # But we need to remove comments to avoid false positives.
        
        # Combined: Remove comments from 'code', THEN Remove modules (by regex) from that.
        
        # 1. Clean comments
        working_code = self._remove_comments(code)
        
        # 2. Blank out module definitions
        # Pattern: module name(...) { ... }
        # Only works if we handle nested braces.
        # Since _extract_modules worked, we can trust it?
        # But _extract_modules worked on RAW code.
        
        # Let's verify: module definitions in `self.modules` are raw.
        # We can iterate matches of module pattern on `working_code` and strip them.
        module_pattern = r'module\s+(\w+)\s*\((.*?)\)\s*\{'
        # We need brace counting to find end.
        
        # Easier: iterate `working_code`, counting braces.
        # If at depth 0, we are at top level.
        # Only extract variables at depth 0!
        
        top_level_code = []
        depth = 0
        current_segment = []
        
        i = 0
        while i < len(working_code):
            char = working_code[i]
            if char == '{':
                if depth == 0:
                    top_level_code.append("".join(current_segment))
                    current_segment = []
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    # End of block (module or if/for).
                    # If it was a module, we skip it.
                    # If it was a global if/for, we conceptually treat it as global scope in SCAD?
                    # Yes, variables in top-level for/if are reachable?
                    # Actually SCAD variables in if() are local scope.
                    # So skipping everything in {} is CORRECT for globals!
                    pass
            elif depth == 0:
                current_segment.append(char)
            i += 1
            
        if current_segment:
            top_level_code.append("".join(current_segment))
            
        final_code = "\n".join(top_level_code)
        
        # Match: variable = value;
        # Use [^;\n] to avoid matching across lines
        # Use ^ anchor to avoid matching named parameters
        var_pattern = r'^\s*(\w+)\s*=\s*([^;\n]+);'
        
        for match in re.finditer(var_pattern, final_code, re.MULTILINE):
            name = match.group(1)
            value = match.group(2).strip()
            self.variables[name] = self._evaluate_expression(value)
    
    def _extract_main_body(self, code: str) -> str:
        """Extract code outside module definitions"""
        # Remove comments first so we can match the code stored in self.modules
        code = self._remove_comments(code)
        
        # Remove all module definitions
        for module_data in self.modules.values():
            code = code.replace(module_data['code'], '')
        
        # Remove top-level variable assignments (they're already extracted)
        # Match: variable = value; but NOT inside parentheses (function parameters)
        # Use a more careful approach: only remove lines that start with var = val;
        lines = code.split('\n')
        filtered_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip if it's a simple variable assignment (no function calls)
            if re.match(r'^\w+\s*=\s*[^()]+;(\s*//.*)?$', stripped):
                continue
            filtered_lines.append(line)
        
        return '\n'.join(filtered_lines).strip()
    
    def _parse_statements(self, code: str, parent_depth: int = 0) -> List[ASTNode]:
        """Parse a block of statements into AST nodes"""
        nodes = []
        
        # Remove comments
        code = self._remove_comments(code)
        
        # Split into top-level statements
        statements = self._split_statements(code)
        
        for stmt in statements:
            stmt = stmt.strip()
            if not stmt:
                continue
                
            node = self._parse_statement(stmt, parent_depth)
            if node:
                nodes.append(node)
        
        return nodes
    
    def _parse_statement(self, stmt: str, parent_depth: int = 0) -> ASTNode:
        """Parse a single statement into an AST node"""
        stmt = stmt.strip()
        
        # Check for primitives (cube, cylinder, sphere, etc.)
        if self._is_primitive(stmt):
            return self._parse_primitive(stmt, parent_depth)
        
        # Check for transforms (translate, rotate, scale, etc.)
        if self._is_transform(stmt):
            return self._parse_transform(stmt, parent_depth)
        
        # Check for boolean operations (union, difference, intersection)
        if self._is_boolean(stmt):
            return self._parse_boolean(stmt, parent_depth)
        
        # Check for conditionals (if)
        if stmt.startswith('if'):
            return self._parse_conditional(stmt, parent_depth)
        
        # Check for loops (for)
        if stmt.startswith('for'):
            return self._parse_loop(stmt, parent_depth)
        
        # Check for module calls
        if self._is_module_call(stmt):
            return self._parse_module_call(stmt, parent_depth)
        
        return None
    
    def _is_primitive(self, stmt: str) -> bool:
        """Check if statement is a primitive shape"""
        primitives = ['cube', 'cylinder', 'sphere', 'polyhedron', 'circle', 'square', 'polygon', 'text']
        return any(stmt.startswith(p + '(') for p in primitives)
    
    def _parse_primitive(self, stmt: str, parent_depth: int) -> ASTNode:
        """Parse a primitive shape"""
        # Extract primitive name and parameters
        match = re.match(r'(\w+)\s*\((.*?)\)', stmt, re.DOTALL)
        if not match:
            return None
        
        name = match.group(1)
        params_str = match.group(2)
        params = self._parse_function_params(params_str)
        
        return ASTNode(
            node_type=NodeType.PRIMITIVE,
            name=name,
            code=stmt,
            params=params,
            header=stmt.split(';')[0], # Primitives are self-contained
            depth=parent_depth
        )
    
    def _is_transform(self, stmt: str) -> bool:
        """Check if statement is a transform"""
        transforms = ['translate', 'rotate', 'scale', 'mirror', 'resize', 'color', 'multmatrix', 'offset']
        return any(stmt.startswith(t + '(') for t in transforms)
    
    def _parse_transform(self, stmt: str, parent_depth: int) -> ASTNode:
        """Parse a transform operation"""
        # Extract transform name, parameters, and child
        match = re.match(r'(\w+)\s*\((.*?)\)\s*(.+)', stmt, re.DOTALL)
        if not match:
            return None
        
        name = match.group(1)
        params_str = match.group(2)
        child_code = match.group(3).strip()
        
        params = self._parse_function_params(params_str)
        
        header = f"{name}({params_str})" 
        
        # Parse child statement
        children = []
        if child_code.startswith('{'):
            # Block of statements
            block = self._extract_block(child_code)
            children = self._parse_statements(block, parent_depth + 1)
        else:
            # Single statement
            child = self._parse_statement(child_code, parent_depth + 1)
            if child:
                children = [child]
        
        return ASTNode(
            node_type=NodeType.TRANSFORM,
            name=name,
            code=stmt,
            params=params,
            header=header,
            children=children,
            depth=parent_depth
        )
    
    def _is_boolean(self, stmt: str) -> bool:
        """Check if statement is a boolean operation"""
        booleans = ['union', 'difference', 'intersection', 'hull', 'minkowski']
        return any(stmt.startswith(b + '(') or stmt.startswith(b + ' ') for b in booleans)
    
    def _parse_boolean(self, stmt: str, parent_depth: int) -> ASTNode:
        """Parse a boolean operation"""
        # Match: union() { ... } or union { ... } (OpenSCAD allows both)
        # Also handle difference(), intersection(), hull(), minkowski()
        
        # Try with parentheses first
        match = re.match(r'(\w+)\s*\(\)\s*\{', stmt, re.DOTALL)
        header = ""
        if match:
            name = match.group(1)
            header = f"{name}()"
        else:
            # Try without parentheses
            match = re.match(r'(\w+)\s*\{', stmt, re.DOTALL)
            if match:
                name = match.group(1)
                header = name 
        
        if not match:
            return None
            
        name = match.group(1)
        
        # Extract the block content using brace matching
        block_start = stmt.find('{')
        if block_start == -1:
            return None
        
        children_code = self._extract_block(stmt[block_start:])
        
        # Parse children
        children = self._parse_statements(children_code, parent_depth + 1)
        
        return ASTNode(
            node_type=NodeType.BOOLEAN,
            name=name,
            code=stmt,
            header=header,
            children=children,
            depth=parent_depth
        )
    
    def _parse_conditional(self, stmt: str, parent_depth: int) -> ASTNode:
        """Parse an if statement"""
        # Extract condition and body
        match = re.match(r'if\s*\((.*?)\)\s*(.+)', stmt, re.DOTALL)
        if not match:
            return None
        
        condition = match.group(1)
        body_code = match.group(2).strip()
        
        # Evaluate condition
        condition_value = self._evaluate_expression(condition)
        
        # Only parse the branch that will execute
        children = []
        if condition_value:
            if body_code.startswith('{'):
                block = self._extract_block(body_code)
                children = self._parse_statements(block, parent_depth + 1)
            else:
                child = self._parse_statement(body_code, parent_depth + 1)
                if child:
                    children = [child]
        
        return ASTNode(
            node_type=NodeType.CONDITIONAL,
            name='if',
            code=stmt,
            params={'condition': condition, 'value': condition_value},
            children=children,
            depth=parent_depth
        )
    
    def _parse_loop(self, stmt: str, parent_depth: int) -> ASTNode:
        """Parse a for loop - unroll into separate nodes"""
        # Extract loop variable, range, and body
        match = re.match(r'for\s*\(\s*(\w+)\s*=\s*\[([^\]]+)\]\s*\)\s*(.+)', stmt, re.DOTALL)
        if not match:
            return None
        
        var_name = match.group(1)
        range_str = match.group(2)
        body_code = match.group(3).strip()
        
        # Parse range
        range_values = self._parse_range(range_str)
        
        # Unroll loop - create separate nodes for each iteration
        children = []
        for value in range_values:
            # Temporarily set loop variable
            old_value = self.variables.get(var_name)
            self.variables[var_name] = value
            
            # Parse body with this value
            if body_code.startswith('{'):
                block = self._extract_block(body_code)
                iteration_nodes = self._parse_statements(block, parent_depth + 1)
            else:
                node = self._parse_statement(body_code, parent_depth + 1)
                iteration_nodes = [node] if node else []
            
            children.extend(iteration_nodes)
            
            # Restore old value
            if old_value is not None:
                self.variables[var_name] = old_value
            else:
                self.variables.pop(var_name, None)
        
        return ASTNode(
            node_type=NodeType.LOOP,
            name='for',
            code=stmt,
            params={'variable': var_name, 'range': range_values},
            children=children,
            depth=parent_depth
        )
    
    def _is_module_call(self, stmt: str) -> bool:
        """Check if statement is a module call"""
        match = re.match(r'(\w+)\s*\(', stmt)
        if match:
            name = match.group(1)
            return name in self.modules
        return False
    
    def _parse_module_call(self, stmt: str, parent_depth: int) -> ASTNode:
        """Parse a module call - inline the module body"""
        match = re.match(r'(\w+)\s*\((.*?)\)', stmt, re.DOTALL)
        if not match:
            return None
        
        name = match.group(1)
        params_str = match.group(2)
        
        if name not in self.modules:
            return None
        
        # Parse call parameters
        call_params = self._parse_function_params(params_str)
        
        # Get module definition
        module = self.modules[name]
        module_params = module['params']  # List of parameter names from definition
        
        # Create parameter mapping: param_name -> value
        param_map = {}
        
        # Handle positional parameters
        pos_idx = 0
        for key, value in call_params.items():
            if key.startswith('_pos_'):
                # Positional parameter - map to module param by position
                if pos_idx < len(module_params):
                    param_name = module_params[pos_idx].split('=')[0].strip()
                    param_map[param_name] = value
                    pos_idx += 1
            else:
                # Named parameter
                param_map[key] = value
        
        # Substitute parameters in module body
        body = module['body']
        for param_name, param_value in param_map.items():
            # Use word boundaries + negative lookbehind to avoid matching inside function names
            # This prevents "height" from matching in "translate" or other function names
            # Pattern: not preceded by letter/underscore, param_name, not followed by letter/underscore/digit
            pattern = r'(?<![a-zA-Z_])' + re.escape(param_name) + r'(?![a-zA-Z_0-9])'
            body = re.sub(pattern, str(param_value), body)
        
        # Parse module body with substituted values
        children = self._parse_statements(body, parent_depth + 1)
        
        return ASTNode(
            node_type=NodeType.MODULE,
            name=name,
            code=stmt,
            params=call_params,
            children=children,
            dependencies={name},
            depth=parent_depth
        )
    
    def _parse_params(self, params_str: str) -> List[str]:
        """Parse module parameter list"""
        if not params_str.strip():
            return []
        return [p.strip() for p in params_str.split(',')]
    
    def _parse_function_params(self, params_str: str) -> Dict[str, Any]:
        """Parse function call parameters"""
        params = {}
        
        if not params_str.strip():
            return params
        
        # Split by comma (but not inside brackets)
        parts = self._smart_split(params_str, ',')
        
        for part in parts:
            part = part.strip()
            if '=' in part:
                # Named parameter
                key, value = part.split('=', 1)
                params[key.strip()] = self._evaluate_expression(value.strip())
            else:
                # Positional parameter (use index as key)
                params[f'_pos_{len(params)}'] = self._evaluate_expression(part)
        
        return params
    
    def _evaluate_expression(self, expr: str) -> Any:
        """Safely evaluate an expression"""
        expr = expr.strip()
        
        # Replace variables
        for var_name, var_value in self.variables.items():
            expr = expr.replace(var_name, str(var_value))
        
        # Try to evaluate as Python expression (safe subset)
        try:
            # Only allow numbers, lists, basic math
            if re.match(r'^[\d\s+\-*/().,\[\]]+$', expr):
                return eval(expr, {"__builtins__": {}}, {})
        except:
            pass
        
        # Return as string if can't evaluate
        return expr
    
    def _parse_range(self, range_str: str) -> List[Any]:
        """Parse a range specification"""
        # Handle [start:end] or [start:step:end] or [val1, val2, val3]
        range_str = range_str.strip()
        
        if ':' in range_str:
            # Range notation
            parts = [self._evaluate_expression(p) for p in range_str.split(':')]
            if len(parts) == 2:
                start, end = parts
                return list(range(int(start), int(end) + 1))
            elif len(parts) == 3:
                start, step, end = parts
                return list(range(int(start), int(end) + 1, int(step)))
        else:
            # List notation
            return [self._evaluate_expression(v) for v in range_str.split(',')]
        
        return []
    
    def _split_statements(self, code: str) -> List[str]:
        """Split code into top-level statements"""
        statements = []
        current = []
        brace_depth = 0
        paren_depth = 0
        
        for char in code:
            current.append(char)
            
            if char == '{':
                brace_depth += 1
            elif char == '}':
                brace_depth -= 1
            elif char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            elif char == ';' and brace_depth == 0 and paren_depth == 0:
                # End of statement
                statements.append(''.join(current))
                current = []
        
        # Add any remaining code
        if current:
            statements.append(''.join(current))
        
        return statements
    
    def _smart_split(self, text: str, delimiter: str) -> List[str]:
        """Split text by delimiter, respecting brackets"""
        parts = []
        current = []
        depth = 0
        
        for char in text:
            if char in '([{':
                depth += 1
            elif char in ')]}':
                depth -= 1
            elif char == delimiter and depth == 0:
                parts.append(''.join(current))
                current = []
                continue
            
            current.append(char)
        
        if current:
            parts.append(''.join(current))
        
        return parts
    
    def _extract_block(self, code: str) -> str:
        """Extract code inside braces"""
        if code.startswith('{'):
            # Find matching closing brace
            depth = 0
            for i, char in enumerate(code):
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        return code[1:i]
        return code
    
    def _resolve_dependencies(self, nodes: List[ASTNode]):
        """Resolve dependencies and calculate depth for each node"""
        # Build dependency graph
        for node in nodes:
            self._collect_dependencies(node)
        
        # Calculate depth (topological sort)
        self._calculate_depth(nodes)
    
    def _collect_dependencies(self, node: ASTNode):
        """Recursively collect dependencies for a node"""
        # Module calls depend on the module
        if node.node_type == NodeType.MODULE:
            node.dependencies.add(node.name)
        
        # Recursively collect from children
        for child in node.children:
            self._collect_dependencies(child)
            node.dependencies.update(child.dependencies)
    
    def _calculate_depth(self, nodes: List[ASTNode]):
        """Calculate depth for topological ordering"""
        def set_depth(node: ASTNode, current_depth: int):
            node.depth = max(node.depth, current_depth)
            for child in node.children:
                set_depth(child, current_depth + 1)
        
        for node in nodes:
            set_depth(node, 0)
    
    def flatten_ast(self, nodes: List[ASTNode]) -> List[ASTNode]:
        """Flatten AST into a list of all nodes (for parallel execution)"""
        flat = []
        
        def flatten(node: ASTNode):
            flat.append(node)
            for child in node.children:
                flatten(child)
        
        for node in nodes:
            flatten(node)
        
        return flat
