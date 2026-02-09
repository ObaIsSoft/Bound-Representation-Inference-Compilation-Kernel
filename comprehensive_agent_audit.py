#!/usr/bin/env python3
"""
Comprehensive Agent Audit - Real-World Production Readiness
Checks all agents for:
1. Hardcoded values that should be database-driven
2. Mock/fallback code that should be removed
3. Missing error handling
4. Interface inconsistencies
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, field

@dataclass
class AgentAuditResult:
    name: str
    path: str
    issues: List[Dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    has_database_calls: bool = False
    has_hardcoded_values: bool = False
    has_fallback_code: bool = False
    has_proper_error_handling: bool = True
    agent_class_found: bool = False
    
def analyze_file(filepath: Path) -> AgentAuditResult:
    """Analyze a single agent file for production readiness issues."""
    result = AgentAuditResult(
        name=filepath.stem,
        path=str(filepath)
    )
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            tree = ast.parse(content)
    except Exception as e:
        result.issues.append({"type": "parse_error", "message": str(e)})
        return result
    
    # Track hardcoded values
    hardcoded_patterns = [
        ('price', ['$2.50', '$30.00', '$100', '2.5', '30.0', '100.0', 
                   '50.0', '75.0', '5000', '2000', '3000']),
        ('dimensions', ['5000', '1000', '100', '10.0']),
        ('time', ['14 days', '7 days', '30 days', '24 hours']),
        ('material', ['"Titanium"', '"Aluminum"', '"Steel"', 
                      "'Titanium'", "'Aluminum'", "'Steel'"]),
    ]
    
    # Check for fallback/mock code patterns
    fallback_patterns = [
        'except Exception',
        'fallback',
        'mock',
        'default',
        'placeholder',
        '# TODO',
        'pass  #',
        'return {',  # Simple dict returns often indicate hardcoded data
    ]
    
    # Database interaction patterns
    db_patterns = [
        'supabase',
        'sqlite',
        'database',
        '.execute(',
        'cursor()',
        'connection',
    ]
    
    has_try_except = False
    has_generic_except = False
    
    for node in ast.walk(tree):
        # Check for class definitions (agent classes)
        if isinstance(node, ast.ClassDef):
            if 'Agent' in node.name or 'Oracle' in node.name or 'Critic' in node.name:
                result.agent_class_found = True
                
        # Check for try/except blocks
        if isinstance(node, ast.Try):
            has_try_except = True
            for handler in node.handlers:
                # Check for generic except:
                if handler.type is None:
                    has_generic_except = True
                    result.has_fallback_code = True
                    result.issues.append({
                        "type": "generic_except",
                        "line": getattr(handler, 'lineno', 0),
                        "message": "Bare except: clause found - may hide errors"
                    })
        
        # Check for string literals (potential hardcoded values)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            val = node.value
            # Check for hardcoded materials
            if val in ['Titanium', 'Aluminum 6061-T6', 'Steel', 'ABS', 'PLA']:
                if 'fallback' not in content[:node.col_offset].lower():
                    result.has_hardcoded_values = True
                    result.warnings.append(f"Hardcoded material: '{val}' at line {node.lineno}")
            
            # Check for hardcoded costs
            if val.startswith('$') and any(c.isdigit() for c in val):
                result.has_hardcoded_values = True
                result.warnings.append(f"Hardcoded cost: '{val}' at line {node.lineno}")
        
        # Check for numeric literals
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            val = node.value
            # Common hardcoded values
            if val in [2.5, 30.0, 50.0, 75.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]:
                result.has_hardcoded_values = True
    
    # Check content for patterns
    content_lower = content.lower()
    
    # Database usage
    for pattern in db_patterns:
        if pattern in content_lower:
            result.has_database_calls = True
            break
    
    # Fallback code detection
    fallback_indicators = [
        'data["materials"] = {',  # Dict assignment in except block
        'data["cost"] = {',
        'data["manufacturing"] = {',
        'except exception as e:',
        'logger.warning(f"',
        'logger.error(f"',
    ]
    for indicator in fallback_indicators:
        if indicator in content:
            result.has_fallback_code = True
            break
    
    # Error handling assessment
    if not has_try_except and result.agent_class_found:
        result.has_proper_error_handling = False
        result.warnings.append("No try/except blocks found - may lack error handling")
    
    if has_generic_except:
        result.has_proper_error_handling = False
    
    return result

def main():
    """Run comprehensive audit on all agents."""
    agents_dir = Path("/Users/obafemi/Documents/dev/brick/backend/agents")
    
    results: List[AgentAuditResult] = []
    
    # Find all agent Python files (exclude __init__.py and adapters for now)
    agent_files = [
        f for f in agents_dir.rglob("*.py")
        if f.name != "__init__.py" 
        and "adapters" not in str(f)
        and "__pycache__" not in str(f)
    ]
    
    print(f"=" * 80)
    print(f"COMPREHENSIVE AGENT AUDIT - {len(agent_files)} Agent Files")
    print(f"=" * 80)
    print()
    
    for filepath in sorted(agent_files):
        result = analyze_file(filepath)
        results.append(result)
    
    # Categorize results
    production_ready = []
    needs_attention = []
    critical_issues = []
    
    for r in results:
        if r.has_fallback_code or r.has_hardcoded_values or r.issues:
            if r.issues and any(i["type"] in ["generic_except", "parse_error"] for i in r.issues):
                critical_issues.append(r)
            else:
                needs_attention.append(r)
        else:
            production_ready.append(r)
    
    # Print summary
    print(f"📊 SUMMARY")
    print(f"-" * 40)
    print(f"✅ Production Ready:     {len(production_ready)}")
    print(f"⚠️  Needs Attention:      {len(needs_attention)}")
    print(f"🚨 Critical Issues:      {len(critical_issues)}")
    print(f"   Total Agents:         {len(results)}")
    print()
    
    # Print production ready agents
    if production_ready:
        print(f"✅ PRODUCTION READY AGENTS ({len(production_ready)})")
        print(f"-" * 40)
        for r in production_ready[:10]:  # Limit output
            print(f"   ✓ {r.name}")
        if len(production_ready) > 10:
            print(f"   ... and {len(production_ready) - 10} more")
        print()
    
    # Print agents needing attention
    if needs_attention:
        print(f"⚠️  AGENTS NEEDING ATTENTION ({len(needs_attention)})")
        print(f"-" * 40)
        for r in needs_attention:
            issues = []
            if r.has_hardcoded_values:
                issues.append("hardcoded values")
            if r.has_fallback_code:
                issues.append("fallback code")
            if not r.has_database_calls:
                issues.append("no DB calls")
            print(f"   ⚠️  {r.name}: {', '.join(issues)}")
            for w in r.warnings[:2]:
                print(f"      - {w}")
        print()
    
    # Print critical issues
    if critical_issues:
        print(f"🚨 CRITICAL ISSUES ({len(critical_issues)})")
        print(f"-" * 40)
        for r in critical_issues:
            print(f"   🚨 {r.name}")
            for issue in r.issues:
                print(f"      [{issue['type']}] {issue.get('message', '')}")
        print()
    
    # Statistics
    print(f"📈 STATISTICS")
    print(f"-" * 40)
    total_with_db = sum(1 for r in results if r.has_database_calls)
    total_with_hardcoded = sum(1 for r in results if r.has_hardcoded_values)
    total_with_fallback = sum(1 for r in results if r.has_fallback_code)
    total_with_classes = sum(1 for r in results if r.agent_class_found)
    
    print(f"   Agents with DB calls:      {total_with_db}/{len(results)} ({100*total_with_db//len(results)}%)")
    print(f"   Agents with hardcoded:     {total_with_hardcoded}/{len(results)} ({100*total_with_hardcoded//len(results)}%)")
    print(f"   Agents with fallback:      {total_with_fallback}/{len(results)} ({100*total_with_fallback//len(results)}%)")
    print(f"   Files with agent classes:  {total_with_classes}/{len(results)} ({100*total_with_classes//len(results)}%)")
    print()
    
    return results

if __name__ == "__main__":
    results = main()
