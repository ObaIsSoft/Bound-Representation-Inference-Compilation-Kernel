"""
SymPy Provider - Symbolic Mathematics

Wraps SymPy for symbolic equation solving.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class SymPyProvider:
    """
    Provider for symbolic mathematics using SymPy library.
    """
    
    def __init__(self):
        """Initialize the provider and import SymPy"""
        self.available = self._check_availability()
        
        if self.available:
            import sympy as sp
            self.sp = sp
            logger.info("SymPyProvider initialized")
        else:
            logger.warning("SymPy not available")
    
    def _check_availability(self) -> bool:
        """
        Check if SymPy is available.
        
        Returns:
            True if library is available
        """
        try:
            import sympy
            return True
        except ImportError:
            return False
    
    def solve(self, equation_str: str, solve_for: str, **known_values) -> Any:
        """
        Solve a symbolic equation.
        
        Args:
            equation_str: Equation as string (e.g., "F = m * a")
            solve_for: Variable to solve for (e.g., "a")
            **known_values: Known variable values
        
        Returns:
            Solution for the specified variable
        """
        if not self.available:
            raise RuntimeError("SymPy not available")
        
        # Parse equation
        lhs, rhs = equation_str.split("=")
        lhs = lhs.strip()
        rhs = rhs.strip()
        
        # Create symbols
        symbols = self.sp.symbols(f"{lhs} {rhs}")
        
        # Create equation
        eq = self.sp.Eq(self.sp.sympify(lhs), self.sp.sympify(rhs))
        
        # Substitute known values
        for var, val in known_values.items():
            eq = eq.subs(var, val)
        
        # Solve
        solution = self.sp.solve(eq, solve_for)
        
        return solution[0] if solution else None
    
    def differentiate(self, expression_str: str, variable: str) -> str:
        """
        Differentiate an expression symbolically.
        
        Args:
            expression_str: Expression as string
            variable: Variable to differentiate with respect to
        
        Returns:
            Derivative as string
        """
        if not self.available:
            raise RuntimeError("SymPy not available")
        
        expr = self.sp.sympify(expression_str)
        var = self.sp.Symbol(variable)
        derivative = self.sp.diff(expr, var)
        
        return str(derivative)
    
    def integrate(self, expression_str: str, variable: str) -> str:
        """
        Integrate an expression symbolically.
        
        Args:
            expression_str: Expression as string
            variable: Variable to integrate with respect to
        
        Returns:
            Integral as string
        """
        if not self.available:
            raise RuntimeError("SymPy not available")
        
        expr = self.sp.sympify(expression_str)
        var = self.sp.Symbol(variable)
        integral = self.sp.integrate(expr, var)
        
        return str(integral)
