"""
SciPy Provider - Numerical Methods

Wraps SciPy for numerical integration and optimization.
"""

import logging
import numpy as np
from typing import Callable, Tuple, Any

logger = logging.getLogger(__name__)


class SciPyProvider:
    """
    Provider for numerical methods using SciPy library.
    """
    
    def __init__(self):
        """Initialize the provider and import SciPy"""
        self.available = self._check_availability()
        
        if self.available:
            from scipy import integrate, optimize
            self.integrate = integrate
            self.optimize = optimize
            logger.info("SciPyProvider initialized")
        else:
            logger.warning("SciPy not available")
    
    def _check_availability(self) -> bool:
        """
        Check if SciPy is available.
        
        Returns:
            True if library is available
        """
        try:
            import scipy
            return True
        except ImportError:
            return False
    
    def integrate_ode(
        self,
        func: Callable,
        initial_state: np.ndarray,
        t_span: Tuple[float, float],
        t_eval: np.ndarray = None
    ) -> Any:
        """
        Integrate an ODE system using solve_ivp.
        
        Args:
            func: ODE function (dy/dt = func(t, y))
            initial_state: Initial conditions
            t_span: Time span (t_start, t_end)
            t_eval: Times at which to store solution
        
        Returns:
            Integration result
        """
        if not self.available:
            raise RuntimeError("SciPy not available")
        
        result = self.integrate.solve_ivp(
            func,
            t_span,
            initial_state,
            t_eval=t_eval,
            method='RK45'  # Runge-Kutta 4/5
        )
        
        return result
    
    def integrate_function(
        self,
        func: Callable,
        a: float,
        b: float
    ) -> Tuple[float, float]:
        """
        Numerically integrate a function.
        
        Args:
            func: Function to integrate
            a: Lower limit
            b: Upper limit
        
        Returns:
            (integral_value, error_estimate)
        """
        if not self.available:
            raise RuntimeError("SciPy not available")
        
        result, error = self.integrate.quad(func, a, b)
        return result, error
    
    def minimize(
        self,
        func: Callable,
        x0: np.ndarray,
        method: str = 'BFGS'
    ) -> Any:
        """
        Minimize a function.
        
        Args:
            func: Objective function
            x0: Initial guess
            method: Optimization method
        
        Returns:
            Optimization result
        """
        if not self.available:
            raise RuntimeError("SciPy not available")
        
        result = self.optimize.minimize(func, x0, method=method)
        return result
