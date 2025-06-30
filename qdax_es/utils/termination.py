import chex
from flax import struct
    
import jax.numpy as jnp
from evosax.core.restart import cma_cond

@struct.dataclass
class RestartParams:
    tol_x: float = 1e-12  # Tolerance for stopping criterion on x
    tol_x_up: float = 1e4  # Tolerance for stopping criterion on x upper bound
    tol_condition_C: float = 1e14  # Tolerance for stopping criterion on condition number of C

CMA_PARAMS = RestartParams()

def cma_criterion(
    state: chex.ArrayTree
) -> bool:
    """Termination criterion specific to CMA-ES strategy. Default tolerances:
    tol_x - 1e-12 
    tol_x_up - 1e4
    tol_condition_C - 1e14
    """
    return cma_cond(
        population=None,
        fitness=None,
        state=state,
        params=None,
        restart_state=None, 
        restart_params=CMA_PARAMS,
    )