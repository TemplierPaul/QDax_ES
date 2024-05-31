import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode
from qdax_es.utils.termination import cma_criterion


class RestartState(PyTreeNode):
    generations: int = 0


class DummyRestarter:
    def init(self):
        return RestartState(generations=0)

    def update(self, emitter_state, scores):
        """
        Update the restart state.
        """
        # Add 1 to the generations
        generations = emitter_state.restart_state.generations + 1
        restart_state = emitter_state.restart_state.replace(generations=generations)
        return emitter_state.replace(restart_state=restart_state)
    
    def restart_criteria(self, emitter_state):
        """
        Check if the restart condition is met.
        """
        return False
    
class FixedGens(DummyRestarter):
    """
    Restart every max_gens generations.
    """
    def __init__(self, max_gens):
        self.max_gens = max_gens
        
    def init(self):
        return RestartState(generations=0)
    
    def update(self, emitter_state, scores):
        """
        Update the restart state.
        """
        # Add 1 to the generations
        generations = emitter_state.restart_state.generations + 1
        generations = jax.numpy.where(generations > self.max_gens, 0, generations)
        restart_state = emitter_state.restart_state.replace(generations=generations)
        return emitter_state.replace(restart_state=restart_state)

    def restart_criteria(self, emitter_state, scores):
        """
        Check if the restart condition is met.
        """
        gens = emitter_state.restart_state.generations
        
        return jax.numpy.where(gens >= self.max_gens, True, False)
        


class CMARestarter(FixedGens):
    """
    Restart when the ES has converged
    """
    def __init__(
            self, 
            min_spread=1e-12, 
            use_cma_criterion=False,
            min_gens=0,
            max_gens=jnp.inf,
            ):
        self.min_spread = min_spread
        self.cma_criterion = lambda s: False
        if use_cma_criterion:
            self.cma_criterion = cma_criterion
        self.min_gens = min_gens
        self.max_gens = max_gens

    def restart_criteria(self, emitter_state, scores):
        """
        Check if the restart condition is met.
        """
        # Stop if min/max fitness spread of recent generation is below threshold
        spread = jnp.max(scores) - jnp.min(scores)
        spread_restart = jnp.where(spread < self.min_spread, True, False)

        cma_restart = self.cma_criterion(emitter_state.es_state)

        max_gens = emitter_state.restart_state.generations >= self.max_gens

        aggregated = jnp.logical_or(spread_restart, cma_restart)
        aggregated = jnp.logical_or(aggregated, max_gens)

        return aggregated
