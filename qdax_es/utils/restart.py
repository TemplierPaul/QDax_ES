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


class ConvergenceRestarter(FixedGens):
    """
    Restart when the ES has converged
    """
    def __init__(self, min_score_spread, max_gens=jnp.inf, min_gens=0):
        super().__init__(max_gens=max_gens)
        self.min_score_spread = min_score_spread
        self.min_gens = min_gens

    def restart_criteria(self, emitter_state, scores):
        """
        Check if the restart condition is met.
        """
        max_gens = emitter_state.restart_state.generations >= self.max_gens
        min_gens = emitter_state.restart_state.generations >= self.min_gens
        converged = jnp.max(scores) - jnp.min(scores) < self.min_score_spread
        return jnp.logical_or(max_gens, jnp.logical_and(min_gens, converged))


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
    
class CombinedRestartState(PyTreeNode):
    restart_states: RestartState

class CombinedRestarter:
    def __init__(self, restarters):
        self.restarters = restarters

    def init(self):
        return CombinedRestartState(
            restart_states=[r.init() for r in self.restarters]
        )

    def update(self, emitter_state, scores):
        """
        Update the restart state.
        """
        restart_states = [
            r.update(emitter_state, scores)
            for r in self.restarters
        ]
        return emitter_state.replace(restart_states=restart_states)
    
    def restart_criteria(self, emitter_state, scores):
        """
        Check if the restart condition is met.
        """
        return jnp.any(
            jnp.array(
                [
                    r.restart_criteria(emitter_state, scores)
                    for r in self.restarters
                ]
            )
        )
    