import jax
from flax.struct import PyTreeNode

class RestartState(PyTreeNode):
    generations: int = 0


class DummyRestarter:
    @classmethod
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
        
    @classmethod
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
        