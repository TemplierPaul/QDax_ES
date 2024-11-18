import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode
from typing import Optional, Tuple, Callable


from qdax.custom_types import Centroid, Descriptor, ExtraScores, Fitness, Genotype, RNGKey
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    get_cells_indices,
)
from qdax.core.emitters.emitter import Emitter, EmitterState

from qdax_es.core.containers.novelty_archive import NoveltyArchive
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
    
    def restart_criteria(
        self,
        emitter_state: Emitter,
        scores: Fitness,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores],
        novelty_archive: NoveltyArchive = None
        ):
        """
        Check if the restart condition is met.
        """
        return False

    def reset(self, emitter_state):
        return emitter_state.replace(restart_state=self.init())
    

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

    def restart_criteria(
        self,
        emitter_state: Emitter,
        scores: Fitness,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores],
        novelty_archive: NoveltyArchive = None
        ):
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

    def restart_criteria(
        self,
        emitter_state: Emitter,
        scores: Fitness,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores],
        novelty_archive: NoveltyArchive = None
        ):
        """
        Check if the restart condition is met.
        """
        # jax.debug.print("diff: {}", jnp.max(scores) - jnp.min(scores))
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

    def restart_criteria(
        self,
        emitter_state: Emitter,
        scores: Fitness,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores],
        novelty_archive: NoveltyArchive = None
        ):
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
    
class StaleRestartState(PyTreeNode):
    staleness: int = 0

class MEMESStaleRestarter(DummyRestarter):
    def __init__(self, Smax=32):
        self.Smax = Smax

    def init(self):
        return StaleRestartState(staleness=0)

    def restart_criteria(
        self,
        emitter_state: Emitter,
        scores: Fitness,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores],
        novelty_archive: NoveltyArchive = None
        ):
        """
        Check if the restart condition is met.
        """
        staleness = emitter_state.restart_state.staleness
        return jnp.where(staleness >= self.Smax, True, False)
    
    def update(self, emitter_state, scores):
        return emitter_state

    def update_staleness(self, emitter_state, added):
        """
        Update the restart state.
        """
        # Check reset
        staleness = emitter_state.restart_state.staleness
        staleness = jax.numpy.where(staleness > self.Smax, 0, staleness)
        # Add 1 to the generations
        staleness = jnp.where(
            added,
            0,
            staleness + 1
        )
        restart_state = emitter_state.restart_state.replace(staleness=staleness)
        return emitter_state.replace(restart_state=restart_state)
    
class DualConvergenceRestarter(FixedGens):
    """
    Restart when both the fitness and the behavior have converged
    """
    def __init__(self, min_score_spread, min_bd_spread, max_gens=jnp.inf, min_gens=0):
        super().__init__(max_gens=max_gens)
        self.min_score_spread = min_score_spread
        self.min_bd_spread = min_bd_spread
        self.min_gens = min_gens

    def restart_criteria(
        self,
        emitter_state: Emitter,
        scores: Fitness,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores],
        novelty_archive: NoveltyArchive = None
        ):
        """
        Check if the restart condition is met.
        """
        max_gens = emitter_state.restart_state.generations >= self.max_gens
        min_gens = emitter_state.restart_state.generations >= self.min_gens
        converged_fit = jnp.max(fitnesses) - jnp.min(fitnesses) < self.min_score_spread
        # check descriptor spread: get min/max for all dimensions
        min_bd = jnp.min(descriptors, axis=0)
        max_bd = jnp.max(descriptors, axis=0)
        bd_spread = max_bd - min_bd
        converged_bd = jnp.all(bd_spread < self.min_bd_spread)
        converged = jnp.logical_and(converged_fit, converged_bd)
        return jnp.logical_or(max_gens, jnp.logical_and(min_gens, converged))