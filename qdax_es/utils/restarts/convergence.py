import jax.numpy as jnp
from typing import Optional

from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
)
from qdax.core.emitters.emitter import Emitter

from qdax_es.core.containers.novelty_archive import NoveltyArchive


from qdax_es.utils.restarts.base import FixedGens

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