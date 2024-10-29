from __future__ import annotations

from functools import partial
from typing import Optional, Tuple, Callable

import jax
import jax.numpy as jnp

from qdax.custom_types import Centroid, Descriptor, ExtraScores, Fitness, Genotype, RNGKey

from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    get_cells_indices,
)
from qdax.core.emitters.cma_pool_emitter import CMAPoolEmitter

from qdax_es.core.containers.novelty_archive import NoveltyArchive
from qdax_es.core.emitters.evosax_emitter import EvosaxEmitterAll

from qdax_es.core.emitters.evosax_base_emitter import EvosaxEmitterState
from qdax_es.utils.restart import RestartState, CMARestarter

from qdax_es.utils.termination import cma_criterion
from qdax.core.emitters.emitter import Emitter, EmitterState

class CMAMERestarter(CMARestarter):
    """
    Restart when the ES has converged
    """
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

        neg_improvement = jnp.all(scores < 0)
        more_than_min = emitter_state.restart_state.generations >= self.min_gens
        neg_improvement = jnp.logical_and(neg_improvement, more_than_min)
        
        cma_restart = super().restart_criteria(
            emitter_state=emitter_state,
            scores=scores,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
            novelty_archive=novelty_archive,)

        return jnp.logical_or(neg_improvement, cma_restart)


class CMAMEEmitter(EvosaxEmitterAll):
    """
    CMA-ME emitter.
    """
    def __init__(
        self,
        centroids: Centroid,
        emitter_type: str = "imp",
        es_hp = {},
        es_type="CMA_ES",
        scoring_fn: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ]=None,
        restarter = None,
    ):
        """
        Initialize the ES emitter.
        """
        if restarter is None:
            use_cma_criterion = es_type == "CMA_ES"
            restarter = CMAMERestarter(
                use_cma_criterion=use_cma_criterion
            )
            if use_cma_criterion:
                print("Using CMA-ES criterion for restart")

        super().__init__(
            centroids=centroids,
            es_hp=es_hp,
            es_type=es_type,
            ns_es=False,
            novelty_archive_size=0,
            scoring_fn=scoring_fn,
            restarter=restarter,
        )

        if emitter_type == "imp":
            self.ranking_criteria = self._cmame_criteria
        else: 
            raise NotImplementedError(f"Unknown emitter type: {emitter_type}. Supported types are: 'imp'")
        self.restart = self._restart_repertoire


    def _cmame_criteria(self,
        emitter_state: EvosaxEmitterState,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores],
        novelty_archive: NoveltyArchive
    ) -> jnp.ndarray:
        """
        Default: Improvement emitter
        """
        return self._improvement_criteria(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

class CMAMEPoolEmitter(CMAPoolEmitter):
    @property
    def evals_per_gen(self):
        """
        Evaluate the population in the main loop for 1 emitter state
        """
        return self._emitter.evals_per_gen