from __future__ import annotations

from functools import partial
from typing import Optional, Tuple, Callable

import jax
import jax.numpy as jnp

from qdax.types import Centroid, Descriptor, ExtraScores, Fitness, Genotype, RNGKey

from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    get_cells_indices,
)

from qdax_es.core.emitters.evosax_emitter import EvosaxEmitterCenter

from qdax_es.core.emitters.evosax_base_emitter import EvosaxEmitterState

class MEESEmitterState(EvosaxEmitterState):
    """
    State of the ME-ES emitter.
    """
    explore_exploit: int = 0 # 0 for explore, 1 for exploit

class MEESEmitter(EvosaxEmitterCenter):
    """
    ME-ES emitter.
    """
    def __init__(
        self,
        batch_size: int,
        centroids: Centroid,
        es_hp = {},
        es_type="CMA_ES",
        ns_es=False,
        novelty_archive_size=1,
        scoring_fn: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ]=None,
    ):
        """
        Initialize the ES emitter.
        """
        super().__init__(
            batch_size=batch_size,
            centroids=centroids,
            es_hp=es_hp,
            es_type=es_type,
            ns_es=ns_es,
            novelty_archive_size=novelty_archive_size,
            scoring_fn=scoring_fn,
        )

        self.ranking_criteria = self._combined_criteria

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def init(
        self, init_genotypes: Genotype, random_key: RNGKey,       
    ):
        state = super().init(init_genotypes, random_key)
        return MEESEmitterState(
            **state,
            explore_exploit=0,
        )

    def _combined_criteria(
        self,
        emitter_state: EvosaxEmitterState,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores],
    ) -> jnp.ndarray:
        """
        NS-ES novelty criteria.
        """

        novelty = emitter_state.novelty_archive.novelty(
                    descriptors, self._config.novelty_nearest_neighbors
                )
        
        # Combine novelty and fitness: ratio = 0 for novelty, 1 for fitness
        ratio = emitter_state.explore_exploit

        scores = fitnesses * ratio + novelty * (1 - ratio)

        return scores