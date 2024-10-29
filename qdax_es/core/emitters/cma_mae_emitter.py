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

from qdax_es.core.emitters.evosax_emitter import EvosaxEmitterAll

from qdax_es.core.emitters.evosax_base_emitter import EvosaxEmitterState
from qdax_es.utils.restart import RestartState, FixedGens

from qdax_es.core.containers.mae_repertoire import MAERepertoire
from qdax_es.core.emitters.cma_me_emitter import CMAMEEmitter

# class EvosaxCMAAnnealingEmitterState

class CMAMEAnnealingEmitter(CMAMEEmitter):
    """
    CMA-Map-annealing emitter.
    In the state, uses previous_fitnesses as previous thresholds.
    """
    def __init__(
        self,
        centroids: Centroid,
        es_hp = {},
        es_type="CMA_ES",
        scoring_fn: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ]=None,
        restarter = None,
        min_threshold= None,
    ):
        self.min_threshold = min_threshold if min_threshold is not None else -jnp.inf
        
        super().__init__(
            centroids=centroids,
            es_hp=es_hp,
            es_type=es_type,
            scoring_fn=scoring_fn,
            restarter=restarter,
        )

    def _post_update_emitter_state(
        self, emitter_state:EvosaxEmitterState, random_key: RNGKey, repertoire: MAERepertoire
    ) -> EvosaxEmitterState:
        return emitter_state.replace(
            random_key=random_key, previous_fitnesses=repertoire.thresholds
        )

    # @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        random_key: RNGKey,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> Tuple[EvosaxEmitterState, RNGKey]:
        
        emitter_state, random_key = super().init(
            random_key=random_key,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )
        
        default_thresholds = jnp.ones(self._centroids.shape[0]) * self.min_threshold
        emitter_state = emitter_state.replace(previous_fitnesses=default_thresholds)

        return emitter_state, random_key
    
    