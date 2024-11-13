from __future__ import annotations

from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire

from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.custom_types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)

import jax.numpy as jnp
from qdax.core.map_elites import MAPElites

class CustomMAPElites(MAPElites):
    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        emitter: Emitter,
        metrics_function: Callable[[MapElitesRepertoire], Metrics],
        repertoire_init=MapElitesRepertoire.init,
    ) -> None:
        self._scoring_function = scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function
        self.repertoire_init = repertoire_init


    # @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        genotypes: Genotype,
        centroids: Centroid,
        random_key: RNGKey,
        repertoire_kwargs: Optional[dict] = {},
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState], RNGKey]:
        """
        Initialize a Map-Elites repertoire with an initial population of genotypes.
        Requires the definition of centroids that can be computed with any method
        such as CVT or Euclidean mapping.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            centroids: tesselation centroids of shape (batch_size, num_descriptors)
            random_key: a random key used for stochastic operations.

        Returns:
            An initialized MAP-Elite repertoire with the initial state of the emitter,
            and a random key.
        """
        # score initial genotypes
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            genotypes, random_key
        )
        with jax.disable_jit():
            # init the repertoire
            repertoire = self.repertoire_init(
                genotypes=genotypes,
                fitnesses=fitnesses,
                descriptors=descriptors,
                centroids=centroids,
                extra_scores=extra_scores,
                **repertoire_kwargs,
            )

            # get initial state of the emitter
            emitter_state, random_key = self._emitter.init(
                random_key=random_key,
                repertoire=repertoire,
                genotypes=genotypes,
                fitnesses=fitnesses,
                descriptors=descriptors,
                extra_scores=extra_scores,
            )

            # update emitter state
            emitter_state = self._emitter.state_update(
                emitter_state=emitter_state,
                repertoire=repertoire,
                genotypes=genotypes,
                fitnesses=fitnesses,
                descriptors=descriptors,
                extra_scores=extra_scores,
            )

        return repertoire, emitter_state, random_key
