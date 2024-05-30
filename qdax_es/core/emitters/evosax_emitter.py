from __future__ import annotations

from functools import partial
from typing import Optional, Tuple, Callable

import jax
import jax.numpy as jnp

from qdax.types import Centroid, Descriptor, ExtraScores, Fitness, Genotype, RNGKey
from qdax.core.emitters.emitter import Emitter, EmitterState


from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    get_cells_indices,
)

from qdax_es.core.emitters.evosax_base_emitter import EvosaxEmitter, EvosaxEmitterState


class EvosaxEmitterAll(EvosaxEmitter):
    """
    Emit the whole population of the ES, like CMA-ME.
    """
    @partial(jax.jit, static_argnames=("self",))
    def emit(
            self,
        repertoire: Optional[MapElitesRepertoire],
        emitter_state: EvosaxEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """
        Generate solutions to be evaluated and added to the archive.
        """
        
        offspring, random_key = self.es_ask(emitter_state, random_key)

        return offspring, random_key
    
    @partial(jax.jit, static_argnames=("self",))
    def state_update(
        self,
        emitter_state: EvosaxEmitterState,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores] = None,
    ) -> Optional[EmitterState]:
        """
        Update the state of the emitter, like ME-ES.
        """

        scores = self.ranking_criteria(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        emitter_state = self.es_tell(emitter_state, genotypes, scores)
        
        # Updating novelty archive
        novelty_archive = emitter_state.novelty_archive.update(descriptors)
        emitter_state = emitter_state.replace(novelty_archive=novelty_archive)
        
        # TODO: Add restart
        emitter_state = self.restarter.update(
            emitter_state,
            scores
        )
        restart_bool = self.restarter.restart_criteria(emitter_state)

        emitter_state = jax.lax.cond(
            restart_bool,
            lambda x: self.restart(repertoire=repertoire, emitter_state=x),
            lambda x: x,
            emitter_state
        )
        
        random_key, subkey = jax.random.split(emitter_state.random_key)
        emitter_state = self._post_update_emitter_state(emitter_state, subkey, repertoire)

        return emitter_state

class EvosaxEmitterCenter(EvosaxEmitter):
    """
    Only emit the center of the ES.
    """
    @partial(jax.jit, static_argnames=("self",))
    def emit(
        self,
        repertoire: Optional[MapElitesRepertoire],
        emitter_state: EvosaxEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """
        Emit the center of the ES.
        """

        es_center = emitter_state.es_state.strategy_state.mean
        offspring = self.reshaper.unflatten(es_center)

        return offspring, random_key
    
    @partial(jax.jit, static_argnames=("self",))
    def state_update(
        self,
        emitter_state: EvosaxEmitterState,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores] = None,
    ) -> Optional[EmitterState]:
        """
        Do an ES step.
        """
        random_key, subkey = jax.random.split(emitter_state.random_key)

        offspring = self.es_ask(
            emitter_state=emitter_state, 
            random_key=random_key
            )
        
        fitnesses, descriptors, extra_scores, random_key = self.scoring_fn(
            genotypes=offspring, 
            random_key=subkey
            )
        
        scores = self.ranking_criteria(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        emitter_state = self.es_tell(
            emitter_state,
            offspring, 
            scores
            )
        
        # Updating novelty archive
        novelty_archive = emitter_state.novelty_archive.update(descriptors)
        emitter_state = emitter_state.replace(novelty_archive=novelty_archive)

        # TODO: add restart

        emitter_state = self._post_update_emitter_state(emitter_state, subkey, repertoire)

        return emitter_state