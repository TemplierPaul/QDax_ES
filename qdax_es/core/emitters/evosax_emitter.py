from __future__ import annotations

from functools import partial
from typing import Optional, Tuple, Callable

import jax
import jax.numpy as jnp

from qdax.custom_types import Centroid, Descriptor, ExtraScores, Fitness, Genotype, RNGKey
from qdax.core.emitters.emitter import Emitter, EmitterState


from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    get_cells_indices,
)

from qdax_es.core.emitters.evosax_base_emitter import EvosaxEmitter, EvosaxEmitterState
from qdax_es.core.containers.novelty_archive import NoveltyArchive


class EvosaxEmitterAll(EvosaxEmitter):
    """
    Emit the whole population of the ES, like CMA-ME.
    """
    @property
    def batch_size(self) -> int:
        """
        Returns:
            the batch size emitted by the emitter.
        """
        return self._batch_size
    
    @property
    def evals_per_gen(self):
        """
        Evaluate the population in the main loop
        """
        return self._batch_size

    # @partial(jax.jit, static_argnames=("self",))
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

        return offspring, {}, random_key
    
    # @partial(jax.jit, static_argnames=("self",))
    # def state_update(
    #     self,
    #     emitter_state: EvosaxEmitterState,
    #     repertoire: MapElitesRepertoire,
    #     genotypes: Genotype,
    #     fitnesses: Fitness,
    #     descriptors: Descriptor,
    #     extra_scores: Optional[ExtraScores] = None,
    # ) -> Optional[EvosaxEmitterState]:
        
    #     """
    #     Update the state of the emitter.
    #     """

    #     scores = self.ranking_criteria(
    #         emitter_state=emitter_state,
    #         repertoire=repertoire,
    #         genotypes=genotypes,
    #         fitnesses=fitnesses,
    #         descriptors=descriptors,
    #         extra_scores=extra_scores,
    #         novelty_archive=emitter_state.novelty_archive,
    #     )

    #     emitter_state = self.es_tell(emitter_state, genotypes, scores)
        
    #     # Updating novelty archive
    #     novelty_archive = emitter_state.novelty_archive.update(descriptors)
    #     emitter_state = emitter_state.replace(novelty_archive=novelty_archive)
        
    #     emitter_state = self.restarter.update(
    #         emitter_state,
    #         scores
    #     )
    #     restart_bool = self.restarter.restart_criteria(
    #         emitter_state=emitter_state,
    #         scores=scores,
    #         repertoire=repertoire,
    #         genotypes=genotypes,
    #         fitnesses=fitnesses,
    #         descriptors=descriptors,
    #         extra_scores=extra_scores,
    #         novelty_archive=emitter_state.novelty_archive,
    #     )

    #     emitter_state = jax.lax.cond(
    #         restart_bool,
    #         lambda x: self.restarter.reset(x),
    #         lambda x: x,
    #         emitter_state
    #     )

    #     emitter_state = jax.lax.cond(
    #         restart_bool,
    #         lambda x: self.restart(repertoire=repertoire, emitter_state=x),
    #         lambda x: x,
    #         emitter_state
    #     )

    #     random_key, subkey = jax.random.split(emitter_state.random_key)
    #     emitter_state = self._post_update_emitter_state(emitter_state, subkey, repertoire)

    #     return emitter_state
    
    # @partial(jax.jit, static_argnames=("self",))
    def state_update(
        self,
        emitter_state: EvosaxEmitterState,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores] = None,
    ) -> Optional[EvosaxEmitterState]:
        
        emitter_state, restart_bool = self.start_state_update(
            emitter_state,
            repertoire,
            genotypes,
            fitnesses,
            descriptors,
            extra_scores,
        )

        emitter_state = self.finish_state_update(
            emitter_state,
            repertoire,
            restart_bool,
        )

        return emitter_state
    
    # @partial(jax.jit, static_argnames=("self",))
    def start_state_update(
        self,
        emitter_state: EvosaxEmitterState,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores] = None,
    ) -> Optional[EvosaxEmitterState]:
        """
        Update the state of the emitter.
        """

        emitter_state = emitter_state.replace(
            emit_count = emitter_state.emit_count + 1
        )
        # jax.debug.print("state_update emit count: {}", emitter_state.emit_count)

        scores = self.ranking_criteria(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
            novelty_archive=emitter_state.novelty_archive,
        )

        emitter_state = self.es_tell(emitter_state, genotypes, scores)
        
        # Updating novelty archive
        novelty_archive = emitter_state.novelty_archive.update(descriptors)
        emitter_state = emitter_state.replace(novelty_archive=novelty_archive)
        
        emitter_state = self.restarter.update(
            emitter_state,
            scores
        )
        restart_bool = self.restarter.restart_criteria(
            emitter_state=emitter_state,
            scores=scores,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
            novelty_archive=emitter_state.novelty_archive,
        )

        emitter_state = jax.lax.cond(
            restart_bool,
            lambda x: self.restarter.reset(x),
            lambda x: x,
            emitter_state
        )

        return emitter_state, restart_bool


    def finish_state_update(
            self,
            emitter_state: EvosaxEmitterState,
            repertoire: MapElitesRepertoire,
            restart_bool: bool,
    ):
        """
        Finish the update with the restart step.
        """
        
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
    @property
    def batch_size(self) -> int:
        """
        Returns:
            the batch size emitted by the emitter.
        """
        return 1
    
    @property
    def evals_per_gen(self):
        """
        Evaluate the center in the main loop and the whole population in the update
        """
        return self._batch_size + 1
    
    # @partial(jax.jit, static_argnames=("self",))
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

        return offspring, {}, random_key
    
    # @partial(jax.jit, static_argnames=("self",))
    def _external_novelty_state_update(
        self,
        emitter_state: EvosaxEmitterState,
        repertoire: MapElitesRepertoire,
        novelty_archive: NoveltyArchive, 
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores] = None,
    ) -> Optional[EmitterState]:
        """
        Do an ES step for a specified novelty archive, return behaviors to update the archive.
        """
        random_key, subkey = jax.random.split(emitter_state.random_key)

        offspring = self.es_ask(
            emitter_state=emitter_state, 
            random_key=random_key
            )
        
        pop_fitnesses, pop_descriptors, pop_extra_scores, random_key = self.scoring_fn(
            genotypes=offspring, 
            random_key=subkey
            )
        
        scores = self.ranking_criteria(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=offspring,
            fitnesses=pop_fitnesses,
            descriptors=pop_descriptors,
            extra_scores=pop_extra_scores,
            novelty_archive=emitter_state.novelty_archive,
        )

        emitter_state = self.es_tell(
            emitter_state,
            offspring, 
            scores
            )
        
        # TODO: add restart
        emitter_state = self.restarter.update(
            emitter_state,
            scores
        )
        
        restart_bool = self.restarter.restart_criteria(
            emitter_state=emitter_state,
            scores=scores,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
            novelty_archive=emitter_state.novelty_archive,
        )

        emitter_state = jax.lax.cond(
            restart_bool,
            lambda x: self.restart(repertoire=repertoire, emitter_state=x),
            lambda x: x,
            emitter_state
        )

        random_key, subkey = jax.random.split(emitter_state.random_key)
        emitter_state = self._post_update_emitter_state(emitter_state, subkey, repertoire)

        return emitter_state, pop_descriptors

    # @partial(jax.jit, static_argnames=("self",))
    def state_update(
        self,
        emitter_state: EvosaxEmitterState,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores] = None,
    ) -> Optional[EmitterState]:
        novelty_archive = emitter_state.novelty_archive

        emitter_state = emitter_state.replace(
            emit_count = emitter_state.emit_count + 1
        )
        # jax.debug.print("state_update emit count: {}", emitter_state.emit_count)

        emitter_state, descriptors = self._external_novelty_state_update(
            emitter_state,
            repertoire,
            novelty_archive,
            genotypes,
            fitnesses,
            descriptors,
            extra_scores,
        )

        # Updating novelty archive: add the center only
        novelty_archive = emitter_state.novelty_archive.update(descriptors)
        emitter_state = emitter_state.replace(novelty_archive=novelty_archive)

        return emitter_state