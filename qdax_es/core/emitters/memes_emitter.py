from __future__ import annotations

from functools import partial
from typing import Optional, Tuple, Callable
from chex import ArrayTree

import jax
import jax.numpy as jnp
import numpy as np 

from qdax.custom_types import Centroid, Descriptor, ExtraScores, Fitness, Genotype, RNGKey

from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    get_cells_indices,
)

from qdax.core.emitters.multi_emitter import MultiEmitter, MultiEmitterState
from qdax.core.emitters.emitter import Emitter, EmitterState

from qdax_es.core.emitters.evosax_base_emitter import EvosaxEmitterState, MultiESEmitterState
from qdax_es.core.emitters.evosax_emitter import EvosaxEmitterCenter
from qdax_es.utils.restart import MEMESStaleRestarter
from qdax_es.core.containers.novelty_archive import NoveltyArchive, DummyNoveltyArchive

@jax.jit
def added_repertoire(
    genotypes: Genotype, descriptors: Descriptor, repertoire: MapElitesRepertoire
) -> jnp.ndarray:
    """Compute if the given genotypes have been added to the repertoire in
    corresponding cell.
    Code taken from the original MEMES implementation: https://github.com/adaptive-intelligent-robotics/MEMES
    
    Args:
        genotypes: genotypes candidate to addition
        descriptors: corresponding descriptors
        repertoire: repertoire considered for addition
    Returns:
        boolean for each genotype
    """
    cells = get_cells_indices(descriptors, repertoire.centroids)
    repertoire_genotypes = jax.tree_util.tree_map(
        lambda x: x[cells], repertoire.genotypes
    )
    added = jax.tree_util.tree_map(
        lambda x, y: jnp.equal(x, y), genotypes, repertoire_genotypes
    )
    added = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (descriptors.shape[0], -1)), added
    )
    added = jax.tree_util.tree_map(lambda x: jnp.all(x, axis=1), added)
    final_added = jnp.array(jax.tree_util.tree_leaves(added))
    final_added = jnp.all(final_added, axis=0)
    return final_added

class MEMESEmitterState(EvosaxEmitterState):
    explore_exploit: int = 0 # 0 for explore, 1 for exploit

class MEMESEmitter(EvosaxEmitterCenter):
    def __init__(
        self,
        centroids: Centroid,
        es_hp = {},
        es_type="CMA_ES",
        scoring_fn: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ]=None,
        restarter = None,
        novelty_nearest_neighbors = 10,
    ):
        """
        Initialize the ES emitter.
        """
        if restarter is None:
            restarter = MEMESStaleRestarter()
        
        super().__init__(
            centroids=centroids,
            es_hp=es_hp,
            es_type=es_type,
            ns_es=False,
            novelty_archive_size=0,
            scoring_fn=scoring_fn,
            restarter=restarter,
        )
        self.novelty_nearest_neighbors = novelty_nearest_neighbors
        self.ranking_criteria = self._combined_criteria
        self.restart = self._restart_repertoire

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self, genotypes: Optional[Genotype], random_key: RNGKey, explore_exploit:int=0
    ) -> Tuple[Optional[MultiESEmitterState], RNGKey]:
        state, random_key = super().init(genotypes, random_key)
        return MEMESEmitterState(
            **state,
            explore_exploit=explore_exploit,
        ), random_key

class MEMESPoolEmitterState(MultiESEmitterState):
    novelty_archive: NoveltyArchive

class MEMESPoolEmitter(Emitter):
    def __init__(
        self,
        pool_size: int,
        centroids: Centroid,
        explore_ratio: float = 0.5,
        es_hp = {},
        es_type="CMA_ES",
        scoring_fn: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ]=None,
        restarter = None,
    ):
        self.pool_size = pool_size
        self.explore_ratio = explore_ratio
        self.emitter = MEMESEmitter(
            centroids=centroids,
            es_hp=es_hp,
            es_type=es_type,
            scoring_fn=scoring_fn,
            restarter=restarter,
        )

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        random_key: RNGKey,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> Tuple[Optional[MultiESEmitterState], RNGKey]:
        # prepare keys for each emitter
        random_key, subkey = jax.random.split(random_key)
        subkeys = jax.random.split(subkey, self.pool_size)

        # Make explore_exploit state booleans (deterministic)
        n_explore = int(self.pool_size * self.explore_ratio)
        n_exploit = self.pool_size - n_explore
        explore_exploit = jnp.concatenate(
            [jnp.zeros(n_explore), jnp.ones(n_exploit)]
        )

        emitter_states, keys = jax.vmap(
            self.emitter.init,
            in_axes=(0, None, None, None, None, None)
        )(
            subkeys,
            repertoire,
            genotypes,
            fitnesses,
            descriptors,
            extra_scores,
        )

        emitter_state = MultiESEmitterState(emitter_states)
        return emitter_state, random_key

    @partial(jax.jit, static_argnames=("self",))
    def emit(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: Optional[MultiESEmitterState],
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """Emit new population. Use all the sub emitters to emit subpopulation
        and gather them.

        Args:
            repertoire: a repertoire of genotypes.
            emitter_state: the current state of the emitter.
            random_key: key for random operations.

        Returns:
            Offsprings and a new random key.
        """

        if emitter_state is None:
            raise ValueError("Emitter state must be initialized before emitting.")

        # prepare subkeys for each sub emitter
        random_key, subkey = jax.random.split(random_key)
        subkeys = jax.random.split(subkey, self.pool_size)

        # jax.debug.print("emitter_states: {}", net_shape(emitter_state.emitter_states))

        # vmap
        all_offsprings, _, keys = jax.vmap(
            lambda s, k: self.emitter.emit(repertoire, s, k),
            in_axes=(0, 0))(
            emitter_state.emitter_states,
            subkeys,
        ) 

        # jax.debug.print("offspring batch: {}", net_shape(all_offsprings))

        # concatenate offsprings together: remove the first dimension
        offsprings = jax.tree_map(
            lambda x: jnp.concatenate(x, axis=0),
            all_offsprings
        )

        # jax.debug.print("offspring batch: {}", net_shape(offsprings))

        return offsprings, {}, random_key
    

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
        # Check if added
        added = added_repertoire(
            genotypes, 
            descriptors,
            repertoire
            )
        
        # Update restart state
        emitter_states = jax.vmap(
            lambda a, s: self.emitter.restarter.update_staleness(s, a),
            in_axes=(0, 0)
        )(
            added,
            emitter_state.emitter_states
        )
        emitter_state = emitter_state.replace(
            emitter_states=emitter_states
        )

        novelty_archive = emitter_states.novelty_archive

        indices = jnp.arange(self.pool_size)
        new_sub_emitter_state, behaviors = jax.vmap(
            lambda i, s, g, f, d, e: self.emitter._external_novelty_state_update(
                emitter_state=s, 
                repertoire=repertoire, 
                novelty_archive=novelty_archive,
                genotypes=g, 
                fitnesses=f, 
                descriptors=d, 
                extra_scores=e),
            in_axes=(0, 0, 0, 0, 0, 0)
        )(
            indices,
            emitter_state.emitter_states,
            genotypes,
            fitnesses,
            descriptors,
            extra_scores,
        )

        jax.debug.print("behaviors: {}", behaviors)

        # Updating novelty archive: add the center only
        novelty_archive = emitter_state.novelty_archive.update(descriptors)
        emitter_state = emitter_state.replace(novelty_archive=novelty_archive)

        return emitter_state