from __future__ import annotations

from functools import partial
from typing import Optional, Tuple, Callable
from chex import ArrayTree

import jax
import jax.numpy as jnp
import numpy as np 

from qdax.types import Centroid, Descriptor, ExtraScores, Fitness, Genotype, RNGKey

from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    get_cells_indices,
)

from qdax.core.emitters.multi_emitter import MultiEmitter, MultiEmitterState
from qdax.core.emitters.emitter import Emitter, EmitterState

from qdax_es.core.emitters.evosax_emitter import EvosaxEmitterAll

from qdax_es.core.emitters.evosax_base_emitter import EvosaxEmitterState
from qdax_es.utils.restart import CMARestarter

EPSILON = 1e-8

# Helper function to get a sub pytree
def _get_sub_pytree(pytree: ArrayTree, start: int, end: int) -> ArrayTree:
    return jax.tree_util.tree_map(lambda x: jax.lax.dynamic_slice_in_dim(x, start, (end-start), 0), pytree)

def split(array, n):
    return jnp.array(jnp.split(array, n, axis=0))
    
def split_tree(tree, n):
    return jax.tree_map(lambda x: split(x, n), tree)

class JEDiEmitterState(EvosaxEmitterState):
    wtfs_alpha: float
    wtfs_target: Descriptor
    # emitter_index: int= 0

def net_shape(net):
    return jax.tree_map(lambda x: x.shape, net)


class JEDiEmitter(EvosaxEmitterAll):
    """
    Emitter for the Quality with Just Enough Diversity (JEDi) algorithm.
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
        wtfs_alpha = 0.5,
    ):
        """
        Initialize the ES emitter.
        """
        if restarter is None:
            use_cma_criterion = es_type == "CMA_ES"
            restarter = CMARestarter(
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

        self.ranking_criteria = self._wtfs_criteria
        self.wtfs_alpha = wtfs_alpha

    def init(
        self, init_genotypes: Genotype, random_key: RNGKey,       
    ):
        state, random_key = super().init(init_genotypes, random_key)
        return JEDiEmitterState(
            **state.__dict__,
            wtfs_alpha=self.wtfs_alpha,
            wtfs_target=jnp.zeros(self._centroids.shape[1]),
        ), random_key


    def _wtfs_criteria(
        self,
        emitter_state: EvosaxEmitterState,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores],
    ) -> jnp.ndarray:
        """
        Weighted Target Fitness Score
        """
        target_bd = emitter_state.wtfs_target
        wtf_alpha = emitter_state.wtfs_alpha

        # jax.debug.print("descriptors: {}", descriptors.shape)
        # jax.debug.print("target_bd: {}", target_bd.shape)

        # Normalized distance
        distance = jnp.linalg.norm(descriptors - target_bd, axis=1)
        # argsort
        min_dist = jnp.min(distance)
        max_dist = jnp.max(distance)
        norm_distance = (distance - min_dist) / (max_dist - min_dist + EPSILON)
        distance_score = 1 - norm_distance  # To minimize distance

        # Normalized fitness
        min_fit = jnp.min(fitnesses)
        max_fit = jnp.max(fitnesses)
        norm_fitnesses = (fitnesses - min_fit) / (max_fit - min_fit + EPSILON)
        # Weighted target fitness
        wtf = (
            1 - wtf_alpha
        ) * norm_fitnesses + wtf_alpha * distance_score
        return wtf

class JEDiPoolEmitterState(EmitterState):
    """State of an emitter than use multiple emitters in a parallel manner.

    WARNING: this is not the emitter state of Multi-Emitter MAP-Elites.

    Args:
        emitter_states: a tuple of emitter states
    """

    emitter_states: EmitterState


class JEDiPoolEmitter(Emitter):
    def __init__(
        self,
        pool_size: int,
        emitter:Emitter,
    ):
        self.pool_size = pool_size
        self.emitter = emitter

        # indexes_separation_batches = self.get_indexes_separation_batches(pool_size, emitter)
        # self.indexes_start_batches = jnp.array(indexes_separation_batches[:-1])
        # self.indexes_end_batches = jnp.array(indexes_separation_batches[1:])

        # self.get_batch = partial(jax.lax.dynamic_slice_in_dim, slice_size=emitter.batch_size, axis=0)
        # self.get_batch = jax.jit(self.get_batch)

    @property
    def batch_size(self) -> int:
        """
        Returns:
            the batch size emitted by the emitter.
        """
        return self.emitter.batch_size * self.pool_size

    def init(
        self, init_genotypes: Optional[Genotype], random_key: RNGKey
    ) -> Tuple[Optional[JEDiPoolEmitterState], RNGKey]:
        
        # prepare keys for each emitter
        random_key, subkey = jax.random.split(random_key)
        subkeys = jax.random.split(subkey, self.pool_size)

        emitter_states, keys = jax.vmap(
            self.emitter.init,
            in_axes=(None, 0)
        )(
            init_genotypes, subkeys
        )

        emitter_state = JEDiPoolEmitterState(emitter_states)
        return emitter_state, random_key

    # @partial(jax.jit, static_argnames=("self",))
    def state_update(
        self,
        emitter_state: EvosaxEmitterState,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores] = None,
    ) -> Optional[JEDiPoolEmitterState]:
        """
        Update the state of the emitters
        """

        if emitter_state is None:
            return None

        # jax.debug.print("states: {}", net_shape(emitter_state.emitter_states))

        split_fitnesses = split(fitnesses, self.pool_size)
        split_descriptors = split(descriptors, self.pool_size)

        # jax.debug.print("split_fitnesses: {}", net_shape(split_fitnesses))
        # jax.debug.print("split_descriptors: {}", net_shape(split_descriptors))

        split_genotypes = split_tree(genotypes, self.pool_size)
        split_extra_scores = split_tree(extra_scores, self.pool_size)

        # jax.debug.print("split_genotypes: {}", net_shape(split_genotypes))
        # jax.debug.print("split_extra_scores: {}", net_shape(split_extra_scores))


        indices = jnp.arange(self.pool_size)
        new_sub_emitter_state, emitter_restart = jax.vmap(
            lambda i, s, g, f, d, e: self.emitter.start_state_update(s, repertoire, g, f, d, e),
            in_axes=(0, 0, 0, 0, 0, 0)
        )(
            indices,
            emitter_state.emitter_states,
            split_genotypes,
            split_fitnesses,
            split_descriptors,
            split_extra_scores,
        )

        # jax.debug.print("new_sub_emitter_state: {}", net_shape(new_sub_emitter_state))

        need_restart = jnp.any(jnp.array(emitter_restart))

        repertoire = jax.lax.cond(
            need_restart,
            lambda x: x.fit_gp(n_steps=100),
            lambda x: x,
            repertoire
        )

        final_emitter_states = jax.vmap(
            lambda i, state, restart: self.emitter.finish_state_update(state, repertoire, restart),
            in_axes=(0, 0, 0)
        )(
            indices,
            new_sub_emitter_state,
            emitter_restart
        )

        # jax.debug.print("final_emitter_states: {}", net_shape(final_emitter_states))

        return JEDiPoolEmitterState(final_emitter_states)

    # @partial(jax.jit, static_argnames=("self",))
    def emit(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: Optional[JEDiPoolEmitterState],
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

        jax.debug.print("emitter_states: {}", net_shape(emitter_state.emitter_states))

        # vmap
        all_offsprings, keys = jax.vmap(
            lambda s, k: self.emitter.emit(repertoire, s, k),
            in_axes=(0, 0))(
            emitter_state.emitter_states,
            subkeys,
        ) 

        jax.debug.print("offspring batch: {}", net_shape(all_offsprings))

        # concatenate offsprings together: remove the first dimension
        offsprings = jax.tree_map(
            lambda x: jnp.concatenate(x, axis=0),
            all_offsprings
        )

        jax.debug.print("offspring batch: {}", net_shape(offsprings))

        return offsprings, random_key