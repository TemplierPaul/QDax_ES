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
from qdax_es.core.emitters.jedi_emitter import (
    JEDiEmitterState,
    net_shape,
    split,
    split_tree,
)
from qdax_es.core.containers.gp_repertoire import GPRepertoire
from qdax_es.utils.pareto_selection import get_pareto_indices, stoch_get_pareto_indices


class UniformJEDiPoolEmitter(Emitter):
    def __init__(
        self,
        pool_size: int,
        emitter:Emitter,
    ):
        self.pool_size = pool_size
        self.emitter = emitter

    @property
    def batch_size(self) -> int:
        """
        Returns:
            the batch size emitted by the emitter.
        """
        return self.emitter.batch_size * self.pool_size
    
    @property
    def evals_per_gen(self):
        """
        Evaluate the population in the main loop for 1 emitter state
        """
        return self.emitter.batch_size * self.pool_size

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

        random_key, subkey = jax.random.split(random_key)
        target_bd_indices = jax.random.choice(
            subkey,
            jnp.arange(len(repertoire.fitnesses)),
            (self.pool_size,),
            replace=False,
        )

        emitter_states = jax.vmap(
            lambda s, i: s.replace(wtfs_target=repertoire.centroids[i]),
            in_axes=(0, 0)
        )(
            emitter_states,
            target_bd_indices
        )
        
        emitter_state = MultiESEmitterState(emitter_states)
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
    ) -> Optional[MultiESEmitterState]:
        """
        Update the state of the emitters
        """

        if emitter_state is None:
            return None

        split_fitnesses = split(fitnesses, self.pool_size)
        split_descriptors = split(descriptors, self.pool_size)

        split_genotypes = split_tree(genotypes, self.pool_size)
        split_extra_scores = split_tree(extra_scores, self.pool_size)

        indices = jnp.arange(self.pool_size)
        new_sub_emitter_state, need_train_gp = jax.vmap(
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

        need_restart = jnp.any(jnp.array(need_train_gp))
        # jax.debug.print("need_restart: {}", need_train_gp.sum())
        target_bd_indices = self.get_target_bd_indices(
            repertoire=repertoire,
            need_restart=need_restart,
            emitter_state = new_sub_emitter_state,
            )
        # jax.debug.print("target BD indices: {}", target_bd_indices)
        
        final_emitter_states = jax.vmap(
            lambda i, state, restart: self.emitter.finish_state_update(state, repertoire, restart, i),
            in_axes=(0, 0, 0)
        )(
            target_bd_indices,
            new_sub_emitter_state,
            need_train_gp
        )

        # jax.debug.print("final_emitter_states: {}", net_shape(final_emitter_states))

        return MultiESEmitterState(final_emitter_states)

    # @partial(jax.jit, static_argnames=("self",))
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
    
    def get_target_bd_indices(
        self,
        repertoire,
        need_restart,
        emitter_state,
        ):
        """
        Reset the target behavior descriptor of the emitters
        """
        # Sample target BD indices
        random_key = emitter_state.random_key[0]

        indices = jax.random.choice(
            random_key,
            jnp.arange(len(repertoire.fitnesses)),
            (self.pool_size,),
            replace=False,
        )

        # jax.debug.print("indices: {}", indices)

        return indices
        

class GPJEDiPoolEmitter(UniformJEDiPoolEmitter):
    def __init__(
        self,
        pool_size: int,
        emitter:Emitter,
        n_steps: int = 1000,
    ):
        self.pool_size = pool_size
        self.emitter = emitter

        self.get_pareto_indices = partial(
            stoch_get_pareto_indices, 
            n_points=self.pool_size, 
            max_depth=10
            )
        self.get_pareto_indices = jax.jit(self.get_pareto_indices)
        self.train_select = jax.jit(partial(self._train_select, n_steps=n_steps))


    def _train_select(self, repertoire, emitter_state, n_steps=1000):
        """
        Train the GP and select targets on the pareto front
        """
        fit_repertoire = repertoire.fit_gp(n_steps=n_steps)
        mean, var = fit_repertoire.batch_predict(repertoire.centroids)
        pareto_indices = self.get_pareto_indices(mean, var, emitter_state.random_key[0])
        return pareto_indices
        

    def get_target_bd_indices(
        self,
        repertoire,
        need_restart,
        emitter_state,
        ):
        """
        Train the GP and select targets on the pareto front if it needs to be trained, else call from the parent class
        """
        # jax.debug.print("need_restart: {}", need_restart)
        return jax.lax.cond(
            need_restart,
            lambda x: self.train_select(repertoire, emitter_state),
            lambda x: super(
                GPJEDiPoolEmitter, self
            ).get_target_bd_indices(repertoire, need_restart, emitter_state),
            None
        )
