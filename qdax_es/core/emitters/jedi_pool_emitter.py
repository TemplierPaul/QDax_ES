from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey

from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
)

from qdax.core.emitters.emitter import Emitter


from qdax_es.core.emitters.evosax_base_emitter import EvosaxEmitterState, MultiESEmitterState
from qdax_es.core.emitters.jedi_emitter import (
    split,
    split_tree,
)


class JEDiPoolEmitter(Emitter):
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
        key: RNGKey,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> Tuple[Optional[MultiESEmitterState], RNGKey]:
        
        # prepare keys for each emitter
        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, self.pool_size)

        emitter_states = jax.vmap(
            self.emitter.init,
            in_axes=(0, None, 0, None, None, None)
        )(
            subkeys,
            repertoire,
            genotypes,
            fitnesses,
            descriptors,
            extra_scores,
        )

        key, subkey = jax.random.split(key)
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
        return emitter_state

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
        
        final_emitter_states = jax.vmap(
            lambda restart, state: self.emitter.finish_state_update(state, repertoire, restart),
            in_axes=(0, 0)
        )(
            need_train_gp,
            new_sub_emitter_state
        )

        # alphas = final_emitter_states.wtfs_alpha
        # jax.debug.print(
        #     "alphas: {}", alphas
        # )

        # jax.debug.print("final_emitter_states: {}", net_shape(final_emitter_states))

        return MultiESEmitterState(final_emitter_states)

    # @partial(jax.jit, static_argnames=("self",))
    def emit(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: Optional[MultiESEmitterState],
        key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """Emit new population. Use all the sub emitters to emit subpopulation
        and gather them.

        Args:
            repertoire: a repertoire of genotypes.
            emitter_state: the current state of the emitter.
            key: key for random operations.

        Returns:
            Offsprings and a new random key.
        """
    
        if emitter_state is None:
            raise ValueError("Emitter state must be initialized before emitting.")

        # prepare subkeys for each sub emitter
        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, self.pool_size)

        # jax.debug.print("emitter_states: {}", net_shape(emitter_state.emitter_states))

        # vmap
        all_offsprings, _ = jax.vmap(
            lambda s, k: self.emitter.emit(repertoire, s, k),
            in_axes=(0, 0))(
            emitter_state.emitter_states,
            subkeys,
        ) 

        # jax.debug.print("offspring batch: {}", net_shape(all_offsprings))

        # concatenate offsprings together: remove the first dimension
        offsprings = jax.tree.map(
            lambda x: jnp.concatenate(x, axis=0),
            all_offsprings
        )

        # jax.debug.print("offspring batch: {}", net_shape(offsprings))

        return offsprings, {}
    
    # def get_target_bd_indices(
    #     self,
    #     repertoire,
    #     need_restart,
    #     emitter_state,
    #     ):
    #     """
    #     Reset the target behavior descriptor of the emitters
    #     """
    #     # Sample target BD indices
    #     key = emitter_state.key[0]

    #     indices = jax.random.choice(
    #         key,
    #         jnp.arange(len(repertoire.fitnesses)),
    #         (self.pool_size,),
    #         replace=False,
    #     )

    #     # jax.debug.print("indices: {}", indices)

    #     return indices
        

# class GPJEDiPoolEmitter(UniformJEDiPoolEmitter):
#     def __init__(
#         self,
#         pool_size: int,
#         emitter:Emitter,
#         n_steps: int = 1000,
#     ):
#         self.pool_size = pool_size
#         self.emitter = emitter

#         self.get_pareto_indices = partial(
#             stoch_get_pareto_indices, 
#             n_points=self.pool_size, 
#             max_depth=10
#             )
#         self.get_pareto_indices = jax.jit(self.get_pareto_indices)
#         self.train_select = jax.jit(partial(self._train_select, n_steps=n_steps))

#     def _train_select(self, repertoire, emitter_state, n_steps=1000):
#         """
#         Train the GP and select targets on the pareto front
#         """
#         fit_repertoire = repertoire.fit_gp(n_steps=n_steps)
#         mean, var = fit_repertoire.batch_predict(repertoire.centroids)
#         print(f"mean: {mean.shape}, var: {var.shape}")
#         pareto_indices = self.get_pareto_indices(mean, var, emitter_state.key[0])
#         return pareto_indices
        
#     def get_target_bd_indices(
#         self,
#         repertoire,
#         need_restart,
#         emitter_state,
#         ):
#         """
#         Train the GP and select targets on the pareto front if it needs to be trained, else call from the parent class
#         """
#         # jax.debug.print("need_restart: {}", need_restart)
#         return jax.lax.cond(
#             need_restart,
#             lambda x: self.train_select(repertoire, emitter_state),
#             lambda x: super(
#                 GPJEDiPoolEmitter, self
#             ).get_target_bd_indices(repertoire, need_restart, emitter_state),
#             None
#         )
