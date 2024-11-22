from __future__ import annotations

from functools import partial
from typing import Optional, Tuple, Callable
from chex import ArrayTree

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

from optax.schedules import linear_schedule

from qdax.custom_types import Centroid, Descriptor, ExtraScores, Fitness, Genotype, RNGKey

from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    get_cells_indices,
)

from qdax.core.emitters.multi_emitter import MultiEmitter, MultiEmitterState
from qdax.core.emitters.emitter import Emitter, EmitterState

from qdax_es.core.containers.novelty_archive import NoveltyArchive, DummyNoveltyArchive
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


def net_shape(net):
    return jax.tree_map(lambda x: x.shape, net)

def get_closest_genotype(
    bd: Descriptor,
    repertoire: MapElitesRepertoire,
):
    """
    Get the genotype closest to the given descriptor.
    """
    mask = repertoire.fitnesses > -jnp.inf
    distances = jnp.where(
        mask,
        jnp.linalg.norm(repertoire.descriptors - bd, axis=1),
        jnp.inf,
    )
    index = jnp.argmin(distances)
    start_bd = repertoire.descriptors[index]
    start_genome = tree_map(
        lambda x: x[index], repertoire.genotypes
    )
    return start_genome, start_bd


class JEDiEmitterState(EvosaxEmitterState):
    wtfs_alpha: float
    wtfs_target: Descriptor


class ConstantScheduler:
    def __init__(self, value):
        self.value = value

    def __call__(self, step):
        return self.value
    
    def __repr__(self):
        return f"ConstantScheduler({self.value})"
    
class LinearScheduler(ConstantScheduler):
    def __init__(self, value, end, steps):
        self.schedule_fn = linear_schedule(value, end, steps)

    def __call__(self, step):
        # jax.debug.print("step: {}, value: {}", step, self.schedule_fn(step))
        return self.schedule_fn(step)



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
        alpha_scheduler = None,
        global_norm=False,
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
        if global_norm:
            self.ranking_criteria = self._global_wtfs_criteria
        self.alpha_scheduler = alpha_scheduler
        self.restart = self._jedi_restart

    def init(
        self,
        random_key: RNGKey,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,       
    ):
        emitter_state, random_key = super().init(
            random_key=random_key,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )
        alpha = self.alpha_scheduler(0)
        return JEDiEmitterState(
            **emitter_state.__dict__,
            wtfs_alpha=alpha,
            wtfs_target=jnp.zeros(self._centroids.shape[1]),
        ), random_key


    def _wtfs_criteria(
        self,
        emitter_state: JEDiEmitterState,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores],
        novelty_archive: NoveltyArchive = None
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
    
    def _global_wtfs_criteria(
        self,
        emitter_state: JEDiEmitterState,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores],
        novelty_archive: NoveltyArchive = None
    ) -> jnp.ndarray:
        """
        Weighted Target Fitness Score with normalization based on global min/max
        """
        target_bd = emitter_state.wtfs_target
        wtf_alpha = emitter_state.wtfs_alpha

        # jax.debug.print("descriptors: {}", descriptors.shape)
        # jax.debug.print("target_bd: {}", target_bd.shape)

        # Normalized distance
        distance = jnp.linalg.norm(descriptors - target_bd, axis=1)
        # argsort
        min_dist = 0
        max_dist = jnp.max(
            jnp.linalg.norm(repertoire.centroids - target_bd, axis=1)
        )

        norm_distance = (distance - min_dist) / (max_dist - min_dist + EPSILON)
        distance_score = 1 - norm_distance  # To minimize distance

        # Normalized fitness
        rep_fitnesses = repertoire.fitnesses
        max_fit = jnp.max(rep_fitnesses)
        max_fit = jnp.maximum(max_fit, jnp.max(fitnesses))

        # replace -inf with + inf
        rep_fitnesses = jnp.where(rep_fitnesses == -jnp.inf, jnp.inf, rep_fitnesses)
        min_fit = jnp.min(rep_fitnesses)
        min_fit = jnp.minimum(min_fit, jnp.min(fitnesses))
        
        norm_fitnesses = (fitnesses - min_fit) / (max_fit - min_fit + EPSILON)
        # Weighted target fitness
        wtf = (
            1 - wtf_alpha
        ) * norm_fitnesses + wtf_alpha * distance_score
        return wtf

    def _jedi_restart(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: JEDiEmitterState,
        target_bd_index: int,
    ):
        """
        JEDi emitter with uniform target selection (no GP).
        """
        # Get centroid based on target_bd_index

        target_bd = repertoire.centroids[target_bd_index]
        # jax.debug.print("emit count: {}", emitter_state.emit_count)
        new_alpha = self.alpha_scheduler(emitter_state.emit_count)
        emitter_state = emitter_state.replace(
            wtfs_target=target_bd,
            wtfs_alpha=new_alpha,
        )
        
        start_genome, start_bd = get_closest_genotype(
            bd=target_bd,
            repertoire=repertoire,
        )
        # jax.debug.print("restart from: {}", start_bd)


        return self.restart_from(
            emitter_state=emitter_state,
            init_genome=start_genome,
        )

    def finish_state_update(
            self,
            emitter_state: JEDiEmitterState,
            repertoire: MapElitesRepertoire,
            restart_bool: bool,
            target_bd_index: int,
    ):
        """
        Finish the update with the restart step.
        """
        
        emitter_state = jax.lax.cond(
            restart_bool,
            lambda x: self.restart(
                repertoire=repertoire, 
                emitter_state=x,
                target_bd_index=target_bd_index,
                ),
            lambda x: x,
            emitter_state
        )

        # print wtfs_target
        # jax.debug.print("wtfs_target: {}", emitter_state.wtfs_target)
        # jax.debug.print("wtfs_alpha: {}", emitter_state.wtfs_alpha)

        random_key, subkey = jax.random.split(emitter_state.random_key)
        emitter_state = self._post_update_emitter_state(emitter_state, subkey, repertoire)

        return emitter_state
