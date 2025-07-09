from __future__ import annotations

from functools import partial
from typing import Optional, Tuple, Callable
from chex import ArrayTree

import jax
import jax.numpy as jnp
# from jax.tree_util import tree.map

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
from qdax_es.utils.target_selection import TargetSelector, TargetSelectorState

EPSILON = 1e-8

# Helper function to get a sub pytree
def _get_sub_pytree(pytree: ArrayTree, start: int, end: int) -> ArrayTree:
    return jax.tree.map(lambda x: jax.lax.dynamic_slice_in_dim(x, start, (end-start), 0), pytree)

def split(array, n):
    return jnp.array(jnp.split(array, n, axis=0))
    
def split_tree(tree, n):
    return jax.tree.map(lambda x: split(x, n), tree)


def net_shape(net):
    return jax.tree.map(lambda x: x.shape, net)

def get_closest_genotype(
    bd: Descriptor,
    repertoire: MapElitesRepertoire,
    # key: RNGKey,
    # tournament_size=None,
):
    """
    Get the genotype closest to the given descriptor.
    """
    mask = jnp.squeeze(repertoire.fitnesses) > -jnp.inf
    # print("Mask shape: ", mask.shape)

    # if tournament_size is None:
    #     tournament_size = mask.sum()

    distances = jnp.where(
        mask,
        jnp.linalg.norm(repertoire.descriptors - bd, axis=1),
        jnp.inf,
    )
    # # print("Distances shape: ", distances.shape)

    # # Select tournament_size random indices with non inf distances
    # random_val = jax.random.uniform(key, shape=distances.shape)
    # # Get top tournament_size values
    # threshold = jnp.sort(random_val, descending=True)[tournament_size-1]

    # distances = jnp.where(
    #     random_val > threshold,
    #     distances,
    #     jnp.inf
    # )

    index = jnp.argmin(distances)
    start_bd = repertoire.descriptors[index]
    start_genome = jax.tree.map(
        lambda x: x[index], repertoire.genotypes
    )
    # print("Start genome shape: ", start_genome.shape)
    # print("Start bd shape: ", start_bd.shape)
    return start_genome, start_bd


class JEDiEmitterState(EvosaxEmitterState):
    wtfs_alpha: float
    wtfs_target: Descriptor
    target_selector_state: TargetSelectorState


class ConstantScheduler:
    def __init__(self, value):
        self.value = value

    def __call__(self, step, key):
        return self.value
    
    def __repr__(self):
        return f"ConstantScheduler({self.value})"
    
class LinearScheduler(ConstantScheduler):
    def __init__(self, value, end, steps):
        self.schedule_fn = linear_schedule(value, end, steps)
        self.value = value
        self.end = end

    def __call__(self, step, key):
        # jax.debug.print("step: {}, value: {}", step, self.schedule_fn(step))
        return self.schedule_fn(step)
    
    def __repr__(self):
        return f"LinearScheduler ({self.value} -> {self.end})"

class RandomScheduler:
    def __call__(self, step, key):
        return jax.random.uniform(key)

    def __repr__(self):
        return f"RandomScheduler"

class JEDiEmitter(EvosaxEmitterAll):
    """
    Emitter for the Quality with Just Enough Diversity (JEDi) algorithm.
    """
    def __init__(
        self,
        centroids: Centroid,
        population_size,
        std_init=0.05,
        es_type="CMA_ES",
        scoring_fn: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ]=None,
        restarter = None,
        alpha_scheduler = None,
        global_norm=False,
        init_variables_func: Callable[[RNGKey], Genotype] = None,
        target_selector: TargetSelector = None
    ):
        """
        Initialize the ES emitter.
        """
        super().__init__(
            centroids=centroids,
            population_size=population_size,
            std_init=std_init,
            es_type=es_type,
            ns_es=False,
            novelty_archive_size=0,
            scoring_fn=scoring_fn,
            restarter=restarter,
            init_variables_func=init_variables_func,
        )

        self.ranking_criteria = self._wtfs_criteria
        if global_norm:
            self.ranking_criteria = self._global_wtfs_criteria
        self.alpha_scheduler = alpha_scheduler
        self.restart = self._jedi_restart
        self.target_selector = target_selector

    def init(
        self,
        key: RNGKey,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,       
    ):
        key, subkey = jax.random.split(key)

        # Add 1 dim to genotypes
        genotypes = jax.tree.map(lambda x: x[None], genotypes)
        emitter_state = super().init(
            key=key,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )
        alpha = self.alpha_scheduler(0, subkey)

        target_selector_state = self.target_selector.init(repertoire)

        return JEDiEmitterState(
            **emitter_state.__dict__,
            wtfs_alpha=alpha,
            wtfs_target=jnp.zeros(self._centroids.shape[1]),
            target_selector_state=target_selector_state
        )

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
    ):
        """
        JEDi emitter with uniform target selection (no GP).
        """
        key, key_t, key_a, key_c = jax.random.split(emitter_state.key, 4)

        target_bd_index = self.target_selector.select(
            emitter_state.target_selector_state,
            repertoire, 
            key_t
            )[0]
        target_bd = repertoire.centroids[target_bd_index]

        # jax.debug.print("emit count: {}", emitter_state.emit_count)
        new_alpha = self.alpha_scheduler(emitter_state.emit_count, key_a)

        emitter_state = emitter_state.replace(
            wtfs_target=target_bd,
            wtfs_alpha=new_alpha,
            key=key
        )
        
        start_genome, start_bd = get_closest_genotype(
            bd=target_bd,
            repertoire=repertoire,
            # key=key_c,
            # tournament_size=64
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
    ):
        """
        Finish the update with the restart step.
        """
        # jax.debug.print("Restart bool: {}", restart_bool)
        
        target_selector_state = self.target_selector.update(
            emitter_state.target_selector_state,
            repertoire=repertoire,
        )
        emitter_state = emitter_state.replace(
            target_selector_state=target_selector_state
        )

        # jax.debug.print(
        #     "Target selector grad steps: {} | {}", 
        #     emitter_state.target_selector_state.n_grad_steps,
        #     emitter_state.target_selector_state.params
        # )

        emitter_state = jax.lax.cond(
            restart_bool,
            self._jedi_restart,
            lambda r, s: s,
            repertoire,
            emitter_state,
        )

        # print wtfs_target
        # jax.debug.print("wtfs_target: {}", emitter_state.wtfs_target)
        # jax.debug.print("wtfs_alpha: {}", emitter_state.wtfs_alpha)

        key, subkey = jax.random.split(emitter_state.key)
        emitter_state = self._post_update_emitter_state(emitter_state, subkey, repertoire)

        return emitter_state
