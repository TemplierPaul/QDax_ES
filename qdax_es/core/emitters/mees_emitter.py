from __future__ import annotations

from functools import partial
from typing import Optional, Tuple, Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from qdax.custom_types import Centroid, Descriptor, ExtraScores, Fitness, Genotype, RNGKey

from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    get_cells_indices,
)

from qdax_es.core.emitters.evosax_emitter import EvosaxEmitterCenter

from qdax_es.core.emitters.evosax_base_emitter import EvosaxEmitterState
from qdax_es.utils.restart import RestartState, FixedGens

@dataclass
class MEESConfig:
    """Configuration for the MAP-Elites-ES emitter.

    Args:
        num_optimizer_steps: frequency of archive-sampling
        novelty_nearest_neighbors
        last_updated_size: number of last updated indiv used to
            choose parents from repertoire
        exploit_num_cell_sample: number of highest-performing cells
            from which to choose parents, when using exploit
        explore_num_cell_sample: number of most-novel cells from
            which to choose parents, when using explore
        use_explore: if False, use only fitness gradient
        use_exploit: if False, use only novelty gradient
    """

    num_optimizer_steps: int = 10
    novelty_nearest_neighbors: int = 10
    last_updated_size: int = 5
    exploit_num_cell_sample: int = 2
    explore_num_cell_sample: int = 5
    use_explore: bool = True
    use_exploit: bool = True

class MEESEmitterState(EvosaxEmitterState):
    """
    State of the ME-ES emitter.
    """
    explore_exploit: int = 0 # 0 for explore, 1 for exploit

class MEESEmitter(EvosaxEmitterCenter):
    """
    ME-ES emitter.
    """
    def __init__(
        self,
        batch_size: int,
        centroids: Centroid,
        mees_config: MEESConfig,
        es_hp = {},
        es_type="CMA_ES",
        novelty_archive_size=1000,
        scoring_fn: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ]=None,
        restarter = None,
    ):
        """
        Initialize the ES emitter.
        """
        self._config = mees_config

        if restarter is None:
            restarter = FixedGens(self._config.num_optimizer_steps)

        super().__init__(
            batch_size=batch_size,
            centroids=centroids,
            es_hp=es_hp,
            es_type=es_type,
            ns_es=False,
            novelty_archive_size=novelty_archive_size,
            scoring_fn=scoring_fn,
            restarter=restarter,
        )
        self.novelty_nearest_neighbors = self._config.novelty_nearest_neighbors
        self.ranking_criteria = self._combined_criteria

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
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
        return MEESEmitterState(
            **state,
            explore_exploit=0,
        ), random_key

    def restart(
        self, 
        repertoire: MapElitesRepertoire,
        emitter_state: EvosaxEmitterState,
    ):
        """
        Restart for explore or exploit
        """
        explore_exploit = 1 - emitter_state.explore_exploit # Flip the explore_exploit
        emitter_state = emitter_state.replace(explore_exploit=explore_exploit)

        # 0: restart explore, 1: restart exploit
        emitter_state = jax.lax.cond(
            explore_exploit == 0,
            lambda x: self._restart_explore(x, repertoire),
            lambda x: self._restart_exploit(x, repertoire),
            emitter_state
        )

        return emitter_state
        


    def _restart_explore(
        self,
        emitter_state: EvosaxEmitterState,
        repertoire: MapElitesRepertoire,
    ):
        """
        Restart for explore: select from the most novel indivs
        """
        # Compute the novelty of all indivs in the archive
        novelties = emitter_state.novelty_archive.novelty(
            repertoire.descriptors, self._config.novelty_nearest_neighbors
        )
        novelties = jnp.where(repertoire.fitnesses > -jnp.inf, novelties, -jnp.inf)




    def _restart_exploit(
        self,
        emitter_state: EvosaxEmitterState,
        repertoire: MapElitesRepertoire,
    ):
        """
        Restart for exploit
        """
        pass