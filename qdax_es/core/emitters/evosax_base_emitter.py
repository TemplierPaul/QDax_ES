from __future__ import annotations

from functools import partial
from typing import Optional, Tuple, Callable
from abc import abstractmethod

import jax
import jax.numpy as jnp

from qdax.custom_types import Centroid, Descriptor, ExtraScores, Fitness, Genotype, RNGKey
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.emitters.cma_emitter import CMAEmitter, CMAEmitterState

from qdax.core.emitters.cma_opt_emitter import CMAOptimizingEmitter
from qdax.core.emitters.cma_rnd_emitter import CMARndEmitter, CMARndEmitterState
from qdax.core.emitters.cma_improvement_emitter import CMAImprovementEmitter

from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    get_cells_indices,
)
from qdax_es.core.containers.novelty_archive import NoveltyArchive, DummyNoveltyArchive

from evosax.algorithms.base import (
    State as EvoState,
    Params as EvoParams,
)
from evosax.algorithms import algorithms as Strategies


from qdax_es.utils.restart import RestartState, DummyRestarter
from qdax_es.utils.evosax_interface import net_shape

class EvosaxEmitterState(EmitterState):
    """
    Emitter state for the ES emitter.

    Args:
        key: a random key to handle stochastic operations. Used for
            state update only, another key is used to emit. This might be
            subject to refactoring discussions in the future.
        cmaes_state: state of the underlying CMA-ES algorithm
        previous_fitnesses: store last fitnesses of the repertoire. Used to
            compute the improvment.
        emit_count: count the number of emission events.
    """

    key: RNGKey
    es_state: EvoState
    es_params: EvoParams
    previous_fitnesses: Fitness
    emit_count: int
    novelty_archive: NoveltyArchive
    restart_state: RestartState


class MultiESEmitterState(EmitterState):
    """State of an emitter than use multiple ES in a parallel manner.

    Args:
        emitter_states: a tree of emitter states
        
    """

    emitter_states: EmitterState

class EvosaxEmitter(Emitter):
    def __init__(
        self,
        centroids: Centroid,
        population_size,
        std_init=0.05,
        es_type="CMA_ES",
        ns_es=False,
        novelty_archive_size=0,
        scoring_fn: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ]=None,
        restarter = None,
        init_variables_func: Callable[[RNGKey], Genotype] = None,
    ):
        """
        Initialize the ES emitter.
        """
        
        self._batch_size = population_size
        self.std_init = std_init
        self.es_type = es_type
        self._centroids = centroids
        self.novelty_archive_size = novelty_archive_size
        self.novelty_nearest_neighbors = 10
        self._num_descriptors = centroids.shape[1]
        self.init_variables_func = init_variables_func

        if restarter is None:
            restarter = DummyRestarter()
            print("No restarter provided. Using DummyRestarter.")
        self.restarter = restarter

        self.scoring_fn = scoring_fn

        # Delay until we have genomes
        self.es = None

        self.ranking_criteria = self._fitness_criteria
        if ns_es:
            self.ranking_criteria = self._novelty_criteria

        self.restart = self._restart_random

    def init(
        self,
        key: RNGKey,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ):
        """
        Initializes the ES emitter

        Args:
            genotypes: initial genotypes to add to the grid.
            key: a random key to handle stochastic operations.

        Returns:
            The initial state of the emitter.
        """
        restart_state = self.restarter.init()
        first_genotype = jax.tree.map(lambda x: x[0], genotypes)

        self.es = Strategies[self.es_type](
            population_size=self._batch_size,
            solution=first_genotype
        )
        print(self.es)

        # Initialize the ES state
        key, init_key = jax.random.split(key)
        es_params = self.es.default_params
        # Replace std_init
        es_params = es_params.replace(
            std_init=self.std_init
        )

        es_state = self.es.init(
            init_key, 
            mean=first_genotype,
            params=es_params
        )   

        genome_dim = self.es._ravel_solution(first_genotype).shape
        print(f"Genome dimension: {genome_dim}")
        # print(f"Net shape: {net_shape(genome_dim)}")

        # Create empty Novelty archive
        if self.novelty_archive_size > 0:
            novelty_archive = NoveltyArchive.init(
                size= self.novelty_archive_size, 
                num_descriptors=self._num_descriptors,
            )
        else:
            novelty_archive = DummyNoveltyArchive()

        # Initialize repertoire with default values
        num_centroids = self._centroids.shape[0]
        default_fitnesses = -jnp.inf * jnp.ones(shape=num_centroids)

        # return the initial state
        key, subkey = jax.random.split(key)

        return (
            EvosaxEmitterState(
                key=subkey,
                es_state=es_state, 
                es_params=es_params,
                previous_fitnesses=default_fitnesses,
                novelty_archive=novelty_archive,
                emit_count=0,
                restart_state=restart_state,
            )
        )
    
    def es_ask(
            self,
            emitter_state: EvosaxEmitterState,
            key: RNGKey,
    ):
        """
        Generate a new population of offspring.
        """
        es_state = emitter_state.es_state
        es_params = emitter_state.es_params

        key, subkey = jax.random.split(key)
        offspring, es_state = self.es.ask(subkey, es_state, es_params)

        return offspring, key
    
    def restart_from(
            self, 
            emitter_state: EvosaxEmitterState,
            init_genome: Genotype,
    ):
        """
        Restart the ES with a new mean.
        """
        
        key = emitter_state.key
        key, subkey = jax.random.split(key)

        es_params = emitter_state.es_params
        es_state = self.es.init(
            subkey,
            mean=init_genome,
            params=es_params
        )

        emitter_state = emitter_state.replace(
            es_state=es_state,
            key=key
        )

        # print("Restart from", type(emitter_state.es_state))

        return emitter_state

    def _restart_random(
            self, 
            repertoire: MapElitesRepertoire,
            emitter_state: EvosaxEmitterState,
    ):
        """
        Restart from a random genome.
        """
        key = emitter_state.key
        key, subkey = jax.random.split(key)

        # Generate a random genotype
        genotypes = self.init_variables_func(subkey)
        first_genotype = jax.tree.map(lambda x: x[0], genotypes)

        emitter_state = self.restart_from(
            emitter_state.replace(key=key),
            first_genotype,
        )

        # print("Random restart", type(emitter_state.es_state))

        return emitter_state
        
    
    def _restart_repertoire(
            self,
            repertoire: MapElitesRepertoire,
            emitter_state: EvosaxEmitterState,
    ):
        """
        Restart from a random cell in the repertoire.
        """
        key = emitter_state.key
        key, subkey = jax.random.split(key)
        random_genotype = repertoire.sample(subkey, 1)

        emitter_state = emitter_state.replace(
            key=key,
        )

        emitter_state = self.restart_from(
            emitter_state,
            random_genotype,
        )

        return emitter_state


    @property
    def evals_per_gen(self):
        raise NotImplementedError("This method should be implemented in the child class")
    
    def es_tell(
            self, 
            emitter_state: EvosaxEmitterState,
            offspring: Genotype,
            fitnesses: Fitness,
    ):
        """
        Update the ES with the fitnesses of the offspring.
        """
        es_state = emitter_state.es_state
        es_params = emitter_state.es_params
        key, subkey = jax.random.split(emitter_state.key)

        fitnesses = - jnp.array(fitnesses) # Maximise

        new_es_state, _ = self.es.tell(
            subkey, 
            population=offspring,
            fitness=fitnesses, 
            state=es_state, 
            params=es_params
            )

        return emitter_state.replace(
            es_state=new_es_state,
            key=key,
        )
    
    """Defines how the genotypes should be sorted. Impacts the update
        of the CMAES state. In the end, this defines the type of CMAES emitter
        used (optimizing, random direction or improvement).

        Args:
            emitter_state: current state of the emitter.
            repertoire: latest repertoire of genotypes.
            genotypes: emitted genotypes.
            fitnesses: corresponding fitnesses.
            descriptors: corresponding fitnesses.
            extra_scores: corresponding extra scores.

        Returns:
            The values to take into account in order to rank the emitted genotypes.
            Here, it's the improvement, or the fitness when the cell was previously
            unoccupied. Additionally, genotypes that discovered a new cell are
            given on offset to be ranked in front of other genotypes.
    """
    

    def _fitness_criteria(
        self,
        emitter_state: EvosaxEmitterState,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores],
        novelty_archive: NoveltyArchive = None
    ) -> jnp.ndarray:
        """
        Use the fitness for standard ES. 
        """
        
        return fitnesses
    
    def _novelty_criteria(
        self,
        emitter_state: EvosaxEmitterState,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores],
        novelty_archive: NoveltyArchive
    ) -> jnp.ndarray:
        """
        NS-ES novelty criteria.
        """

        novelty = novelty_archive.novelty(
                    descriptors, self.novelty_nearest_neighbors
                )
        return novelty

    def _combined_criteria(
        self,
        emitter_state: EvosaxEmitterState,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores],
        novelty_archive: NoveltyArchive
    ) -> jnp.ndarray:
        """
        NS-ES novelty criteria.
        """

        novelty = novelty_archive.novelty(
                    descriptors, self.novelty_nearest_neighbors
                )
        
        # Combine novelty and fitness: ratio = 0 for novelty, 1 for fitness
        ratio = emitter_state.explore_exploit

        scores = fitnesses * ratio + novelty * (1 - ratio)

        return scores

    
    def _improvement_criteria(
        self,
        emitter_state: EvosaxEmitterState,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores],
        novelty_archive: NoveltyArchive = None
    ) -> jnp.ndarray:
        """
        Improvement criteria.
        """
        # Compute the improvements - needed for re-init condition
        indices = get_cells_indices(descriptors, repertoire.centroids)
        improvements = fitnesses - emitter_state.previous_fitnesses[indices]

        # condition for being a new cell
        condition = improvements == jnp.inf

        # criteria: fitness if new cell, improvement else
        ranking_criteria = jnp.where(condition, fitnesses, improvements)

        # make sure to have all the new cells first
        new_cell_offset = jnp.max(ranking_criteria) - jnp.min(ranking_criteria)

        ranking_criteria = jnp.where(
            condition, ranking_criteria + new_cell_offset, ranking_criteria
        )

        return ranking_criteria


    def _post_update_emitter_state(
        self, emitter_state, key: RNGKey, repertoire: MapElitesRepertoire
    ) -> EvosaxEmitterState:
        # From (1024, 1) to (1024,)
        previous_fitnesses = jnp.ravel(repertoire.fitnesses)
        return emitter_state.replace(
            key=key, previous_fitnesses=previous_fitnesses
        )
    
   