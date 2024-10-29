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

from evosax import EvoState, EvoParams, Strategies

from qdax_es.utils.evosax_interface import ANNReshaper, DummyReshaper

from qdax_es.utils.restart import RestartState, DummyRestarter

class EvosaxEmitterState(EmitterState):
    """
    Emitter state for the CMA-ME emitter.

    Args:
        random_key: a random key to handle stochastic operations. Used for
            state update only, another key is used to emit. This might be
            subject to refactoring discussions in the future.
        cmaes_state: state of the underlying CMA-ES algorithm
        previous_fitnesses: store last fitnesses of the repertoire. Used to
            compute the improvment.
        emit_count: count the number of emission events.
    """

    random_key: RNGKey
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
        es_hp = {},
        es_type="CMA_ES",
        ns_es=False,
        novelty_archive_size=0,
        scoring_fn: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ]=None,
        restarter = None,
    ):
        """
        Initialize the ES emitter.
        """
        
        self._batch_size = es_hp["popsize"]
        self.es_hp = es_hp
        self.es_type = es_type
        self._centroids = centroids
        self.novelty_archive_size = novelty_archive_size
        self.novelty_nearest_neighbors = 10
        self._num_descriptors = centroids.shape[1]

        if restarter is None:
            restarter = DummyRestarter()
            print("No restarter provided. Using DummyRestarter.")
        self.restarter = restarter

        self.scoring_fn = scoring_fn

        # Delay until we have genomes
        self.es = None
        self.reshaper = None

        self.ranking_criteria = self._fitness_criteria
        if ns_es:
            self.ranking_criteria = self._novelty_criteria


        self.restart = self._restart_random

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
        """
        Initializes the ES emitter

        Args:
            genotypes: initial genotypes to add to the grid.
            random_key: a random key to handle stochastic operations.

        Returns:
            The initial state of the emitter.
        """
        restart_state = self.restarter.init()

        # Check if initial genotypes are ANNs or vectors
        if isinstance(genotypes, jnp.ndarray):
            print("Using DummyReshaper")
            self.reshaper = DummyReshaper()
        else:
            print("Using ANNReshaper")
            if jax.tree_util.tree_leaves(genotypes)[0].shape[0] > 1:
                genotypes = jax.tree_util.tree_map(
                    lambda x: x[0],
                    genotypes,
                )
            self.reshaper = ANNReshaper.init(genotypes)

        self.es = Strategies[self.es_type](
            num_dims=self.reshaper.genotype_dim,
            # popsize=self.batch_size,
            **self.es_hp,
        )
        print(self.es)

        # Initialize the ES state
        random_key, init_key = jax.random.split(random_key)
        es_params = self.es.default_params
        es_state = self.es.initialize(
            init_key, params=es_params
        )

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
        random_key, subkey = jax.random.split(random_key)

        return (
            EvosaxEmitterState(
                random_key=subkey,
                es_state=es_state, 
                es_params=es_params,
                previous_fitnesses=default_fitnesses,
                novelty_archive=novelty_archive,
                emit_count=0,
                restart_state=restart_state,
            ),
            random_key,
        )
    
    def es_ask(
            self,
            emitter_state: EvosaxEmitterState,
            random_key: RNGKey,
    ):
        """
        Generate a new population of offspring.
        """
        es_state = emitter_state.es_state
        es_params = emitter_state.es_params

        random_key, subkey = jax.random.split(random_key)
        genomes, es_state = self.es.ask(subkey, es_state, es_params)

        offspring = jax.vmap(self.reshaper.unflatten)(genomes)

        return offspring, random_key
    
    def restart_from(
            self, 
            emitter_state: EvosaxEmitterState,
            init_genome: Genotype,
    ):
        """
        Restart the ES with a new mean.
        """
        init_mean = self.reshaper.flatten(init_genome)
        
        random_key = emitter_state.random_key
        random_key, subkey = jax.random.split(random_key)

        es_params = self.es.default_params
        es_state = self.es.initialize(
            subkey, params=es_params
        )
        es_state = es_state.replace(mean=init_mean)

        return emitter_state.replace(
            es_state=es_state,
            random_key=random_key,
        )

    def _restart_random(
            self, 
            repertoire: MapElitesRepertoire,
            emitter_state: EvosaxEmitterState,
    ):
        """
        Restart from a random genome.
        """
        random_key = emitter_state.random_key
        random_key, subkey = jax.random.split(random_key)

        es_params = self.es.default_params
        es_state = self.es.initialize(
            subkey, params=es_params
        )

        return emitter_state.replace(
            es_state=es_state,
            random_key=random_key,
        )
    
    def _restart_repertoire(
            self,
            repertoire: MapElitesRepertoire,
            emitter_state: EvosaxEmitterState,
    ):
        """
        Restart from a random cell in the repertoire.
        """
        random_key = emitter_state.random_key
        random_genotype, random_key = repertoire.sample(random_key, 1)

        emitter_state = emitter_state.replace(
            random_key=random_key,
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
        genomes = jax.vmap(self.reshaper.flatten)(offspring)
        
        es_state = emitter_state.es_state
        es_params = emitter_state.es_params

        fitnesses = - jnp.array(fitnesses) # Maximise

        new_es_state = self.es.tell(genomes, fitnesses, es_state, es_params)

        return emitter_state.replace(
            es_state=new_es_state,
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
        self, emitter_state, random_key: RNGKey, repertoire: MapElitesRepertoire
    ) -> EvosaxEmitterState:
        return emitter_state.replace(
            random_key=random_key, previous_fitnesses=repertoire.fitnesses
        )
    
    # @partial(jax.jit, static_argnames=("self",))
    @abstractmethod
    def emit(
        self,
        repertoire: Optional[MapElitesRepertoire],
        emitter_state: EvosaxEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """
        Generate solutions to be evaluated and added to the archive.
        """
        pass


    # @partial(jax.jit, static_argnames=("self",))
    @abstractmethod
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
        Update the state of the emitter.
        """
        pass