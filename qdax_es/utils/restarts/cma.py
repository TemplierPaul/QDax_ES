import chex
import jax.numpy as jnp
from flax.struct import dataclass
from typing import Optional

from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
)
from qdax.core.emitters.emitter import Emitter

from qdax_es.core.containers.novelty_archive import NoveltyArchive

from evosax.algorithms.distribution_based.cma_es import eigen_decomposition as cma_eigen_decomposition
from evosax.algorithms.distribution_based.sep_cma_es import eigen_decomposition as sep_cma_eigen_decomposition

from qdax_es.utils.restarts.base import FixedGens


@dataclass
class RestartParams:
    tol_x: float = 1e-12  # Tolerance for stopping criterion on x
    tol_x_up: float = 1e4  # Tolerance for stopping criterion on x upper bound
    tol_mean: float = 1e-12  # Tolerance for stopping criterion on mean
    tol_condition_C: float = 1e14  # Tolerance for stopping criterion on condition number of C


class CMARestarter(FixedGens):
    """
    Restart when the ES has converged
    """
    def __init__(
            self, 
            min_gens=0,
            max_gens=jnp.inf,
            es_type= "CMA_ES",
            params: RestartParams = RestartParams()
            ):
        self.eigen_decomposition = cma_eigen_decomposition
        if es_type.lower() == "sep_cma_es":
            print("Using SEP-CMA-ES eigen decomposition")
            self.eigen_decomposition = self._sep_cma_eigen_decomposition
        self.min_gens = min_gens
        self.max_gens = max_gens
        self.params = params

    def _sep_cma_eigen_decomposition(self, C):
        C, D = sep_cma_eigen_decomposition(
            C,
        )
        B = jnp.eye(C.shape[0])  # B is identity in SEP-CMA-ES
        return C, B, D
    
    def cma_criterion(self, state: chex.ArrayTree):
        restart_params = self.params
    
        dC = jnp.diag(state.C)
        C, B, D = self.eigen_decomposition(
            state.C,
        )

        # Stop if std of normal distribution is smaller than tolx in all coordinates
        # and pc is smaller than tolx in all components.
        cond_s_1 = jnp.all(state.std * dC < restart_params.tol_x)
        cond_s_2 = jnp.all(state.std * state.p_c < restart_params.tol_x)
        cond_1 = jnp.logical_and(cond_s_1, cond_s_2)

        # Stop if std diverges
        cond_2 = state.std * jnp.max(D) > restart_params.tol_x_up

        # Stop if adding 0.2 std does not change mean.
        cond_no_coord_change = jnp.allclose(
            state.mean,
            state.mean + (0.2 * state.std * jnp.sqrt(dC)),
            atol=restart_params.tol_mean,
        )
        cond_3 = cond_no_coord_change

        # Stop if adding 0.1 std in principal directions of C does not change mean.
        cond_no_axis_change = jnp.allclose(
            state.mean,
            state.mean + (0.1 * state.std * D[0] * B[:, 0]),
            atol=restart_params.tol_mean,
        )
        cond_4 = cond_no_axis_change

        # Stop if the condition number of the covariance matrix exceeds 1e14.
        cond_condition_cov = jnp.max(D) / jnp.min(D) > restart_params.tol_condition_C
        cond_5 = cond_condition_cov

        return cond_1 | cond_2 | cond_3 | cond_4 | cond_5

    def restart_criteria(
        self,
        emitter_state: Emitter,
        scores: Fitness,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores],
        novelty_archive: NoveltyArchive = None
        ):
        """
        Check if the restart condition is met.
        """
        # Stop if min/max fitness spread of recent generation is below threshold
        # spread = jnp.max(scores) - jnp.min(scores)
        # spread_restart = jnp.where(spread < self.min_spread, True, False)

        # Check if the cma_criterion is met
        cma_restart = self.cma_criterion(emitter_state.es_state)

        # Check if the max generations have been reached
        max_gens = emitter_state.restart_state.generations >= self.max_gens

        return cma_restart | max_gens
    