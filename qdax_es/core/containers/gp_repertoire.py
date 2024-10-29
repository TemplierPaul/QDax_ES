import jax
import jax.numpy as jnp
from jax import jit
import optax 
from functools import partial
from flax.struct import dataclass as fdataclass
from qdax.custom_types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    RNGKey,
    Metrics,
)
import matplotlib.pyplot as plt
from typing import Callable, List, Optional, Tuple, Union

from qdax_es.core.containers.count_repertoire import CountMapElitesRepertoire
from qdax.utils.plotting import plot_2d_map_elites_repertoire
from qdax_es.utils.count_plots import plot_2d_count, plot_archive_value
from qdax_es.utils.gaussian_process import GPState, train_gp, gp_predict, gp_batch_predict

class GPRepertoire(CountMapElitesRepertoire):
    gp_state: GPState = None
    
    @classmethod
    def init(
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        centroids: Centroid,
        extra_scores: Optional[ExtraScores] = None,
        weighted: bool = False,
    ) -> CountMapElitesRepertoire:
        """Initialize a repertoire"""
        repertoire = super().init(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
            extra_scores=extra_scores,
        )
        gp_state = GPState.init_from_repertoire(repertoire, weighted)
        return repertoire.replace(gp_state=gp_state)

    @jit
    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
        batch_of_extra_scores: Optional[ExtraScores] = None,
    ) -> CountMapElitesRepertoire:
        # Super add
        new_repertoire = super().add(
            batch_of_genotypes,
            batch_of_descriptors,
            batch_of_fitnesses,
            batch_of_extra_scores,
        )
        # set ls_scaler
        return new_repertoire.replace(
            gp_state=self.gp_state,
        )
    
    @jit
    def __add__(
        self,
        other_repertoire: CountMapElitesRepertoire,
    ) -> CountMapElitesRepertoire:
        # Super add
        new_repertoire = super().__add__(other_repertoire)
        # set ls_scaler
        return new_repertoire.replace(
            gp_state=self.gp_state,
        )

    @partial(jit, static_argnames=("n_steps",))
    def fit_gp(self, n_steps: int = 1000):
        gp_state = GPState.init_from_repertoire(self, self.gp_state.weighted)
        fit_gp_state = train_gp(gp_state, num_steps=n_steps)
        return self.replace(gp_state=fit_gp_state)

    @jit
    def predict(self, x_new):
        return gp_predict(self.gp_state, x_new)
    
    @jit
    def batch_predict(self, x_new):
        return gp_batch_predict(self.gp_state, x_new)


    def plot(
            self,
            min_bd,
            max_bd,
            title='GP',
            plot_gp=True,
            cfg=None,
            ):
        """Plot the repertoire"""
        if plot_gp:
            fig, axes = plt.subplot_mosaic("""
                    AB
                    CD
                    """,
                    figsize=(20, 15),
                )
        else:
            fig, axes = plt.subplot_mosaic("""
                AB
                """,
                figsize=(20, 8),
            )
        try:
            vmin, vmax = None, None
            if cfg is not None:
                vmin, vmax = cfg.task.plotting.fitness_bounds
            _, axes["A"] = plot_2d_map_elites_repertoire(
                centroids=self.centroids,
                repertoire_fitnesses=self.fitnesses,
                minval=min_bd,
                maxval=max_bd,
                repertoire_descriptors=self.descriptors,
                ax=axes["A"],
                vmin=vmin,
                vmax=vmax,
            )

            vmin, vmax = None, None
            if cfg is not None:
                vmin, vmax = 0, cfg.task.plotting.max_eval_cell
            axes["B"] = plot_2d_count(
                self, 
                min_bd, 
                max_bd, 
                log_scale=True, 
                ax=axes["B"],
                colormap="plasma",
                vmin=vmin,
                vmax=vmax,
                )
            
            if plot_gp:
                # print(f"Plot GP LS: {self.gp_params.lengthscale}")
                means, covs = self.batch_predict(self.centroids)

                _, axes["C"] = plot_archive_value(
                    self, 
                    means, 
                    min_bd, 
                    max_bd,
                    ax=axes["C"],
                    title="GP mean"
                )
                _, axes["D"] = plot_archive_value(
                    self, 
                    covs, 
                    min_bd, 
                    max_bd,
                    ax=axes["D"],
                    title="GP variance"
                )
            plt.suptitle(title, fontsize=20)
        except Exception as e:
            # raise e
            print("Failed plotting")

        return fig, axes
    
    def plot_gp(
            self,
            min_bd,
            max_bd,
    ):
        """Plot only GP as 2 separate plots"""
    
        means, covs = self.batch_predict(self.centroids)
        # Plot GP mean
        mean_fig, mean_ax = plt.subplots(figsize=(10, 10))
        mean_ax = plot_archive_value(
            self, 
            means, 
            min_bd, 
            max_bd,
            ax=mean_ax,
            title="GP mean"
        )

        # Plot GP variance
        var_fig, var_ax = plt.subplots(figsize=(10, 10))
        var_ax = plot_archive_value(
            self, 
            covs, 
            min_bd, 
            max_bd,
            ax=var_ax,
            title="GP variance"
        )

        return mean_fig, var_fig
