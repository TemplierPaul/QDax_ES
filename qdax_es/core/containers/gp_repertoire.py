import jax
import jax.numpy as jnp
from jax import jit
import optax 
from functools import partial
from flax.struct import dataclass as fdataclass
from qdax.types import (
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
    n_steps: int = 1000
    
    @classmethod
    def init(
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        centroids: Centroid,
        extra_scores: Optional[ExtraScores] = None,
        weighted: bool = False,
        n_steps: int = 1000,
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
        return repertoire.replace(gp_state=gp_state, n_steps=n_steps)

   
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
            n_steps=self.n_steps
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
            n_steps=self.n_steps
        )

    @partial(jit, static_argnames=("n_steps",))
    def fit_gp(self, n_steps: int = 1000):
        fit_gp_state = train_gp(self.gp_state, num_steps=n_steps)
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
            ):
        """Plot the repertoire"""
        if plot_gp:
            fig, axes = plt.subplot_mosaic("""
                    AB
                    CD
                    """,
                    figsize=(15, 15),
                )
        else:
            fig, axes = plt.subplot_mosaic("""
                AB
                """,
                figsize=(15, 8),
            )
        try:
            axes["A"] = plot_2d_map_elites_repertoire(
                centroids=self.centroids,
                repertoire_fitnesses=self.fitnesses,
                minval=min_bd,
                maxval=max_bd,
                repertoire_descriptors=self.descriptors,
                ax=axes["A"],
            )
            axes["B"] = plot_2d_count(
                self, 
                min_bd, 
                max_bd, 
                log_scale=True, 
                ax=axes["B"]
                )
            
            if plot_gp:
                print(f"Plot GP LS: {self.gp_params.lengthscale}")
                means, covs = self.batch_predict(self.centroids)

                axes["C"] = plot_archive_value(
                    self, 
                    means, 
                    min_bd, 
                    max_bd,
                    ax=axes["C"],
                    title="GP mean"
                )
                axes["D"] = plot_archive_value(
                    self, 
                    covs, 
                    min_bd, 
                    max_bd,
                    ax=axes["D"],
                    title="GP variance"
                )
            plt.suptitle(title, fontsize=20)
        except:
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
