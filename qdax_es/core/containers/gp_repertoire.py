import jax
import jax.numpy as jnp
from jax import jit
import optax 
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
from typing import Callable, List, Optional, Tuple, Union

from jedies.core.containers.count_repertoire import CountMapElitesRepertoire
from qdax.utils.plotting import plot_2d_map_elites_repertoire
from jedies.jedi.jedi_plotting import plot_2d_count
from jedies.utils.plotting import plot_archive_value
import matplotlib.pyplot as plt

@fdataclass
class RBFParams:
    sigma: float = 1.0
    lengthscale: float = 1.0
    obs_noise_sigma: float = 1.0

@jit
def rbf_kernel(params, x1, x2):
    """RBF kernel with x in R^D"""
    sigma = params.sigma
    lengthscale = params.lengthscale
    return sigma**2 * jnp.exp(-0.5 * jnp.sum((x1 - x2)**2) / (lengthscale**2))

# Mattern 32 kernel
@jit
def mattern32_kernel(params, x1, x2):
    """Mattern 32 kernel with x in R^D"""
    sigma = params.sigma
    lengthscale = params.lengthscale
    return sigma**2 * (1 + jnp.sqrt(3) * jnp.sum((x1 - x2)**2) / lengthscale) * jnp.exp(-jnp.sqrt(3) * jnp.sum((x1 - x2)**2) / lengthscale)

# @jit
# def no_ls_rbf_kernel(params, x1, x2):
#     """RBF kernel with x in R^D"""
#     sigma = params.sigma
#     lengthscale = 1.0
#     return sigma**2 * jnp.exp(-0.5 * jnp.sum((x1 - x2)**2) / (lengthscale**2))

kernel_func = rbf_kernel
# kernel_func = mattern32_kernel

@jit
def random_params(key):
    keys = jax.random.split(key, 3)
    sigma = jax.random.uniform(keys[0], minval=0.1, maxval=10)
    lengthscale = jax.random.uniform(keys[1], minval=0.1, maxval=10)
    obs_noise_sigma = jax.random.uniform(keys[2], minval=0.1, maxval=10)
    return RBFParams(sigma, lengthscale, obs_noise_sigma)


class GPRepertoire(CountMapElitesRepertoire):
    gp_params: RBFParams = None
    Kinv: jnp.ndarray = None
    ls_scaler: float = 1.0

    @jax.jit
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
            ls_scaler=self.ls_scaler
        )
    
    @jax.jit
    def __add__(
        self,
        other_repertoire: CountMapElitesRepertoire,
    ) -> CountMapElitesRepertoire:
        # Super add
        new_repertoire = super().__add__(other_repertoire)
        # set ls_scaler
        return new_repertoire.replace(
            ls_scaler=self.ls_scaler
        )

    @jit
    def compute_K(self, X, params):
        """Compute the kernel matrix K using vmap"""
        K = jax.vmap(lambda x1: jax.vmap(lambda x2: kernel_func(params, x1, x2))(X))(X)
        return K + params.obs_noise_sigma**2 * jnp.eye(K.shape[0])

    @jit
    def GPpredict(self, x_new, X, Y, Kinv, params):
        """Predict mean and variance of GP at x_new"""
        Y_mean = jnp.mean(Y)
        Y_norm = Y - Y_mean
        Kx = jax.vmap(lambda x: kernel_func(params, x_new, x))(X)

        # compute mean prediction
        f_mean = Y_mean + Kx.T @ Kinv @ Y_norm

        # compute variance prediction
        kxx = kernel_func(params, x_new, x_new)
        f_var = kxx - Kx.T @ Kinv @ Kx
        return f_mean, f_var
    
    @jit
    def neg_marginal_likelihood(self, params, X, Y):
        """Compute the marginal likelihood of the data given the hyperparameters"""
        K = self.compute_K(X, params)
        Kinv = jnp.linalg.inv(K)
        Y_mean = jnp.mean(Y)
        Y_norm = Y - Y_mean
        data_fit = Y_norm.T @ Kinv @ Y_norm
        complexity_penalty = jnp.log(jnp.linalg.det(K))
        constant_term = len(X) * jnp.log(2 * jnp.pi)
        log_marginal_likelihood = -0.5 * (data_fit + complexity_penalty + constant_term)
        return - log_marginal_likelihood
    
    def fit_gp(self, n_steps=1000):
        grad_neg_marginal_likelihood = jit(jax.grad(self.neg_marginal_likelihood))

        def train_loop(params, X, Y, optimizer, opt_state):
            """Train the GP"""
            # get gradients
            grads = grad_neg_marginal_likelihood(params, X, Y)

            # update parameters 
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            
            return params, opt_state

        params = RBFParams()

        # maximize the marginal likelihood
        learning_rate = 0.01
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)

        fit = self.fitnesses
        is_empty = fit == -jnp.inf
        fit = fit[~is_empty]
        bd = self.descriptors[~is_empty]

        # rewrite loop with scan
        def loop_scan(carry, i):
            params, opt_state = carry
            params, opt_state = train_loop(params, bd, fit, optimizer, opt_state)
            return (params, opt_state), None
        
        (params, opt_state), _ = jax.lax.scan(
            loop_scan, 
            (params, opt_state), 
            jnp.arange(n_steps)
            )

        K = self.compute_K(bd, params)
        Kinv = jnp.linalg.inv(K)

        new_repertoire = GPRepertoire(
            descriptors=self.descriptors,
            fitnesses=self.fitnesses,
            genotypes=self.genotypes,
            centroids=self.centroids,
            count=self.count,
            gp_params=params,
            Kinv=Kinv,
            ls_scaler=self.ls_scaler,
        )
        return new_repertoire

    def predict(self, x_new):
        fit = self.fitnesses
        is_empty = fit == -jnp.inf
        fit = fit[~is_empty]
        bd = self.descriptors[~is_empty]

        params = self.gp_params.replace(
            lengthscale=self.gp_params.lengthscale * self.ls_scaler
        )
        print(f"Predict: Scaled lengthscale ({self.ls_scaler}): {params.lengthscale}")

        K = self.compute_K(bd, params)
        Kinv = jnp.linalg.inv(K)

        mean, var = self.GPpredict(x_new, bd, fit, Kinv, params)
        return mean, var
    
    def batch_predict(self, x_new):
        fit = self.fitnesses
        is_empty = fit == -jnp.inf
        fit = fit[~is_empty]
        bd = self.descriptors[~is_empty]

        params = self.gp_params.replace(
            lengthscale=self.gp_params.lengthscale * self.ls_scaler
        )
        print(f"Batch predict: Scaled lengthscale ({self.ls_scaler}): {params.lengthscale}")

        K = self.compute_K(bd, params)
        Kinv = jnp.linalg.inv(K)

        # vmap over x_new
        mean, var = jax.vmap(lambda x: self.GPpredict(x, bd, fit, Kinv, params))(x_new)
        return mean, var


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



    
    @classmethod
    def import_repertoire(cls, repertoire):
        """Turn a repertoire into a GPRepertoire"""
        new_repertoire = cls(
            descriptors=repertoire.descriptors,
            fitnesses=repertoire.fitnesses,
            genotypes=repertoire.genotypes,
            centroids=repertoire.centroids,
            count=repertoire.count,
            ls_scaler=repertoire.ls_scaler,
        )
        return new_repertoire