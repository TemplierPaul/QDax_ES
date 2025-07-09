import jax
import jax.numpy as jnp
from functools import partial
import optax 
from flax.struct import dataclass as fdataclass
import matplotlib.pyplot as plt

from qdax_es.utils.gaussian_processes.base_gp import GaussianProcess, GPParams
from qdax_es.utils.pareto_selection import stoch_get_pareto_indices, get_pareto_depths
from qdax_es.utils.count_plots import plot_archive_value


EMPTY_WEIGHT = 1e4
DEFAULT_X = 100
EPSILON = 1e-4
JITTER = 1e-6

def sample_centroids(repertoire, key, tournament_size):
    # Select tournament_size points from the repertoire
    indices = jax.random.choice(key, jnp.arange(len(repertoire.fitnesses)), (tournament_size,), replace=False)
    return jax.tree.map(
            lambda x: x[indices],
            repertoire,
        )

def masked_mean(x, mask):
    return jnp.sum(jnp.where(mask, x, 0)) / jnp.sum(mask)

@fdataclass
class TargetSelectorState:
    pass

class TargetSelector:
    def __init__(self, n_points):
        self._select = partial(self._select, n_points=n_points)

    def init(self, repertoire):
        return TargetSelectorState()
    
    def update(self, selector_state, repertoire):
        return selector_state

    def _select(self, selector_state, repertoire, key, n_points):
        return 0
    
    @classmethod
    def _plot(
        cls, 
        selector_state, 
        repertoire, 
        ax, 
        min_descriptor: jnp.ndarray,
        max_descriptor: jnp.ndarray,
        gp
        ):
        return ax

class UniformTargetSelector(TargetSelector):
    def _select(self, selector_state, repertoire, key, n_points):
        return select_uniform(repertoire, key, n_points)

def select_uniform(repertoire, key, n_points):
    """
    Sample n_points uniformly from the repertoire.
    """
    indices = jax.random.choice(key, jnp.arange(len(repertoire.fitnesses)), (n_points,), replace=False)
    return indices

@fdataclass
class GPSelectorState(TargetSelectorState):
    params: GPParams
    opt_state: optax.OptState
    x: jnp.ndarray
    y: jnp.ndarray
    n_grad_steps: int = 0

class GPSelector(TargetSelector):
    def __init__(self, gp: GaussianProcess, pareto_depth: int, gp_steps: int, n_points: int, tournament_size=1024):
        self.update = partial(
            self._update,
            gp=gp,
            gp_steps=gp_steps
        )
        
        self.select = partial(
            self._select,
            gp=gp,
            pareto_depth=pareto_depth,
            n_points=n_points,
            tournament_size=tournament_size
        )

        self.init = partial(
            self.init,
            gp=gp,
            tournament_size=tournament_size
        )
        self.tournament_size = tournament_size

        self.plot = partial(self._plot, gp=gp)

        print(f"Initialized GPSelector with tournament size {self.tournament_size}")

    @classmethod
    def init(cls, repertoire, gp: GaussianProcess, tournament_size=1024):
        params_init = GPParams(
                        kernel_params={'sigma': 0.0, 'length_scale': 0.0},  # log(1) = 0, will become ~1 after softplus
                        noise_var=-2.0  # log(0.1) ≈ -2.3, will become ~0.1 after softplus
                    )
        params_init, x, y = gp.from_repertoire(
            repertoire=repertoire,
            params=params_init,
        )

        assert tournament_size <= repertoire.centroids.shape[0]

        trainable_params = params_init._learnable_params()
        opt_state = gp.optimizer.init(trainable_params)
        return GPSelectorState(
            params=params_init,
            opt_state=opt_state,
            x=x,
            y=y,
            n_grad_steps=0
        )

    @classmethod
    def _update(cls, selector_state:GPSelectorState, repertoire, gp, gp_steps):
        params, X, y = gp.from_repertoire(
            repertoire=repertoire,
            params=selector_state.params,
        )
        params, opt_state = gp.fit_one(X=X, y=y, params=params, opt_state=selector_state.opt_state)
        return GPSelectorState(
            params=params,
            opt_state=opt_state,
            x=X,
            y=y,
            n_grad_steps=selector_state.n_grad_steps + gp_steps
        )
    
    @classmethod
    def _select(cls, selector_state, repertoire, key, n_points, gp, pareto_depth, tournament_size):
        key_t, key_p = jax.random.split(key)

        params, x, y = gp.from_repertoire(
            repertoire=repertoire,
            params=selector_state.params,
        )

        # Sample centroids for the tournament
        selected = sample_centroids(
            repertoire=repertoire,
            key=key_t,
            tournament_size=tournament_size
        )
        centroids = selected.centroids
        # print(f"Selected {centroids.shape} centroids for the tournament")

        mean, var = gp.predict(params, x, y, centroids)

        mean_nans = jnp.sum(jnp.isnan(mean))
        var_nans = jnp.sum(jnp.isnan(var))
        # jax.debug.print("GP predictions NaNs: mean={}, var={}", mean_nans, var_nans)
        mean = mean.squeeze()

        pareto_indices = stoch_get_pareto_indices(
            f1=mean,
            f2=var,
            key=key_p,
            n_points=n_points,
            max_depth=pareto_depth,
        )

        return pareto_indices

    @classmethod
    def _plot(
        cls, 
        selector_state, 
        repertoire, 
        axes, 
        min_descriptor: jnp.ndarray,
        max_descriptor: jnp.ndarray,
        gp
        ):
        print(selector_state.params)
        params, x, y = gp.from_repertoire(
            repertoire=repertoire,
            params=selector_state.params,
        )
        mean, var = gp.predict(params, x, y, repertoire.centroids)
        # Check for nans
        if jnp.isnan(mean).any() or jnp.isnan(var).any():
            mean_nans = jnp.sum(jnp.isnan(mean))
            var_nans = jnp.sum(jnp.isnan(var))
            print(f"GP predictions contain NaNs: mean={mean_nans}, var={var_nans}")
            print("Params: ", selector_state.params)

            return axes

        # Plot GP mean
        axes[0] = plot_archive_value(
            repertoire, 
            mean, 
            min_descriptor, 
            max_descriptor,
            ax=axes[0],
            title="GP Mean"
        )

        # Plot GP var
        axes[1] = plot_archive_value(
            repertoire, 
            var, 
            min_descriptor, 
            max_descriptor,
            ax=axes[1],
            title="GP Variance"
        )

        # Plot Pareto front
        pareto_depth = - get_pareto_depths(mean, var)
        print("Pareto Depth", pareto_depth.min(), pareto_depth.max())

        axes[2] = plot_archive_value(
            repertoire, 
            pareto_depth, 
            min_descriptor, 
            max_descriptor,
            ax=axes[2],
            title="Pareto front"
        )
        return axes

class GPCountSelector(GPSelector):
    @classmethod
    def _select(cls, selector_state, repertoire, key, n_points, gp, pareto_depth, tournament_size):
        key_t, key_p = jax.random.split(key)
        # Sample centroids for the tournament
        selected = sample_centroids(
            repertoire=repertoire,
            key=key_t,
            tournament_size=tournament_size
        )
        centroids = selected.centroids
        counts = selected.count
        # print(f"Selected {centroids.shape} centroids for the tournament")

        mean, _ = gp.predict(selector_state.params, selector_state.x, selector_state.y, centroids)
        mean = mean.squeeze()

        # print(f"Mean shape: {mean.shape}, Counts shape: {counts.shape}")

        pareto_indices = stoch_get_pareto_indices(
            f1=mean,
            f2= - counts,
            key=key_p,
            n_points=n_points,
            max_depth=pareto_depth,
        )

        return pareto_indices
    
    @classmethod
    def _plot(
        cls, 
        selector_state, 
        repertoire, 
        axes, 
        min_descriptor: jnp.ndarray,
        max_descriptor: jnp.ndarray,
        gp
        ):
        params, x, y = gp.from_repertoire(
            repertoire=repertoire,
            params=selector_state.params,
        )
        mean, var = gp.predict(params, x, y, repertoire.centroids)

        # Plot GP mean
        axes[0] = plot_archive_value(
            repertoire, 
            mean, 
            min_descriptor, 
            max_descriptor,
            ax=axes[0],
            title="GP Mean"
        )

        # Plot GP var
        axes[1] = plot_archive_value(
            repertoire, 
            var, 
            min_descriptor, 
            max_descriptor,
            ax=axes[1],
            title="GP Variance"
        )

        # Plot Pareto front
        counts = repertoire.count
        pareto_depth = - get_pareto_depths(mean, - counts)
        print("Pareto Depth", pareto_depth.min(), pareto_depth.max())

        axes[2] = plot_archive_value(
            repertoire, 
            pareto_depth, 
            min_descriptor, 
            max_descriptor,
            ax=axes[2],
            title="Pareto front"
        )
        return axes
