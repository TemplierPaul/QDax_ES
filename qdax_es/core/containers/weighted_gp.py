import jax
import jax.numpy as jnp
from jax import jit
import optax 
from flax.struct import dataclass as fdataclass

from qdax_es.core.containers.gp_repertoire import (
    GPRepertoire, 
    rbf_kernel, 
    RBFParams, 
    random_params
    )

class WeightedGPRepertoire(GPRepertoire):
    weights: jnp.ndarray = None

    @jit
    def compute_K(self, X, params):
        """Compute the kernel matrix K using vmap"""
        K = jax.vmap(lambda x1: jax.vmap(lambda x2: rbf_kernel(params, x1, x2))(X))(X)
        return K + params.obs_noise_sigma**2 * self.weights

    def _fit_weighted_gp(self, n_steps=1000):
        grad_neg_marginal_likelihood = jit(jax.grad(self.neg_marginal_likelihood))

        def train_loop(params, X, Y, optimizer, opt_state):
            """Train the GP"""
            # get gradients
            grads = grad_neg_marginal_likelihood(params, X, Y)

            # update parameters 
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            
            return params, opt_state

        fit = self.fitnesses
        is_empty = fit == -jnp.inf
        fit = fit[~is_empty]
        bd = self.descriptors[~is_empty]

        # maximize the marginal likelihood
        learning_rate = 0.01
        optimizer = optax.adam(learning_rate)

        # Test initial params to make sure they are valid
        n_tests = 100
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, num=n_tests)
        init_params = jax.vmap(random_params)(keys)

        def test_init(params):
            """Test if the init params are valid"""
            opt_state = optimizer.init(params)
            params, opt_state = train_loop(params, bd, fit, optimizer, opt_state)
            return ~jnp.isnan(params.sigma) 
        
        valid_params = jax.vmap(test_init)(init_params)
        indices = jnp.arange(len(valid_params))
        valid_params = indices[valid_params]
        # pick first valid one
        index = valid_params[0]

        params = jax.tree_util.tree_map(
            lambda x: x[index],
            init_params,
        )

        # Init point
        opt_state = optimizer.init(params)

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

        new_repertoire = WeightedGPRepertoire(
            descriptors=self.descriptors,
            fitnesses=self.fitnesses,
            genotypes=self.genotypes,
            centroids=self.centroids,
            count=self.count,
            gp_params=params,
            Kinv=Kinv,
            weights=self.weights,
            ls_scaler=self.ls_scaler,
        )
        return new_repertoire

    def fit_gp(self, n_steps=1000):
        fit = self.fitnesses
        is_empty = fit == -jnp.inf
        count = self.count[~is_empty]
        w = jnp.diag(1 / count)
        # Set weights
        new_repertoire = WeightedGPRepertoire(
            descriptors=self.descriptors,
            fitnesses=self.fitnesses,
            genotypes=self.genotypes,
            centroids=self.centroids,
            count=self.count,
            gp_params=self.gp_params,
            Kinv=self.Kinv,
            weights=w,
            ls_scaler=self.ls_scaler,
        )
        return new_repertoire._fit_weighted_gp(n_steps=n_steps)
