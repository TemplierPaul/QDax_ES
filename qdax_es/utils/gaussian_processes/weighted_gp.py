import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from jax.scipy.linalg import cholesky, solve_triangular
from typing import Callable, Tuple, NamedTuple
import optax
from flax.struct import dataclass as fdataclass

# Import base GP implementation
from qdax_es.utils.gaussian_processes.base_gp import (
    GaussianProcess, 
    GPParams, 
    constrain_params,
    masked_mean,
    DEFAULT_X,
    EPSILON,
    JITTER,
    EMPTY_WEIGHT,
)

@fdataclass
class WeightedGPParams(GPParams):
    """Parameters for the Weighted Gaussian Process"""
    weights: jnp.ndarray

    def _learnable_params(self):
        """Return only the trainable parameters (kernel and noise)"""
        return GPParams(
            kernel_params=self.kernel_params,
            noise_var=self.noise_var
        )

class WeightedGaussianProcess(GaussianProcess):
    """
    Weighted Gaussian Process for N-dimensional to 1D regression.
    Extends the base GP to incorporate per-point weights.
    """    

    def __init__(self, kernel_fn: Callable, max_count=1e4):
        """
        Initialize GP with a kernel function.
        
        Args:
            kernel_fn: Function that takes (params, x1, x2) and returns covariance
        """
        # super init with kernel function
        super().__init__(kernel_fn)
        self.max_count = max_count
        self.jit_loss_and_grad_weighted = jit(self._loss_and_grad_weighted)


    def _log_likelihood_weighted(self, params: GPParams, X: jnp.ndarray, y: jnp.ndarray, weights: jnp.ndarray) -> float:
        """
        Compute weighted log marginal likelihood.
        
        Args:
            params: Weighted GP parameters
            X: Input data (N, D)
            y: Target data (N,)
            
        Returns:
            Log marginal likelihood
        """
        # Apply parameter constraints to base params
        base_params = params._learnable_params()
        base_params = constrain_params(base_params)
        
        N = X.shape[0]
        
        # Compute kernel matrix
        K = self._compute_kernel_matrix(base_params.kernel_params, X)
        
        # Add weighted noise and jitter for numerical stability
        jitter = 1e-6
        # Weights are incorporated as diagonal scaling of noise
        weighted_noise = base_params.noise_var * weights + jitter * jnp.eye(N)
        K += weighted_noise
        
        # Try Cholesky decomposition with error handling
        try:
            L = cholesky(K, lower=True)
        except:
            # If Cholesky fails, add more jitter
            K += 1e-3 * jnp.eye(N)
            L = cholesky(K, lower=True)
        
        # Solve L @ alpha = y
        alpha = solve_triangular(L, y, lower=True)
        
        # Log likelihood
        log_det = 2 * jnp.sum(jnp.log(jnp.diag(L)))
        quad_form = jnp.sum(alpha**2)
        
        log_likelihood = -0.5 * (quad_form + log_det + N * jnp.log(2 * jnp.pi))
        
        # Return negative inf if we get NaN
        return jnp.where(jnp.isnan(log_likelihood), -jnp.inf, log_likelihood)
    
    def _predict(self, params: WeightedGPParams, X: jnp.ndarray, y: jnp.ndarray, 
                         X_new: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Make weighted predictions at new points.
        
        Args:
            params: Weighted GP parameters
            X: Training inputs (N, D)
            y: Training targets (N,)
            X_new: Test inputs (M, D)
            
        Returns:
            mean: Predictive mean (M,)
            var: Predictive variance (M,)
        """
        # Apply parameter constraints to base params
        base_params = params._learnable_params()
        base_params = constrain_params(base_params)
        
        N = X.shape[0]
        
        # Compute training kernel matrix with weights
        K = self._compute_kernel_matrix(base_params.kernel_params, X)
        jitter = 1e-6
        weighted_noise = base_params.noise_var * params.weights + jitter * jnp.eye(N)
        K += weighted_noise
        
        # Cholesky decomposition
        L = cholesky(K, lower=True)
        
        # Solve for alpha
        alpha = solve_triangular(L, y, lower=True)
        alpha = solve_triangular(L.T, alpha, lower=False)
        
        def predict_single(x_new):
            # Compute k_star
            k_star = self._compute_kernel_vector(base_params.kernel_params, X, x_new)
            
            # Predictive mean
            mean = jnp.dot(k_star, alpha)
            
            # Predictive variance
            v = solve_triangular(L, k_star, lower=True)
            k_star_star = self.kernel_fn(base_params.kernel_params, x_new, x_new)
            var = k_star_star - jnp.sum(v**2)
            
            # Ensure variance is positive
            var = jnp.maximum(var, 1e-8)
            
            return mean, var
        
        mean, var = vmap(predict_single)(X_new)
        return mean, var

    def _loss_and_grad_weighted(self, params: WeightedGPParams, X: jnp.ndarray, y: jnp.ndarray, weights: jnp.ndarray) -> Tuple[float, WeightedGPParams]:
        """Compute weighted loss and gradients - this can be JIT compiled"""
        return jax.value_and_grad(lambda p: -self._log_likelihood_weighted(p, X, y, weights))(params)

    def fit_one(self, X: jnp.ndarray, y: jnp.ndarray, params: WeightedGPParams,
                        opt_state: optax.OptState) -> Tuple[WeightedGPParams, optax.OptState]:
        """Single optimization step for weighted GP"""
        # Only update kernel_params and noise_var, keep weights fixed
        trainable_params = params._learnable_params()
        # print(f"Training with params: {trainable_params}")
        
        # Use JIT compiled loss and grad computation
        loss, grads = self.jit_loss_and_grad_weighted(trainable_params, X, y, params.weights)
        
        # print("Grads:", grads)
        # print("Opt state:", opt_state)
        # Apply optimizer update
        updates, new_opt_state = self.optimizer.update(grads, opt_state, trainable_params)
        new_trainable_params = optax.apply_updates(trainable_params, updates)
        
        # Reconstruct full params with updated trainable params and original weights
        new_params = WeightedGPParams(
            kernel_params=new_trainable_params.kernel_params,
            noise_var=new_trainable_params.noise_var,
            weights=params.weights
        )
        
        # Skip the update if loss is nan
        params, opt_state = jax.lax.cond(
            jnp.isnan(loss) | jnp.isinf(loss),
            lambda: (params, opt_state),
            lambda: (new_params, new_opt_state),
        )
        
        return params, opt_state
    
    def fit_weighted(self, X: jnp.ndarray, y: jnp.ndarray, weights: jnp.ndarray,
                    params_init: WeightedGPParams = None, n_steps: int = 1000) -> WeightedGPParams:
        """
        Fit the weighted GP by optimizing hyperparameters.
        
        Args:
            X: Training inputs (N, D)
            y: Training targets (N,)
            weights: Per-point weights (N, N) diagonal matrix or (N,) vector
            params_init: Initial parameters
            n_steps: Number of optimization steps
            learning_rate: Learning rate for Adam optimizer
            
        Returns:
            Optimized parameters
        """
        # Handle weights format
        if weights.ndim == 1:
            weights = jnp.diag(weights)
        
        if params_init is None:
            params_init = WeightedGPParams(
                kernel_params={'sigma': 0.0, 'length_scale': 0.0},
                noise_var=-2.0,
                weights=weights
            )
        
        # Initialize optimizer with only trainable parameters
        trainable_params = params_init._learnable_params()
        opt_state = self.optimizer.init(trainable_params)
        
        params = params_init
        
        for i in range(n_steps):
            params, opt_state = self.fit_one(X, y, params, opt_state)
        
        # Return constrained parameters
        base_params = GPParams(
            kernel_params=params.kernel_params,
            noise_var=params.noise_var
        )
        constrained_base = constrain_params(base_params)
        
        return WeightedGPParams(
            kernel_params=constrained_base.kernel_params,
            noise_var=constrained_base.noise_var,
            weights=params.weights
        )

    def fit_predict_weighted(self, X: jnp.ndarray, y: jnp.ndarray, weights: jnp.ndarray,
                            params_init: WeightedGPParams = None, n_steps: int = 1000, 
                            learning_rate: float = 0.01) -> Tuple[WeightedGPParams, jnp.ndarray, jnp.ndarray]:
        """
        Fit the weighted GP and make predictions.
        """
        params = self.fit_weighted(X, y, weights, params_init, n_steps, learning_rate)
        y_pred, y_var = self.predict(params, X, y, X)
        return params, y_pred, y_var

    def from_repertoire(self, repertoire, params: GPParams):
        # Super method to handle repertoire data
        params, x, y = super().from_repertoire(repertoire, params)

        count = repertoire.count
        mask = count > 0

        # Normalize count
        count_factor = jnp.where(
            count.max() > self.max_count,
            self.max_count / count.max(),
            1
        )
        count = count * count_factor
        count = jnp.clip(count, 1e-3, 1e3)

        ## THIS REMOVES THE WEIGHTING, TO BE FIXED
        # count = jnp.where(mask, 1.0, 0.0)
        
        # Compute weights
        weights = jnp.where(mask, 1/count, EMPTY_WEIGHT)
        weights = jnp.diag(weights)

        params = WeightedGPParams(
            kernel_params=params.kernel_params,
            noise_var=params.noise_var,
            weights=weights
        )

        return params, x, y