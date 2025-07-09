import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from jax.scipy.linalg import cholesky, solve_triangular
from typing import Callable, Tuple, NamedTuple
import optax
from flax.struct import dataclass as fdataclass

EMPTY_WEIGHT = 1e4
DEFAULT_X = 100
EPSILON = 1e-4
JITTER = 1e-6

@fdataclass
class GPParams:
    """Parameters for the Gaussian Process"""
    kernel_params: dict
    noise_var: float

    def _learnable_params(self):
        """Return only the trainable parameters (kernel and noise)"""
        return self

def constrain_params(params: GPParams) -> GPParams:
    """Apply constraints to ensure parameters stay in valid ranges"""
    # Use softplus to ensure positive parameters
    constrained_kernel_params = {}
    for key, value in params.kernel_params.items():
        if key in ['sigma', 'length_scale']:
            # Softplus with lower bound
            constrained_kernel_params[key] = jnp.log(1 + jnp.exp(value)) + 1e-6
        else:
            constrained_kernel_params[key] = value
    
    noise_var = jnp.log(1 + jnp.exp(params.noise_var)) + 1e-6
    
    return GPParams(
        kernel_params=constrained_kernel_params,
        noise_var=noise_var
    )

def masked_mean(x, mask):
    return jnp.sum(jnp.where(mask, x, 0)) / jnp.sum(mask)

class GaussianProcess:
    """
    Gaussian Process for N-dimensional to 1D regression.
    Fully jittable implementation using JAX.
    """
    
    def __init__(self, kernel_fn: Callable, learning_rate=0.01):
        """
        Initialize GP with a kernel function.
        
        Args:
            kernel_fn: Function that takes (params, x1, x2) and returns covariance
        """
        self.kernel_fn = kernel_fn
        self.optimizer = optax.adam(learning_rate)
        
        # JIT compile core methods
        self.jit_log_likelihood = jit(self._log_likelihood)
        self.jit_predict = jit(self._predict)
        self.jit_loss_and_grad = jit(self._loss_and_grad)
    
    def _log_likelihood(self, params: GPParams, X: jnp.ndarray, y: jnp.ndarray) -> float:
        """
        Compute log marginal likelihood.
        
        Args:
            params: GP parameters
            X: Input data (N, D)
            y: Target data (N,)
            
        Returns:
            Log marginal likelihood
        """
        # Apply parameter constraints
        params = constrain_params(params)
        
        N = X.shape[0]
        
        # Compute kernel matrix
        K = self._compute_kernel_matrix(params.kernel_params, X)
        
        # Add noise and jitter for numerical stability
        jitter = 1e-6
        K += (params.noise_var + jitter) * jnp.eye(N)
        
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
    
    def _compute_kernel_matrix(self, kernel_params: dict, X: jnp.ndarray) -> jnp.ndarray:
        """Compute full kernel matrix K(X, X)"""
        def kernel_row(xi):
            return vmap(lambda xj: self.kernel_fn(kernel_params, xi, xj))(X)
        
        return vmap(kernel_row)(X)
    
    def _compute_kernel_vector(self, kernel_params: dict, X: jnp.ndarray, x_new: jnp.ndarray) -> jnp.ndarray:
        """Compute kernel vector k(X, x_new)"""
        return vmap(lambda xi: self.kernel_fn(kernel_params, xi, x_new))(X)
    
    def _predict(self, params: GPParams, X: jnp.ndarray, y: jnp.ndarray, 
                X_new: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Make predictions at new points.
        
        Args:
            params: GP parameters
            X: Training inputs (N, D)
            y: Training targets (N,)
            X_new: Test inputs (M, D)
            
        Returns:
            mean: Predictive mean (M,)
            var: Predictive variance (M,)
        """
        # Apply parameter constraints
        params = constrain_params(params)
        
        N = X.shape[0]
        
        # Compute training kernel matrix
        K = self._compute_kernel_matrix(params.kernel_params, X)
        jitter = 1e-6
        K += (params.noise_var + jitter) * jnp.eye(N)
        
        # Cholesky decomposition
        L = cholesky(K, lower=True)
        
        # Solve for alpha
        alpha = solve_triangular(L, y, lower=True)
        alpha = solve_triangular(L.T, alpha, lower=False)
        
        def predict_single(x_new):
            # Compute k_star
            k_star = self._compute_kernel_vector(params.kernel_params, X, x_new)
            
            # Predictive mean
            mean = jnp.dot(k_star, alpha)
            
            # Predictive variance
            v = solve_triangular(L, k_star, lower=True)
            k_star_star = self.kernel_fn(params.kernel_params, x_new, x_new)
            var = k_star_star - jnp.sum(v**2)
            
            # Ensure variance is positive
            var = jnp.maximum(var, 1e-8)
            
            return mean, var
        
        mean, var = vmap(predict_single)(X_new)
        return mean, var
    
    def _loss_and_grad(self, params: GPParams, X: jnp.ndarray, y: jnp.ndarray) -> Tuple[float, GPParams]:
        """Compute loss and gradients - this can be JIT compiled"""
        return jax.value_and_grad(lambda p: -self._log_likelihood(p, X, y))(params)
    
    def fit_one(self, X: jnp.ndarray, y: jnp.ndarray, params: GPParams, opt_state: optax.OptState) -> Tuple[GPParams, list]:
        # Use JIT compiled loss and grad computation
        loss, grads = self.jit_loss_and_grad(params, X, y)
        
        # Apply optimizer update
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # Skip the update if loss is nan
        params, opt_state = jax.lax.cond(
            jnp.isnan(loss) | jnp.isinf(loss),
            lambda: (params, opt_state),
            lambda: (new_params, new_opt_state),
        )
        # jax.debug.print("Step: {}, Loss: {}", i, loss)
        return params, opt_state
    
    def fit(self, X: jnp.ndarray, y: jnp.ndarray, params_init: GPParams = None, 
           n_steps: int = 1000, learning_rate: float = 0.01) -> GPParams:
        """
        Fit the GP by optimizing hyperparameters.
        
        Args:
            X: Training inputs (N, D)
            y: Training targets (N,)
            params_init: Initial parameters
            n_steps: Number of optimization steps
            learning_rate: Learning rate for Adam optimizer
            
        Returns:
            Optimized parameters and loss history
        """
        if params_init is None:
            params_init = GPParams(
                kernel_params={'sigma': 0.0, 'length_scale': 0.0},  # log(1) = 0, will become ~1 after softplus
                noise_var=-2.0  # log(0.1) ≈ -2.3, will become ~0.1 after softplus
            )

        # Initialize with unconstrained parameters (will be constrained in loss function)
        opt_state = self.optimizer.init(params_init)
        
        params = params_init
        
        for i in range(n_steps):
            params, opt_state = self.fit_one(X, y, params, opt_state)
        
        # Return constrained parameters
        return constrain_params(params)
    
    def from_repertoire(self, repertoire, params: GPParams):
        x = repertoire.descriptors
        y = jnp.squeeze(repertoire.fitnesses)

        mask = y != -jnp.inf

        # Make up fake x values for masked data
        default_x = jnp.ones(x.shape[0]) * DEFAULT_X
        x = jax.vmap(
            lambda i: jnp.where(mask, i, default_x + i),
            in_axes=1,
            out_axes=1
        )(x)

        # Make up fake y values for masked data
        default_y = masked_mean(y, mask)

        y = jnp.where(mask, y, default_y)

        # Norm y
        y_min, y_max = y.min(), y.max()
        y = (y-y_min) / (y_max - y_min + EPSILON)
        return params, x, y
    
    def predict(self, params: GPParams, X: jnp.ndarray, y: jnp.ndarray, 
               X_new: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Make predictions (wrapper for jitted version)"""
        return self.jit_predict(params, X, y, X_new)
    
    def log_likelihood(self, params: GPParams, X: jnp.ndarray, y: jnp.ndarray) -> float:
        """Compute log likelihood (wrapper for jitted version)"""
        return self.jit_log_likelihood(params, X, y)
    
    def fit_predict(self, X: jnp.ndarray, y: jnp.ndarray, params_init: GPParams = None, 
           n_steps: int = 1000, learning_rate: float = 0.01) -> Tuple[GPParams, jnp.ndarray, jnp.ndarray]:
        """
        Fit the GP and make predictions.
        """
        params = self.fit(X, y, params_init, n_steps, learning_rate)
        # print(f"Fitted params: {params}")
        y_pred, y_var = self.predict(params, X, y, X)
        return params, y_pred, y_var


# Kernel functions
def rbf_kernel(params: dict, x1: jnp.ndarray, x2: jnp.ndarray) -> float:
    """
    RBF (Gaussian) kernel: k(x1, x2) = sigma^2 * exp(-||x1 - x2||^2 / (2 * l^2))
    
    Args:
        params: {'sigma': output_scale, 'length_scale': length_scale}
        x1, x2: Input vectors
    """
    diff = x1 - x2
    distance_sq = jnp.sum(diff**2)
    return params['sigma']**2 * jnp.exp(-distance_sq / (2 * params['length_scale']**2))

def matern32_kernel(params: dict, x1: jnp.ndarray, x2: jnp.ndarray) -> float:
    """
    Matérn 3/2 kernel
    """
    diff = x1 - x2
    distance = jnp.sqrt(jnp.sum(diff**2) + 1e-12)  # Add small epsilon for numerical stability
    sqrt3_d_l = jnp.sqrt(3) * distance / params['length_scale']
    return params['sigma']**2 * (1 + sqrt3_d_l) * jnp.exp(-sqrt3_d_l)

def linear_kernel(params: dict, x1: jnp.ndarray, x2: jnp.ndarray) -> float:
    """
    Linear kernel: k(x1, x2) = sigma^2 * (x1^T x2 + c)
    """
    return params['sigma']**2 * (jnp.dot(x1, x2) + params.get('c', 0.0))

def polynomial_kernel(params: dict, x1: jnp.ndarray, x2: jnp.ndarray) -> float:
    """
    Polynomial kernel: k(x1, x2) = sigma^2 * (x1^T x2 + c)^degree
    """
    dot_prod = jnp.dot(x1, x2) + params.get('c', 1.0)
    return params['sigma']**2 * (dot_prod ** params['degree'])

