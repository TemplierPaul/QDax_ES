from functools import partial

import jax
import jax.numpy as jnp
from jax import jit

from flax.struct import dataclass as fdataclass
from flax.struct import PyTreeNode
import optax 

EMPTY_WEIGHT = 1e3
DEFAULT_X = 10

learning_rate = 0.01
optimizer = optax.adam(learning_rate)

@fdataclass
class RBFParams:
    sigma: float = 1.0
    lengthscale: float = 1.0
    obs_noise_sigma: float = 1.0

    @classmethod
    def random_params(cls, key):
        keys = jax.random.split(key, 3)
        sigma = jax.random.uniform(keys[0], minval=0, maxval=1)
        lengthscale = jax.random.uniform(keys[1], minval=0, maxval=1)
        obs_noise_sigma = jax.random.uniform(keys[2], minval=0, maxval=1)
        return cls(sigma, lengthscale, obs_noise_sigma)

@jit
def rbf_kernel(params, x1, x2):
    """RBF kernel with x in R^D"""
    sigma = params.sigma
    lengthscale = params.lengthscale
    return sigma**2 * jnp.exp(-0.5 * jnp.sum((x1 - x2)**2) / (lengthscale**2))


def masked_mean(x, mask):
    return jnp.sum(jnp.where(mask, x, 0)) / jnp.sum(mask)

class GPState(PyTreeNode):
    kernel_params: RBFParams
    Kinv: jnp.ndarray
    x: jnp.ndarray = None
    y: jnp.ndarray = None
    weights: jnp.ndarray = None
    weighted: bool=False 
    mask: jnp.ndarray = None

    @classmethod
    def init(cls, x, y, weighted, count, empty_weight = EMPTY_WEIGHT):
        kernel_params = RBFParams()
        mask = count > 0

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

        weights = jnp.where(
            weighted,
            jnp.where(mask, 1/count, empty_weight), # 1/count is the weight
            jnp.where(mask, 1.0, empty_weight),
        )
        weights = jnp.diag(weights)

        default_Kinv = jnp.eye(x.shape[0])

        return cls(
            kernel_params=kernel_params,
            Kinv=default_Kinv,
            x=x,
            y=y,
            weights=weights,
            weighted=weighted,
            mask=mask
        )
    
    @classmethod
    def init_from_repertoire(cls, repertoire, weighted=False):
        x = repertoire.descriptors
        y = repertoire.fitnesses
        count = repertoire.count
        return cls.init(x, y, weighted, count)
    

@jit
def compute_K(params, X, weights):
    """Compute the kernel matrix K using vmap"""
    # K = jax.vmap(rbf_kernel, in_axes=(None, 0, 1))(params, X, X)
    K = jax.vmap(lambda x1: jax.vmap(lambda x2: rbf_kernel(params, x1, x2))(X))(X)
    return K + params.obs_noise_sigma**2 * weights

@jit
def neg_marginal_likelihood(params, X, Y, weights):
    K = compute_K(params, X, weights) 
    Kinv = jnp.linalg.inv(K)

    Y_mean = jnp.mean(Y)
    Y_norm = Y - Y_mean
    # jax.debug.print("Mean {}, new mean {}", Y_mean, jnp.mean(Y_norm))

    data_fit = Y_norm.T @ Kinv @ Y_norm

    complexity_penalty = jnp.log(jnp.linalg.det(K))

    n = jnp.sum(1/jnp.diag(weights))
    constant_term = n * jnp.log(2 * jnp.pi)

    log_marginal_likelihood = -0.5 * (data_fit + complexity_penalty + constant_term)
    return - log_marginal_likelihood

grad_neg_marginal_likelihood = jit(jax.grad(neg_marginal_likelihood))

@jit
def train_loop(gp_state, opt_state):
    grads = grad_neg_marginal_likelihood(
        gp_state.kernel_params, 
        gp_state.x, 
        gp_state.y, 
        gp_state.weights,
        )
    # jax.debug.print("grads {}", grads)
    
    # update parameters 
    updates, opt_state = optimizer.update(grads, opt_state)
    # jax.debug.print("updates {}", updates)
    params = optax.apply_updates(
        gp_state.kernel_params, 
        updates
        )
    # jax.debug.print("params {}", params)
    
    return gp_state.replace(kernel_params=params), opt_state
        
@jit
def train_loop_scan(carry, _):
    # Unroll the loop
    gp_state, opt_state, is_nan = carry
    new_gp_state, opt_state = jax.lax.cond(
        is_nan,
        lambda x: (gp_state, opt_state),
        lambda x: train_loop(gp_state, opt_state),
        None
    )

    # check nan
    new_nan = jnp.isnan(new_gp_state.kernel_params.sigma)
    is_nan = jnp.logical_or(is_nan, new_nan)

    return (gp_state, opt_state, is_nan), None

@jit
def get_init_state(gp_state):
    # Test initial params to make sure they are valid
    n_tests = 128
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, num=n_tests)
    init_params = jax.vmap(RBFParams.random_params)(keys)

    def test_init(gp_state, params):
        """Test if the init params are valid"""
        gp_state = gp_state.replace(kernel_params=params)
        opt_state = optimizer.init(params)
        gp_state, opt_state = train_loop(
            gp_state, opt_state)
        return ~jnp.isnan(gp_state.kernel_params.sigma) 
    
    valid_params = jax.vmap(
    test_init,
    in_axes=(None, 0)
    )(
        gp_state,
        init_params
        )
    # jax.debug.print("Valid params: {}", valid_params.sum())

    # pick first valid one
    index = jnp.argmax(valid_params)

    params = jax.tree_util.tree_map(
        lambda x: x[index],
        init_params,
    )
    return gp_state.replace(kernel_params=params)

@partial(jit, static_argnames=("num_steps",))
def train_gp(gp_state, num_steps):
    gp_state = get_init_state(gp_state)
    opt_state = optimizer.init(gp_state.kernel_params)

    carry = (gp_state, opt_state, False)
    (gp_state, opt_state, is_nan), _ = jax.lax.scan(
        train_loop_scan,
        carry,
        jnp.arange(num_steps),
    )
    return gp_state

def gp_predict(gp_state, x_new):
    Kinv = gp_state.Kinv
    X, Y = gp_state.x, gp_state.y
    params = gp_state.kernel_params

    Y_mean = masked_mean(Y, gp_state.mask)
    # jax.debug.print("Y_mean {}", Y_mean)
    Y_norm = Y - Y_mean
    # jax.debug.print("Y_norm {}", Y_norm)

    Kx = jax.vmap(lambda x: rbf_kernel(params, x_new, x))(X)

    # compute mean prediction
    f_mean = Y_mean + Kx.T @ Kinv @ Y_norm

    # compute variance prediction
    kxx = rbf_kernel(params, x_new, x_new)
    f_var = kxx - Kx.T @ Kinv @ Kx
    return f_mean, f_var

def gp_batch_predict(gp_state, x_new):
    return jax.vmap(partial(gp_predict, gp_state))(x_new)