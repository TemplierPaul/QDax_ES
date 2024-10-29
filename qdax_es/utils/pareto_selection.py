import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

def pareto_filter(f1, f2):
    def is_dominated(x):
        """Check if a point is dominated."""
        return jnp.any((f1 > x[0]) & (f2 > x[1]))

    # zip f1 and f2
    f = jnp.vstack([f1, f2]).T
    dominated = jax.vmap(is_dominated)(f)
    return ~ dominated

def get_pareto_indices(f1, f2, n_points= 10, max_depth=10):
    def pareto_indices_scan(carry, depth):
        pareto_depth, f1, f2 = carry
        
        # Get non-dominated condition
        f = pareto_filter(f1, f2)
        
        # Update points on the current pareto front
        pareto_depth = jnp.where(f, depth, pareto_depth)
        # Ignore points already in the pareto front
        f1 = jnp.where(pareto_depth == jnp.inf, f1, -jnp.inf)
        f2 = jnp.where(pareto_depth == jnp.inf, f2, -jnp.inf)

        return (pareto_depth, f1, f2), pareto_depth
    
    # Initial state
    carry = (jnp.inf * jnp.ones(f1.shape), f1, f2)
    depths = jnp.arange(max_depth)
    carry, _ = jax.lax.scan(pareto_indices_scan, carry, depths)
    (pareto_depth, f1, f2) = carry

    # Count pareto_depth == 0
    first_front = jnp.sum(pareto_depth < 1)
    # jax.debug.print("First pareto front: {}", first_front)

    # Get indices of the pareto front
    pareto_indices = jnp.argsort(pareto_depth)

    # Get first n_points
    return pareto_indices[:n_points]

def stoch_get_pareto_indices(f1, f2, random_key, n_points= 10, max_depth=10):
    # jax.debug.print("f1: {}", f1.shape)
    # jax.debug.print("f2: {}", f2.shape)
    def pareto_indices_scan(carry, depth):
        pareto_depth, f1, f2 = carry
        
        # Get non-dominated condition
        f = pareto_filter(f1, f2)
        
        # Update points on the current pareto front
        pareto_depth = jnp.where(f, depth, pareto_depth)
        # Ignore points already in the pareto front
        f1 = jnp.where(pareto_depth == jnp.inf, f1, -jnp.inf)
        f2 = jnp.where(pareto_depth == jnp.inf, f2, -jnp.inf)

        return (pareto_depth, f1, f2), pareto_depth
    
    # Initial state
    carry = (jnp.inf * jnp.ones(f1.shape), f1, f2)
    depths = jnp.arange(max_depth)
    carry, _ = jax.lax.scan(pareto_indices_scan, carry, depths)
    (pareto_depth, f1, f2) = carry
    # jax.debug.print("Pareto depth: {}", pareto_depth.shape)

    # Add uniform noise to pareto_depth
    noise = jax.random.uniform(
        key=random_key, shape=f1.shape) * 1e-2
    # jax.debug.print("Noise: {}", noise.shape)

    pareto_depth += noise

    # Count pareto_depth == 0
    first_front = jnp.sum(pareto_depth < 1)
    # jax.debug.print("First pareto front: {}", first_front)

    # Get indices of the pareto front
    pareto_indices = jnp.argsort(pareto_depth)

    # Get first n_points
    return pareto_indices[:n_points]