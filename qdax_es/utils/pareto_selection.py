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

def get_pareto_depths(f1, f2, max_depth=None):
    # Get all the pareto fronts
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
    if max_depth is None:
        max_depth = len(f1)  # Default to the number of points
    carry = (jnp.inf * jnp.ones(f1.shape), f1, f2)
    depths = jnp.arange(max_depth)

    # carry, _ = jax.lax.scan(pareto_indices_scan, carry, depths)
    # In a for loop instead
    for depth in depths:
        carry, _ = pareto_indices_scan(carry, depth)
        (pareto_depth, f1, f2) = carry
        # Check if any points are not in a front
        if jnp.all(pareto_depth < jnp.inf):
            print(f"All points are in a front at depth {depth}.")
            break

    return pareto_depth

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

def stoch_get_pareto_indices(f1, f2, key, n_points= 10, max_depth=10):
    """
    Sample n_points from the first pareto fronts. Solutions are shuffled in each pareto front
    by adding a random noise.
    """
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
    # print(f1.shape, f2.shape)
    carry = (jnp.inf * jnp.ones(f1.shape), f1, f2)
    depths = jnp.arange(max_depth)
    carry, _ = jax.lax.scan(pareto_indices_scan, carry, depths)
    (pareto_depth, f1, f2) = carry
    # jax.debug.print("Pareto depth: {}", pareto_depth.shape)

    # Add uniform noise to pareto_depth
    noise = jax.random.uniform(
        key=key, shape=f1.shape) * 1e-2
    # jax.debug.print("Noise: {}", noise.shape)

    pareto_depth += noise

    # Count pareto_depth == 0
    first_front = jnp.sum(pareto_depth < 1)
    # jax.debug.print("First pareto front: {}", first_front)

    # Get indices of the pareto front
    pareto_indices = jnp.argsort(pareto_depth)

    # Get first n_points
    return pareto_indices[:n_points]