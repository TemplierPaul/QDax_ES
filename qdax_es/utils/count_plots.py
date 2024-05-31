from typing import Any, Dict, Iterable, List, Optional, Tuple
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import numpy as np

from qdax.utils.plotting import plot_2d_map_elites_repertoire, get_voronoi_finite_polygons_2d

from mpl_toolkits.axes_grid1 import make_axes_locatable


# Customize matplotlib params
font_size = 20
mpl_params = {
    "axes.labelsize": font_size,
    "axes.titlesize": font_size,
    "legend.fontsize": font_size,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "font.size": font_size,
    "text.usetex": False,
    "axes.titlepad": 10,
}

def plot_2d_count(
        repertoire, 
        min_bd, 
        max_bd, 
        log_scale=True, 
        ax=None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        colormap: str = "viridis",
        ):
    # Replace 0 by -inf in count
    count = repertoire.count
    count = jnp.where(count == 0, -jnp.inf, count)
    title = "Number of solutions tried per cell"
    if log_scale:
        # make log 10 scale where not -inf
        count = jnp.where(count != -jnp.inf, jnp.log10(count), count)
        title += " (log10)"

    # set the parameters
    mpl.rcParams.update(mpl_params)

        
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plot_2d_map_elites_repertoire(
        centroids=repertoire.centroids,
        repertoire_fitnesses=count,
        minval=min_bd,
        maxval=max_bd,
        repertoire_descriptors=repertoire.descriptors,
        ax=ax,
        # title=title,
        vmin=vmin,
        vmax=vmax,
        # colormap=colormap,
    )
    return ax


def plot_archive_value(
    repertoire,
    y,
    minval: jnp.ndarray,
    maxval: jnp.ndarray,
    title=None,
    ax: Optional[plt.Axes] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    centroids = repertoire.centroids
    # TODO: check it and fix it if needed
    grid_empty = y == -jnp.inf
    num_descriptors = centroids.shape[1]
    if num_descriptors != 2:
        raise NotImplementedError("Grid plot supports 2 descriptors only for now.")

    my_cmap = cm.viridis

    fitnesses = y
    if vmin is None:
        vmin = float(jnp.min(fitnesses[~grid_empty]))
    if vmax is None:
        vmax = float(jnp.max(fitnesses[~grid_empty]))

    mpl.rcParams.update(mpl_params)
    # create the plot object
    fig = None
    if ax is None:
        fig, ax = plt.subplots(facecolor="white", edgecolor="white")

    assert (
        len(np.array(minval).shape) < 2
    ), f"minval : {minval} should be float or couple of floats"
    assert (
        len(np.array(maxval).shape) < 2
    ), f"maxval : {maxval} should be float or couple of floats"

    if len(np.array(minval).shape) == 0 and len(np.array(maxval).shape) == 0:
        ax.set_xlim(minval, maxval)
        ax.set_ylim(minval, maxval)
    else:
        ax.set_xlim(minval[0], maxval[0])
        ax.set_ylim(minval[1], maxval[1])

    ax.set(adjustable="box", aspect="equal")

    # create the regions and vertices from centroids
    regions, vertices = get_voronoi_finite_polygons_2d(centroids)

    norm = Normalize(vmin=vmin, vmax=vmax)

    # fill the plot with contours
    for region in regions:
        polygon = vertices[region]
        ax.fill(*zip(*polygon), alpha=0.05, edgecolor="black", facecolor="white", lw=1)

    # fill the plot with the colors
    for idx, fitness in enumerate(fitnesses):
        if fitness > -jnp.inf:
            region = regions[idx]
            polygon = vertices[region]

            ax.fill(*zip(*polygon), alpha=0.8, color=my_cmap(norm(fitness)))

    # if descriptors are specified, add points location
    if repertoire.descriptors is not None:
        descriptors = repertoire.descriptors[~grid_empty]
        ax.scatter(
            descriptors[:, 0],
            descriptors[:, 1],
            c=fitnesses[~grid_empty],
            cmap=my_cmap,
            s=10,
            zorder=0,
        )

    # aesthetic
    ax.set_xlabel("Behavior Dimension 1")
    ax.set_ylabel("Behavior Dimension 2")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=my_cmap), cax=cax)
    cbar.ax.tick_params(labelsize=font_size)
    if title is not None:
        ax.set_title(title)
    ax.set_aspect("equal")
    return fig, ax
