import jax 
import jax.numpy as jnp
import hydra 
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt

from qdax.core.map_elites import MAPElites
from qdax_bench.utils.setup import setup_qd
from qdax_bench.utils.plotting import plot_2d_map_elites_repertoire

from qdax_es.core.containers.count_repertoire import CountMapElitesRepertoire
from qdax_es.core.emitters.jedi_emitter import ConstantScheduler, LinearScheduler, RandomScheduler
from qdax_es.utils.count_plots import plot_2d_count, plot_archive_value
from qdax_es.utils.target_selection import GPSelector, GPSelectorState

class JEDiFactory:
    def build(self, cfg):
        task = cfg["task"]
        algo = cfg["algo"]

        (
            centroids, 
            min_bd, 
            max_bd, 
            scoring_fn, 
            metrics_fn, 
            init_variables_func, 
            key
        ) = setup_qd(cfg)

        batch_size = algo["params"]["population_size"] * algo["pool_size"]
        num_iterations = int(task["total_evaluations"] / batch_size / cfg["num_loops"])
        print("Iterations per step: ", num_iterations)
        print("Iterations: ", num_iterations * cfg["num_loops"])

        # EvosaxEmitterAll
        repertoire_init_fn = hydra.utils.instantiate(
            algo["repertoire_init"]
        )
        print("Repertoire init: ", repertoire_init_fn)

        if cfg["algo"]["params"]["alpha"] == "decay":
            alpha_scheduler = LinearScheduler(0.8, 0.0, num_iterations * cfg["num_loops"])
        elif cfg["algo"]["params"]["alpha"] == "random":
            alpha_scheduler = RandomScheduler()
        else:
            # Assert it is int or float
            assert isinstance(cfg["algo"]["params"]["alpha"], (int, float)), f"Alpha should be int or float if constant, got {cfg['algo']['params']['alpha']}"
            alpha_scheduler = ConstantScheduler(cfg["algo"]["params"]["alpha"])

        emitter_func = hydra.utils.instantiate(algo["emitter"])

        internal_emitter = emitter_func(
            centroids=centroids,
            init_variables_func=init_variables_func,
            alpha_scheduler=alpha_scheduler,
        )

        pool_emitter = hydra.utils.instantiate(cfg["algo"]["pool_emitter"])
        print("Pool emitter: ", pool_emitter)

        emitter = pool_emitter(
            emitter=internal_emitter
            )

        repertoire_init_fn = hydra.utils.instantiate(cfg["algo"]["repertoire_init"])
        print("Repertoire init: ", repertoire_init_fn)

        map_elites = MAPElites(
            scoring_function=scoring_fn,
            emitter=emitter,
            metrics_function=metrics_fn,
            repertoire_init=repertoire_init_fn
        )

        # with jax.disable_jit():
        key, var_key, subkey = jax.random.split(key, 3)

        init_variables = init_variables_func(var_key)

        # repertoire, emitter_state, init_metrics = map_elites.init(
        #     init_variables, 
        #     centroids, 
        #     subkey,
        # )

        plot_prefix = f"JEDi_" + str(cfg['algo']['params']['alpha'])

        return (
            min_bd, 
            max_bd, 
            key, 
            map_elites, 
            emitter, 
            init_variables, 
            centroids,
            plot_prefix,
            scoring_fn,    
            )

def plot_jedi(
        cfg, 
        emitter_state,
        metrics: Dict,
        repertoire: CountMapElitesRepertoire,
        min_descriptor: jnp.ndarray,
        max_descriptor: jnp.ndarray,
        title: str = "Final GP",
        path: str = "JEDi_plots"
    ):
    print("Plotting JEDi")
    n_cols = 2
    jedi_emitter_state = jax.tree.map(
        lambda x: x[0], 
        emitter_state.emitter_states
    )
    plot_gp = isinstance(jedi_emitter_state.target_selector_state, GPSelectorState)
    if plot_gp:
        n_cols = 5
        print("Plotting GP")

    fig, axes = plt.subplots(nrows=1, ncols=n_cols, figsize=(n_cols * 10, 10))

    # Fitness
    _, axes[0] = plot_2d_map_elites_repertoire(
        centroids=repertoire.centroids,
        repertoire_fitnesses=repertoire.fitnesses,
        minval=min_descriptor,
        maxval=max_descriptor,
        repertoire_descriptors=repertoire.descriptors,
        ax=axes[0],
    )
    max_fit = jnp.max(repertoire.fitnesses)
    axes[0].set_title(f"Fitness (max: {max_fit:.2f})")

    # Count
    axes[1] = plot_2d_count(
        repertoire, 
        min_descriptor, 
        max_descriptor, 
        log_scale=True, 
        ax=axes[1],
        colormap="plasma",
        )
    
    if plot_gp:
        target_selector = hydra.utils.instantiate(cfg.algo.target.target_selector)
        axes = target_selector.plot(
            jedi_emitter_state.target_selector_state,
            repertoire=repertoire,
            axes = axes[2:],
            min_descriptor=min_descriptor, 
            max_descriptor=max_descriptor,
        )
    plt.suptitle(title)
    plt.savefig(f"{cfg.plots_dir}/{path}", bbox_inches="tight")

    return fig, axes

    


