import hydra

from typing import Dict, Tuple
from typing import Any, Dict, Iterable, List, Optional, Tuple

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.80"
# Jax floating point precision
# os.environ["JAX_ENABLE_X64"] = "True"

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

import jax 
import jax.numpy as jnp
from typing import Dict
from warnings import warn

from qdax.core.map_elites import MAPElites
from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids, MapElitesRepertoire

from qdax import environments
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.tasks.brax_envs import (
    scoring_function_brax_envs,
    reset_based_scoring_function_brax_envs,
    make_policy_network_play_step_fn_brax
)
from qdax.custom_types import EnvState, Params, RNGKey
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    compute_cvt_centroids,
)

from qdax_es.core.custom_repertoire_mapelites import CustomMAPElites
from qdax_es.core.containers.gp_repertoire import GPRepertoire
from qdax_es.core.containers.mae_repertoire import MAERepertoire

from qdax_es.utils.setup import setup_qd
from qdax_es.core.emitters.jedi_emitter import JEDiEmitter
from qdax_es.core.emitters.jedi_pool_emitter import GPJEDiPoolEmitter, UniformJEDiPoolEmitter
from qdax_es.utils.restart import FixedGens, ConvergenceRestarter, DualConvergenceRestarter

from qdax_es.utils.count_plots import plot_archive_value
from qdax_es.utils.plotting import plot_map_elites_results 
import matplotlib
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from evosax import Strategies

from factories.jedi import JEDi_factory, plot_results_jedi
from factories.cmame import CMAME_factory, plot_results_cmame

# Check there is a gpu
assert jax.device_count() > 0, "No GPU found"


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main_jedi(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    task = cfg.task
    algo = cfg.algo
    import os
    os.makedirs(cfg.plots_dir, exist_ok=True)
    
    if algo.algo == "jedi":
        (
            min_bd, 
            max_bd, 
            random_key, 
            map_elites, 
            emitter, 
            repertoire, 
            emitter_state,
            plot_prefix
            ) = JEDi_factory(cfg)
        
        plot_results = plot_results_jedi

    elif algo.algo in ["cmame"]:
        print(algo.plotting.algo_name)
        (
            min_bd, 
            max_bd, 
            random_key, 
            map_elites, 
            emitter, 
            repertoire, 
            emitter_state,
            plot_prefix
            ) = CMAME_factory(cfg)
        
        plot_results = plot_results_cmame
        
    else:
        raise ValueError(f"algo.type should be jedi, got {algo.type}")

    num_iterations = int(task.total_evaluations / emitter.batch_size / cfg.steps) 

    # scan_update = jax.jit(map_elites.scan_update)
    metrics = {}
    iter = 0
    for step in tqdm(range(cfg.steps)):
        (repertoire, emitter_state, random_key,), step_metrics = jax.lax.scan(
            map_elites.scan_update,
            (repertoire, emitter_state, random_key),
            (),
            length=int(num_iterations),
        )
        if metrics:
            for k in step_metrics.keys():
                metrics[k] = jnp.concatenate([metrics[k], step_metrics[k]])
        else:
            metrics = step_metrics
        iter += num_iterations

        plot_results(
            repertoire,
            emitter_state,
            cfg,
            min_bd, 
            max_bd,
            step
        )


    # ## Plot results

    # env_steps = jnp.arange(num_iterations * cfg.steps) * emitter.batch_size * cfg.task.episode_length 
    evals = jnp.arange(num_iterations * cfg.steps) * emitter.batch_size

    fig, axes = plot_map_elites_results(
        evals=evals,
        metrics=metrics,
        repertoire=repertoire,
        min_bd=min_bd,
        max_bd=max_bd,
    )

    # main title
    plt.suptitle(f"{cfg.algo.plotting.algo_name} in {cfg.task.plotting.task_name}", fontsize=20)

    # udpate this variable to save your results locally
    savefig = True
    if savefig:
        # figname = os.path.join(cfg.plots_dir, f"{cfg.task.env_name}_{'W' if cfg.algo.weighted_gp else ''}JEDi_"  + str(cfg.algo.wtfs_alpha) + ".png")
        figname = f"{cfg.plots_dir}/{cfg.task.env_name}/{plot_prefix}"+ "_results.png"
        # create folder if it does not exist
        import os
        os.makedirs(os.path.dirname(figname), exist_ok=True)
        print("Save figure in: ", figname)
        plt.savefig(figname, bbox_inches="tight")

    plot_results(
            repertoire,
            emitter_state,
            cfg,
            min_bd, 
            max_bd,
            step="end"
        )


if __name__ == "__main__":
    main_jedi()
