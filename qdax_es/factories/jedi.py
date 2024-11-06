import jax 
import os
import matplotlib.pyplot as plt
from qdax_es.core.custom_repertoire_mapelites import CustomMAPElites
from qdax_es.core.containers.gp_repertoire import GPRepertoire

from qdax_es.utils.setup import setup_qd
from qdax_es.core.emitters.jedi_emitter import JEDiEmitter
from qdax_es.core.emitters.jedi_pool_emitter import (
    UniformJEDiPoolEmitter,
    GPJEDiPoolEmitter,
    )
from qdax_es.utils.restart import FixedGens
import wandb

from evosax import Strategies

JEDI_EMITTERS = {
    "UniformJEDiPoolEmitter": UniformJEDiPoolEmitter,
    "GPJEDiPoolEmitter": GPJEDiPoolEmitter,
}

def JEDi_factory(cfg):
    task = cfg.task
    algo = cfg.algo
    assert algo.algo == "jedi", f"algo.type should be jedi, got {algo.type}"

    batch_size = task.es_params.popsize * algo.pool_size
    initial_batch = batch_size
    num_iterations = int(task.total_evaluations / batch_size / cfg.steps) 
    print("Iterations per step: ", num_iterations)
    print("Iterations: ", num_iterations*cfg.steps)

    assert task.es_params.es_type in Strategies, f"{task.es_params.es_type} is not one of {Strategies.keys()}"

    print("Algo: ", cfg.algo.plotting.algo_name)   

    setup_config = {
        "seed": cfg.seed,
        "env": task.env_name,
        "episode_length": task.episode_length,
        "stochastic": task.stochastic,
        "policy_hidden_layer_sizes": task.network.policy_hidden_layer_sizes,
        "activation": task.network.activation,
        "initial_batch": initial_batch,
        "num_init_cvt_samples": algo.archive.num_init_cvt_samples,
        "num_centroids": algo.archive.num_centroids,
    }
    (
        centroids, 
        min_bd, 
        max_bd, 
        scoring_fn, 
        metrics_fn, 
        init_variables, 
        random_key
    ) = setup_qd(setup_config)

    if algo.restarter.type == "FixedGens":
        restarter = FixedGens(cfg.algo.params.gens)
        
    else:
        raise ValueError(f"Unknown restarter type: {algo.restarter.type}")
    
    es_params = {
        k:v for k,v in task.es_params.items() if k != "es_type"}

    emitter = JEDiEmitter(
        centroids=centroids,
        es_hp=es_params,
        es_type=task.es_params.es_type,
        wtfs_alpha = cfg.algo.params.alpha,
        restarter=restarter,
        global_norm=algo.global_norm,
    )

    emitter_class = JEDI_EMITTERS[algo.gp.emitter]
    emitter_args = algo.gp.args if hasattr(algo.gp, "args") else {}
    if algo.gp.emitter != "UniformJEDiPoolEmitter":
        emitter_args["n_steps"] = algo.gp.n_steps
    
    emitter = emitter_class(
        pool_size=algo.pool_size,
        emitter=emitter,
        **emitter_args
    )

    scoring_fn = jax.jit(scoring_fn)
    map_elites = CustomMAPElites(
        scoring_function=scoring_fn,
        emitter=emitter,
        metrics_function=metrics_fn,
        repertoire_type=GPRepertoire,
    )
    repertoire_kwargs = {
        "weighted": algo.gp.weighted,
    }

    # with jax.disable_jit():
    repertoire, emitter_state, random_key = map_elites.init(
        init_variables, 
        centroids, 
        random_key,
        repertoire_kwargs=repertoire_kwargs
    )

    plot_prefix = f"{'W' if algo.gp.weighted else ''}JEDi_" + str(cfg.algo.params.alpha)

    return (min_bd, 
            max_bd, 
            random_key, 
            map_elites, 
            emitter, 
            repertoire, 
            emitter_state,
            plot_prefix)

def plot_results_jedi(
    repertoire: GPRepertoire,
    emitter_state,
    cfg,
    min_bd, 
    max_bd,
    step,
    wandb_run=None
    ):
    final_repertoire = repertoire.fit_gp()
    fig, axes = final_repertoire.plot(min_bd, max_bd, cfg=cfg)
    plt.suptitle(f"{cfg.algo.plotting.algo_name} in {cfg.task.plotting.task_name}", fontsize=20)


    current_target_bd = jax.vmap(
        lambda e: e.wtfs_target,
    )(
        emitter_state.emitter_states
    )

    ax = axes["C"]
    ax.scatter(current_target_bd[:, 0], current_target_bd[:, 1], c="red", marker="x")
    ax = axes["D"]
    ax.scatter(current_target_bd[:, 0], current_target_bd[:, 1], c="red", marker="x")

    # ax.legend()

    # Save fig with step number
    plot_prefix = f"{'W' if cfg.algo.gp.weighted else ''}JEDi_" + str(cfg.algo.params.alpha)
    figname = f"{cfg.plots_dir}/{cfg.task.env_name}/{plot_prefix}"+ "_count_" + str(step) + ".png"
    
    if wandb_run is not None:
        wandb_run.log({f"step_{step}": wandb.Image(fig)})

    # create folder if it does not exist
    os.makedirs(os.path.dirname(figname), exist_ok=True)
    print("Save figure in: ", figname)
    plt.savefig(figname, bbox_inches='tight')
    plt.close()