import os 
import matplotlib.pyplot as plt
import jax 
from evosax import Strategies

from qdax_es.core.custom_repertoire_mapelites import CustomMAPElites
from qdax_es.core.containers.count_repertoire import CountMapElitesRepertoire
from qdax_es.core.containers.mae_repertoire import MAERepertoire
from qdax_es.core.emitters.cma_me_emitter import CMAMEEmitter, CMAMEPoolEmitter
from qdax_es.core.emitters.cma_mae_emitter import CMAMEAnnealingEmitter

from qdax_es.utils.setup import setup_qd

def CMAME_factory(cfg):
    task = cfg.task
    algo = cfg.algo

    batch_size = task.es_params.popsize
    initial_batch = batch_size
    num_iterations = int(task.total_evaluations / batch_size / cfg.steps) 
    print("Iterations per step: ", num_iterations)
    print("Iterations: ", num_iterations*cfg.steps)

    assert task.es_params.es_type in Strategies, f"{task.es_params.es_type} is not one of {Strategies.keys()}"


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

    es_params = {
        k:v for k,v in task.es_params.items() if k != "es_type"}
    
    if algo.annealing.use_mae: # Annealing
        repertoire_kwargs = {
            "min_threshold" : algo.annealing.min_threshold,
            "archive_learning_rate" : algo.annealing.archive_learning_rate,
        }

        emitter = CMAMEAnnealingEmitter(
            centroids=centroids,
            emitter_type=algo.emitter_type,
            es_hp=es_params,
            es_type=task.es_params.es_type,
        )

        repertoire_type = MAERepertoire
    else: # No Annealing
        repertoire_kwargs = {}

        emitter = CMAMEEmitter(
            centroids=centroids,
            emitter_type=algo.emitter_type,
            es_hp=es_params,
            es_type=task.es_params.es_type,
        )
        repertoire_type = CountMapElitesRepertoire

    emitter = CMAMEPoolEmitter(
        num_states=algo.pool_size,
        emitter=emitter
    )

    map_elites = CustomMAPElites(
        scoring_function=scoring_fn,
        emitter=emitter,
        metrics_function=metrics_fn,
        repertoire_type=repertoire_type,
    )
    
    # with jax.disable_jit():
    repertoire, emitter_state, random_key = map_elites.init(
        init_variables, 
        centroids, 
        random_key,
        repertoire_kwargs=repertoire_kwargs
    )

    plot_prefix = f"CMAM{'A' if algo.annealing.use_mae else ''}E_" + str(algo.emitter_type)

    return (min_bd, 
            max_bd, 
            random_key, 
            map_elites, 
            emitter, 
            repertoire, 
            emitter_state,
            plot_prefix)

def plot_results_cmame(
    repertoire: CountMapElitesRepertoire,
    emitter_state,
    cfg,
    min_bd, 
    max_bd,
    step
    ):
    fig, axes = repertoire.plot(min_bd, max_bd)

    # ax.legend()

    # Save fig with step number
    plot_prefix = f"CMAM{'A' if cfg.algo.annealing.use_mae else ''}E_" + str(cfg.algo.emitter_type)
    figname = f"{cfg.plots_dir}/{cfg.task.env_name}/{plot_prefix}"+ "_count_" + str(step) + ".png"
    
    # create folder if it does not exist
    os.makedirs(os.path.dirname(figname), exist_ok=True)
    print("Save figure in: ", figname)
    plt.savefig(figname)
    plt.close()