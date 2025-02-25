import os 
import matplotlib.pyplot as plt
import jax 
from evosax import Strategies
import hydra 
import wandb
import warnings

from qdax_es.core.custom_repertoire_mapelites import CustomMAPElites
from qdax_es.core.containers.count_repertoire import CountMapElitesRepertoire
from qdax_es.core.containers.mae_repertoire import MAERepertoire
from qdax_es.core.emitters.cma_me_emitter import CMAMEEmitter, CMAMEPoolEmitter
from qdax_es.core.emitters.cma_mae_emitter import CMAMEAnnealingEmitter

from qdax_es.utils.setup import setup_qd

class CMAMEFactory:
    def build(self, cfg):
        task = cfg.task
        algo = cfg.algo
        # assert algo.algo == "jedi", f"algo.algo should be jedi, got {algo.algo}"

        batch_size = task.es_params.popsize * algo.pool_size
        initial_batch = batch_size
        num_iterations = int(task.total_evaluations / batch_size / cfg.steps) 
        print("Iterations per step: ", num_iterations)
        print("Iterations: ", num_iterations*cfg.steps)

        if hasattr(task, "legacy_spring"):
            legacy_spring = task.legacy_spring
        else:
            legacy_spring = False
            warnings.warn("Legacy spring not set. Defaulting to False")

        assert task.es_params.es_type in Strategies, f"{task.es_params.es_type} is not one of {Strategies.keys()}"

        print("Algo: ", cfg.algo.plotting.algo_name)

        setup_config = {
            "seed": cfg.seed,
            "env": task.env_name,
            "descriptors": task.descriptors,
            "episode_length": task.episode_length,
            "stochastic": task.stochastic,
            "legacy_spring": legacy_spring,
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
            key
        ) = setup_qd(setup_config)

        es_params = {
            k:v for k,v in task.es_params.items() if k != "es_type"}
        
        repertoire_init = hydra.utils.instantiate(cfg.algo.repertoire_init)
        print("Repertoire init: ", repertoire_init)

        internal_emitter_func = hydra.utils.instantiate(cfg.algo.emitter)

        internal_emitter = internal_emitter_func(
            centroids=centroids,
            es_hp=es_params,
            es_type=task.es_params.es_type,
        )

        print("Emitter: ", internal_emitter)
    
        emitter = CMAMEPoolEmitter(
            num_states=algo.pool_size,
            emitter=internal_emitter
        )

        map_elites = CustomMAPElites(
            scoring_function=scoring_fn,
            emitter=emitter,
            metrics_function=metrics_fn,
            repertoire_init=repertoire_init,
        )
        
        # with jax.disable_jit():
        key, subkey = jax.random.split(key)

        repertoire, emitter_state = map_elites.init(
            init_variables, 
            centroids, 
            key,
            repertoire_kwargs={}
        )

        plot_prefix = algo.plotting.algo_name.replace(" ", "_")

        return (min_bd, 
                max_bd, 
                key, 
                map_elites, 
                emitter, 
                repertoire, 
                emitter_state,
                plot_prefix,
                scoring_fn,
                )

    def plot_results(
        self,
        repertoire: CountMapElitesRepertoire,
        emitter_state,
        cfg,
        min_bd, 
        max_bd,
        step,
        wandb_run=None
        ):
        fig, axes = repertoire.plot(min_bd, max_bd, cfg=cfg)
        plt.suptitle(f"{cfg.algo.plotting.algo_name} in {cfg.task.plotting.task_name}", fontsize=20)

        # Save fig with step number
        plot_prefix = cfg.algo.plotting.algo_name.replace(" ", "_")
        figname = f"{cfg.plots_dir}/{cfg.task.env_name}/{plot_prefix}"+ "_count_" + str(step) + ".png"
        
        if wandb_run is not None:
            wandb_run.log({f"step_{step}": wandb.Image(fig)})

        # create folder if it does not exist
        os.makedirs(os.path.dirname(figname), exist_ok=True)
        print("Save figure in: ", figname)
        plt.savefig(figname, bbox_inches='tight')
        plt.close()