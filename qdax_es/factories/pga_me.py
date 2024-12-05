import os 
import matplotlib.pyplot as plt
import jax 
from evosax import Strategies
import hydra 
import wandb


from qdax_es.core.custom_repertoire_mapelites import CustomMAPElites
from qdax_es.core.containers.count_repertoire import CountMapElitesRepertoire
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax_es.utils.setup import setup_pga

class PGAMEFactory:
    def build(self, cfg):
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
            random_key,
            env,
            policy_network
        ) = setup_pga(setup_config)


        repertoire_init = hydra.utils.instantiate(cfg.algo.repertoire_init)
        print("Repertoire init: ", repertoire_init)

        emitter = hydra.utils.instantiate(cfg.algo.emitter)(
            policy_network=policy_network,
            env=env,
        )
        print("Emitter: ", emitter)

        map_elites = CustomMAPElites(
            scoring_function=scoring_fn,
            emitter=emitter,
            metrics_function=metrics_fn,
            repertoire_init=repertoire_init,
        )

        # with jax.disable_jit():
        repertoire, emitter_state, random_key = map_elites.init(
            init_variables, 
            centroids, 
            random_key,
            repertoire_kwargs={}
        )

        plot_prefix = algo.plotting.algo_name.replace(" ", "_")

        return (min_bd, 
                max_bd, 
                random_key, 
                map_elites, 
                emitter, 
                repertoire, 
                emitter_state,
                plot_prefix)

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