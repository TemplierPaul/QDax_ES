import jax 
import os
import matplotlib.pyplot as plt
from qdax_es.core.custom_repertoire_mapelites import CustomMAPElites
from qdax_es.core.containers.gp_repertoire import GPRepertoire

from qdax_es.utils.setup import setup_qd
from qdax_es.core.emitters.jedi_emitter import JEDiEmitter, ConstantScheduler, LinearScheduler
from qdax_es.utils.restart import FixedGens
import wandb
import hydra 
import warnings


from evosax import Strategies

class JEDiFactory:
    def build(self, cfg):
        task = cfg.task
        algo = cfg.algo
        assert algo.algo == "jedi", f"algo.algo should be jedi, got {algo.algo}"

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

        restarter = hydra.utils.instantiate(cfg.algo.restarter)
        print("Restarter: ", restarter)

        if cfg.algo.params.alpha == "decay":
            alpha_scheduler = LinearScheduler(0.8, 0.0, num_iterations*cfg.steps)
        else:
            # Assert it is int or float
            assert isinstance(cfg.algo.params.alpha, (int, float)), f"Alpha should be int or float if constant, got {cfg.algo.params.alpha}"
            alpha_scheduler = ConstantScheduler(cfg.algo.params.alpha)

        es_params = {
            k:v for k,v in task.es_params.items() if k != "es_type"}

        internal_emitter = JEDiEmitter(
            centroids=centroids,
            es_hp=es_params,
            es_type=task.es_params.es_type,
            alpha_scheduler = alpha_scheduler,
            restarter=restarter,
            global_norm=algo.global_norm,
        )

        pool_emitter = hydra.utils.instantiate(cfg.algo.gp.emitter)
        print("Pool emitter: ", pool_emitter)

        emitter = pool_emitter(
            emitter=internal_emitter
            )
        
        repertoire_init = hydra.utils.instantiate(cfg.algo.repertoire_init)
        print("Repertoire init: ", repertoire_init)

        scoring_fn = jax.jit(scoring_fn)
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
            subkey,
        )

        plot_prefix = f"{algo.gp.jedi_prefix}JEDi_" + str(cfg.algo.params.alpha)

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
        plot_prefix = f"{cfg.algo.gp.jedi_prefix}JEDi_" + str(cfg.algo.params.alpha)
        figname = f"{cfg.plots_dir}/{cfg.task.env_name}/{plot_prefix}"+ "_count_" + str(step) + ".png"
        
        if wandb_run is not None:
            wandb_run.log({f"step_{step}": wandb.Image(fig)})

        # create folder if it does not exist
        os.makedirs(os.path.dirname(figname), exist_ok=True)
        print("Save figure in: ", figname)
        plt.savefig(figname, bbox_inches='tight')
        plt.close()