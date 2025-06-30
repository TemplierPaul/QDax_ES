import jax 
import hydra 

from qdax.core.map_elites import MAPElites

from qdax_bench.utils.setup import setup_qd


class ESFactory:
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

        # emitter = hydra.utils.instantiate(algo["emitter"])()
        # print("Emitter: ", emitter)
        
        # EvosaxEmitterAll
        repertoire_init_fn = hydra.utils.instantiate(
                algo["repertoire_init"]
            )
        print("Repertoire init: ", repertoire_init_fn)

        emitter_func = hydra.utils.instantiate(algo["emitter"])

        emitter = emitter_func(
            centroids=centroids,
            init_variables_func=init_variables_func,
        )
        print("Emitter: ", emitter)

        map_elites = MAPElites(
            scoring_function=scoring_fn,
            emitter=emitter,
            metrics_function=metrics_fn,
            repertoire_init=repertoire_init_fn
        )

        # with jax.disable_jit():
        key, var_key, subkey = jax.random.split(key, 3)

        init_variables = init_variables_func(var_key)

        repertoire, emitter_state, init_metrics = map_elites.init(
            init_variables, 
            centroids, 
            subkey,
        )

        plot_prefix = algo["plotting"]["algo_name"].replace(" ", "_")

        return (
            min_bd, 
            max_bd, 
            key, 
            map_elites, 
            emitter, 
            repertoire, 
            emitter_state,
            init_metrics,
            plot_prefix,
            scoring_fn,    
            )

    # def plot_results(
    #     self,
    #     repertoire: CountMapElitesRepertoire,
    #     emitter_state,
    #     cfg,
    #     min_bd, 
    #     max_bd,
    #     step,
    #     wandb_run=None
    #     ):
    #     fig, axes = repertoire.plot(min_bd, max_bd, cfg=cfg)
    #     plt.suptitle(f"{cfg.algo.plotting.algo_name} in {cfg.task.plotting.task_name}", fontsize=20)

    #     # Save fig with step number
    #     plot_prefix = cfg.algo.plotting.algo_name.replace(" ", "_")
    #     figname = f"{cfg.plots_dir}/{cfg.task.env_name}/{plot_prefix}"+ "_count_" + str(step) + ".png"
        
    #     if wandb_run is not None:
    #         wandb_run.log({f"step_{step}": wandb.Image(fig)})

    #     # create folder if it does not exist
    #     os.makedirs(os.path.dirname(figname), exist_ok=True)
    #     print("Save figure in: ", figname)
    #     plt.savefig(figname, bbox_inches='tight')
    #     plt.close()