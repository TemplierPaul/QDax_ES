import hydra

from typing import Dict
from typing import Dict

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1"
# Jax floating point precision
# os.environ["JAX_ENABLE_X64"] = "True"

import matplotlib.pyplot as plt

import jax 
import jax.numpy as jnp
from typing import Dict

from qdax_es.utils.plotting import plot_map_elites_results 
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from qdax_es.factories.jedi import JEDi_factory, plot_results_jedi
from qdax_es.factories.cmame import CMAME_factory, plot_results_cmame

# Check there is a gpu
assert jax.device_count() > 0, "No GPU found"
import wandb

def set_env_params(cfg: DictConfig) -> Dict:
    if "env_params" not in cfg.algo.keys():
        return cfg
    env_params = cfg.algo.env_params.defaults
    if cfg.task.env_name in cfg.algo.env_params.keys():
        for k, v in cfg.algo.env_params[cfg.task.env_name].items():
            env_params[k] = v
    cfg.algo.env_params = env_params
    return cfg

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # cfg = set_env_params(cfg)
    print(OmegaConf.to_yaml(cfg))
    task = cfg.task
    algo = cfg.algo
    import os
    os.makedirs(cfg.plots_dir, exist_ok=True)
    
    wandb_run = None
    if cfg.wandb.use:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        wandb_run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, config=cfg_dict)

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
        raise NotImplementedError(f"algo.type not supported, got {algo.type}")

    num_iterations = int(task.total_evaluations / emitter.batch_size / cfg.steps) 

    update = jax.jit(map_elites.update)
    metrics = {}
    iter = 0
    for step in tqdm(range(cfg.steps)):
        for gen in range(num_iterations):
            repertoire, emitter_state, step_metrics, random_key = update(
                repertoire, emitter_state, random_key
            )
            step_metrics = {k: v for k, v in step_metrics.items()}
            step_metrics['generation'] = step * num_iterations + gen + 1
            step_metrics['evaluations'] = step_metrics['generation'] * emitter.batch_size

            # print(step_metrics)
            # print(type(step_metrics))

            if metrics == {}:
                metrics = step_metrics.copy()
            
            else:
                for k in step_metrics.keys():
                    metrics[k] = jnp.append(metrics[k], step_metrics[k])

            if cfg.wandb.use:
                wandb_run.log(step_metrics)

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

    if cfg.wandb.use:
        # Log the figure to wandb
        wandb_run.log({"results": wandb.Image(fig)})

    plot_results(
            repertoire,
            emitter_state,
            cfg,
            min_bd, 
            max_bd,
            step="end",
            wandb_run=wandb_run
        )

    if wandb_run:
        wandb_run.finish()
    # Return last max fitness
    return jnp.max(repertoire.fitnesses)


if __name__ == "__main__":
    try:
        wandb.setup()
    except:
        print("Wandb not available")
    main()
