import math
import functools
from typing import Dict, Tuple

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.80"

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import jax 
import jax.numpy as jnp
from typing import Dict

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

from qdax_es.utils.count_plots import plot_archive_value

from qdax_es.utils.setup import setup_qd

# from jax.config import config
# config.update("jax_debug_nans", True)

# ES params
es_pop = 32
sigma_g = .1

# JEDi params
pool_size = 4 
es_gens = 10
wtfs_alpha = 0.3
weighted_gp = True

batch_size = es_pop * pool_size 
print("batch_size", batch_size)
initial_batch = batch_size

env_name = "kheperax_pointmaze"
episode_length = 250
stochastic = False
total_evaluations = 1e6
seed = 42
policy_hidden_layer_sizes = (8, )
activation = "relu"

num_init_cvt_samples = 50000
num_centroids = 100

es_type = "Sep_CMA_ES"

num_iterations = int(total_evaluations / batch_size) 
print("Iterations: ", num_iterations)

from evosax import Strategies
assert es_type in Strategies, f"{es_type} is not one of {Strategies.keys()}"

# ## Setup

es_params = {
    "sigma_init": sigma_g,
    "popsize": es_pop,
}

repertoire_kwargs = {
    "n_steps": 100,
    "weighted": weighted_gp,
}

setup_config = {
    "seed": seed, 
    "env": env_name,
    "episode_length": episode_length,
    "stochastic": stochastic,
    "policy_hidden_layer_sizes": policy_hidden_layer_sizes,
    "activation": activation,
    "initial_batch": initial_batch,
    "num_init_cvt_samples": num_init_cvt_samples,
    "num_centroids": num_centroids,
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

# ## Emitter

from qdax_es.core.emitters.jedi_emitter import JEDiEmitter
from qdax_es.core.emitters.jedi_pool_emitter import GPJEDiPoolEmitter, UniformJEDiPoolEmitter
from qdax_es.utils.restart import FixedGens, ConvergenceRestarter, DualConvergenceRestarter

# restarter = FixedGens(es_gens)
# restarter = ConvergenceRestarter(
#     min_score_spread=0.2,
#     min_gens=3,
#     # max_gens=100
#     )
restarter = DualConvergenceRestarter(
    min_score_spread=3,
    min_bd_spread=0.1,
    min_gens=3,
    max_gens=100
    )

emitter = JEDiEmitter(
    centroids=centroids,
    es_hp=es_params,
    es_type=es_type,
    wtfs_alpha = wtfs_alpha,
    restarter=restarter,
)

emitter = GPJEDiPoolEmitter(
    pool_size=pool_size,
    emitter=emitter
)

repertoire_type = GPRepertoire

map_elites = CustomMAPElites(
    scoring_function=scoring_fn,
    emitter=emitter,
    metrics_function=metrics_fn,
    repertoire_type=repertoire_type,
)

with jax.disable_jit():
    repertoire, emitter_state, random_key = map_elites.init(
        init_variables, 
        centroids, 
        random_key,
        repertoire_kwargs=repertoire_kwargs
    )

# with jax.disable_jit():
# map_elites.update(repertoire, emitter_state, random_key);

(repertoire, emitter_state, random_key,), metrics = jax.lax.scan(
    map_elites.scan_update,
    (repertoire, emitter_state, random_key),
    (),
    length=num_iterations,
)

for k, v in metrics.items():
    print(f"{k} after {num_iterations * batch_size}: {v[-1]}")

repertoire.total_count

emitter_state.emitter_states.restart_state

# ## Plot results

env_steps = jnp.arange(num_iterations) * emitter.batch_size * episode_length

from qdax.utils.plotting import plot_map_elites_results
import matplotlib


fig, axes = plot_map_elites_results(
    env_steps=env_steps,
    metrics=metrics,
    repertoire=repertoire,
    min_bd=min_bd,
    max_bd=max_bd,
)

# main title
plt.suptitle(f"{env_name} task with {es_type} for JEDi", fontsize=20)

# udpate this variable to save your results locally
savefig = True
if savefig:
    figname = f"./plots/{env_name}/DyRJEDi_"  + str(wtfs_alpha) + "_logs.png"
    # create folder if it does not exist
    import os
    os.makedirs(os.path.dirname(figname), exist_ok=True)
    print("Save figure in: ", figname)
    plt.savefig(figname)

final_repertoire = repertoire.fit_gp(10)
fig, axes = final_repertoire.plot(-1, 1);

if savefig:
    figname = f"./plots/{env_name}/DyRJEDi_"  + str(wtfs_alpha) + "_count.png"
    # create folder if it does not exist
    import os
    os.makedirs(os.path.dirname(figname), exist_ok=True)
    print("Save figure in: ", figname)
    plt.savefig(figname)




