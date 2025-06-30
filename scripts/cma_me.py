import math
import functools
from typing import Dict, Tuple

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import jax 
import jax.numpy as jnp
from typing import Dict

import matplotlib

from qdax.utils.plotting import plot_map_elites_results
from qdax.core.map_elites import MAPElites
from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids, MapElitesRepertoire

from qdax import environments
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.tasks.brax_envs import scoring_function_brax_envs
from qdax.custom_types import EnvState, Params, RNGKey
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    compute_cvt_centroids,
)
from evosax import Strategies

from qdax_es.core.custom_repertoire_mapelites import CustomMAPElites
from qdax_es.core.containers.count_repertoire import CountMapElitesRepertoire

from qdax_es.utils.setup import setup_qd

env_name = "kheperax_pointmaze"
stochastic = False
episode_length = 250
seed = 42
num_init_cvt_samples = 1000
num_centroids = 1000
# min_bd = -1.0
# max_bd = 1.0

policy_hidden_layer_sizes = (8,)
activation = "relu"

emitter_type = "imp" #@param["opt", "imp", "rnd"]
es_type = "Sep_CMA_ES"
pool_size = 4 #@param {type:"integer"}
batch_size = 128
sigma_g = .1

num_iterations = 1000

assert es_type in Strategies, f"{es_type} is not one of {Strategies.keys()}"

# Setup QD

setup_config = {
    "seed": seed, 
    "env": env_name,
    "episode_length": episode_length,
    "stochastic": stochastic,
    "policy_hidden_layer_sizes": policy_hidden_layer_sizes,
    "activation": activation,
    "initial_batch": batch_size,
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
    key
) = setup_qd(setup_config)


# ## Emitters

es_params = {
    "sigma_init": sigma_g,
    "popsize": batch_size,
}

from qdax_es.core.emitters.cma_me_emitter import CMAMEEmitter, CMAMEPoolEmitter

emitter = CMAMEEmitter(
    centroids=centroids,
    es_hp=es_params,
    es_type=es_type,
)

emitter = CMAMEPoolEmitter(
    num_states=pool_size,
    emitter=emitter
)

map_elites = CustomMAPElites(
    scoring_function=scoring_fn,
    emitter=emitter,
    metrics_function=metrics_fn,
    repertoire_type=CountMapElitesRepertoire,
)

with jax.disable_jit():
    repertoire, emitter_state, key = map_elites.init(
        init_variables, centroids, key
    )


(repertoire, emitter_state, key,), metrics = jax.lax.scan(
    map_elites.scan_update,
    (repertoire, emitter_state, key),
    (),
    length=num_iterations,
)

for k, v in metrics.items():
    print(f"{k} after {num_iterations * batch_size}: {v[-1]}")

# ## Plot results

env_steps = jnp.arange(num_iterations) * emitter.batch_size * episode_length

fig, axes = plot_map_elites_results(
    env_steps=env_steps,
    metrics=metrics,
    repertoire=repertoire,
    min_bd=min_bd,
    max_bd=max_bd,
)

# main title
plt.suptitle(f"{env_name} task with {es_type} and CMA-ME {emitter_type} emitter", fontsize=20)

# udpate this variable to save your results locally
savefig = True
if savefig:
    figname = f"./plots/{env_name}/{es_type}_ME_"  + emitter_type + ".png"
    # create folder if it does not exist
    import os
    os.makedirs(os.path.dirname(figname), exist_ok=True)
    print("Save figure in: ", figname)
    plt.savefig(figname)

# Count plot 
import qdax_es.utils.count_plots as plotmodule
import importlib
plotmodule = importlib.reload(plotmodule)

log_scale=True
plotmodule.plot_2d_count(repertoire, min_bd, max_bd, log_scale=log_scale)
plt.suptitle(f"{env_name} task with {es_type} and {emitter_type} emitter", fontsize=20)

if savefig:
    figname = f"./plots/{env_name}/{es_type}_ME_"  + emitter_type + "_count.png"
    # create folder if it does not exist
    import os
    os.makedirs(os.path.dirname(figname), exist_ok=True)
    print("Save figure in: ", figname)
    plt.savefig(figname)




