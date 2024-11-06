import functools

import jax
import jax.numpy as jnp
from qdax_es.environments import create_no_legacy
import flax.linen as nn
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.tasks.brax_envs import (
    make_policy_network_play_step_fn_brax,
    reset_based_scoring_function_brax_envs,
)
from qdax import environments

from qdax_es.utils.env_bd import get_bd_bounds
from qdax_es.core.containers.count_repertoire import CountMapElitesRepertoire, count_qd_metrics
from qdax_es.core.custom_repertoire_mapelites import CustomMAPElites

from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    compute_cvt_centroids,
)

def create_task(config, random_key):
    if "kheperax" in config["env"]:
        from kheperax.tasks.final_distance import FinalDistKheperaxTask
        from kheperax.tasks.target import TargetKheperaxConfig
        from kheperax.tasks.quad import make_quad_config

        map_name = config["env"].replace("kheperax_", "").replace("kheperax-", "")
        # Define Task configuration
        if "quad_" in map_name:
            base_map_name = map_name.replace("quad_", "")
            # print(f"Kheperax Quad: Using {base_map_name} as base map")
            # config_kheperax = QuadKheperaxConfig.get_default_for_map(base_map_name)
            config_kheperax = TargetKheperaxConfig.get_default_for_map(map_name)
            config_kheperax = make_quad_config(config_kheperax)
            qd_offset = 2 * jnp.sqrt(2) * 100 + config["episode_length"]
        else:
            # print(f"Kheperax: Using {map_name} as base map")
            config_kheperax = TargetKheperaxConfig.get_default_for_map(map_name)
            qd_offset = jnp.sqrt(2) * 100 + config["episode_length"]

        
        config_kheperax.episode_length = config["episode_length"]
        config_kheperax.mlp_policy_hidden_layer_sizes = config["policy_hidden_layer_sizes"]

        (
            env,
            policy_network,
            scoring_fn,
        ) = FinalDistKheperaxTask.create_default_task(
            config_kheperax,
            random_key=random_key,
        )
        return env, policy_network, scoring_fn, qd_offset

    else:
        env = create_no_legacy(config["env"], episode_length=config["episode_length"])

        if env.behavior_descriptor_length != 2:
            # warn
            print("Plotting only works for 2D BDs")
            config["plot"] = False

        # Init policy network
        activations = {
            "relu": nn.relu,
            "tanh": jnp.tanh,
            "sigmoid": jax.nn.sigmoid,
            "sort": jnp.sort,
        }
        if config["activation"] not in activations:
            raise NotImplementedError(
                f"Activation {config['activation']} not implemented, choose one of {activations.keys()}"
            )

        activation = activations[config["activation"]]

        policy_layer_sizes = config["policy_hidden_layer_sizes"] + (env.action_size,)
        policy_network = MLP(
            layer_sizes=policy_layer_sizes,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            final_activation=jnp.tanh,
            activation=activation,
        )
        # Prepare the play function
        if config["stochastic"]:
            play_reset_fn = env.reset
        else:
            env_seed = jax.random.PRNGKey(config["seed"])
            init_state = env.reset(env_seed)
            def play_reset_fn(_):
                return init_state
            
        seeds = jax.random.split(random_key, num=10)
        init_states = jax.vmap(play_reset_fn)(seeds)
        # print(init_states.obs)

        # Prepare the scoring function
        from qdax_es.environments import get_qd_params
        bd_extraction_fn, reward_offset = get_qd_params(config["env"])
        qd_offset = reward_offset * config["episode_length"]

        play_step_fn = make_policy_network_play_step_fn_brax(env, policy_network)

        scoring_fn = functools.partial(
            reset_based_scoring_function_brax_envs,
            episode_length=config["episode_length"],
            play_reset_fn=play_reset_fn,
            play_step_fn=play_step_fn,
            behavior_descriptor_extractor=bd_extraction_fn,
        )

        return env, policy_network, scoring_fn, qd_offset
    

def setup_qd(config):
    random_key = jax.random.PRNGKey(config["seed"])

    (
        env,
        policy_network,
        scoring_fn,
        reward_offset
    ) = create_task(
        config, 
        random_key=random_key,
    )

    bd_bounds = get_bd_bounds(config["env"], n_dim=env.behavior_descriptor_length)
    min_bd = bd_bounds["minval"]
    max_bd = bd_bounds["maxval"]


    config["video_recording"] = {
        "env": env,
        "policy_network": policy_network,
    }

    # Init population of controllers
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=config["initial_batch"])
    fake_batch = jnp.zeros(shape=(config["initial_batch"], env.observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

    metrics_function = functools.partial(
        count_qd_metrics,
        qd_offset=reward_offset,
    )

    # Compute the centroids
    centroids, random_key = compute_cvt_centroids(
        num_descriptors=env.behavior_descriptor_length,
        num_init_cvt_samples=config["num_init_cvt_samples"],
        num_centroids=config["num_centroids"],
        minval=min_bd,
        maxval=max_bd,
        random_key=random_key,
    )

    return centroids, min_bd, max_bd, scoring_fn, metrics_function, init_variables, random_key