import functools
import warnings
import jax
import jax.numpy as jnp
# from qdax.tasks.brax.common import descriptor_extractor
import flax.linen as nn
from qdax.core.neuroevolution.networks.networks import MLP
# from qdax.tasks.brax_envs import (
    # create_brax_scoring_fn,
# )
# from qdax import environments

from qdax.utils.metrics import CSVLogger, default_qd_metrics
from qdax.core.containers.mapelites_repertoire import (
    compute_cvt_centroids,
)

# from qdax.environments import create
import qdax.tasks.brax.v1 as brax_v1
import qdax.tasks.brax.v2 as brax_v2
from qdax.tasks.brax.v1.env_creators import create_brax_scoring_fn as create_brax_scoring_fn_v1
from qdax.tasks.brax.v2.env_creators import create_brax_scoring_fn as create_brax_scoring_fn_v2


def create_task(config, key):
    if "kheperax" in config["task"]["env_name"]:
        from kheperax.tasks.final_distance import FinalDistKheperaxTask
        from kheperax.tasks.target import TargetKheperaxConfig
        from kheperax.tasks.quad import make_quad_config

        map_name = config["task"]["env_name"].replace("kheperax_", "").replace("kheperax-", "")
        # Define Task configuration
        if "quad_" in map_name:
            # base_map_name = map_name.replace("quad_", "")
            # print(f"Kheperax Quad: Using {base_map_name} as base map")
            # config_kheperax = QuadKheperaxConfig.get_default_for_map(base_map_name)
            config_kheperax = TargetKheperaxConfig.get_default_for_map(map_name)
            config_kheperax = make_quad_config(config_kheperax)
            qd_offset = 2 * jnp.sqrt(2) * 100 + config["task"]["episode_length"]
        else:
            # print(f"Kheperax: Using {map_name} as base map")
            config_kheperax = TargetKheperaxConfig.get_default_for_map(map_name)
            qd_offset = jnp.sqrt(2) * 100 + config["task"]["episode_length"]

        
        config_kheperax.episode_length = config["task"]["episode_length"]
        config_kheperax.mlp_policy_hidden_layer_sizes = tuple(config["task"]["network"]["policy_hidden_layer_sizes"])

        (
            env,
            policy_network,
            scoring_fn,
        ) = FinalDistKheperaxTask.create_default_task(
            config_kheperax,
            random_key=key,
        )
        def fixed_scoring_fn(*args, **kwargs):
            # Remove the key from the return
            return scoring_fn(*args, **kwargs)[:3]
        
        return env, policy_network, fixed_scoring_fn, qd_offset, env.behavior_descriptor_length

    else:
        if config["task"]["brax"]["version"] == "v1":
            brax = brax_v1
            create_brax_scoring_fn = create_brax_scoring_fn_v1
            env = brax_v1.create(
                config["task"]["env_name"], 
                episode_length=config["task"]["episode_length"],
                legacy_spring=config["task"]["brax"]["legacy_spring"],
                )
        elif config["task"]["brax"]["version"] == "v2":
            brax = brax_v2
            create_brax_scoring_fn = create_brax_scoring_fn_v2
            env = brax.create(
                config["task"]["env_name"],
                episode_length=config["task"]["episode_length"],
                backend=config["task"]["brax"]["backend"],
            )
        else:
            raise NotImplementedError(
                f"brax version {config["task"]['brax']["version"]} not supported, choose one of ['v1', 'v2']"
            )

        if env.descriptor_length != 2:
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
        if config["task"]["network"]["activation"] not in activations:
            raise NotImplementedError(
                f"Activation {config['activation']} not implemented, choose one of {activations.keys()}"
            )

        activation = activations[config["task"]["network"]["activation"]]

        policy_layer_sizes = config["task"]["network"]["policy_hidden_layer_sizes"] + [env.action_size]
        policy_layer_sizes = tuple(policy_layer_sizes)
        policy_network = MLP(
            layer_sizes=policy_layer_sizes,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            final_activation=jnp.tanh,
            activation=activation,
        )
        
        # Prepare the scoring function
        reward_offset = brax.reward_offset[config["task"]["env_name"]]
        qd_offset = reward_offset * config["task"]["episode_length"]

        descriptor_extraction_fn = brax.descriptor_extractor[config["task"]["env_name"]]
        scoring_fn = create_brax_scoring_fn(
            env=env,
            policy_network=policy_network,
            descriptor_extraction_fn=descriptor_extraction_fn,
            key=key,
            episode_length=config["task"]["episode_length"],
            deterministic=not config["task"]["stochastic"],
        )

        return env, policy_network, scoring_fn, qd_offset, env.descriptor_length
    

def setup_pga(config):
    key = jax.random.PRNGKey(config["seed"])
    key, subkey = jax.random.split(key)

    (
        env,
        policy_network,
        scoring_fn,
        reward_offset,
        descriptor_length
    ) = create_task(
        config, 
        key=subkey,
    )

    min_bd = config["task"]["descriptors"]["minval"]
    max_bd = config["task"]["descriptors"]["maxval"]

    config["video_recording"] = {
        "env": env,
        "policy_network": policy_network,
    }

    # Init population of controllers
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=config["initial_batch"])
    fake_batch = jnp.zeros(shape=(config["initial_batch"], env.observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

    metrics_function = functools.partial(
        default_qd_metrics,
        qd_offset=reward_offset,
    )

    # Compute the centroids
    key, subkey = jax.random.split(key)
    centroids = compute_cvt_centroids(
        num_descriptors=descriptor_length,
        num_init_cvt_samples=config["algo"]["archive"]["num_init_cvt_samples"],
        num_centroids=config["algo"]["archive"]["num_centroids"],
        minval=min_bd,
        maxval=max_bd,
        key=subkey,
    )

    return centroids, min_bd, max_bd, scoring_fn, metrics_function, init_variables, key, env, policy_network

def setup_qd(config):
    centroids, min_bd, max_bd, scoring_fn, metrics_function, init_variables, key, env, policy_network = setup_pga(config)
    return centroids, min_bd, max_bd, scoring_fn, metrics_function, init_variables, key