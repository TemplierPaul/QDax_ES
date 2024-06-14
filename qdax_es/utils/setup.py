
import jax
import jax.numpy as jnp
from qdax_es.environments import create_no_legacy
import flax.linen as nn
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.tasks.brax_envs import (
    make_policy_network_play_step_fn_brax,
    reset_based_scoring_function_brax_envs,
)
import functools


def create_task(config, random_key):
    if "kheperax" in config["env"]:
        from kheperax.final_distance import FinalDistKheperaxTask
        from kheperax.target import TargetKheperaxConfig
        from kheperax.quad_task import QuadKheperaxConfig

        map_name = config["env"].replace("kheperax_", "")
        # Define Task configuration
        if "quad_" in map_name:
            base_map_name = map_name.replace("quad_", "")
            # print(f"Kheperax Quad: Using {base_map_name} as base map")
            config_kheperax = QuadKheperaxConfig.get_map(base_map_name)
            qd_offset = 2 * jnp.sqrt(2) * 100 + config["episode_length"]
        else:
            # print(f"Kheperax: Using {map_name} as base map")
            config_kheperax = TargetKheperaxConfig.get_map(map_name)
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
        print(init_states.obs)

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