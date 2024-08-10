from qdax.environments.bd_extractors import (
    get_feet_contact_proportion,
    get_final_xy_position,
)
from qdax_es.environments import behavior_descriptor_extractor, _qdax_custom_envs, reward_offset
from qdax_es.environments import get_qd_params, custom_to_base

_feet_nb = {
    "hopper": 1,
    "halfcheetah": 2,
    "walker2d": 2,
    "humanoid": 2,
    "ant": 6,
}

def get_bd_bounds(env_name, n_dim=None):
    if "kheperax" in env_name:
        if "quad" in env_name:
            return {"minval": [-1.0, -1.0], "maxval": [1.0, 1.0]}
        else:
            return {"minval": [0.0, 0.0], "maxval": [1.0, 1.0]}
    elif env_name == "reacher_qd":
        return {"minval": [-1.0, -1.0], "maxval": [1.0, 1.0]}
    elif env_name == "pusher_qd":
        return {"minval": [-1.0, -1.0], "maxval": [1.0, 1.0]}
    
    env_name = custom_to_base(env_name)
    if behavior_descriptor_extractor[env_name] == get_final_xy_position:
        if env_name in _qdax_custom_envs:
            bd_bounds = _qdax_custom_envs[env_name]["kwargs"][0]
        elif env_name == "pointmaze":
            bd_bounds = {"minval": [-1.0, -1.0], "maxval": [1.0, 1.0]}
        elif "custom" in env_name:
            bd_bounds = {"minval": [0.0, 0.0], "maxval": [1.0, 1.0]}

    elif behavior_descriptor_extractor[env_name] == get_feet_contact_proportion:
        if env_name in _qdax_custom_envs:
            base_env = _qdax_custom_envs[env_name]["env"]
            if n_dim is None and base_env in _feet_nb:
                n_dim = _feet_nb[base_env]
            bd_bounds = {
                "minval": [0.0] * n_dim,
                "maxval": [1.0] * n_dim,
            }

    else:
        bd_bounds = {"minval": [0.0], "maxval": [1.0]}
    return bd_bounds