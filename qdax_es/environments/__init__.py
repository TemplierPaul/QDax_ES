import functools
from typing import Any, Callable, List, Optional, Union

import brax
from brax.v1.envs import Env, _envs
from brax.v1.envs.wrappers import (
    AutoResetWrapper,
    EpisodeWrapper,
    EvalWrapper,
    VectorWrapper,
)


from qdax.environments.base_wrappers import QDEnv, StateDescriptorResetWrapper
from qdax.environments.descriptor_extractors import (
    get_feet_contact_proportion,
    get_final_xy_position,
)
from qdax.environments.exploration_wrappers import MazeWrapper, TrapWrapper
from qdax.environments.humanoidtrap import HumanoidTrap
from qdax.environments.init_state_wrapper import FixedInitialStateWrapper
from qdax.environments.locomotion_wrappers import (
    FeetContactWrapper,
    NoForwardRewardWrapper,
    XYPositionWrapper,
)
from qdax.environments.pointmaze import PointMaze
from qdax.environments.wrappers import CompletedEvalWrapper


# experimentally determinated offset (except for antmaze)
# should be sufficient to have only positive rewards but no guarantee
reward_offset = {
    "pointmaze": 2.3431,
    "anttrap": 3.38,
    "humanoidtrap": 0.0,
    "antnotrap": 3.38,
    "antmaze": 40.32,
    "ant_omni": 3.0,
    "humanoid_omni": 0.0,
    "ant_uni": 3.24,
    "humanoid_uni": 0.0,
    "halfcheetah_uni": 9.231,
    "hopper_uni": 0.9,
    "walker2d_uni": 1.413,
}

behavior_descriptor_extractor = {
    "pointmaze": get_final_xy_position,
    "anttrap": get_final_xy_position,
    "humanoidtrap": get_final_xy_position,
    "antnotrap": get_final_xy_position,
    "antmaze": get_final_xy_position,
    "ant_omni": get_final_xy_position,
    "humanoid_omni": get_final_xy_position,
    "ant_uni": get_feet_contact_proportion,
    "humanoid_uni": get_feet_contact_proportion,
    "halfcheetah_uni": get_feet_contact_proportion,
    "hopper_uni": get_feet_contact_proportion,
    "walker2d_uni": get_feet_contact_proportion,
}

def custom_to_base(env_name):
    if "custom" in env_name:
        args = env_name.split("_")
        if len(args) == 3:
            _, base_env_name, param = args
        elif len(args) == 4:
            base_env_name = args[1] + "_" + args[2]
            param = args[3]
        else:
            raise NotImplementedError(f"{base_env_name} does not exist!")
        
        return base_env_name
    else:
        return env_name

def get_qd_params(env_name):
    env_name = custom_to_base(env_name)
    if env_name in behavior_descriptor_extractor:
        return behavior_descriptor_extractor[env_name], reward_offset[env_name]
    else:
        raise NotImplementedError(f"{env_name} does not exist!")

_qdax_envs = {
    "pointmaze": PointMaze,
    "humanoid_w_trap": HumanoidTrap,
}

_qdax_custom_envs = {
    "anttrap": {
        "env": "ant",
        "wrappers": [XYPositionWrapper, TrapWrapper],
        "kwargs": [{"minval": [0.0, -8.0], "maxval": [30.0, 8.0]}, {}],
    },
    "humanoidtrap": {
        "env": "humanoid_w_trap",
        "wrappers": [XYPositionWrapper],
        "kwargs": [{"minval": [0.0, -8.0], "maxval": [30.0, 8.0]}],
    },
    "antnotrap": {
        "env": "ant",
        "wrappers": [XYPositionWrapper],
        "kwargs": [{"minval": [0.0, -8.0], "maxval": [70.0, 8.0]}],
    },
    "antmaze": {
        "env": "ant",
        "wrappers": [XYPositionWrapper, MazeWrapper],
        "kwargs": [{"minval": [-5.0, -5.0], "maxval": [40.0, 40.0]}, {}],
    },
    "ant_omni": {
        "env": "ant",
        "wrappers": [XYPositionWrapper, NoForwardRewardWrapper],
        "kwargs": [{"minval": [-30.0, -30.0], "maxval": [30.0, 30.0]}, {}],
    },
    "humanoid_omni": {
        "env": "humanoid",
        "wrappers": [XYPositionWrapper, NoForwardRewardWrapper],
        "kwargs": [{"minval": [-30.0, -30.0], "maxval": [30.0, 30.0]}, {}],
    },
    "ant_uni": {"env": "ant", "wrappers": [FeetContactWrapper], "kwargs": [{}, {}]},
    "humanoid_uni": {
        "env": "humanoid",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}, {}],
    },
    "halfcheetah_uni": {
        "env": "halfcheetah",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}, {}],
    },
    "hopper_uni": {
        "env": "hopper",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}, {}],
    },
    "walker2d_uni": {
        "env": "walker2d",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}, {}],
    },
}


def create(
    env_name: str,
    episode_length: int = 1000,
    action_repeat: int = 1,
    auto_reset: bool = True,
    batch_size: Optional[int] = None,
    eval_metrics: bool = False,
    fixed_init_state: bool = False,
    qdax_wrappers_kwargs: Optional[List] = None,
    legacy_spring=False,
    **kwargs: Any,
) -> Union[Env, QDEnv]:
    """Creates an Env with a specified brax system.
    Please use namespace to avoid confusion between this function and
    brax.envs.create.
    """

    if env_name in _envs.keys():
        env = _envs[env_name](legacy_spring=legacy_spring, **kwargs)
    elif env_name in _qdax_envs.keys():
        env = _qdax_envs[env_name](**kwargs)
    elif env_name in _qdax_custom_envs.keys():
        base_env_name = _qdax_custom_envs[env_name]["env"]
        if base_env_name in _envs.keys():
            env = _envs[base_env_name](legacy_spring=legacy_spring, **kwargs)
        elif base_env_name in _qdax_envs.keys():
            env = _qdax_envs[base_env_name](**kwargs)  # type: ignore

    else:
        raise NotImplementedError(f"{env_name} env does not exist!")

    if env_name in _qdax_custom_envs.keys():
        # roll with qdax wrappers
        wrappers = _qdax_custom_envs[env_name]["wrappers"]
        if qdax_wrappers_kwargs is None:
            kwargs_list = _qdax_custom_envs[env_name]["kwargs"]
        else:
            kwargs_list = qdax_wrappers_kwargs
        for wrapper, kwargs in zip(wrappers, kwargs_list):  # type: ignore
            env = wrapper(env, base_env_name, **kwargs)  # type: ignore

    if episode_length is not None:
        env = EpisodeWrapper(env, episode_length, action_repeat)
    if batch_size:
        env = VectorWrapper(env, batch_size)
    if fixed_init_state:
        # retrieve the base env
        if env_name not in _qdax_custom_envs.keys():
            base_env_name = env_name
        # wrap the env
        env = FixedInitialStateWrapper(env, base_env_name=base_env_name)  # type: ignore
    if auto_reset:
        env = AutoResetWrapper(env)
        if env_name in _qdax_custom_envs.keys():
            env = StateDescriptorResetWrapper(env)
    if eval_metrics:
        env = EvalWrapper(env)
        env = CompletedEvalWrapper(env)

    return env


def create_fn_no_legacy(env_name: str, **kwargs: Any) -> Callable[..., brax.envs.Env]:
    """Returns a function that when called, creates an Env.
    Please use namespace to avoid confusion between this function and
    brax.envs.create_fn.
    """
    return functools.partial(create, env_name, **kwargs)
