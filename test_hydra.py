import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
# Register the "python" resolver

def clean_algo_cfg(cfg):
    # Resolve all interpolations (e.g., ${oc.select:...})
    resolved = OmegaConf.to_container(cfg.algo, resolve=True)

    # Remove internal helper sections
    for key in ["group_defaults", "env_params"]:
        resolved.pop(key, None)

    # Convert back to OmegaConf if needed
    return OmegaConf.create(resolved)

def oc_if(cond: bool, true_val, false_val):
    cond = str(cond).strip()
    if cond.lower() == "true":
        return true_val
    elif cond.lower() == "false":
        return false_val
    else:
        raise ValueError(f"Invalid condition: {cond}")

OmegaConf.register_new_resolver("oc.if", oc_if)


@hydra.main(version_base=None, config_path="qdax_es/configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    cfg.algo = clean_algo_cfg(cfg)

    print(OmegaConf.to_yaml(cfg))

    resolved = OmegaConf.to_container(cfg, resolve=True)

    print(resolved["task"]["plotting"]["task_name"])


if __name__ == "__main__":
    main()
