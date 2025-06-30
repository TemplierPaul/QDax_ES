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

# def remove_task_depth(cfg):
#     # Remove the "task" section from the configuration
    

@hydra.main(version_base=None, config_path="qdax_es/configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    cfg.algo = clean_algo_cfg(cfg)

    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
