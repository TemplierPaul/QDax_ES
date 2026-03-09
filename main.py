import hydra

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1"

import jax 

from omegaconf import DictConfig

# Check there is a gpu
assert jax.device_count() > 0, "No GPU found"

from qdax_bench.main_func import run as bench_run

def main_func(cfg: DictConfig) -> None:
    bench_run(cfg)

if __name__ == "__main__":
    main = hydra.main(version_base=None, config_path="qdax_es/configs", config_name="config")(main_func)
    main()
