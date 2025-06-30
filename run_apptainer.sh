SINGULARITY_TMP_DIR=$(mktemp -d -p "$(pwd)")

singularity -d run \
		--bind ./output/:/workdir/output/ \
		--cleanenv \
		--containall \
		--home /tmp/ \
		--no-home \
		--nv \
		--pwd /workdir/ \
		--workdir $SINGULARITY_TMP_DIR \
		apptainer/container_2024-12-05_132228_0ef85245115cf0968daa189feb884f077994bca9.sif algo=pga_me wandb.use=false hydra=hpc_basic task=kh_pointmaze

# singularity -d run --bind $HOME/QDax_ES/container_2024-11-20_143305_c88f432d378b907178415d455ba0df8073c25e0c/output/:/workdir/output/ --cleanenv --containall --home /tmp/ --no-home --nv --pwd /workdir/ --workdir $SINGULARITY_TMP_DIR $HOME/QDax_ES/container_2024-11-20_143305_c88f432d378b907178415d455ba0df8073c25e0c/container_2024-11-20_143305_c88f432d378b907178415d455ba0df8073c25e0c.sif +commit=c88f432d378b907178415d455ba0df8073c25e0c hydra=hpc_local algo=jedi

# singularity shell --bind $HOME/QDax_ES/container_2024-11-20_143305_c88f432d378b907178415d455ba0df8073c25e0c/output/:/workdir/output/ --cleanenv --containall --home /tmp/ --no-home --nv --pwd /workdir/ --workdir $SINGULARITY_TMP_DIR $HOME/QDax_ES/container_2024-11-20_143305_c88f432d378b907178415d455ba0df8073c25e0c/container_2024-11-20_143305_c88f432d378b907178415d455ba0df8073c25e0c.sif
