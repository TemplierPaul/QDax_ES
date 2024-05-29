from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from qdax.types import Centroid, Descriptor, ExtraScores, Fitness, Genotype, RNGKey
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.emitters.cma_emitter import CMAEmitter, CMAEmitterState

from qdax.core.emitters.cma_opt_emitter import CMAOptimizingEmitter
from qdax.core.emitters.cma_rnd_emitter import CMARndEmitter, CMARndEmitterState
from qdax.core.emitters.cma_improvement_emitter import CMAImprovementEmitter

from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    get_cells_indices,
)

from evosax import EvoState, EvoParams, Strategies

from qdax_es.utils.core.utils.termination import cma_criterion
