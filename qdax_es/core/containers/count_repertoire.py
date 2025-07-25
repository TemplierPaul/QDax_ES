from __future__ import annotations

import warnings
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

from brax.io import html
import flax
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from qdax.custom_types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    RNGKey,
    Metrics,
)

from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    get_cells_indices,
)

from qdax.utils.plotting import plot_2d_map_elites_repertoire
import matplotlib.pyplot as plt
from typing_extensions import TypeAlias

from qdax_es.utils.count_plots import plot_2d_count

Count: TypeAlias = jnp.ndarray


class CountMapElitesRepertoire(MapElitesRepertoire):
    """
    ME repertoire that counts how many times a solution has been proposed for each cell.
    """

    count: Count

    @property
    def total_count(self):
        return self.count.sum()

    def add(  # type: ignore
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
        batch_of_extra_scores: Optional[ExtraScores] = None,
    ) -> MapElitesRepertoire:
        """
        Add a batch of elements to the repertoire.

        Args:
            batch_of_genotypes: a batch of genotypes to be added to the repertoire.
                Similarly to the self.genotypes argument, this is a PyTree in which
                the leaves have a shape (batch_size, num_features)
            batch_of_descriptors: an array that contains the descriptors of the
                aforementioned genotypes. Its shape is (batch_size, num_descriptors)
            batch_of_fitnesses: an array that contains the fitnesses of the
                aforementioned genotypes. Its shape is (batch_size,)
            batch_of_extra_scores: unused tree that contains the extra_scores of
                aforementioned genotypes.

        Returns:
            The updated MAP-Elites repertoire.
        """

        if batch_of_extra_scores is None:
            batch_of_extra_scores = {}

        filtered_batch_of_extra_scores = self.filter_extra_scores(batch_of_extra_scores)

        batch_of_indices = get_cells_indices(batch_of_descriptors, self.centroids)
        batch_of_indices = jnp.expand_dims(batch_of_indices, axis=-1)


        count = self.count + jnp.bincount(
            batch_of_indices.squeeze(axis=-1),
            minlength=len(self.count),
            length=len(self.count),
        )

        num_centroids = self.centroids.shape[0]
        batch_of_fitnesses = jnp.reshape(
            batch_of_fitnesses, (batch_of_descriptors.shape[0], 1)
        )

        # get fitness segment max
        best_fitnesses = jax.ops.segment_max(
            batch_of_fitnesses,
            batch_of_indices.astype(jnp.int32).squeeze(axis=-1),
            num_segments=num_centroids,
        )

        cond_values = jnp.take_along_axis(best_fitnesses, batch_of_indices, 0)

        # put dominated fitness to -jnp.inf
        batch_of_fitnesses = jnp.where(
            batch_of_fitnesses == cond_values, batch_of_fitnesses, -jnp.inf
        )

        # get addition condition
        current_fitnesses = jnp.take_along_axis(self.fitnesses, batch_of_indices, 0)
        addition_condition = batch_of_fitnesses > current_fitnesses

        # assign fake position when relevant : num_centroids is out of bound
        batch_of_indices = jnp.where(
            addition_condition, batch_of_indices, num_centroids
        )

        # create new repertoire
        new_repertoire_genotypes = jax.tree.map(
            lambda repertoire_genotypes, new_genotypes: repertoire_genotypes.at[
                batch_of_indices.squeeze(axis=-1)
            ].set(new_genotypes),
            self.genotypes,
            batch_of_genotypes,
        )

        # compute new fitness and descriptors
        new_fitnesses = self.fitnesses.at[batch_of_indices.squeeze(axis=-1)].set(
            batch_of_fitnesses
        )
        new_descriptors = self.descriptors.at[batch_of_indices.squeeze(axis=-1)].set(
            batch_of_descriptors
        )

        # update extra scores
        new_extra_scores = jax.tree.map(
            lambda repertoire_scores, new_scores: repertoire_scores.at[
                batch_of_indices.squeeze(axis=-1)
            ].set(new_scores),
            self.extra_scores,
            filtered_batch_of_extra_scores,
        )

        new_repertoire = self.__class__(
            genotypes=new_repertoire_genotypes,
            fitnesses=new_fitnesses,
            descriptors=new_descriptors,
            centroids=self.centroids,
            count=count,
            extra_scores=new_extra_scores,
            keys_extra_scores=self.keys_extra_scores,
        )

        return new_repertoire

    def __add__(
        self,
        other_repertoire: CountMapElitesRepertoire,
    ) -> CountMapElitesRepertoire:
        """
        Add another repertoire to the current one.
        """
        to_replace = other_repertoire.fitnesses > self.fitnesses

        new_genotypes = jax.vmap(
            lambda i: jax.tree.map(
                lambda x, y: to_replace[i] * y[i] + (1 - to_replace[i]) * x[i],
                self.genotypes,
                other_repertoire.genotypes,
            )
        )(jnp.arange(to_replace.shape[0]))

        new_fitnesses = jnp.where(
            to_replace, other_repertoire.fitnesses, self.fitnesses
        )

        new_descriptors = jax.vmap(
            lambda i: jax.tree.map(
                lambda x, y: to_replace[i] * y[i] + (1 - to_replace[i]) * x[i],
                self.descriptors,
                other_repertoire.descriptors,
            )
        )(jnp.arange(to_replace.shape[0]))

        new_count = self.count + other_repertoire.count

        return self.__class__(
            genotypes=new_genotypes,
            fitnesses=new_fitnesses,
            descriptors=new_descriptors,
            centroids=self.centroids,
            count=new_count,
        )

    @classmethod
    def init_default(
        cls,
        genotype: Genotype,
        centroids: Centroid,
        one_extra_score: Optional[ExtraScores] = None,
        keys_extra_scores: Tuple[str, ...] = (),
    ) -> CountMapElitesRepertoire:
        """Initialize a Map-Elites repertoire with an initial population of
        genotypes. Requires the definition of centroids that can be computed
        with any method such as CVT or Euclidean mapping.

        Note: this function has been kept outside of the object MapElites, so
        it can be called easily called from other modules.

        Args:
            genotype: the typical genotype that will be stored.
            centroids: the centroids of the repertoire
            keys_extra_scores: keys of the extra scores to store in the repertoire
            one_extra_score: a single extra score to be used for initialization.
        Returns:
            A repertoire filled with default values.
        """

        if one_extra_score is None:
            one_extra_score = {}

        one_extra_score = {
            key: value
            for key, value in one_extra_score.items()
            if key in keys_extra_scores
        }

        # get number of centroids
        num_centroids = centroids.shape[0]

        # default fitness is -inf
        default_fitnesses = -jnp.inf * jnp.ones(shape=(num_centroids, 1))

        # default genotypes is all 0
        default_genotypes = jax.tree.map(
            lambda x: jnp.zeros(shape=(num_centroids,) + x.shape, dtype=x.dtype),
            genotype,
        )

        # default descriptor is all zeros
        default_descriptors = jnp.zeros_like(centroids)

        # default extra scores is empty dict
        default_extra_scores = jax.tree.map(
            lambda x: jnp.zeros(shape=(num_centroids,) + x.shape, dtype=x.dtype),
            one_extra_score,
        )

        default_count = jnp.zeros(shape=num_centroids)

        return cls(
            genotypes=default_genotypes,
            fitnesses=default_fitnesses,
            descriptors=default_descriptors,
            centroids=centroids,
            count=default_count,
            extra_scores=default_extra_scores,
            keys_extra_scores=keys_extra_scores,
        )
    
    
    def plot(
            self,
            min_bd,
            max_bd,
            title='Repertoire',
            plot_gp=False,
            cfg=None,
            ):
        if plot_gp:
            print("No GP to plot")
        """Plot the repertoire"""
        fig, axes = plt.subplot_mosaic("""
            AB
            """,
            figsize=(15, 8),
        )

        try:
            vmin, vmax = None, None
            if cfg is not None:
                vmin, vmax = cfg.task.plotting.fitness_bounds
            _, axes["A"] = plot_2d_map_elites_repertoire(
                centroids=self.centroids,
                repertoire_fitnesses=self.fitnesses,
                minval=min_bd,
                maxval=max_bd,
                repertoire_descriptors=self.descriptors,
                ax=axes["A"],
                vmin=vmin,
                vmax=vmax,
            )
            max_fit = jnp.max(self.fitnesses)
            axes["A"].set_title(f"Fitness (max: {max_fit:.2f})")

            vmin, vmax = None, None
            if cfg is not None:
                vmin, vmax = 0, cfg.task.plotting.max_eval_cell
            axes["B"] = plot_2d_count(
                self, 
                min_bd, 
                max_bd, 
                log_scale=True, 
                ax=axes["B"],
                colormap="plasma",
                vmin=vmin,
                vmax=vmax,
                )
            
            plt.suptitle(title, fontsize=20)
        except:
            print("Failed plotting")

        return fig, axes

    # def record_video(self, env, policy_network):
    #     """Record a video of the best individual in the repertoire."""
    #     best_idx = jnp.argmax(self.fitnesses)

    #     elite = jax.tree.map(lambda x: x[best_idx], self.genotypes)

    #     jit_env_reset = jax.jit(env.reset)
    #     jit_env_step = jax.jit(env.step)
    #     jit_inference_fn = jax.jit(policy_network.apply)

    # rollout = []
    # rng = jax.random.PRNGKey(seed=1)
    # state = jit_env_reset(rng=rng)
    # while not state.done:
    #     rollout.append(state)
    #     action = jit_inference_fn(elite, state.obs)
    #     state = jit_env_step(state, action)

    # return html.render(env.sys, [s.qp for s in rollout[:500]])

    def record_video(self, config):
        if "kheperax" in config["env"]:
            return self._kheperax_video(config)
        else:
            return self._brax_video(config)

    def _brax_video(self, config):
        """Record a video of the best individual in the repertoire with brax"""
        best_idx = jnp.argmax(self.fitnesses)
        elite = jax.tree.map(lambda x: x[best_idx], self.genotypes)

        env = config["video_recording"]["env"]
        policy_network = config["video_recording"]["policy_network"]
        seed = config["seed"]
        episode_length = config["episode_length"]

        jit_env_reset = jax.jit(env.reset)
        jit_env_step = jax.jit(env.step)
        jit_inference_fn = jax.jit(policy_network.apply)

        rollout = []
        rng = jax.random.PRNGKey(seed=seed)
        state = jit_env_reset(rng=rng)
        for _ in range(episode_length):
            rollout.append(state)
            action = jit_inference_fn(elite, state.obs)
            state = jit_env_step(state, action)
            if state.done:
                break

        return html.render(env.sys, [s.qp for s in rollout])
    
    def _kheperax_video(self, config):
        """Record a video of the best individual in the repertoire with Kheperax"""
        best_idx = jnp.argmax(self.fitnesses)
        elite = jax.tree.map(lambda x: x[best_idx], self.genotypes)

        env = config["video_recording"]["env"]
        policy_network = config["video_recording"]["policy_network"]
        seed = config["seed"]
        episode_length = config["episode_length"]

        jit_env_reset = jax.jit(env.reset)
        jit_env_step = jax.jit(env.step)
        jit_inference_fn = jax.jit(policy_network.apply)

        rollout = []
        rng = jax.random.PRNGKey(seed=seed)
        state = jit_env_reset(rng=rng)
        base_image = env.create_image(state)

        for _ in range(episode_length):
            # Render
            image = env.add_robot(base_image, state)
            image = env.add_lasers(image, state)
            image = env.render_rgb_image(image, flip=True)
            rollout.append(image)

            # Update state
            if state.done:
                break
            action = jit_inference_fn(elite, state.obs)
            state = jit_env_step(state, action)

        return rollout


def count_qd_metrics(repertoire: CountMapElitesRepertoire, qd_offset: float) -> Metrics:
    """Compute the usual QD metrics that one can retrieve
    from a MAP Elites repertoire.

    Args:
        repertoire: a MAP-Elites repertoire
        qd_offset: an offset used to ensure that the QD score
            will be positive and increasing with the number
            of individuals.

    Returns:
        a dictionary containing the QD score (sum of fitnesses
            modified to be all positive), the max fitness of the
            repertoire, the coverage (number of niche filled in
            the repertoire).
    """

    # get metrics
    repertoire_empty = repertoire.fitnesses == -jnp.inf
    qd_score = jnp.sum(repertoire.fitnesses, where=~repertoire_empty)
    qd_score += qd_offset * jnp.sum(1.0 - repertoire_empty)
    coverage = 100 * jnp.mean(1.0 - repertoire_empty)
    max_fitness = jnp.max(repertoire.fitnesses)

    min_count = jnp.min(repertoire.count)
    max_count = jnp.max(repertoire.count)
    mean_count = jnp.mean(repertoire.count)
    median_count = jnp.median(repertoire.count)
    std_count = jnp.std(repertoire.count)

    return {
        "qd_score": qd_score,
        "max_fitness": max_fitness,
        "coverage": coverage,
        "min_count": min_count,
        "max_count": max_count,
        "mean_count": mean_count,
        "median_count": median_count,
        "std_count": std_count,
    }

