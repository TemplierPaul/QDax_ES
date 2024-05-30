import jax
import jax.numpy as jnp

import jax
import jax.numpy as jnp


from flax.struct import PyTreeNode
from jax.tree_util import PyTreeDef
from jax.tree_util import tree_flatten, tree_unflatten


def net_shape(net):
    return jax.tree_map(lambda x: x.shape, net)

class DummyReshaper(PyTreeNode):
    """ A placeholder reshaper that does nothing"""
    genotype_dim: int

    def flatten(self, network):
        return network

    def unflatten(self, vect):
        return vect

    @classmethod
    def init(self, network):
        # Network is a vector
        genotype_dim = network.shape[0]
        return DummyReshaper(
            genotype_dim=genotype_dim
        )


class ANNReshaper(DummyReshaper):
    """A class to reshape a network into a vector of floats and back"""
    split_indices: jnp.ndarray
    layer_shapes: jnp.ndarray
    tree_def: PyTreeDef

    def flatten(self, network):
        """Flatten a network into a vector of floats """
        flat_variables, _ = tree_flatten(network)
        # print("Flatten", flat_variables)
        vect = jnp.concatenate([jnp.ravel(x) for x in flat_variables])
        return vect

    def unflatten(self, vect):
        """Unflatten a vector of floats into a network"""
        # print("Unflatten", vect.shape)
        split_genome = jnp.split(vect, self.split_indices)
        # Reshape to the original shape
        split_genome = [x.reshape(s) for x, s in zip(split_genome, self.layer_shapes)]

        # Unflatten the tree
        new_net = tree_unflatten(self.tree_def, split_genome)
        return new_net

    # class method to create Reshaper
    @classmethod
    def init(self, network):
        """Initialize a reshaper from a network"""
        # print(net_shape(network))
        flat_variables, tree_def = tree_flatten(network)
        layer_shapes = [x.shape for x in flat_variables]
        layer_shapes = tuple(layer_shapes)
        # print("Layer shapes", layer_shapes)

        sizes = [x.size for x in flat_variables]
        sizes = jnp.array(sizes)

        genotype_dim = jnp.sum(sizes)

        split_indices = jnp.cumsum(sizes)[:-1]
        split_indices = tuple(split_indices.tolist())

        print(f"Genotype dim: {genotype_dim}")

        return ANNReshaper(
            split_indices=split_indices,
            layer_shapes=layer_shapes,
            tree_def=tree_def,
            genotype_dim=genotype_dim,
        )
