import numpy as np

import nengo.utils.numpy as npext
from nengo import Connection, Ensemble, Network, Node, Probe


def seed_network(network, base_rng=np.random):
    """Generate `seeds` and `seeded` dictionaries for all objects in a network.

    This process is meant to mimic the one used by the Nengo builder, so that
    the same seeds can be generated without having to build the whole network.
    """
    def get_seed(obj, rng):
        # Generate a seed no matter what, so that setting a seed or not on
        # one object doesn't affect the seeds of other objects.
        seed = rng.randint(npext.maxint)
        return (seed if not hasattr(obj, 'seed') or obj.seed is None
                else obj.seed)

    def _seed_network(network, seeds, seeded):
        # assign seeds to children
        rng = np.random.RandomState(seeds[network])

        # Put probes last so that they don't influence other seeds
        sorted_types = (Connection, Ensemble, Network, Node, Probe)
        assert all(tp in sorted_types for tp in network.objects)
        for obj_type in sorted_types:
            for obj in network.objects[obj_type]:
                seeded[obj] = (seeded[network] or
                               getattr(obj, 'seed', None) is not None)
                seeds[obj] = get_seed(obj, rng)

        # assign seeds to subnetworks
        for subnetwork in network.networks:
            _seed_network(subnetwork, seeds, seeded)

    # seed this base network
    seeds = {}
    seeded = {}
    seeds[network] = get_seed(network, base_rng)
    seeded[network] = getattr(network, 'seed', None) is not None

    # seed all sub-objects
    _seed_network(network, seeds, seeded)

    return seeds, seeded
