import collections
import logging
import warnings

import numpy as np

from nengo import Network, Ensemble, Connection, Node, Probe
from nengo.exceptions import BuildError
import nengo.utils.numpy as npext

from nengo_loihi.builder.interneurons import (
    Preset10Interneurons,
    OnOffInterneurons,
)
from nengo_loihi.compartments import CxGroup
from nengo_loihi.inputs import CxSpikeInput


logger = logging.getLogger(__name__)


class CxModel(object):

    def __init__(self, dt=0.001, label=None):
        self.dt = dt
        self.label = label

        self.cx_inputs = collections.OrderedDict()
        self.cx_groups = collections.OrderedDict()

    def add_input(self, input):
        assert isinstance(input, CxSpikeInput)
        assert input not in self.cx_inputs
        self.cx_inputs[input] = len(self.cx_inputs)

    def add_group(self, group):
        assert isinstance(group, CxGroup)
        assert group not in self.cx_groups
        self.cx_groups[group] = len(self.cx_groups)

    def discretize(self):
        for group in self.cx_groups:
            group.discretize()

    def validate(self):
        if len(self.cx_groups) == 0:
            raise BuildError("No neurons marked for execution on-chip. "
                             "Please mark some ensembles as on-chip.")

        for group in self.cx_groups:
            group.validate()


class Model(CxModel):
    """The data structure for the chip/simulator.

    This is a subclass of CxModel, which defines methods for adding ensembles,
    discretizing, and tracking the simulator. This class handles build
    functions and keeping track of chip/host communication.

    Parameters
    ----------
    dt : float, optional (Default: 0.001)
        The length of a simulator timestep, in seconds.
    label : str, optional (Default: None)
        A name or description to differentiate models.
    builder : Builder, optional (Default: None)
        A `.Builder` instance to keep track of build functions.
        If None, the default builder will be used.

    Attributes
    ----------
    builder : Builder
        The build functions used by this model.
    dt : float
        The length of a simulator timestep, in seconds.
    label : str or None
        A name or description to differentiate models.
    objs : dict
        Dictionary mapping from Nengo objects to Nengo Loihi objects.
    params : dict
        Mapping from objects to namedtuples containing parameters generated
        in the build process.
    probes : list
        List of all probes. Probes must be added to this list in the build
        process, as this list is used by Simulator.
    seeded : dict
        All objects are assigned a seed, whether the user defined the seed
        or it was automatically generated. 'seeded' keeps track of whether
        the seed is user-defined. We consider the seed to be user-defined
        if it was set directly on the object, or if a seed was set on the
        network in which the object resides, or if a seed was set on any
        ancestor network of the network in which the object resides.
    seeds : dict
        Mapping from objects to the integer seed assigned to that object.
    """
    def __init__(self, dt=0.001, label=None, builder=None):
        super(Model, self).__init__(dt=dt, label=label)

        self.objs = collections.defaultdict(dict)
        self.params = {}  # Holds data generated when building objects
        self.probes = []
        self.probe_conns = {}

        self.seeds = {}
        self.seeded = {}

        self.builder = Builder() if builder is None else builder
        self.build_callback = None

        # --- other (typically standard) parameters
        # Filter on intermediary neurons
        self.inter_tau = 0.005
        # ^TODO: how to choose this filter? Even though the input is spikes,
        # it may not be absolutely necessary since tau_rc provides a filter,
        # and maybe we don't want double filtering if connection has a filter

        # self.inter_neurons = NoisyInterneurons(10, dt=dt)
        # self.inter_neurons = Preset5Interneurons(dt=dt)
        self.inter_neurons = Preset10Interneurons(dt=dt)
        self.node_neurons = OnOffInterneurons(dt=dt)

        # voltage threshold for non-spiking neurons (i.e. voltage decoders)
        self.vth_nonspiking = 10

        # limit for clipping intercepts, to avoid neurons with high gains
        self.intercept_limit = 0.95

        # scaling for PES errors, before rounding and clipping to -127..127
        self.pes_error_scale = 100.

        # Will be provided by Simulator
        self.chip2host_params = {}

    def __getstate__(self):
        raise NotImplementedError("Can't pickle nengo_loihi.builder.Model")

    def __setstate__(self, state):
        raise NotImplementedError("Can't pickle nengo_loihi.builder.Model")

    def __str__(self):
        return "Model: %s" % self.label

    def build(self, obj, *args, **kwargs):
        built = self.builder.build(self, obj, *args, **kwargs)
        if self.build_callback is not None:
            self.build_callback(obj)
        return built

    def has_built(self, obj):
        return obj in self.params


class Builder(object):
    """Fills in the Loihi Model object based on the Nengo Network."""

    builders = {}  # Methods that build different components

    @classmethod
    def build(cls, model, obj, *args, **kwargs):
        if model.has_built(obj):
            warnings.warn("Object %s has already been built." % obj)
            return None

        for obj_cls in type(obj).__mro__:
            if obj_cls in cls.builders:
                break
        else:
            raise BuildError(
                "Cannot build object of type %r" % type(obj).__name__)

        return cls.builders[obj_cls](model, obj, *args, **kwargs)

    @classmethod
    def register(cls, nengo_class):
        """Register methods to build Nengo objects into Model."""

        def register_builder(build_fn):
            if nengo_class in cls.builders:
                warnings.warn("Type '%s' already has a builder. Overwriting."
                              % nengo_class)
            cls.builders[nengo_class] = build_fn
            return build_fn
        return register_builder


@Builder.register(Network)
def build_network(model, network):
    def get_seed(obj, rng):
        return (rng.randint(npext.maxint)
                if not hasattr(obj, 'seed') or obj.seed is None else obj.seed)

    if network not in model.seeds:
        model.seeded[network] = getattr(network, 'seed', None) is not None
        model.seeds[network] = get_seed(network, np.random)

    # # Set config
    # old_config = model.config
    # model.config = network.config

    # assign seeds to children
    rng = np.random.RandomState(model.seeds[network])
    # Put probes last so that they don't influence other seeds
    sorted_types = (Connection, Ensemble, Network, Node, Probe)
    assert all(tp in sorted_types for tp in network.objects)
    for obj_type in sorted_types:
        for obj in network.objects[obj_type]:
            model.seeded[obj] = (model.seeded[network]
                                 or getattr(obj, 'seed', None) is not None)
            model.seeds[obj] = get_seed(obj, rng)

    logger.debug("Network step 1: Building ensembles and nodes")
    for obj in network.ensembles + network.nodes:
        model.build(obj)

    logger.debug("Network step 2: Building subnetworks")
    for subnetwork in network.networks:
        model.build(subnetwork)

    logger.debug("Network step 3: Building connections")
    for conn in network.connections:
        model.build(conn)

    logger.debug("Network step 4: Building probes")
    for probe in network.probes:
        model.build(probe)

    # # Unset config
    # model.config = old_config
    model.params[network] = None
