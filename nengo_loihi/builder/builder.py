import collections
import logging
import warnings


from nengo.exceptions import BuildError

from nengo_loihi.builder.decode_neurons import (
    Preset10DecodeNeurons,
    OnOffDecodeNeurons,
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
        # Filter on decode neurons
        self.decode_tau = 0.005
        # ^TODO: how to choose this filter? Even though the input is spikes,
        # it may not be absolutely necessary since tau_rc provides a filter,
        # and maybe we don't want double filtering if connection has a filter

        self.decode_neurons = Preset10DecodeNeurons(dt=dt)
        self.node_neurons = OnOffDecodeNeurons(dt=dt)

        # voltage threshold for non-spiking neurons (i.e. voltage decoders)
        self.vth_nonspiking = 10

        # limit for clipping intercepts, to avoid neurons with high gains
        self.intercept_limit = 0.95

        # scaling for PES errors, before rounding and clipping to -127..127
        self.pes_error_scale = 100.

        # learning weight exponent for PES (controls the maximum weight
        # magnitude/weight resolution)
        self.pes_wgt_exp = 4

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
