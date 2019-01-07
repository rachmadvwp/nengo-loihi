from nengo_loihi.axons import AxonGroup
from nengo_loihi.compartments import CompartmentGroup
from nengo_loihi.probes import ProbeGroup
from nengo_loihi.synapses import SynapseGroup


class NeuronGroup(object):
    """Class holding Loihi objects that can be placed on the chip.

    Typically representing a Nengo Ensemble, this can also represent
    interneurons connecting Ensembles or a special decoding ensemble.

    Before ``discretize`` has been called, most parameters in this class are
    floating-point values. Calling ``discretize`` converts them to integer
    values inplace, for use on Loihi.

    Attributes
    ----------
    n_neurons : int
        The number of neurons in the group.
    label : string
        A label for the group (for debugging purposes).
    axons : AxonGroup
        Axons objects outputting from these neurons.
    compartments : CompartmentGroup
        Compartments object representing all compartments for these neurons.
    synapses : SynapseGroup
        Synapses objects projecting to these neurons.
    probes : ProbeGroup
        Probes recording information from these neurons.
    """
    def __init__(self, n_neurons, label=None):
        self.n_neurons = n_neurons
        self.label = label

        self.axons = AxonGroup()
        self.compartments = CompartmentGroup(n_compartments=n_neurons)
        self.synapses = SynapseGroup()
        self.probes = ProbeGroup()

    def __str__(self):
        return "%s(%s)" % (
            type(self).__name__, self.label if self.label else '')

    def add_synapses(self, synapses, name=None):
        self.synapses.add(synapses, name=name)

    def add_axons(self, axons, name=None):
        self.axons.add(axons, name=name)

    def add_probe(self, probe):
        self.probes.add(probe)

    def discretize(self):
        w_max = self.synapses.max_weight()
        p = self.compartments.discretize(w_max)
        self.synapses.discretize(w_max, p['w_scale'], p['w_exp'])
        self.probes.discretize(p['v_scale'][0])

    def validate(self):
        self.compartments.validate()
        self.axons.validate()
        self.synapses.validate()
        self.probes.validate()
