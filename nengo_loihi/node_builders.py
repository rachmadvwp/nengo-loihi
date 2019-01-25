from nengo import Node
from nengo.utils.compat import is_integer

from nengo_loihi.builder import Builder
from nengo_loihi.io_objects import (
    ChipReceiveNode,
)


@Builder.register(Node)
def build_node(model, node):
    if isinstance(node, ChipReceiveNode):
        spike_input = SpikeInput(node.raw_dimensions, label=node.label)
        model.add_input(spike_input)
        model.objs[node]['out'] = spike_input
        node.cx_spike_input = spike_input
    else:
        raise NotImplementedError()


class SpikeInput(object):
    def __init__(self, n_neurons, label=None):
        self.n_neurons = n_neurons
        self.label = label

        self.spikes = {}  # map sim timestep index to list of spike inds
        self.axons = []
        self.probes = []

    def add_axons(self, axons):
        self.axons.append(axons)

    def add_probe(self, probe):
        if probe.target is None:
            probe.target = self
        assert probe.target is self
        self.probes.append(probe)

    def add_spikes(self, ti, spike_idxs):
        assert is_integer(ti)
        ti = int(ti)
        assert ti > 0, "Spike times must be >= 1 (got %d)" % ti
        assert ti not in self.spikes
        self.spikes[ti] = spike_idxs

    def clear_spikes(self):
        self.spikes.clear()

    def spike_times(self):
        return sorted(self.spikes)

    def spike_idxs(self, ti):
        return self.spikes.get(ti, [])