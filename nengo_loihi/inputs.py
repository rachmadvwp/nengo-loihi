from __future__ import division

import collections

import numpy as np
import nengo
from nengo.exceptions import SimulationError
from nengo.utils.compat import is_integer


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


class PESModulatoryTarget(object):
    def __init__(self, target):
        self.target = target
        self.errors = collections.OrderedDict()

    def clear(self):
        self.errors.clear()

    def receive(self, t, x):
        assert len(self.errors) == 0 or t >= next(reversed(self.errors))
        if t in self.errors:
            self.errors[t] += x
        else:
            self.errors[t] = np.array(x)

    def collect_errors(self):
        for t, x in self.errors.items():
            yield (self.target, t, x)


class HostSendNode(nengo.Node):
    """For sending host->chip messages"""

    def __init__(self, dimensions):
        self.queue = []
        super(HostSendNode, self).__init__(self.update,
                                           size_in=dimensions, size_out=0)

    def update(self, t, x):
        assert len(self.queue) == 0 or t > self.queue[-1][0]
        self.queue.append((t, x))


class HostReceiveNode(nengo.Node):
    """For receiving chip->host messages"""

    def __init__(self, dimensions):
        self.queue = [(0, np.zeros(dimensions))]
        self.queue_index = 0
        super(HostReceiveNode, self).__init__(self.update,
                                              size_in=0, size_out=dimensions)

    def update(self, t):
        while (len(self.queue) > self.queue_index + 1
               and self.queue[self.queue_index][0] < t):
            self.queue_index += 1
        return self.queue[self.queue_index][1]

    def receive(self, t, x):
        self.queue.append((t, x))


class ChipReceiveNode(nengo.Node):
    """For receiving host->chip messages"""

    def __init__(self, dimensions, size_out, **kwargs):
        self.raw_dimensions = dimensions
        self.spikes = []
        self.cx_spike_input = None  # set by builder
        super(ChipReceiveNode, self).__init__(
            self.update, size_in=0, size_out=size_out, **kwargs)

    def clear(self):
        self.spikes.clear()

    def receive(self, t, x):
        assert len(self.spikes) == 0 or t > self.spikes[-1][0]
        assert x.ndim == 1
        self.spikes.append((t, x.nonzero()[0]))

    def update(self, t):
        raise SimulationError("ChipReceiveNodes should not be run")

    def collect_spikes(self):
        assert self.cx_spike_input is not None
        for t, x in self.spikes:
            yield (self.cx_spike_input, t, x)


class ChipReceiveNeurons(ChipReceiveNode):
    """Passes spikes directly (no on-off neuron encoding)"""
    def __init__(self, dimensions, neuron_type=None, **kwargs):
        self.neuron_type = neuron_type
        super(ChipReceiveNeurons, self).__init__(
            dimensions, dimensions, **kwargs)
