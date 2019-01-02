class Probe(object):
    _slice = slice

    def __init__(self, target=None, key=None, slice=None, weights=None,
                 synapse=None):
        self.target = target
        self.key = key
        self.slice = slice if slice is not None else self._slice(None)
        self.weights = weights
        self.synapse = synapse
        self.use_snip = False
        self.snip_info = None

    def validate(self):
        pass


class ProbeGroup(object):
    """A group of probes, typically belonging to a NeuronGroup.

    Attributes
    ----------
    probes : list of Probes
        A list of all probes in this group.
    """

    def __init__(self):
        self.probes = []

    def __iter__(self):
        return iter(self.probes)

    def __len__(self):
        return len(self.probes)

    def add(self, probe):
        self.probes.append(probe)

    def discretize(self, v_scale):
        for p in self.probes:
            if p.key == 'v' and p.weights is not None:
                p.weights /= v_scale

    def validate(self):
        for probe in self:
            probe.validate()
