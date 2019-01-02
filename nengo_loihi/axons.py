import numpy as np

from nengo.exceptions import BuildError

from nengo_loihi.synapses import Synapses


class Axons(object):
    """A group of axons, targeting a specific Synapses object.

    Attributes
    ----------
    cx_atoms : list of length ``group.n``
        Atom (weight index) associated with each group compartment.
    cx_to_axon_map : list of length ``group.n``
        Index of the axon in `target` targeted by each group compartment.
    group : CompartmentGroup
        Parent CompartmentGroup for this object (set in `add_axons`).
    n_axons : int
        The number of outgoing axons.
    target : Synapses
        Target synapses for these axons.
    """

    class Spike(object):
        """A spike, targeting a particular axon within a Synapses object.

        The Synapses target is implicit, given by the Axons object that
        creates this Spike.

        Parameters
        ----------
        axon_id : int
            The index of the axon within the targeted Synapses object.
        atom : int, optional (Default: 0)
            An index into the target Synapses weights. This allows spikes
            targeting a particular axon to use different weights.
        """

        __slots__ = ['axon_id', 'atom']

        def __init__(self, axon_id, atom=0):
            self.axon_id = axon_id
            self.atom = atom

        def __repr__(self):
            return "%s(axon_id=%d, atom=%d)" % (
                type(self).__name__, self.axon_id, self.atom)

    def __init__(self, n_axons, label=None):
        self.n_axons = n_axons
        self.label = label
        self.group = None

        self.target = None
        self.cx_to_axon_map = None
        self.cx_atoms = None

    def __str__(self):
        return "%s(%s)" % (
            type(self).__name__, self.label if self.label else '')

    @property
    def pop_type(self):
        return self.target.pop_type

    @property
    def slots_per_axon(self):
        """The number of axonCfg slots occupied by each axon."""
        return 2 if self.pop_type == 32 else 1

    def axon_slots(self):
        """The total number of axonCfg slots used by all axons."""
        return self.slots_per_axon * self.n_axons

    def set_axon_map(self, cx_to_axon_map, cx_atoms=None):
        self.cx_to_axon_map = cx_to_axon_map
        self.cx_atoms = cx_atoms

    def map_cx_axons(self, cx_idxs):
        return (self.cx_to_axon_map[cx_idxs]
                if self.cx_to_axon_map is not None else cx_idxs)

    def map_cx_atoms(self, cx_idxs):
        return (self.cx_atoms[cx_idxs] if self.cx_atoms is not None else
                [0 for _ in cx_idxs])

    def map_cx_spikes(self, cx_idxs):
        axon_ids = self.map_cx_axons(cx_idxs)
        atoms = self.map_cx_atoms(cx_idxs)
        return [self.Spike(axon_id, atom=atom) if axon_id >= 0 else None
                for axon_id, atom in zip(axon_ids, atoms)]

    def validate(self):
        if isinstance(self.target, Synapses):
            if self.cx_atoms is not None:
                cx_idxs = np.arange(len(self.cx_atoms))
                axon_ids = self.map_cx_axons(cx_idxs)
                for atom, axon_id in zip(self.cx_atoms, axon_ids):
                    n_populations = self.target.axon_populations(axon_id)
                    assert 0 <= atom < n_populations


class AxonGroup(object):
    """A group of axons, typically belonging to a NeuronGroup.

    Attributes
    ----------
    named_axons : {string: Axons}
        Maps names for axons to the axons themselves.
    axons : list of Axons
        A list of all axons in this group.
    """

    def __init__(self):
        self.axons = []
        self.named_axons = {}

    def __iter__(self):
        return iter(self.axons)

    def __len__(self):
        return len(self.axons)

    def add(self, axons, name=None):
        """Add an Axons object to this group."""
        self.axons.append(axons)
        if name is not None:
            assert name not in self.named_axons
            self.named_axons[name] = axons

    def _validate_structure(self):
        OUT_AXONS_MAX = 4096
        n_axons = sum(a.axon_slots() for a in self.axons)
        if n_axons > OUT_AXONS_MAX:
            raise BuildError("Output axons (%d) exceeded max (%d)" % (
                n_axons, OUT_AXONS_MAX))

    def validate(self):
        self._validate_structure()
        for axons in self:
            axons.validate()
