from __future__ import division

import warnings

import numpy as np
from nengo.exceptions import BuildError
from nengo.utils.compat import is_iterable

from nengo_loihi.discretize import (
    array_to_int,
    learn_overflow_bits,
    tracing_mag_int_frac,
)
from nengo_loihi.utils import Profile, shift


class SynapseFmt(Profile):
    INDEX_BITS_MAP = [0, 6, 7, 8, 9, 10, 11, 12]
    WEIGHT_BITS_MAP = [0, 1, 2, 3, 4, 5, 6, 8]

    params = ('wgtLimitMant', 'wgtLimitExp', 'wgtExp', 'discMaxWgt',
              'learningCfg', 'tagBits', 'dlyBits', 'wgtBits',
              'reuseSynData', 'numSynapses', 'cIdxOffset', 'cIdxMult',
              'skipBits', 'idxBits', 'synType', 'fanoutType',
              'compression', 'stdpProfile', 'ignoreDly')

    def __init__(self, wgtLimitMant=0, wgtLimitExp=0, wgtExp=0, discMaxWgt=0,
                 learningCfg=0, tagBits=0, dlyBits=0, wgtBits=0,
                 reuseSynData=0, numSynapses=0, cIdxOffset=0, cIdxMult=0,
                 skipBits=0, idxBits=0, synType=0, fanoutType=0,
                 compression=0, stdpProfile=0, ignoreDly=0):
        self.wgtLimitMant = wgtLimitMant
        self.wgtLimitExp = wgtLimitExp
        self.wgtExp = wgtExp
        self.discMaxWgt = discMaxWgt
        self.learningCfg = learningCfg
        self.tagBits = tagBits
        self.dlyBits = dlyBits
        self.wgtBits = wgtBits
        self.reuseSynData = reuseSynData
        self.numSynapses = numSynapses
        self.cIdxOffset = cIdxOffset
        self.cIdxMult = cIdxMult
        self.skipBits = skipBits
        self.idxBits = idxBits
        self.synType = synType
        self.fanoutType = fanoutType
        self.compression = compression
        self.stdpProfile = stdpProfile
        self.ignoreDly = ignoreDly

    @classmethod
    def get_realWgtExp(cls, wgtExp):
        return 6 + wgtExp

    @classmethod
    def get_scale(cls, wgtExp):
        return 2**cls.get_realWgtExp(wgtExp)

    @property
    def realWgtExp(self):
        return self.get_realWgtExp(self.wgtExp)

    @property
    def scale(self):
        return self.get_scale(self.wgtExp)

    @property
    def realWgtBits(self):
        return self.WEIGHT_BITS_MAP[self.wgtBits]

    @property
    def realIdxBits(self):
        return self.INDEX_BITS_MAP[self.idxBits]

    @property
    def isMixed(self):
        return self.fanoutType == 1

    @property
    def shift_bits(self):
        """Number of bits the -256..255 weight is right-shifted by."""
        return 8 - self.realWgtBits + self.isMixed

    def bits_per_axon(self, n_weights):
        """For an axon with n weights, compute the weight memory bits used"""
        bits_per_weight = self.realWgtBits + self.dlyBits + self.tagBits
        if self.compression == 0:
            bits_per_weight += self.realIdxBits
        elif self.compression == 3:
            pass
        else:
            raise NotImplementedError("Compression %s" % (self.compression,))

        SYNAPSE_FMT_IDX_BITS = 4
        N_SYNAPSES_BITS = 6
        bits = 0
        synapses_per_group = self.numSynapses + 1
        for i in range(0, n_weights, synapses_per_group):
            n = min(n_weights - i, synapses_per_group)
            bits_i = n*bits_per_weight + SYNAPSE_FMT_IDX_BITS + N_SYNAPSES_BITS
            bits_i = -64 * (-bits_i // 64)
            # ^ round up to nearest 64 (size of one int64 memory unit)
            bits += bits_i

        return bits

    def set(self, **kwargs):
        for key, value in kwargs.items():
            assert hasattr(self, key)
            setattr(self, key, value)

    def validate(self, core=None):
        assert -7 <= self.wgtExp <= 7
        assert 0 <= self.tagBits < 4
        assert 0 <= self.dlyBits < 8
        assert 1 <= self.wgtBits < 8
        assert 0 <= self.cIdxOffset < 16
        assert 0 <= self.cIdxMult < 16
        assert 0 <= self.idxBits < 8
        assert 1 <= self.fanoutType < 4

    def discretize_weights(
            self, w, dtype=np.int32, lossy_shift=True, check_result=True):
        """Takes weights and returns their quantized values with wgtExp.

        The actual weight to be put on the chip is this returned value
        divided by the ``scale`` attribute.

        Parameters
        ----------
        w : float ndarray
            Weights to be discretized, in the range -255 to 255.
        dtype : np.dtype, optional (Default: np.int32)
            Data type for discretized weights.
        lossy_shift : bool, optional (Default: True)
            Whether to mimic the two-part weight shift that currently happens
            on the chip, which can lose information for small wgtExp.
        check_results : bool, optional (Default: True)
            Whether to check that the discretized weights fall in
            the valid range for weights on the chip (-256 to 255).
        """
        s = self.shift_bits
        m = 2**(8 - s) - 1

        w = np.round(w / 2.**s).clip(-m, m).astype(dtype)
        s2 = s + self.wgtExp

        if lossy_shift:
            if s2 < 0:
                warnings.warn("Lost %d extra bits in weight rounding" % (-s2,))

                # Round before `s2` right shift. Just shifting would floor
                # everything resulting in weights biased towards being smaller.
                w = (np.round(w * 2.**s2) / 2**s2).clip(-m, m).astype(dtype)

            shift(w, s2, out=w)
            np.left_shift(w, 6, out=w)
        else:
            shift(w, 6 + s2, out=w)

        if check_result:
            ws = w // self.scale
            assert np.all(ws <= 255) and np.all(ws >= -256)

        return w


class Synapses(object):
    """A group of Loihi synapses that share some properties.

    Attributes
    ----------
    n_axons : int
        Number of input axons to this group of synapses.
    synapse_fmt : SynapseFmt
        The synapse format object for these synapses.
    weights : (n_axons,) list of (n_populations, n_compartments) ndarray
        The synapse weights. Organized as a list of arrays so each axon
        can have a different number of target compartments.
    indices : (population, axon, compartment) ndarray
        The synapse indices.
    axon_cx_bases : list or None
        List providing ax cx_base (compartment offset) for each input axon.
    axon_to_weight_map : dict or None
        Map from input axon index to weight index, to allow weights to be
        re-used by axons. If None, the weight index for an input axon is the
        axon index.
    learning : bool
        Whether synaptic tracing and learning is enabled for these synapses.
    learning_rate : float
        The learning rate.
    learning_wgt_exp : int
        The weight exponent used on this connection if learning is enabled.
    tracing_tau : int
        Decay time constant for the learning trace, in timesteps (not seconds).
    tracing_mag : float
        Magnitude by which the learning trace is increased for each spike.
    pop_type : int (0, 16, 32)
        Whether these synapses are discrete (0), pop16, or pop32. This
        determines the type of axons these synapses can connect to.
    """
    def __init__(self, n_axons, label=None):
        self.n_axons = n_axons
        self.label = label
        self.synapse_fmt = None
        self.weights = None
        self.indices = None
        self.axon_cx_bases = None
        self.axon_to_weight_map = None

        self.learning = False
        self.learning_rate = 1.
        self.learning_wgt_exp = None
        self.tracing_tau = None
        self.tracing_mag = None
        self.pop_type = 0  # one of (0, 16, 32) for discrete, pop16, pop32

    def __str__(self):
        return "%s(%s)" % (
            type(self).__name__, self.label if self.label else '')

    def size(self):
        return sum(w.size for w in self.weights)

    def bits(self):
        return sum(self.synapse_fmt.bits_per_axon(w.size)
                   for w in self.weights)

    def max_abs_weight(self):
        return max(np.abs(w).max() if w.size > 0 else -np.inf
                   for w in self.weights)

    def max_ind(self):
        return max(i.max() if len(i) > 0 else -1 for i in self.indices)

    def idx_bits(self):
        idxBits = int(np.ceil(np.log2(self.max_ind() + 1)))
        assert idxBits <= SynapseFmt.INDEX_BITS_MAP[-1], (
            "idxBits out of range, ensemble too large?")
        idxBits = next(i for i, v in enumerate(SynapseFmt.INDEX_BITS_MAP)
                       if v >= idxBits)
        return idxBits

    def idxs_per_synapse(self):
        return 2 if self.learning else 1

    def atom_bits_extra(self):
        atom_bits = self.atom_bits()
        assert atom_bits <= 9, "Cannot have more than 9 atom bits"
        return max(atom_bits - 5, 0)  # has 5 bits by default

    def atom_bits(self):
        max_populations = max(w.shape[0] for w in self.weights)
        return int(np.ceil(np.log2(max_populations)))

    def axon_bits(self):
        if self.pop_type == 16:
            return 10 - self.atom_bits_extra()
        else:
            return 12

    def axon_populations(self, axon_idx):
        weight_idx = self.axon_weight_idx(axon_idx)
        return self.weights[weight_idx].shape[0]

    def axon_weight_idx(self, axon_idx):
        return (self.axon_to_weight_map[axon_idx]
                if self.axon_to_weight_map is not None else axon_idx)

    def axon_weights_indices(self, axon_idx, atom=0):
        weight_idx = self.axon_weight_idx(axon_idx)
        w = self.weights[weight_idx]
        i = self.indices[weight_idx]
        return w[atom, :], i[atom, :]

    def axon_cx_base(self, axon_idx):
        if self.axon_cx_bases is None:
            return 0
        cx_base = self.axon_cx_bases[axon_idx]
        return cx_base if cx_base > -1024 else None

    def _set_weights_indices(self, weights, indices=None):
        weights = [np.array(w, copy=False, dtype=np.float32, ndmin=2)
                   for w in weights]
        assert all(w.ndim == 2 for w in weights), (
            "Weights must be shape (n_axons,) (n_populations, n_compartments)")
        assert all(w.shape[0] == weights[0].shape[0] for w in weights), (
            "All axon weights must have the same number of populations")
        self.weights = weights

        if indices is None:
            indices = [np.zeros((w.shape[0], 1), dtype=np.int32)
                       + np.arange(w.shape[1], dtype=np.int32)
                       for w in self.weights]
        indices = [np.array(i, copy=False, dtype=np.int32, ndmin=2)
                   for i in indices]
        assert all(i.ndim == 2 for i in indices), (
            "Indices must be shape (n_axons,) (n_populations, n_compartments)")
        assert all(i.shape == w.shape for i, w in zip(indices, weights)), (
            "Indices shapes must match weights shapes")
        assert len(weights) == len(indices)
        self.indices = indices

    def set_full_weights(self, weights):
        self._set_weights_indices(weights)
        assert len(self.weights) == self.n_axons, (
            "Full weights must have different weights for each axon")

        idxBits = self.idx_bits()
        self.format(compression=3, idxBits=idxBits, fanoutType=1,
                    numSynapses=63, wgtBits=7)

    def set_diagonal_weights(self, diag):
        weights = diag.ravel()
        indices = list(range(len(weights)))
        self._set_weights_indices(weights, indices)
        assert len(self.weights) == self.n_axons

        idxBits = self.idx_bits()
        self.format(compression=3, idxBits=idxBits, fanoutType=1,
                    numSynapses=63, wgtBits=7)

    def set_population_weights(
            self,
            weights,
            indices,
            axon_to_weight_map,
            cx_bases,
            pop_type=None
    ):
        self._set_weights_indices(weights, indices)
        self.axon_to_weight_map = axon_to_weight_map
        self.axon_cx_bases = cx_bases
        self.pop_type = 16 if pop_type is None else pop_type

        idxBits = self.idx_bits()
        self.format(compression=0,
                    idxBits=idxBits,
                    fanoutType=1,
                    numSynapses=63,
                    wgtBits=7)

    def set_learning(
            self, learning_rate=1., tracing_tau=2, tracing_mag=1.0, wgt_exp=4):
        assert tracing_tau == int(tracing_tau), "tracing_tau must be integer"

        self.learning = True
        self.tracing_tau = int(tracing_tau)
        self.tracing_mag = tracing_mag
        self.format(learningCfg=1, stdpProfile=0)
        # ^ stdpProfile hard-coded for now (see hardware.builder)

        self.train_epoch = 2
        self.learn_epoch_k = 1
        self.learn_epoch = self.train_epoch * 2**self.learn_epoch_k

        self.learning_rate = learning_rate * self.learn_epoch
        self.learning_wgt_exp = wgt_exp

    def format(self, **kwargs):
        if self.synapse_fmt is None:
            self.synapse_fmt = SynapseFmt()
        self.synapse_fmt.set(**kwargs)

    def validate(self):
        self.synapse_fmt.validate()
        if self.axon_cx_bases is not None:
            assert np.all(self.axon_cx_bases < 256), "CxBase cannot be > 256"
        if self.pop_type == 16:
            if self.axon_cx_bases is not None:
                assert np.all(self.axon_cx_bases % 4 == 0)


class SynapseGroup(object):
    """A group of synapses, typically belonging to a NeuronGroup.

    Attributes
    ----------
    named_synapses : {string: Synapses}
        Maps names for synapses to the synapses themselves.
    synapses : list of Synapses
        A list of all synapses in this group.
    """

    def __init__(self):
        self.synapses = []
        self.named_synapses = {}

    def __iter__(self):
        return iter(self.synapses)

    def __len__(self):
        return len(self.synapses)

    def by_name(self, name):
        return self.named_synapses[name]

    def has_name(self, name):
        return name in self.named_synapses

    def add(self, synapses, name=None):
        """Add a Synapses object to this group."""
        self.synapses.append(synapses)
        if name is not None:
            assert name not in self.named_synapses
            self.named_synapses[name] = synapses

        self._validate_structure()

    def max_weight(self):
        w_maxs = [s.max_abs_weight() for s in self.synapses]
        return max(w_maxs) if len(w_maxs) > 0 else 0

    def discretize(self, w_max, w_scale, w_exp):
        for i, synapse in enumerate(self.synapses):
            w_max_i = synapse.max_abs_weight()
            if synapse.learning:
                w_exp2 = synapse.learning_wgt_exp
                dw_exp = w_exp - w_exp2
            elif w_max_i > 1e-16:
                dw_exp = int(np.floor(np.log2(w_max / w_max_i)))
                assert dw_exp >= 0
                w_exp2 = max(w_exp - dw_exp, -6)
            else:
                w_exp2 = -6
                dw_exp = w_exp - w_exp2
            synapse.format(wgtExp=w_exp2)
            for w, idxs in zip(synapse.weights, synapse.indices):
                ws = w_scale[idxs] if is_iterable(w_scale) else w_scale
                array_to_int(w, synapse.synapse_fmt.discretize_weights(
                    w * ws * 2 ** dw_exp))

            # discretize learning
            if synapse.learning:
                synapse.tracing_tau = int(np.round(synapse.tracing_tau))

                if is_iterable(w_scale):
                    assert np.all(w_scale == w_scale[0])
                w_scale_i = w_scale[0] if is_iterable(w_scale) else w_scale

                # incorporate weight scale and difference in weight exponents
                # to learning rate, since these affect speed at which we learn
                ws = w_scale_i * 2 ** dw_exp
                synapse.learning_rate *= ws

                # Loihi down-scales learning factors based on the number of
                # overflow bits. Increasing learning rate maintains true rate.
                synapse.learning_rate *= 2 ** learn_overflow_bits(2)

                # TODO: Currently, Loihi learning rate fixed at 2**-7.
                # We should explore adjusting it for better performance.
                lscale = 2 ** -7 / synapse.learning_rate
                synapse.learning_rate *= lscale
                synapse.tracing_mag /= lscale

                # discretize learning rate into mantissa and exponent
                lr_exp = int(np.floor(np.log2(synapse.learning_rate)))
                lr_int = int(np.round(synapse.learning_rate * 2 ** (-lr_exp)))
                synapse.learning_rate = lr_int * 2 ** lr_exp
                synapse._lr_int = lr_int
                synapse._lr_exp = lr_exp
                assert lr_exp >= -7

                # discretize tracing mag into integer and fractional components
                mag_int, mag_frac = tracing_mag_int_frac(synapse.tracing_mag)
                if mag_int > 127:
                    warnings.warn("Trace increment exceeds upper limit "
                                  "(learning rate may be too large)")
                    mag_int = 127
                    mag_frac = 127
                synapse.tracing_mag = mag_int + mag_frac / 128.

    def _validate_structure(self):
        IN_AXONS_MAX = 4096
        n_axons = sum(s.n_axons for s in self)
        if n_axons > IN_AXONS_MAX:
            raise BuildError("Input axons (%d) exceeded max (%d)" % (
                n_axons, IN_AXONS_MAX))

        MAX_SYNAPSE_BITS = 16384*64
        synapse_bits = sum(s.bits() for s in self)
        if synapse_bits > MAX_SYNAPSE_BITS:
            raise BuildError("Total synapse bits (%d) exceeded max (%d)" % (
                synapse_bits, MAX_SYNAPSE_BITS))

    def validate(self):
        self._validate_structure()
        for synapses in self:
            synapses.validate()
