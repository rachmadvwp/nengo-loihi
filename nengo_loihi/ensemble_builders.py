import warnings

import nengo
from nengo import Ensemble
from nengo.builder.ensemble import BuiltEnsemble, gen_eval_points
from nengo.dists import Distribution, get_samples
from nengo.exceptions import BuildError
from nengo.utils.compat import is_iterable
import nengo.utils.numpy as npext
import numpy as np

from nengo_loihi.builder import Builder
# from nengo_loihi.neuron_builders import NeuronGroup
# from nengo_loihi.connection_builders import (
#     Synapses,
#     SynapseFmt,
#     AxonGroup,
#     SynapseGroup
# )
from nengo_loihi.discretize import (
    array_to_int,
    BIAS_MAX,
    bias_to_manexp,
    decay_magnitude,
    learn_overflow_bits,
    Q_BITS,
    shift,
    tracing_mag_int_frac,
    VTH_MAX,
    vth_to_manexp,
)
from nengo_loihi.probe_builders import ProbeGroup


def get_gain_bias(ens, rng=np.random, intercept_limit=1.0):
    # Modified from the Nengo version to handle `intercept_limit`

    if ens.gain is not None and ens.bias is not None:
        gain = get_samples(ens.gain, ens.n_neurons, rng=rng)
        bias = get_samples(ens.bias, ens.n_neurons, rng=rng)
        max_rates, intercepts = ens.neuron_type.max_rates_intercepts(
            gain, bias)
    elif ens.gain is not None or ens.bias is not None:
        # TODO: handle this instead of error
        raise NotImplementedError("gain or bias set for %s, but not both. "
                                  "Solving for one given the other is not "
                                  "implemented yet." % ens)
    else:
        int_distorarray = ens.intercepts
        if isinstance(int_distorarray, nengo.dists.Uniform):
            if int_distorarray.high > intercept_limit:
                warnings.warn(
                    "Intercepts are larger than intercept limit (%g). "
                    "High intercept values cause issues when discretizing "
                    "the model for running on Loihi." % intercept_limit)
                int_distorarray = nengo.dists.Uniform(
                    min(int_distorarray.low, intercept_limit),
                    min(int_distorarray.high, intercept_limit))

        max_rates = get_samples(ens.max_rates, ens.n_neurons, rng=rng)
        intercepts = get_samples(int_distorarray, ens.n_neurons, rng=rng)

        if np.any(intercepts > intercept_limit):
            intercepts[intercepts > intercept_limit] = intercept_limit
            warnings.warn(
                "Intercepts are larger than intercept limit (%g). "
                "High intercept values cause issues when discretizing "
                "the model for running on Loihi." % intercept_limit)

        gain, bias = ens.neuron_type.gain_bias(max_rates, intercepts)
        if gain is not None and (
                not np.all(np.isfinite(gain)) or np.any(gain <= 0.)):
            raise BuildError(
                "The specified intercepts for %s lead to neurons with "
                "negative or non-finite gain. Please adjust the intercepts so "
                "that all gains are positive. For most neuron types (e.g., "
                "LIF neurons) this is achieved by reducing the maximum "
                "intercept value to below 1." % ens)

    return gain, bias, max_rates, intercepts


@Builder.register(Ensemble)
def build_ensemble(model, ens):

    # Create random number generator
    rng = np.random.RandomState(model.seeds[ens])

    eval_points = gen_eval_points(ens, ens.eval_points, rng=rng)

    # Set up encoders
    if isinstance(ens.neuron_type, nengo.Direct):
        encoders = np.identity(ens.dimensions)
    elif isinstance(ens.encoders, Distribution):
        encoders = get_samples(
            ens.encoders, ens.n_neurons, ens.dimensions, rng=rng)
    else:
        encoders = npext.array(ens.encoders, min_dims=2, dtype=np.float64)
    if ens.normalize_encoders:
        encoders /= npext.norm(encoders, axis=1, keepdims=True)

    # Build the neurons
    gain, bias, max_rates, intercepts = get_gain_bias(
        ens, rng, model.intercept_limit)

    group = NeuronGroup(ens.n_neurons, label='%s' % ens)
    group.compartments.bias[:] = bias
    model.build(ens.neuron_type, ens.neurons, group)

    # set default filter just in case no other filter gets set
    group.compartments.configure_default_filter(model.decode_tau, dt=model.dt)

    if ens.noise is not None:
        raise NotImplementedError("Ensemble noise not implemented")

    # Scale the encoders
    if isinstance(ens.neuron_type, nengo.Direct):
        raise NotImplementedError("Direct neurons not implemented")
        # scaled_encoders = encoders
    else:
        # to keep scaling reasonable, we don't include the radius
        # scaled_encoders = encoders * (gain / ens.radius)[:, np.newaxis]
        scaled_encoders = encoders * gain[:, np.newaxis]

    model.add_group(group)

    model.objs[ens]['in'] = group
    model.objs[ens]['out'] = group
    model.objs[ens.neurons]['in'] = group
    model.objs[ens.neurons]['out'] = group
    model.params[ens] = BuiltEnsemble(
        eval_points=eval_points,
        encoders=encoders,
        intercepts=intercepts,
        max_rates=max_rates,
        scaled_encoders=scaled_encoders,
        gain=gain,
        bias=bias)


@Builder.register(nengo.neurons.NeuronType)
def build_neurons(model, neurontype, neurons, group):
    # If we haven't registered a builder for a specific type, then it cannot
    # be simulated on Loihi.
    raise BuildError(
        "The neuron type %r cannot be simulated on Loihi. Please either "
        "switch to a supported neuron type like LIF or "
        "SpikingRectifiedLinear, or explicitly mark ensembles using this "
        "neuron type as off-chip with\n"
        "  net.config[ensembles].on_chip = False")


@Builder.register(nengo.LIF)
def build_lif(model, lif, neurons, group):
    group.compartments.configure_lif(
        tau_rc=lif.tau_rc,
        tau_ref=lif.tau_ref,
        dt=model.dt)


@Builder.register(nengo.SpikingRectifiedLinear)
def build_relu(model, relu, neurons, group):
    group.compartments.configure_relu(
        vth=1./model.dt,  # so input == 1 -> neuron fires 1/dt steps -> 1 Hz
        dt=model.dt)


class CompartmentGroup(object):
    """Class holding Loihi objects that can be placed on the chip or Lakemont.

    Typically an ensemble or node, can be a special decoding ensemble. Once
    implemented, SNIPS might use this as well.

    Before ``discretize`` has been called, most parameters in this class are
    floating-point values. Calling ``discretize`` converts them to integer
    values inplace, for use on Loihi.

    Attributes
    ----------
    n_compartments : int
        The number of compartments in the group.
    label : string
        A label for the group (for debugging purposes).
    decayU : (n,) ndarray
        Input (synapse) decay constant for each compartment.
    decayV : (n,) ndarray
        Voltage decay constant for each compartment.
    tau_s : float or None
        Time constant used to set decayU. None if decayU has not been set.
    scaleU : bool
        Scale input (U) by decayU so that the integral of U is
        the same before and after filtering.
    scaleV : bool
        Scale voltage (V) by decayV so that the integral of V is
        the same before and after filtering.
    refractDelay : (n,) ndarray
        Compartment refractory delays, in time steps.
    vth : (n,) ndarray
        Compartment voltage thresholds.
    bias : (n,) ndarray
        Compartment biases.
    enableNoise : (n,) ndarray
        Whether to enable noise for each compartment.
    vmin : float or int (range [-2**23 + 1, 0])
        Minimum voltage for all compartments, in Loihi voltage units.
    vmax : float or int (range [2**9 - 1, 2**23 - 1])
        Maximum voltage for all compartments, in Loihi voltage units.
    noiseMantOffset0 : float or int
        Offset for noise generation.
    noiseExp0 : float or int
        Exponent for noise generation. Floating point values are base 10
        in units of current or voltage. Integer values are in base 2.
    noiseAtDenOrVm : {0, 1}
        Inject noise into current (0) or voltage (1).
    """
    # threshold at which U/V scaling is allowed
    DECAY_SCALE_TH = 0.5 / 2**12  # half of one decay scaling unit

    def __init__(self, n_compartments, label=None):
        self.n_compartments = n_compartments
        self.label = label

        # parameters specific to compartments/group
        self.decayU = np.ones(n_compartments, dtype=np.float32)
        # ^ default to no filter
        self.decayV = np.zeros(n_compartments, dtype=np.float32)
        # ^ default to integration
        self.tau_s = None
        self.scaleU = True
        self.scaleV = False

        self.refractDelay = np.zeros(n_compartments, dtype=np.int32)
        self.vth = np.zeros(n_compartments, dtype=np.float32)
        self.bias = np.zeros(n_compartments, dtype=np.float32)
        self.enableNoise = np.zeros(n_compartments, dtype=bool)

        # parameters common to core
        self.vmin = 0
        self.vmax = np.inf
        self.noiseMantOffset0 = 0
        self.noiseExp0 = 0
        self.noiseAtDendOrVm = 0

    def __str__(self):
        return "%s(%s)" % (
            type(self).__name__, self.label if self.label else '')

    def configure_default_filter(self, tau_s, dt=0.001):
        """Set the default Lowpass synaptic input filter for compartments.

        Parameters
        ----------
        tau_s : float
            `nengo.Lowpass` synapse time constant for filtering.
        dt : float
            Simulator time step.
        """
        if self.tau_s is None:  # don't overwrite a non-default filter
            self._configure_filter(tau_s, dt=dt)

    def configure_filter(self, tau_s, dt=0.001):
        """Set Lowpass synaptic input filter for compartments.

        Parameters
        ----------
        tau_s : float
            `nengo.Lowpass` synapse time constant for filtering.
        dt : float
            Simulator time step.
        """
        if self.tau_s is not None and tau_s < self.tau_s:
            warnings.warn("tau_s is already set to %g, which is larger than "
                          "%g. Using %g." % (self.tau_s, tau_s, self.tau_s))
            return
        elif self.tau_s is not None and tau_s > self.tau_s:
            warnings.warn(
                "tau_s is currently %g, which is smaller than %g. Overwriting "
                "tau_s with %g." % (self.tau_s, tau_s, tau_s))
        self._configure_filter(tau_s, dt=dt)
        self.tau_s = tau_s

    def _configure_filter(self, tau_s, dt):
        decayU = 1 if tau_s == 0 else -np.expm1(-dt/np.asarray(tau_s))
        self.decayU[:] = decayU
        self.scaleU = decayU > self.DECAY_SCALE_TH
        if not self.scaleU:
            raise BuildError(
                "Current (U) scaling is required. Perhaps a synapse time "
                "constant is too large in your model.")

    def configure_lif(self, tau_rc=0.02, tau_ref=0.001, vth=1, dt=0.001):
        self.decayV[:] = -np.expm1(-dt/np.asarray(tau_rc))
        self.refractDelay[:] = np.round(tau_ref / dt) + 1
        self.vth[:] = vth
        self.vmin = 0
        self.vmax = np.inf
        self.scaleV = np.all(self.decayV > self.DECAY_SCALE_TH)
        if not self.scaleV:
            raise BuildError(
                "Voltage (V) scaling is required with LIF neurons. Perhaps "
                "the neuron tau_rc time constant is too large.")

    def configure_relu(self, tau_ref=0.0, vth=1, dt=0.001):
        self.decayV[:] = 0.
        self.refractDelay[:] = np.round(tau_ref / dt) + 1
        self.vth[:] = vth
        self.vmin = 0
        self.vmax = np.inf
        self.scaleV = False

    def configure_nonspiking(self, tau_ref=0.0, vth=1, dt=0.001):
        self.decayV[:] = 1.
        self.refractDelay[:] = 1
        self.vth[:] = vth
        self.vmin = 0
        self.vmax = np.inf
        self.scaleV = False

    def discretize(self, w_max):
        # --- discretize decayU and decayV
        # subtract 1 from decayU here because it gets added back by the chip
        decayU = self.decayU * (2**12 - 1) - 1
        array_to_int(self.decayU, np.clip(decayU, 0, 2**12 - 1))
        array_to_int(self.decayV, self.decayV * (2**12 - 1))

        # Compute factors for current and voltage decay. These factors
        # counteract the fact that for longer decays, the current (or voltage)
        # created by a single spike has a larger integral.
        u_infactor = (1. / decay_magnitude(self.decayU, x0=2**21, offset=1)
                      if self.scaleU else np.ones(self.decayU.shape))
        v_infactor = (1. / decay_magnitude(self.decayV, x0=2**21)
                      if self.scaleV else np.ones(self.decayV.shape))
        self.scaleU = False
        self.scaleV = False

        # --- vmin and vmax
        vmine = np.clip(np.round(np.log2(-self.vmin + 1)), 0, 2**5-1)
        self.vmin = -2**vmine + 1
        vmaxe = np.clip(np.round((np.log2(self.vmax + 1) - 9)*0.5), 0, 2**3-1)
        self.vmax = 2**(9 + 2*vmaxe) - 1

        # --- discretize weights and vth
        # To avoid overflow, we can either lower vth_max or lower w_exp_max.
        # Lowering vth_max is more robust, but has the downside that it may
        # force smaller w_exp on connections than necessary, potentially
        # leading to lost weight bits (see SynapseFmt.discretize_weights).
        # Lowering w_exp_max can let us keep vth_max higher, but overflow
        # is still be possible on connections with many small inputs (uncommon)
        vth_max = VTH_MAX
        w_exp_max = 0

        b_max = np.abs(self.bias).max()
        w_exp = 0

        if w_max > 1e-8:
            w_scale = (255. / w_max)
            s_scale = 1. / (u_infactor * v_infactor)

            for w_exp in range(w_exp_max, -8, -1):
                v_scale = s_scale * w_scale * SynapseFmt.get_scale(w_exp)
                b_scale = v_scale * v_infactor
                vth = np.round(self.vth * v_scale)
                bias = np.round(self.bias * b_scale)
                if (vth <= vth_max).all() and (np.abs(bias) <= BIAS_MAX).all():
                    break
            else:
                raise BuildError("Could not find appropriate weight exponent")
        elif b_max > 1e-8:
            b_scale = BIAS_MAX / b_max
            while b_scale*b_max > 1:
                v_scale = b_scale / v_infactor
                w_scale = b_scale * u_infactor / SynapseFmt.get_scale(w_exp)
                vth = np.round(self.vth * v_scale)
                bias = np.round(self.bias * b_scale)
                if np.all(vth <= vth_max):
                    break

                b_scale /= 2.
            else:
                raise BuildError("Could not find appropriate bias scaling")
        else:
            # reduce vth_max in this case to avoid overflow since we're setting
            # all vth to vth_max (esp. in learning with zeroed initial weights)
            vth_max = min(vth_max, 2**Q_BITS - 1)
            v_scale = np.array([vth_max / (self.vth.max() + 1)])
            vth = np.round(self.vth * v_scale)
            b_scale = v_scale * v_infactor
            bias = np.round(self.bias * b_scale)
            w_scale = (v_scale * v_infactor * u_infactor
                       / SynapseFmt.get_scale(w_exp))

        vth_man, vth_exp = vth_to_manexp(vth)
        array_to_int(self.vth, vth_man * 2**vth_exp)

        bias_man, bias_exp = bias_to_manexp(bias)
        array_to_int(self.bias, bias_man * 2**bias_exp)

        # --- noise
        assert (v_scale[0] == v_scale).all()
        noiseExp0 = np.round(np.log2(10.**self.noiseExp0 * v_scale[0]))
        if noiseExp0 < 0:
            warnings.warn("Noise amplitude falls below lower limit")
        if noiseExp0 > 23:
            warnings.warn(
                "Noise amplitude exceeds upper limit (%d > 23)" % (noiseExp0,))
        self.noiseExp0 = int(np.clip(noiseExp0, 0, 23))
        self.noiseMantOffset0 = int(np.round(2*self.noiseMantOffset0))

        return dict(w_max=w_max,
                    w_scale=w_scale,
                    w_exp=w_exp,
                    v_scale=v_scale)

    def validate(self):
        N_CX_MAX = 1024
        if self.n_compartments > N_CX_MAX:
            raise BuildError("Number of compartments (%d) exceeded max (%d)" %
                             (self.n_compartments, N_CX_MAX))


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


class Profile(object):
    def __eq__(self, obj):
        return isinstance(obj, type(self)) and all(
            self.__dict__[key] == obj.__dict__[key] for key in self.params)

    def __hash__(self):
        return hash(tuple(self.__dict__[key] for key in self.params))


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


class Axons(object):
    """A group of axons, targeting a specific Synapses object.

    Attributes
    ----------
    cx_atoms : list of length ``group.n``
        Atom (weight index) associated with each group compartment.
    cx_to_axon_map : list of length ``group.n``
        Index of the axon in `target` targeted by each group compartment.
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
