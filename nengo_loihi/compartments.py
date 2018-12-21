from __future__ import division

import warnings

import numpy as np
from nengo.exceptions import BuildError
from nengo.utils.compat import is_iterable

from nengo_loihi.discretize import (
    BIAS_MAX,
    bias_to_manexp,
    decay_magnitude,
    learn_overflow_bits,
    tracing_mag_int_frac,
    Q_BITS,
    VTH_MAX,
    vth_to_manexp,
)
from nengo_loihi.synapses import SynapseFmt


class CxGroup(object):
    """Class holding Loihi objects that can be placed on the chip or Lakemont.

    Typically an ensemble or node, can be a special decoding ensemble. Once
    implemented, SNIPS might use this as well.

    Before ``discretize`` has been called, most parameters in this class are
    floating-point values. Calling ``discretize`` converts them to integer
    values inplace, for use on Loihi.

    Attributes
    ----------
    n : int
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
    synapses : list of CxSynapse
        CxSynapse objects projecting to these compartments.
    named_synapses : dict
        Dictionary mapping names to CxSynapse objects.
    axons : list of CxAxon
        CxAxon objects outputting from these compartments.
    named_axons : dict
        Dictionary mapping names to CxAxon objects.
    probes : list of CxProbe
        CxProbes recording information from these compartments.
    location : {"core", "cpu"}
        Whether these compartments are on a Loihi core
        or handled by the Loihi x86 processor (CPU).
    """
    # threshold at which U/V scaling is allowed
    DECAY_SCALE_TH = 0.5 / 2**12  # half of one decay scaling unit

    def __init__(self, n, label=None, location='core'):
        self.n = n
        self.label = label

        self.decayU = np.ones(n, dtype=np.float32)  # default to no filter
        self.decayV = np.zeros(n, dtype=np.float32)  # default to integration
        self.tau_s = None
        self.scaleU = True
        self.scaleV = False

        self.refractDelay = np.zeros(n, dtype=np.int32)
        self.vth = np.zeros(n, dtype=np.float32)
        self.bias = np.zeros(n, dtype=np.float32)
        self.enableNoise = np.zeros(n, dtype=bool)

        # parameters common to core
        self.vmin = 0
        self.vmax = np.inf
        self.noiseMantOffset0 = 0
        self.noiseExp0 = 0
        self.noiseAtDendOrVm = 0

        self.synapses = []
        self.named_synapses = {}
        self.axons = []
        self.named_axons = {}
        self.probes = []

        assert location in ('core', 'cpu')
        self.location = location

    def __str__(self):
        return "%s(%s)" % (
            type(self).__name__, self.label if self.label else '')

    def add_synapses(self, synapses, name=None):
        """Add a CxSynapses object to ensemble."""

        assert synapses.group is None
        synapses.group = self
        self.synapses.append(synapses)
        if name is not None:
            assert name not in self.named_synapses
            self.named_synapses[name] = synapses

    def add_axons(self, axons, name=None):
        """Add a CxAxons object to ensemble."""

        assert axons.group is None
        axons.group = self
        self.axons.append(axons)
        if name is not None:
            assert name not in self.named_axons
            self.named_axons[name] = axons

    def add_probe(self, probe):
        """Add a CxProbe object to ensemble."""
        if probe.target is None:
            probe.target = self
        assert probe.target is self
        self.probes.append(probe)

    def configure_default_filter(self, tau_s, dt=0.001):
        """Set the default Lowpass synaptic input filter for Cx.

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
        """Set Lowpass synaptic input filter for Cx to time constant tau_s.

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

    def discretize(self):  # noqa C901
        def discretize(target, value):
            assert target.dtype == np.float32
            # new = np.round(target * scale).astype(np.int32)
            new = np.round(value).astype(np.int32)
            target.dtype = np.int32
            target[:] = new

        # --- discretize decayU and decayV
        # subtract 1 from decayU here because it gets added back by the chip
        decayU = self.decayU * (2**12 - 1) - 1
        discretize(self.decayU, np.clip(decayU, 0, 2**12 - 1))
        discretize(self.decayV, self.decayV * (2**12 - 1))

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
        # To avoid overflow, we can either lower vth_max or lower wgtExp_max.
        # Lowering vth_max is more robust, but has the downside that it may
        # force smaller wgtExp on connections than necessary, potentially
        # leading to lost weight bits (see SynapseFmt.discretize_weights).
        # Lowering wgtExp_max can let us keep vth_max higher, but overflow
        # is still be possible on connections with many small inputs (uncommon)
        vth_max = VTH_MAX
        wgtExp_max = 0

        w_maxs = [s.max_abs_weight() for s in self.synapses]
        w_max = max(w_maxs) if len(w_maxs) > 0 else 0
        b_max = np.abs(self.bias).max()
        wgtExp = 0

        if w_max > 1e-8:
            w_scale = (255. / w_max)
            s_scale = 1. / (u_infactor * v_infactor)

            for wgtExp in range(wgtExp_max, -8, -1):
                v_scale = s_scale * w_scale * SynapseFmt.get_scale(wgtExp)
                b_scale = v_scale * v_infactor
                vth = np.round(self.vth * v_scale)
                bias = np.round(self.bias * b_scale)
                if (vth <= vth_max).all() and (np.abs(bias) <= BIAS_MAX).all():
                    break
            else:
                raise BuildError("Could not find appropriate wgtExp")
        elif b_max > 1e-8:
            b_scale = BIAS_MAX / b_max
            while b_scale*b_max > 1:
                v_scale = b_scale / v_infactor
                w_scale = b_scale * u_infactor / SynapseFmt.get_scale(wgtExp)
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
                       / SynapseFmt.get_scale(wgtExp))

        vth_man, vth_exp = vth_to_manexp(vth)
        discretize(self.vth, vth_man * 2**vth_exp)

        bias_man, bias_exp = bias_to_manexp(bias)
        discretize(self.bias, bias_man * 2**bias_exp)

        for i, synapse in enumerate(self.synapses):
            if synapse.tracing:
                wgtExp2 = synapse.learning_wgt_exp
                dWgtExp = wgtExp - wgtExp2
            elif w_maxs[i] > 1e-16:
                dWgtExp = int(np.floor(np.log2(w_max / w_maxs[i])))
                assert dWgtExp >= 0
                wgtExp2 = max(wgtExp - dWgtExp, -6)
            else:
                wgtExp2 = -6
                dWgtExp = wgtExp - wgtExp2
            synapse.format(wgtExp=wgtExp2)
            for w, idxs in zip(synapse.weights, synapse.indices):
                ws = w_scale[idxs] if is_iterable(w_scale) else w_scale
                discretize(w, synapse.synapse_fmt.discretize_weights(
                    w * ws * 2**dWgtExp))

            # discretize learning
            if synapse.tracing:
                synapse.tracing_tau = int(np.round(synapse.tracing_tau))

                if is_iterable(w_scale):
                    assert np.all(w_scale == w_scale[0])
                w_scale_i = w_scale[0] if is_iterable(w_scale) else w_scale

                # incorporate weight scale and difference in weight exponents
                # to learning rate, since these affect speed at which we learn
                ws = w_scale_i * 2**dWgtExp
                synapse.learning_rate *= ws

                # Loihi down-scales learning factors based on the number of
                # overflow bits. Increasing learning rate maintains true rate.
                synapse.learning_rate *= 2**learn_overflow_bits(2)

                # TODO: Currently, Loihi learning rate fixed at 2**-7.
                # We should explore adjusting it for better performance.
                lscale = 2**-7 / synapse.learning_rate
                synapse.learning_rate *= lscale
                synapse.tracing_mag /= lscale

                # discretize learning rate into mantissa and exponent
                lr_exp = int(np.floor(np.log2(synapse.learning_rate)))
                lr_int = int(np.round(synapse.learning_rate * 2**(-lr_exp)))
                synapse.learning_rate = lr_int * 2**lr_exp
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

        for p in self.probes:
            if p.key == 'v' and p.weights is not None:
                p.weights /= v_scale[0]

    def validate(self):
        if self.location == 'cpu':
            return  # none of these checks currently apply to Lakemont

        N_CX_MAX = 1024
        if self.n > N_CX_MAX:
            raise BuildError("Number of compartments (%d) exceeded max (%d)" %
                             (self.n, N_CX_MAX))

        IN_AXONS_MAX = 4096
        n_axons = sum(s.n_axons for s in self.synapses)
        if n_axons > IN_AXONS_MAX:
            raise BuildError("Input axons (%d) exceeded max (%d)" % (
                n_axons, IN_AXONS_MAX))

        MAX_SYNAPSE_BITS = 16384*64
        synapse_bits = sum(s.bits() for s in self.synapses)
        if synapse_bits > MAX_SYNAPSE_BITS:
            raise BuildError("Total synapse bits (%d) exceeded max (%d)" % (
                synapse_bits, MAX_SYNAPSE_BITS))

        OUT_AXONS_MAX = 4096
        n_axons = sum(a.axon_slots() for a in self.axons)
        if n_axons > OUT_AXONS_MAX:
            raise BuildError("Output axons (%d) exceeded max (%d)" % (
                n_axons, OUT_AXONS_MAX))

        for synapses in self.synapses:
            synapses.validate()

        for axons in self.axons:
            axons.validate()

        for probe in self.probes:
            probe.validate()
