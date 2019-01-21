from __future__ import division

import warnings

import numpy as np
from nengo.exceptions import BuildError

from nengo_loihi.discretize import (
    array_to_int,
    BIAS_MAX,
    bias_to_manexp,
    decay_magnitude,
    Q_BITS,
    VTH_MAX,
    vth_to_manexp,
)
from nengo_loihi.axons import AxonGroup
from nengo_loihi.probes import ProbeGroup
from nengo_loihi.synapses import SynapseFmt, SynapseGroup


class CompartmentGroup(object):
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
    axons : AxonGroup
        Axons objects outputting from these compartments.
    synapses : SynapseGroup
        Synapses objects projecting to these compartments.
    probes : ProbeGroup
        Probes recording information from these compartments.
    """
    # threshold at which U/V scaling is allowed
    DECAY_SCALE_TH = 0.5 / 2**12  # half of one decay scaling unit

    def __init__(self, n, label=None):
        self.n = n
        self.label = label

        self.axons = AxonGroup()
        self.synapses = SynapseGroup()
        self.probes = ProbeGroup()

        # parameters specific to compartments/group
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

    def discretize(self):  # noqa C901
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

        w_maxs = [s.max_abs_weight() for s in self.synapses]
        w_max = max(w_maxs) if len(w_maxs) > 0 else 0
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

        # --- discretize constituents
        self.synapses.discretize(w_max, w_scale, w_exp)
        self.probes.discretize(v_scale[0])

    def validate(self):
        N_CX_MAX = 1024
        if self.n > N_CX_MAX:
            raise BuildError("Number of compartments (%d) exceeded max (%d)" %
                             (self.n, N_CX_MAX))

        self.axons.validate()
        self.synapses.validate()
        self.probes.validate()
