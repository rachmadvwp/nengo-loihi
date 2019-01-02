from __future__ import division

import collections
import logging
import warnings

from nengo.exceptions import SimulationError
from nengo.utils.compat import iteritems, itervalues, range
import numpy as np

from nengo_loihi.discretize import (
    decay_int,
    LEARN_FRAC,
    learn_overflow_bits,
    overflow_signed,
    scale_pes_errors,
    shift,
    Q_BITS,
    U_BITS,
)
from nengo_loihi.probes import Probe

logger = logging.getLogger(__name__)


class EmulatorInterface(object):
    """Software emulator for Loihi chip behaviour.

    Parameters
    ----------
    model : Model
        Model specification that will be simulated.
    seed : int, optional (Default: None)
        A seed for all stochastic operations done in this simulator.
    """

    strict = False

    def __init__(self, model, seed=None):
        self.closed = True
        model.validate()

        if seed is None:
            seed = np.random.randint(2**31 - 1)
        self.seed = seed
        logger.debug("EmulatorInterface seed: %d", seed)
        self.rng = np.random.RandomState(self.seed)

        self.group_info = GroupInfo(model.groups)
        self.inputs = list(model.inputs)
        logger.debug("EmulatorInterface dtype: %s", self.group_info.dtype)

        self.compartments = CompartmentState(
            self.group_info, strict=self.strict)
        self.synapses = SynapseState(
            self.group_info,
            pes_error_scale=getattr(model, 'pes_error_scale', 1.),
            strict=self.strict,
        )
        self.axons = AxonState(self.group_info)
        self.probes = ProbeState(self.group_info, self.inputs, model.dt)

        self.t = 0
        self._chip2host_sent_steps = 0
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.closed = True

        # remove references to states to free memory (except probes)
        self.group_info = None
        self.inputs = None
        self.compartments = None
        self.synapses = None
        self.axons = None

    def chip2host(self, probes_receivers=None):
        if probes_receivers is None:
            probes_receivers = {}

        increment = 0
        for probe, receiver in probes_receivers.items():
            inc = self.probes.send(probe, self._chip2host_sent_steps, receiver)
            increment = inc if increment == 0 else increment
            assert inc == 0 or increment == inc

        self._chip2host_sent_steps += increment

    def host2chip(self, spikes, errors):
        for spike_input, t, spike_idxs in spikes:
            spike_input.add_spikes(t, spike_idxs)

        self.synapses.update_pes_errors(errors)

    def run_steps(self, steps):
        """Simulate for the given number of ``dt`` steps.

        Parameters
        ----------
        steps : int
            Number of steps to run the simulation for.
        """
        for _ in range(steps):
            self.step()

    def step(self):
        """Advance the simulation by 1 step (``dt`` seconds)."""
        self.t += 1
        self.compartments.advance_input()
        self.synapses.inject_current(
            self.t, self.inputs, self.axons, self.compartments.spiked)
        self.synapses.update_input(self.compartments.input)
        self.synapses.update_traces(self.t, self.rng)
        self.synapses.update_weights(self.t, self.rng)
        self.compartments.update(self.rng)
        self.probes.update(self.t, self.compartments)

    def get_probe_output(self, probe):
        return self.probes[probe]


class GroupInfo(object):
    """Provide information about all the NeuronGroups in the model.

    Attributes
    ----------
    dtype : dtype
        Datatype of the groups. Either ``np.float32`` if the groups are not
        discretized or ``np.int32`` if they are. All groups are the same.
    groups : list of NeuronGroup
        List of all the groups in the model.
    n_compartments : int
        Total number of compartments across all groups.
    slices : dict of {NeuronGroup: slice}
        Maps each group to a slice for that group's compartments with respect
        to all compartments. Used to slice into any array storing data across
        all compartments.
    """

    def __init__(self, groups):
        self.groups = list(groups)
        self.slices = {}

        assert self.dtype in (np.float32, np.int32)

        start_ix = end_ix = 0
        for group in self.groups:
            end_ix += group.n
            self.slices[group] = slice(start_ix, end_ix)
            assert group.vth.dtype == self.dtype
            assert group.bias.dtype == self.dtype
            start_ix = end_ix

        self.n_compartments = end_ix

    @property
    def dtype(self):
        return self.groups[0].vth.dtype


class IterableState(object):
    """Base class for aspects of the emulator state.

    This class takes the name of a NeuronGroup attribute as the ``group_key``,
    and maps these objects to their parent groups and slices.

    Attributes
    ----------
    dtype : dtype
        Datatype of the state elements (given by the GroupInfo datatype).
    group_map : dict of {item: group}
        Maps an item (determined by ``group_key``) to the group it belongs to.
    n_compartments : int
        The total number of neuron compartments (given by GroupInfo).
    slices : dict of {item: slice}
        Maps an item to the ``group_info.slice`` for the group it belongs to.
    strict : bool (Default: True)
        Whether "undesired" chip effects (ex. overflow) raise errors (``True``)
        or whether they only raise warnings (``False``).
    """

    def __init__(self, group_info, group_key, strict=True):
        self.n_compartments = group_info.n_compartments
        self.dtype = group_info.dtype
        self.strict = strict

        if group_key == "compartments":
            self.group_map = {group: group for group in group_info.groups}
            self.slices = {
                group: group_info.slices[group]
                for group in group_info.groups
            }
        else:
            self.group_map = {
                item: group
                for group in group_info.groups
                for item in getattr(group, group_key)
            }
            self.slices = {
                item: group_info.slices[group]
                for group in group_info.groups
                for item in getattr(group, group_key)
            }

    def __contains__(self, item):
        return item in self.slices

    def __getitem__(self, key):
        return self.slices[key]

    def __iter__(self):
        for obj in self.slices:
            yield obj

    def __len__(self):
        return len(self.slices)

    def error(self, msg):
        if self.strict:
            raise SimulationError(msg)
        else:
            warnings.warn(msg)

    def items(self):
        return iteritems(self.slices)


class CompartmentState(IterableState):
    """State representing the Compartments of all groups."""

    MAX_DELAY = 1  # delay not yet implemented

    def __init__(self, group_info, strict=True):
        super(CompartmentState, self).__init__(
            group_info, "compartments", strict=strict)

        # Initialize NumPy arrays to store compartment-related data
        self.input = np.zeros(
            (self.MAX_DELAY, self.n_compartments), dtype=self.dtype)
        self.current = np.zeros(self.n_compartments, dtype=self.dtype)
        self.voltage = np.zeros(self.n_compartments, dtype=self.dtype)
        self.spiked = np.zeros(self.n_compartments, dtype=bool)
        self.spike_count = np.zeros(self.n_compartments, dtype=np.int32)
        self.ref_count = np.zeros(self.n_compartments, dtype=np.int32)

        self.decay_u = np.full(self.n_compartments, np.nan, dtype=self.dtype)
        self.decay_v = np.full(self.n_compartments, np.nan, dtype=self.dtype)
        self.scale_u = np.ones(self.n_compartments, dtype=self.dtype)
        self.scale_v = np.ones(self.n_compartments, dtype=self.dtype)

        self.vth = np.full(self.n_compartments, np.nan, dtype=self.dtype)
        self.vmin = np.full(self.n_compartments, np.nan, dtype=self.dtype)
        self.vmax = np.full(self.n_compartments, np.nan, dtype=self.dtype)

        self.bias = np.full(self.n_compartments, np.nan, dtype=self.dtype)
        self.ref = np.full(self.n_compartments, np.nan, dtype=self.dtype)

        # Fill in arrays with parameters from CompartmentGroups
        for compartment, sl in self.items():
            self.decay_u[sl] = compartment.decayU
            self.decay_v[sl] = compartment.decayV
            if compartment.scaleU:
                self.scale_u[sl] = compartment.decayU
            if compartment.scaleV:
                self.scale_v[sl] = compartment.decayV
            self.vth[sl] = compartment.vth
            self.vmin[sl] = compartment.vmin
            self.vmax[sl] = compartment.vmax
            self.bias[sl] = compartment.bias
            self.ref[sl] = compartment.refractDelay

        assert not np.any(np.isnan(self.decay_u))
        assert not np.any(np.isnan(self.decay_v))
        assert not np.any(np.isnan(self.vth))
        assert not np.any(np.isnan(self.vmin))
        assert not np.any(np.isnan(self.vmax))
        assert not np.any(np.isnan(self.bias))
        assert not np.any(np.isnan(self.ref))

        if self.dtype == np.int32:
            assert (self.scale_u == 1).all()
            assert (self.scale_v == 1).all()
            self._decay_current = (
                lambda x, u: decay_int(x, self.decay_u, offset=1) + u)
            self._decay_voltage = lambda x, u: decay_int(x, self.decay_v) + u

            def overflow(x, bits, name=None):
                _, o = overflow_signed(x, bits=bits, out=x)
                if np.any(o):
                    self.error("Overflow" + (" in %s" % name if name else ""))

        elif self.dtype == np.float32:
            def decay_float(x, u, d, s):
                return (1 - d)*x + s*u

            self._decay_current = lambda x, u: decay_float(
                x, u, d=self.decay_u, s=self.scale_u)
            self._decay_voltage = lambda x, u: decay_float(
                x, u, d=self.decay_v, s=self.scale_v)

            def overflow(x, bits, name=None):
                pass  # do not do overflow in floating point
        else:
            raise ValueError("dtype %r not supported" % self.dtype)

        self._overflow = overflow

        self.noise = NoiseState(group_info)

    def advance_input(self):
        self.input[:-1] = self.input[1:]
        self.input[-1] = 0

    def update(self, rng):
        noise = self.noise.sample(rng)
        q0 = self.input[0, :]
        q0[~self.noise.target_u] += noise[~self.noise.target_u]
        self._overflow(q0, bits=Q_BITS, name="q0")

        self.current[:] = self._decay_current(self.current, q0)
        self._overflow(self.current, bits=U_BITS, name="current")
        u2 = self.current + self.bias
        u2[self.noise.target_u] += noise[self.noise.target_u]
        self._overflow(u2, bits=U_BITS, name="u2")

        self.voltage[:] = self._decay_voltage(self.voltage, u2)
        # We have not been able to create V overflow on the chip, so we do
        # not include it here. See github.com/nengo/nengo-loihi/issues/130
        # self.overflow(self.v, bits=V_BIT, name="V")

        np.clip(self.voltage, self.vmin, self.vmax, out=self.voltage)
        self.voltage[self.ref_count > 0] = 0
        # TODO^: don't zero voltage in case neuron is saving overshoot

        self.spiked[:] = (self.voltage > self.vth)
        self.voltage[self.spiked] = 0
        self.ref_count[self.spiked] = self.ref[self.spiked]
        # decrement ref_count
        np.clip(self.ref_count - 1, 0, None, out=self.ref_count)

        self.spike_count[self.spiked] += 1


class NoiseState(IterableState):
    """State representing the noise parameters for all compartments."""

    def __init__(self, group_info):
        super(NoiseState, self).__init__(group_info, "compartments")
        self.enabled = np.full(self.n_compartments, np.nan, dtype=bool)
        self.exp = np.full(self.n_compartments, np.nan, dtype=self.dtype)
        self.mant_offset = np.full(self.n_compartments, np.nan,
                                   dtype=self.dtype)
        self.target_u = np.full(self.n_compartments, np.nan, dtype=bool)

        # Fill in arrays with parameters from CompartmentGroups
        for compartment, sl in self.items():
            self.enabled[sl] = compartment.enableNoise
            self.exp[sl] = compartment.noiseExp0
            self.mant_offset[sl] = compartment.noiseMantOffset0
            self.target_u[sl] = compartment.noiseAtDendOrVm

        if self.dtype == np.int32:
            if np.any(self.exp < 7):
                warnings.warn("Noise amplitude falls below lower limit")
                self.exp[self.exp < 7] = 7

            self.mult = np.where(self.enabled, 2**(self.exp - 7), 0)
            self.mant_offset *= 64

            def uniform(rng, n=self.n_compartments):
                return rng.randint(-128, 128, size=n, dtype=np.int32)

        elif self.dtype == np.float32:
            self.mult = np.where(self.enabled, 10.**self.exp, 0)

            def uniform(rng, n=self.n_compartments):
                return rng.uniform(-1, 1, size=n).astype(np.float32)
        else:
            raise ValueError("dtype %r not supported" % self.dtype)

        assert not np.any(np.isnan(self.enabled))
        assert not np.any(np.isnan(self.exp))
        assert not np.any(np.isnan(self.mant_offset))
        assert not np.any(np.isnan(self.target_u))
        assert not np.any(np.isnan(self.mult))

        self._uniform = uniform

    def sample(self, rng):
        x = self._uniform(rng)
        return (x + self.mant_offset) * self.mult


class SynapseState(IterableState):
    """State representing all synapses.

    Attributes
    ----------
    pes_error_scale : float
        Scaling for the errors of PES learning rules.
    pes_errors : {Synapses: ndarray(n_neurons / 2)}
        Maps synapses to PES learning rule errors for those synapses.
    spikes_in : {Synapses: list}
        Maps synapses to a queue of input spikes targeting those synapses.
    traces : {Synapses: ndarray(Synapses.n_axons)}
        Maps synapses to trace values for each of their axons.
    trace_spikes : {Synapses: set}
        Maps synapses to a queue of input spikes waiting to be added to those
        synapses traces.
    """

    def __init__(self, group_info,  # noqa: C901
                 pes_error_scale=1.,
                 strict=True
             ):
        super(SynapseState, self).__init__(
            group_info, "synapses", strict=strict)

        self.pes_error_scale = pes_error_scale

        self.spikes_in = {}
        self.traces = {}
        self.trace_spikes = {}
        self.pes_errors = {}
        for synapses in self.slices:
            n = synapses.n_axons
            self.spikes_in[synapses] = []

            if synapses.learning:
                self.traces[synapses] = np.zeros(n, dtype=self.dtype)
                self.trace_spikes[synapses] = set()
                self.pes_errors[synapses] = np.zeros(
                    self.group_map[synapses].n // 2, dtype=self.dtype)
                # ^ Currently, PES learning only happens on Nodes, where we
                # have pairs of on/off neurons. Therefore, the number of error
                # dimensions is half the number of neurons.

        if self.dtype == np.int32:
            def stochastic_round(x, dtype=self.dtype, rng=None,
                                 clip=None, name="values"):
                x_sign = np.sign(x).astype(dtype)
                x_frac, x_int = np.modf(np.abs(x))
                p = rng.rand(*x.shape)
                y = x_int.astype(dtype) + (x_frac > p)
                if clip is not None:
                    q = y > clip
                    if np.any(q):
                        warnings.warn("Clipping %s" % name)
                    y[q] = clip
                return x_sign * y

            def trace_round(x, rng=None):
                return stochastic_round(x, rng=rng, clip=127,
                                        name="synapse trace")

            def weight_update(synapses, delta_ws, rng=None):
                synapse_fmt = synapses.synapse_fmt
                wgt_exp = synapse_fmt.realWgtExp
                shift_bits = synapse_fmt.shift_bits
                overflow = learn_overflow_bits(n_factors=2)
                for w, delta_w in zip(synapses.weights, delta_ws):
                    product = shift(
                        delta_w * synapses._lr_int,
                        LEARN_FRAC + synapses._lr_exp - overflow)
                    learn_w = shift(w, LEARN_FRAC - wgt_exp) + product
                    learn_w[:] = stochastic_round(
                        learn_w * 2**(-LEARN_FRAC - shift_bits),
                        clip=2**(8 - shift_bits) - 1,
                        rng=rng,
                        name="learning weights")
                    w[:] = np.left_shift(learn_w, wgt_exp + shift_bits)

        elif self.dtype == np.float32:
            def trace_round(x, rng=None):
                return x  # no rounding

            def weight_update(synapses, delta_ws, rng=None):
                for w, delta_w in zip(synapses.weights, delta_ws):
                    w += synapses.learning_rate * delta_w
        else:
            raise ValueError("dtype %r not supported" % self.dtype)

        self._trace_round = trace_round
        self._weight_update = weight_update

    def inject_current(self, t, spike_inputs, all_axons, spiked):
        # --- clear spikes going in to each synapse
        for spike_queue in itervalues(self.spikes_in):
            spike_queue.clear()

        # --- inputs pass spikes to synapses
        if t >= 2:  # input spikes take one time-step to arrive
            for spike_input in spike_inputs:
                cx_idxs = spike_input.spike_idxs(t - 1)
                for axons in spike_input.axons:
                    spikes = axons.map_cx_spikes(cx_idxs)
                    self.spikes_in[axons.target].extend(spikes)

        # --- axons pass spikes to synapses
        for axons, a_idx in all_axons.items():
            cx_idxs = spiked[a_idx].nonzero()[0]
            spikes = axons.map_cx_spikes(cx_idxs)
            self.spikes_in[axons.target].extend(spikes)

    def update_input(self, input):
        for synapses, s_slice in self.items():
            qb = input[:, s_slice]

            for spike in self.spikes_in[synapses]:
                cx_base = synapses.axon_cx_base(spike.axon_id)
                if cx_base is None:
                    continue

                weights, indices = synapses.axon_weights_indices(
                    spike.axon_id, atom=spike.atom)
                qb[0, cx_base + indices] += weights

    def update_pes_errors(self, errors):
        # TODO: these are sent every timestep, but learning only happens every
        # `tepoch * 2**learn_k` timesteps (see Synapses). Need to average.
        for pes_errors in self.pes_errors.values():
            pes_errors[:] = 0

        for synapses, _, e in errors:
            pes_errors = self.pes_errors[synapses]
            assert pes_errors.shape == e.shape
            pes_errors += scale_pes_errors(e, scale=self.pes_error_scale)

    def update_weights(self, t, rng):
        for synapses, pes_error in iteritems(self.pes_errors):
            if t % synapses.learn_epoch == 0:
                trace = self.traces[synapses]
                e = np.hstack([-pes_error, pes_error])
                delta_w = np.outer(trace, e)
                self._weight_update(synapses, delta_w, rng=rng)

    def update_traces(self, t, rng):
        for synapses in self.traces:
            trace_spikes = self.trace_spikes.get(synapses, None)
            if trace_spikes is not None:
                for spike in self.spikes_in[synapses]:
                    if spike.axon_id in trace_spikes:
                        self.error("Synaptic trace spikes lost")
                    trace_spikes.add(spike.axon_id)

            trace = self.traces.get(synapses, None)
            if trace is not None and t % synapses.train_epoch == 0:
                tau = synapses.tracing_tau
                decay = np.exp(-synapses.train_epoch / tau)
                trace1 = decay * trace
                trace1[list(trace_spikes)] += synapses.tracing_mag
                trace[:] = self._trace_round(trace1, rng=rng)
                trace_spikes.clear()


class AxonState(IterableState):
    """State representing all (output) Axons."""

    def __init__(self, group_info):
        super(AxonState, self).__init__(group_info, "axons")


class ProbeState(object):
    """State representing all probes.

    Attributes
    ----------
    dt : float
        Time constant of the Emulator.
    filters : {nengo_loihi.Probe: function}
        Maps Probes to the filtering function for that probe.
    filter_pos : {nengo_loihi.Probe: int}
        Maps Probes to the position of their filter in the data.
    group_probes : {nengo_loihi.Probe: slice}
        Maps Probes to the GroupInfo slice for the group they are probing.
    input_probes : {nengo_loihi.Probe: SpikeInput}
        Maps Probes to the SpikeInput that they are probing.
    """

    def __init__(self, group_info, inputs, dt):
        self.dt = dt
        self.input_probes = {}
        for spike_input in inputs:
            for probe in spike_input.probes:
                assert probe.key == 'spiked'
                self.input_probes[probe] = spike_input
        self.other_probes = {}
        for group in group_info.groups:
            for probe in group.probes.probes:
                self.other_probes[probe] = group_info.slices[group]

        self.filters = {}
        self.filter_pos = {}
        for probe, spike_input in iteritems(self.input_probes):
            if probe.synapse is not None:
                self.filters[probe] = probe.synapse.make_step(
                    shape=spike_input.spikes[0][probe.slice].shape[0],
                    dt=self.dt,
                    rng=None,
                    dtype=spike_input.spikes.dtype,
                )
                self.filter_pos[probe] = 0

        for probe, sl in iteritems(self.other_probes):
            if probe.synapse is not None:
                size = (sl.stop - sl.start if probe.weights is None
                        else probe.weights.shape[1])
                self.filters[probe] = probe.synapse.make_step(
                    shape_in=(size,),
                    shape_out=(size,),
                    dt=self.dt,
                    rng=None,
                    dtype=np.float32,
                )
                self.filter_pos[probe] = 0

        self.outputs = collections.defaultdict(list)

    def __getitem__(self, probe):
        assert isinstance(probe, Probe)
        out = np.asarray(self.outputs[probe], dtype=np.float32)
        out = out if probe.weights is None else np.dot(out, probe.weights)
        return self._filter(probe, out) if probe in self.filters else out

    def _filter(self, probe, data):
        dt = self.dt
        i = self.filter_pos[probe]
        step = self.filters[probe]
        filt_data = np.zeros_like(data)
        for k, x in enumerate(data):
            filt_data[k] = step((i + k) * dt, x)
        self.filter_pos[probe] = i + k
        return filt_data

    def send(self, probe, already_sent, receiver):
        """Send probed data to the receiver node.

        Returns
        -------
        steps : int
            The number of steps sent to the receiver.
        """
        x = self.outputs[probe][already_sent:]

        if len(x) > 0:
            if probe.weights is not None:
                x = np.dot(x, probe.weights)
            for j, xx in enumerate(x):
                receiver.receive(self.dt * (already_sent + j + 2), xx)
        return len(x)

    def update(self, t, compartments):
        for probe, spike_input in iteritems(self.input_probes):
            assert probe.key == 'spiked'
            output = spike_input.spikes[t][probe.slice].copy()
            self.outputs[probe].append(output)

        for probe, out_idx in iteritems(self.other_probes):
            p_slice = probe.slice
            assert hasattr(compartments, probe.key)
            output = getattr(compartments, probe.key)[out_idx][p_slice].copy()
            self.outputs[probe].append(output)
