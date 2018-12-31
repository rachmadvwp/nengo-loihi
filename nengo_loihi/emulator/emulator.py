from __future__ import division

# import collections
import logging
import warnings

import numpy as np
# import nengo
from nengo.exceptions import SimulationError
# from nengo.utils.compat import is_integer, is_iterable


from nengo_loihi.discretize import (
    decay_int,
    LEARN_FRAC,
    learn_overflow_bits,
    overflow_signed,
    scale_pes_errors,
    Q_BITS,
    U_BITS,
)
from nengo_loihi.probes import Probe
from nengo_loihi.utils import shift


logger = logging.getLogger(__name__)


class Emulator(object):
    """Software emulator for Loihi chip.

    Parameters
    ----------
    model : Model
        Model specification that will be simulated.
    seed : int, optional (Default: None)
        A seed for all stochastic operations done in this simulator.
    """

    strict = False

    def __init__(self, model, seed=None):
        self.closed = False

        self.build(model, seed=seed)

        self._chip2host_sent_steps = 0
        self._probe_filters = {}
        self._probe_filter_pos = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @classmethod
    def error(cls, msg):
        if cls.strict:
            raise SimulationError(msg)
        else:
            warnings.warn(msg)

    def build(self, model, seed=None):  # noqa: C901
        """Set up NumPy arrays to emulate chip memory and I/O."""
        model.validate()

        if seed is None:
            seed = np.random.randint(2**31 - 1)

        logger.debug("Emulator seed: %d", seed)
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.t = 0

        self.model = model
        self.inputs = list(self.model.inputs)
        self.groups = sorted(self.model.groups,
                             key=lambda g: g.location == 'cpu')
        self.probe_outputs = {}
        for obj in self.inputs + self.groups:
            for probe in obj.probes:
                self.probe_outputs[probe] = []

        self.n_cx = sum(group.n for group in self.groups)
        self.group_cxs = {}
        cx_slice = None
        i0, i1 = 0, 0
        for group in self.groups:
            if group.location == 'cpu' and cx_slice is None:
                cx_slice = slice(0, i0)

            i1 = i0 + group.n
            self.group_cxs[group] = slice(i0, i1)
            i0 = i1

        self.cx_slice = slice(0, i0) if cx_slice is None else cx_slice
        self.cpu_slice = slice(self.cx_slice.stop, i1)

        # --- allocate group memory
        group_dtype = self.groups[0].vth.dtype
        assert group_dtype in (np.float32, np.int32)
        for group in self.groups:
            assert group.vth.dtype == group_dtype
            assert group.bias.dtype == group_dtype

        logger.debug("Emulator dtype: %s", group_dtype)

        MAX_DELAY = 1  # don't do delay yet
        self.q = np.zeros((MAX_DELAY, self.n_cx), dtype=group_dtype)
        self.u = np.zeros(self.n_cx, dtype=group_dtype)
        self.v = np.zeros(self.n_cx, dtype=group_dtype)
        self.s = np.zeros(self.n_cx, dtype=bool)  # spiked
        self.c = np.zeros(self.n_cx, dtype=np.int32)  # spike counter
        self.w = np.zeros(self.n_cx, dtype=np.int32)  # ref period counter

        # --- allocate group parameters
        self.decayU = np.hstack([group.decayU for group in self.groups])
        self.decayV = np.hstack([group.decayV for group in self.groups])
        self.scaleU = np.hstack([
            group.decayU if group.scaleU else np.ones_like(group.decayU)
            for group in self.groups])
        self.scaleV = np.hstack([
            group.decayV if group.scaleV else np.ones_like(group.decayV)
            for group in self.groups])

        if group_dtype == np.int32:
            assert (self.scaleU == 1).all()
            assert (self.scaleV == 1).all()
            self.decayU_fn = (
                lambda x, u: decay_int(x, self.decayU, offset=1) + u)
            self.decayV_fn = lambda x, u: decay_int(x, self.decayV) + u

            def overflow(x, bits, name=None):
                _, o = overflow_signed(x, bits=bits, out=x)
                if np.any(o):
                    self.error("Overflow" + (" in %s" % name if name else ""))
        elif group_dtype == np.float32:
            def decay_float(x, u, d, s):
                return (1 - d)*x + s*u

            self.decayU_fn = lambda x, u: decay_float(
                x, u, d=self.decayU, s=self.scaleU)
            self.decayV_fn = lambda x, u: decay_float(
                x, u, d=self.decayV, s=self.scaleV)

            def overflow(x, bits, name=None):
                pass  # do not do overflow in floating point

        self.overflow = overflow

        ones = lambda n: np.ones(n, dtype=group_dtype)
        self.vth = np.hstack([group.vth for group in self.groups])
        self.vmin = np.hstack([
            group.vmin*ones(group.n) for group in self.groups])
        self.vmax = np.hstack([
            group.vmax*ones(group.n) for group in self.groups])

        self.bias = np.hstack([group.bias for group in self.groups])
        self.ref = np.hstack([group.refractDelay for group in self.groups])

        # --- allocate synapse memory
        self.axons_in = {synapses: [] for group in self.groups
                         for synapses in group.synapses}

        learning_synapses = [
            synapses for group in self.groups
            for synapses in group.synapses if synapses.tracing]
        self.z = {synapses: np.zeros(synapses.n_axons, dtype=group_dtype)
                  for synapses in learning_synapses}  # synapse traces
        self.z_spikes = {synapses: set() for synapses in learning_synapses}
        self.pes_errors = {synapses: np.zeros(group.n//2, dtype=group_dtype)
                           for synapses in learning_synapses}
        # ^ Currently, PES learning only happens on Nodes, where we have
        # pairs of on/off neurons. Therefore, the number of error dimensions
        # is half the number of neurons.
        self.pes_error_scale = getattr(model, 'pes_error_scale', 1.)

        if group_dtype == np.int32:
            def stochastic_round(x, dtype=group_dtype, rng=self.rng,
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

            def trace_round(x, dtype=group_dtype, rng=self.rng):
                return stochastic_round(x, dtype=dtype, rng=rng,
                                        clip=127, name="synapse trace")

            def weight_update(synapses, delta_ws):
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
                        clip=2**(8 - shift_bits) - 1, name="learning weights")
                    w[:] = np.left_shift(learn_w, wgt_exp + shift_bits)

        elif group_dtype == np.float32:
            def trace_round(x, dtype=group_dtype):
                return x  # no rounding

            def weight_update(synapses, delta_ws):
                for w, delta_w in zip(synapses.weights, delta_ws):
                    w[:] += delta_w

        self.trace_round = trace_round
        self.weight_update = weight_update

        # --- noise
        enableNoise = np.hstack([
            group.enableNoise*ones(group.n) for group in self.groups])
        noiseExp0 = np.hstack([
            group.noiseExp0*ones(group.n) for group in self.groups])
        noiseMantOffset0 = np.hstack([
            group.noiseMantOffset0*ones(group.n) for group in self.groups])
        noiseTarget = np.hstack([
            group.noiseAtDendOrVm*ones(group.n) for group in self.groups])
        if group_dtype == np.int32:
            if np.any(noiseExp0 < 7):
                warnings.warn("Noise amplitude falls below lower limit")
            noiseExp0[noiseExp0 < 7] = 7
            noiseMult = np.where(enableNoise, 2**(noiseExp0 - 7), 0)

            def noiseGen(n=self.n_cx, rng=self.rng):
                x = rng.randint(-128, 128, size=n)
                return (x + 64*noiseMantOffset0) * noiseMult
        elif group_dtype == np.float32:
            noiseMult = np.where(enableNoise, 10.**noiseExp0, 0)

            def noiseGen(n=self.n_cx, rng=self.rng):
                x = rng.uniform(-1, 1, size=n)
                return (x + noiseMantOffset0) * noiseMult

        self.noiseGen = noiseGen
        self.noiseTarget = noiseTarget

    def clear(self):
        """Clear all signals set in `build` (to free up memory)"""
        self.q = None
        self.u = None
        self.v = None
        self.s = None
        self.c = None
        self.w = None

        self.vth = None
        self.vmin = None
        self.vmax = None

        self.bias = None
        self.ref = None
        self.a_in = None
        self.z = None

        self.noiseGen = None
        self.noiseTarget = None

    def close(self):
        self.closed = True
        self.clear()

    def chip2host(self, probes_receivers=None):
        if probes_receivers is None:
            probes_receivers = {}

        increment = None
        for probe, receiver in probes_receivers.items():
            # extract the probe data from the simulator
            x = self.probe_outputs[probe][self._chip2host_sent_steps:]
            if len(x) > 0:
                if increment is None:
                    increment = len(x)
                else:
                    assert increment == len(x)
                if probe.weights is not None:
                    x = np.dot(x, probe.weights)
                for j in range(len(x)):
                    receiver.receive(
                        self.model.dt * (self._chip2host_sent_steps + j + 2),
                        x[j]
                    )
        if increment is not None:
            self._chip2host_sent_steps += increment

    def host2chip(self, spikes, errors):
        for cx_spike_input, t, spike_idxs in spikes:
            cx_spike_input.add_spikes(t, spike_idxs)

        # TODO: these are sent every timestep, but learning only happens every
        # `tepoch * 2**learn_k` timesteps (see Synapses). Need to average.
        for pes_errors in self.pes_errors.values():
            pes_errors[:] = 0

        for synapses, t, e in errors:
            pes_errors = self.pes_errors[synapses]
            assert pes_errors.shape == e.shape
            pes_errors += scale_pes_errors(e, scale=self.pes_error_scale)

    def step(self):  # noqa: C901
        """Advance the simulation by 1 step (``dt`` seconds)."""
        self.t += 1

        # --- connections
        self.q[:-1] = self.q[1:]  # advance delays
        self.q[-1] = 0

        # --- clear spikes going in to each synapse
        for axons_in_spikes in self.axons_in.values():
            axons_in_spikes.clear()

        # --- inputs pass spikes to synapses
        if self.t >= 2:  # input spikes take one time-step to arrive
            for input in self.inputs:
                cx_idxs = input.spike_idxs(self.t - 1)
                for axons in input.axons:
                    spikes = axons.map_cx_spikes(cx_idxs)
                    self.axons_in[axons.target].extend(spikes)

        # --- axons pass spikes to synapses
        for group in self.groups:
            cx_idxs = self.s[self.group_cxs[group]].nonzero()[0]
            for axons in group.axons:
                spikes = axons.map_cx_spikes(cx_idxs)
                self.axons_in[axons.target].extend(spikes)

        # --- synapse spikes use weights to modify compartment input
        for group in self.groups:
            for synapses in group.synapses:
                b_slice = self.group_cxs[synapses.group]
                qb = self.q[:, b_slice]
                # delays = np.zeros(qb.shape[1], dtype=np.int32)

                for spike in self.axons_in[synapses]:
                    # qb[0, indices[spike.axon_id]] += weights[spike.axon_id]
                    cx_base = synapses.axon_cx_base(spike.axon_id)
                    if cx_base is None:
                        continue

                    weights, indices = synapses.axon_weights_indices(
                        spike.axon_id, atom=spike.atom)
                    qb[0, cx_base + indices] += weights

                # --- learning trace
                z_spikes = self.z_spikes.get(synapses, None)
                if z_spikes is not None:
                    for spike in self.axons_in[synapses]:
                        if spike.axon_id in z_spikes:
                            self.error("Synaptic trace spikes lost")
                        z_spikes.add(spike.axon_id)

                z = self.z.get(synapses, None)
                if z is not None and self.t % synapses.train_epoch == 0:
                    tau = synapses.tracing_tau
                    decay = np.exp(-synapses.train_epoch / tau)
                    zi = decay*z
                    zi[list(z_spikes)] += synapses.tracing_mag
                    z[:] = self.trace_round(zi)
                    z_spikes.clear()

                # --- learning update
                pes_e = self.pes_errors.get(synapses, None)
                if pes_e is not None and self.t % synapses.learn_epoch == 0:
                    assert z is not None
                    x = np.hstack([-pes_e, pes_e])
                    delta_w = np.outer(z, x)
                    self.weight_update(synapses, delta_w)

        # --- updates
        q0 = self.q[0, :]

        noise = self.noiseGen()
        q0[self.noiseTarget == 0] += noise[self.noiseTarget == 0]
        self.overflow(q0, bits=Q_BITS, name="q0")

        self.u[:] = self.decayU_fn(self.u[:], q0)
        self.overflow(self.u, bits=U_BITS, name="U")
        u2 = self.u + self.bias
        u2[self.noiseTarget == 1] += noise[self.noiseTarget == 1]
        self.overflow(u2, bits=U_BITS, name="u2")

        self.v[:] = self.decayV_fn(self.v, u2)
        # We have not been able to create V overflow on the chip, so we do
        # not include it here. See github.com/nengo/nengo-loihi/issues/130
        # self.overflow(self.v, bits=V_BIT, name="V")

        np.clip(self.v, self.vmin, self.vmax, out=self.v)
        self.v[self.w > 0] = 0
        # TODO^: don't zero voltage in case neuron is saving overshoot

        self.s[:] = (self.v > self.vth)

        cx = self.cx_slice
        cpu = self.cpu_slice
        self.v[cx][self.s[cx]] = 0
        self.v[cpu][self.s[cpu]] -= self.vth[cpu][self.s[cpu]]

        self.w[self.s] = self.ref[self.s]
        np.clip(self.w - 1, 0, None, out=self.w)  # decrement w

        self.c[self.s] += 1

        # --- probes
        for input in self.inputs:
            for probe in input.probes:
                assert probe.key == 's'
                p_slice = probe.slice
                x = input.spikes[self.t][p_slice].copy()
                self.probe_outputs[probe].append(x)

        for group in self.groups:
            for probe in group.probes:
                x_slice = self.group_cxs[probe.target]
                p_slice = probe.slice
                assert hasattr(self, probe.key), "probe key not found"
                x = getattr(self, probe.key)[x_slice][p_slice].copy()
                self.probe_outputs[probe].append(x)

    def run_steps(self, steps):
        """Simulate for the given number of ``dt`` steps.

        Parameters
        ----------
        steps : int
            Number of steps to run the simulation for.
        """
        for _ in range(steps):
            self.step()

    def _filter_probe(self, probe, data):
        dt = self.model.dt
        i = self._probe_filter_pos.get(probe, 0)
        if i == 0:
            shape = data[0].shape
            synapse = probe.synapse
            rng = None
            step = (synapse.make_step(shape, shape, dt, rng, dtype=data.dtype)
                    if synapse is not None else None)
            self._probe_filters[probe] = step
        else:
            step = self._probe_filters[probe]

        if step is None:
            self._probe_filter_pos[probe] = i + len(data)
            return data
        else:
            filt_data = np.zeros_like(data)
            for k, x in enumerate(data):
                filt_data[k] = step((i + k) * dt, x)

            self._probe_filter_pos[probe] = i + k
            return filt_data

    def get_probe_output(self, probe):
        assert isinstance(probe, Probe)
        x = np.asarray(self.probe_outputs[probe], dtype=np.float32)
        x = x if probe.weights is None else np.dot(x, probe.weights)
        return self._filter_probe(probe, x)
