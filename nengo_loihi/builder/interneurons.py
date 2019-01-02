import numpy as np

import nengo

from nengo_loihi.compartments import CompartmentGroup
from nengo_loihi.synapses import Synapses


class Interneurons(object):
    def __init__(self, n, dt=0.001):
        from nengo_loihi.neurons import LoihiSpikingRectifiedLinear
        self.neuron_type = LoihiSpikingRectifiedLinear()

        self.dt = dt

        # firing rate of inter neurons
        self._inter_rate = None

        # number of inter neurons
        self.n = n

    @property
    def inter_rate(self):
        return (1. / (self.dt * self.n) if self._inter_rate is None else
                self._inter_rate)

    @inter_rate.setter
    def inter_rate(self, inter_rate):
        self._inter_rate = inter_rate

    def inter_scale(self):
        """Scaling applied to input from interneurons.

        Such that if all ``n`` interneurons are firing at
        their max rate ``inter_rate``, then the total output when
        averaged over time will be 1.
        """
        return 1. / (self.dt * self.inter_rate * self.n)

    def get_post_encoders(self, encoders):
        """Take interneuron raw output and encode to population"""
        raise NotImplementedError()

    def get_post_inds(self, inds, d):
        """Index into post encoders"""
        raise NotImplementedError()

    def get_compartments(self, weights, comp_label=None, syn_label=None):
        raise NotImplementedError()


class OnOffInterneurons(Interneurons):
    def __init__(self, dt=0.001):
        super(OnOffInterneurons, self).__init__(1, dt=dt)

    def get_ensemble(self, dim):
        from nengo_loihi.neurons import NIF
        assert self.n == 1
        ens = nengo.Ensemble(
            2 * dim, dim,
            neuron_type=NIF(tau_ref=0.0),
            encoders=np.vstack([np.eye(dim), -np.eye(dim)]),
            max_rates=np.ones(dim * 2) * self.inter_rate,
            intercepts=np.ones(dim * 2) * -1,
            add_to_container=False)
        return ens

    def get_post_encoders(self, encoders):
        encoders = encoders * self.inter_scale()
        return np.vstack([encoders.T, -encoders.T])


class EqualDecoderInterneurons(Interneurons):
    def get_post_inds(self, inds, d):
        return np.concatenate([inds, inds + d] * self.n)


class NoisyInterneurons(EqualDecoderInterneurons):
    def __init__(self, *args, **kwargs):
        super(NoisyInterneurons, self).__init__(*args, **kwargs)

        # noise exponent for inter neurons
        self.inter_noise_exp = -2

        self.gain = None
        self.bias = None

    def fix_parameters(self):
        self.gain = 0.5 * self.dt * self.inter_rate
        self.bias = self.gain

    def inter_scale(self):
        self.fix_parameters()
        return 1. / (self.dt * self.inter_rate * self.n)

    def get_post_encoders(self, encoders):
        encoders = encoders * self.inter_scale()
        return np.vstack([encoders.T, -encoders.T])

    def get_compartments(self, weights, comp_label=None, syn_label=None):
        self.fix_parameters()
        d, n = weights.shape
        cx = CompartmentGroup(2 * d * self.n, label=comp_label)
        cx.configure_relu(dt=self.dt)
        cx.bias[:] = self.bias * np.ones(d * 2 * self.n)
        if self.inter_noise_exp > -30:
            cx.enableNoise[:] = 1
            cx.noiseExp0 = self.inter_noise_exp
            cx.noiseAtDendOrVm = 1

        # TODO: can eliminate the weight copies (tiling) by using input indices
        # to target input axons
        syn = Synapses(n, label=syn_label)
        weights2 = self.gain * np.vstack([weights, -weights] * self.n).T
        syn.set_full_weights(weights2)
        cx.add_synapses(syn)

        return cx, syn


class PresetInterneurons(EqualDecoderInterneurons):
    def fix_parameters(self):
        raise NotImplementedError()

    def inter_scale(self):
        self.fix_parameters()
        return self.out_gain

    def get_post_encoders(self, encoders):
        encoders = encoders * self.inter_scale()
        return np.vstack([encoders.T, -encoders.T])

    def get_compartments(self, weights, comp_label=None, syn_label=None):
        self.fix_parameters()
        d, n = weights.shape
        cx = CompartmentGroup(self.n * 2 * d, label=comp_label)
        cx.configure_relu(dt=self.dt)
        cx.bias[:] = self.bias.repeat(d)

        syn = Synapses(n, label=syn_label)
        weights2 = []
        for ga, gb in self.gain.reshape(self.n, 2):
            weights2.extend([ga*weights.T, -gb*weights.T])
        weights2 = np.hstack(weights2)
        syn.set_full_weights(weights2)
        cx.add_synapses(syn)

        return cx, syn


class Preset5Interneurons(PresetInterneurons):
    def __init__(self, **kwargs):
        super(Preset5Interneurons, self).__init__(5, **kwargs)

    def fix_parameters(self):
        assert self.n == 5
        intercepts = np.linspace(-0.8, 0.8, self.n)
        max_rates = np.linspace(70, 160, self.n)[::-1]
        gain, bias = self.neuron_type.gain_bias(max_rates, intercepts)

        target_point = 0.85
        gain, bias = self.neuron_type.gain_bias(max_rates, intercepts)
        target_rates = self.neuron_type.rates(target_point, gain, bias)
        target_rate = target_rates.sum()
        # TODO: why does this 1.1 factor help??
        self.out_gain = 1.1 * target_point / (self.dt * target_rate)

        self.gain = gain.repeat(2) * self.dt
        self.bias = bias.repeat(2) * self.dt


class Preset10Interneurons(PresetInterneurons):
    def __init__(self, **kwargs):
        super(Preset10Interneurons, self).__init__(10, **kwargs)

    def fix_parameters(self):
        # Parameters determined by hyperopt
        assert self.n == 10
        intercepts = np.linspace(-0.612, 0.448, self.n)
        max_rates = np.linspace(98, 148, self.n)[::-1]
        gain, bias = self.neuron_type.gain_bias(max_rates, intercepts)

        target_point = 0.67
        gain, bias = self.neuron_type.gain_bias(max_rates, intercepts)
        target_rates = self.neuron_type.rates(target_point, gain, bias)
        target_rate = target_rates.sum()
        self.out_gain = target_point / (self.dt * target_rate)

        self.gain = gain.repeat(2) * self.dt
        self.bias = bias.repeat(2) * self.dt
