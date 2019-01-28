import warnings

import nengo
from nengo import Ensemble
from nengo.builder.ensemble import BuiltEnsemble, gen_eval_points
from nengo.dists import Distribution, get_samples
from nengo.exceptions import BuildError
import nengo.utils.numpy as npext
import numpy as np

from nengo_loihi.builder import Builder
from nengo_loihi.segment import LoihiSegment


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

    segment = LoihiSegment(ens.n_neurons, label='%s' % ens)
    segment.compartments.bias[:] = bias
    model.build(ens.neuron_type, ens.neurons, segment)

    # set default filter just in case no other filter gets set
    segment.compartments.configure_default_filter(
        model.decode_tau, dt=model.dt)

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

    model.add_segment(segment)

    model.objs[ens]['in'] = segment
    model.objs[ens]['out'] = segment
    model.objs[ens.neurons]['in'] = segment
    model.objs[ens.neurons]['out'] = segment
    model.params[ens] = BuiltEnsemble(
        eval_points=eval_points,
        encoders=encoders,
        intercepts=intercepts,
        max_rates=max_rates,
        scaled_encoders=scaled_encoders,
        gain=gain,
        bias=bias)


@Builder.register(nengo.neurons.NeuronType)
def build_neurons(model, neurontype, neurons, segment):
    # If we haven't registered a builder for a specific type, then it cannot
    # be simulated on Loihi.
    raise BuildError(
        "The neuron type %r cannot be simulated on Loihi. Please either "
        "switch to a supported neuron type like LIF or "
        "SpikingRectifiedLinear, or explicitly mark ensembles using this "
        "neuron type as off-chip with\n"
        "  net.config[ensembles].on_chip = False")


@Builder.register(nengo.LIF)
def build_lif(model, lif, neurons, segment):
    segment.compartments.configure_lif(
        tau_rc=lif.tau_rc,
        tau_ref=lif.tau_ref,
        dt=model.dt)


@Builder.register(nengo.SpikingRectifiedLinear)
def build_relu(model, relu, neurons, segment):
    segment.compartments.configure_relu(
        vth=1./model.dt,  # so input == 1 -> neuron fires 1/dt steps -> 1 Hz
        dt=model.dt)
