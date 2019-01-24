import nengo
from nengo import Ensemble, Connection, Node
from nengo.builder.connection import BuiltConnection
from nengo.dists import get_samples
from nengo.ensemble import Neurons
from nengo.exceptions import BuildError, ValidationError
from nengo.solvers import NoSolver, Solver
import numpy as np

from nengo_loihi import conv
from nengo_loihi.builder.builder import Builder
from nengo_loihi.builder.ensemble import (
    build_decode_neuron_encoders,
    gen_eval_points,
)
from nengo_loihi.axons import CxAxons
from nengo_loihi.compartments import CxGroup
from nengo_loihi.io_objects import ChipReceiveNeurons, CxSpikeInput
from nengo_loihi.probes import CxProbe
from nengo_loihi.synapses import CxSynapses
from nengo_loihi.neurons import loihi_rates


def get_eval_points(model, conn, rng):
    if conn.eval_points is None:
        view = model.params[conn.pre_obj].eval_points.view()
        view.setflags(write=False)
        return view
    else:
        return gen_eval_points(
            conn.pre_obj, conn.eval_points, rng, conn.scale_eval_points)


def get_targets(conn, eval_points):
    if conn.function is None:
        targets = eval_points[:, conn.pre_slice]
    elif isinstance(conn.function, np.ndarray):
        targets = conn.function
    else:
        targets = np.zeros((len(eval_points), conn.size_mid))
        for i, ep in enumerate(eval_points[:, conn.pre_slice]):
            out = conn.function(ep)
            if out is None:
                raise BuildError("Building %s: Connection function returned "
                                 "None. Cannot solve for decoders." % (conn,))
            targets[i] = out

    return targets


def build_decoders(model, conn, rng, transform):
    encoders = model.params[conn.pre_obj].encoders
    gain = model.params[conn.pre_obj].gain
    bias = model.params[conn.pre_obj].bias

    eval_points = get_eval_points(model, conn, rng)
    targets = get_targets(conn, eval_points)

    x = np.dot(eval_points, encoders.T / conn.pre_obj.radius)
    E = None
    if conn.solver.weights:
        E = model.params[conn.post_obj].scaled_encoders.T[conn.post_slice]
        # include transform in solved weights
        targets = multiply(targets, transform.T)

    # wrapped_solver = (model.decoder_cache.wrap_solver(solve_for_decoders)
    #                   if model.seeded[conn] else solve_for_decoders)
    # decoders, solver_info = wrapped_solver(
    decoders, solver_info = solve_for_decoders(
        conn, gain, bias, x, targets, rng=rng, dt=model.dt, E=E)

    weights = (decoders.T if conn.solver.weights else
               multiply(transform, decoders.T))
    return eval_points, weights, solver_info


def solve_for_decoders(conn, gain, bias, x, targets, rng, dt, E=None):
    activities = loihi_rates(conn.pre_obj.neuron_type, x, gain, bias, dt)
    if np.count_nonzero(activities) == 0:
        raise BuildError(
            "Building %s: 'activities' matrix is all zero for %s. "
            "This is because no evaluation points fall in the firing "
            "ranges of any neurons." % (conn, conn.pre_obj))

    decoders, solver_info = conn.solver(activities, targets, rng=rng, E=E)
    return decoders, solver_info


def multiply(x, y):
    if x.ndim <= 2 and y.ndim < 2:
        return x * y
    elif x.ndim < 2 and y.ndim == 2:
        return x.reshape(-1, 1) * y
    elif x.ndim == 2 and y.ndim == 2:
        return np.dot(x, y)
    else:
        raise BuildError("Tensors not supported (x.ndim = %d, y.ndim = %d)"
                         % (x.ndim, y.ndim))


@Builder.register(Solver)
def build_solver(model, solver, conn, rng, transform):
    return build_decoders(model, conn, rng, transform)


@Builder.register(NoSolver)
def build_no_solver(model, solver, conn, rng, transform):
    activities = np.zeros((1, conn.pre_obj.n_neurons))
    targets = np.zeros((1, conn.size_mid))
    E = np.zeros((1, conn.post_obj.n_neurons)) if solver.weights else None
    # No need to invoke the cache for NoSolver
    decoders, solver_info = conn.solver(activities, targets, rng=rng, E=E)
    weights = (decoders.T if conn.solver.weights else
               multiply(transform, decoders.T))
    return None, weights, solver_info


@Builder.register(Connection)  # noqa: C901
def build_connection(model, conn):
    if isinstance(conn.transform, conv.Conv2D):
        # TODO: integrate these into the same function
        conv.build_conv2d_connection(model, conn)
        return

    # Create random number generator
    rng = np.random.RandomState(model.seeds[conn])

    pre_cx = model.objs[conn.pre_obj]['out']
    post_cx = model.objs[conn.post_obj]['in']
    assert isinstance(pre_cx, (CxGroup, CxSpikeInput))
    assert isinstance(post_cx, (CxGroup, CxProbe))

    weights = None
    eval_points = None
    solver_info = None
    neuron_type = None

    # Sample transform if given a distribution
    transform = get_samples(
        conn.transform, conn.size_out, d=conn.size_mid, rng=rng)

    tau_s = 0.0  # `synapse is None` gets mapped to `tau_s = 0.0`
    if isinstance(conn.synapse, nengo.synapses.Lowpass):
        tau_s = conn.synapse.tau
    elif conn.synapse is not None:
        raise NotImplementedError("Cannot handle non-Lowpass synapses")

    needs_decode_neurons = False
    target_encoders = None
    if isinstance(conn.pre_obj, Node):
        assert conn.pre_slice == slice(None)

        if np.array_equal(transform, np.array(1.)):
            # TODO: this identity transform may be avoidable
            transform = np.eye(conn.pre.size_out)
        else:
            assert transform.ndim == 2, "transform shape not handled yet"
            assert transform.shape[1] == conn.pre.size_out

        assert transform.shape[1] == conn.pre.size_out
        if isinstance(conn.pre_obj, ChipReceiveNeurons):
            weights = transform / model.dt
            neuron_type = conn.pre_obj.neuron_type
        else:
            # input is on-off neuron encoded, so double/flip transform
            weights = np.column_stack([transform, -transform])
            target_encoders = 'node_encoders'
    elif (isinstance(conn.pre_obj, Ensemble)
          and isinstance(conn.pre_obj.neuron_type, nengo.Direct)):
        raise NotImplementedError()
    elif isinstance(conn.pre_obj, Ensemble):  # Normal decoded connection
        eval_points, weights, solver_info = model.build(
            conn.solver, conn, rng, transform)

        # the decoder solver assumes a spike height of 1/dt; that isn't the
        # case on loihi, so we need to undo that scaling
        weights = weights / model.dt

        neuron_type = conn.pre_obj.neuron_type

        if not conn.solver.weights:
            needs_decode_neurons = True
    elif isinstance(conn.pre_obj, Neurons):
        assert conn.pre_slice == slice(None)
        assert transform.ndim == 2, "transform shape not handled yet"
        weights = transform / model.dt
        neuron_type = conn.pre_obj.ensemble.neuron_type
    else:
        raise NotImplementedError("Connection from type %r" % (
            type(conn.pre_obj),))

    if neuron_type is not None and hasattr(neuron_type, 'amplitude'):
        weights = weights * neuron_type.amplitude

    mid_cx = pre_cx
    mid_axon_inds = None
    post_tau = tau_s
    if needs_decode_neurons and not isinstance(conn.post_obj, Neurons):
        # --- add decode neurons
        assert weights.ndim == 2
        d, n = weights.shape

        if isinstance(post_cx, CxProbe):
            # use non-spiking decode neurons for voltage probing
            assert post_cx.target is None
            assert conn.post_slice == slice(None)

            # use the same scaling as the ensemble does, to get good
            #  decodes.  Note that this assumes that the decoded value
            #  is in the range -radius to radius, which is usually true.
            weights = weights / conn.pre_obj.radius

            gain = 1
            dec_cx = CxGroup(2 * d, label='%s' % conn, location='core')
            dec_cx.configure_nonspiking(dt=model.dt, vth=model.vth_nonspiking)
            dec_cx.bias[:] = 0
            model.add_group(dec_cx)
            model.objs[conn]['decoded'] = dec_cx

            dec_syn = CxSynapses(n, label="probe_decoders")
            weights2 = gain * np.vstack([weights, -weights]).T

            dec_syn.set_full_weights(weights2)
            dec_cx.add_synapses(dec_syn)
            model.objs[conn]['decoders'] = dec_syn
        else:
            # use spiking decode neurons for on-chip connection
            if isinstance(conn.post_obj, Ensemble):
                # loihi encoders don't include radius, so handle scaling here
                weights = weights / conn.post_obj.radius

            post_d = conn.post_obj.size_in
            post_inds = np.arange(post_d, dtype=np.int32)[conn.post_slice]
            assert weights.shape[0] == len(post_inds) == conn.size_out == d
            mid_axon_inds = model.decode_neurons.get_post_inds(
                post_inds, post_d)

            target_encoders = 'decode_neuron_encoders'
            dec_cx, dec_syn = model.decode_neurons.get_cx(
                weights, cx_label="%s" % conn, syn_label="decoders")

            model.add_group(dec_cx)
            model.objs[conn]['decoded'] = dec_cx
            model.objs[conn]['decoders'] = dec_syn

        # use tau_s for filter into decode neurons, decode_tau for filter out
        dec_cx.configure_filter(tau_s, dt=model.dt)
        post_tau = model.decode_tau

        dec_ax0 = CxAxons(n, label="decoders")
        dec_ax0.target = dec_syn
        pre_cx.add_axons(dec_ax0)
        model.objs[conn]['decode_axons'] = dec_ax0

        if conn.learning_rule_type is not None:
            rule_type = conn.learning_rule_type
            if isinstance(rule_type, nengo.PES):
                if not isinstance(rule_type.pre_synapse,
                                  nengo.synapses.Lowpass):
                    raise ValidationError(
                        "Loihi only supports `Lowpass` pre-synapses for "
                        "learning rules", attr='pre_synapse', obj=rule_type)

                tracing_tau = rule_type.pre_synapse.tau / model.dt

                # Nengo builder scales PES learning rate by `dt / n_neurons`
                n_neurons = (conn.pre_obj.n_neurons
                             if isinstance(conn.pre_obj, Ensemble)
                             else conn.pre_obj.size_in)
                learning_rate = rule_type.learning_rate * model.dt / n_neurons

                # Account for scaling to put integer error in range [-127, 127]
                learning_rate /= model.pes_error_scale

                # Tracing mag set so that the magnitude of the pre trace
                # is independent of the pre tau. `dt` factor accounts for
                # Nengo's `dt` spike scaling. Where is the second `dt` from?
                # Maybe the fact that post decode neurons have `vth = 1/dt`?
                tracing_mag = -np.expm1(-1. / tracing_tau) / model.dt**2

                # learning weight exponent controls the maximum weight
                # magnitude/weight resolution
                wgt_exp = model.pes_wgt_exp

                dec_syn.set_learning(
                    learning_rate=learning_rate,
                    tracing_mag=tracing_mag,
                    tracing_tau=tracing_tau,
                    wgt_exp=wgt_exp,
                )
            else:
                raise NotImplementedError()

        mid_cx = dec_cx

    if isinstance(post_cx, CxProbe):
        assert post_cx.target is None
        assert conn.post_slice == slice(None)
        post_cx.target = mid_cx
        mid_cx.add_probe(post_cx)
    elif isinstance(conn.post_obj, Neurons):
        assert isinstance(post_cx, CxGroup)
        assert conn.post_slice == slice(None)
        if weights is None:
            raise NotImplementedError("Need weights for connection to neurons")
        else:
            assert weights.ndim == 2
            n2, n1 = weights.shape
            assert post_cx.n == n2

            syn = CxSynapses(n1, label="neuron_weights")
            gain = model.params[conn.post_obj.ensemble].gain
            syn.set_full_weights(weights.T * gain)
            post_cx.add_synapses(syn)
            model.objs[conn]['weights'] = syn

        ax = CxAxons(mid_cx.n, label="neuron_weights")
        ax.target = syn
        mid_cx.add_axons(ax)

        post_cx.configure_filter(post_tau, dt=model.dt)

        if conn.learning_rule_type is not None:
            raise NotImplementedError()
    elif isinstance(conn.post_obj, Ensemble) and conn.solver.weights:
        assert isinstance(post_cx, CxGroup)
        assert weights.ndim == 2
        n2, n1 = weights.shape
        assert post_cx.n == n2

        # loihi encoders don't include radius, so handle scaling here
        weights = weights / conn.post_obj.radius

        syn = CxSynapses(n1, label="%s::decoder_weights" % conn)
        syn.set_full_weights(weights.T)
        post_cx.add_synapses(syn)
        model.objs[conn]['weights'] = syn

        ax = CxAxons(n1, label="decoder_weights")
        ax.target = syn
        mid_cx.add_axons(ax)

        post_cx.configure_filter(post_tau, dt=model.dt)

        if conn.learning_rule_type is not None:
            raise NotImplementedError()
    elif isinstance(conn.post_obj, Ensemble):
        assert target_encoders is not None
        if target_encoders not in post_cx.named_synapses:
            build_decode_neuron_encoders(
                model, conn.post_obj, kind=target_encoders)

        mid_ax = CxAxons(mid_cx.n, label="encoders")
        mid_ax.target = post_cx.named_synapses[target_encoders]
        mid_ax.set_axon_map(mid_axon_inds)
        mid_cx.add_axons(mid_ax)
        model.objs[conn]['mid_axons'] = mid_ax

        post_cx.configure_filter(post_tau, dt=model.dt)
    elif isinstance(conn.post_obj, Node):
        raise NotImplementedError()
    else:
        raise NotImplementedError()

    model.params[conn] = BuiltConnection(
        eval_points=eval_points,
        solver_info=solver_info,
        transform=transform,
        weights=weights)
