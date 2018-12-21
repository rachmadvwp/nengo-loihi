import nengo
from nengo import Ensemble, Connection, Node, Probe
from nengo.connection import LearningRule
from nengo.ensemble import Neurons
from nengo.exceptions import BuildError
import numpy as np

from nengo_loihi.builder.builder import Builder
from nengo_loihi.probes import CxProbe


def conn_probe(model, probe):
    # Connection probes create a connection from the target, and probe
    # the resulting signal (used when you want to probe the default
    # output of an object, which may not have a predefined signal)

    synapse = 0  # Removed internal filtering

    # get any extra arguments if this probe was created to send data
    #  to an off-chip Node via the splitter

    kwargs = model.chip2host_params.get(probe, None)
    if kwargs is not None:
        # this probe is for sending data to a Node

        # determine the dimensionality
        input_dim = probe.target.size_out
        func = kwargs['function']
        if func is not None:
            if callable(func):
                input_dim = np.asarray(
                    func(np.zeros(input_dim, dtype=np.float64))).size
            else:
                input_dim = len(func[0])
        transform = kwargs['transform']
        transform = np.asarray(transform, dtype=np.float64)
        if transform.ndim <= 1:
            output_dim = input_dim
        elif transform.ndim == 2:
            assert transform.shape[1] == input_dim
            output_dim = transform.shape[0]
        else:
            raise NotImplementedError()

        target = nengo.Node(size_in=output_dim, add_to_container=False)

        conn = Connection(probe.target, target, synapse=synapse,
                          solver=probe.solver, add_to_container=False,
                          **kwargs
                          )
        model.probe_conns[probe] = conn
    else:
        conn = Connection(probe.target, probe, synapse=synapse,
                          solver=probe.solver, add_to_container=False,
                          )
        target = probe

    # Set connection's seed to probe's (which isn't used elsewhere)
    model.seeded[conn] = model.seeded[probe]
    model.seeds[conn] = model.seeds[probe]

    d = conn.size_out
    if isinstance(probe.target, Ensemble):
        # probed values are scaled by the target ensemble's radius
        scale = probe.target.radius
        w = np.diag(scale * np.ones(d))
        weights = np.vstack([w, -w])
    else:
        raise NotImplementedError(
            "Nodes cannot be onchip, connections not yet probeable")

    cx_probe = CxProbe(key='v', weights=weights, synapse=probe.synapse)
    model.objs[target]['in'] = cx_probe
    model.objs[target]['out'] = cx_probe

    # add an extra entry for simulator.run_steps to read data out
    model.objs[probe]['out'] = cx_probe

    # Build the connection
    model.build(conn)


def signal_probe(model, key, probe):
    kwargs = model.chip2host_params.get(probe, None)
    weights = None
    if kwargs is not None:
        if kwargs['function'] is not None:
            raise BuildError("Functions not supported for signal probe")
        weights = kwargs['transform'].T / model.dt

    if isinstance(probe.target, nengo.ensemble.Neurons):
        if probe.attr == 'output':
            if weights is None:
                # spike probes should give values of 1.0/dt on spike events
                weights = 1.0 / model.dt

            if hasattr(probe.target.ensemble.neuron_type, 'amplitude'):
                weights = weights * probe.target.ensemble.neuron_type.amplitude

    # Signal probes directly probe a target signal
    target = model.objs[probe.obj]['out']

    cx_probe = CxProbe(
        target=target, key=key, slice=probe.slice,
        synapse=probe.synapse, weights=weights)
    target.add_probe(cx_probe)
    model.objs[probe]['in'] = target
    model.objs[probe]['out'] = cx_probe


probemap = {
    Ensemble: {'decoded_output': None,
               'input': 'q'},
    Neurons: {'output': 's',
              'spikes': 's',
              'voltage': 'v',
              'input': 'u'},
    Node: {'output': None},
    Connection: {'output': 'weighted',
                 'input': 'in'},
    LearningRule: {},  # make LR signals probeable, but no mapping required
}


@Builder.register(Probe)
def build_probe(model, probe):
    # find the right parent class in `objtypes`, using `isinstance`
    for nengotype, probeables in probemap.items():
        if isinstance(probe.obj, nengotype):
            break
    else:
        raise BuildError(
            "Type %r is not probeable" % type(probe.obj).__name__)

    key = probeables[probe.attr] if probe.attr in probeables else probe.attr
    if key is None:
        conn_probe(model, probe)
    else:
        signal_probe(model, key, probe)

    model.probes.append(probe)

    # Simulator will fill this list with probe data during simulation
    model.params[probe] = []
