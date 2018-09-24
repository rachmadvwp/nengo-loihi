import nengo
import numpy as np
import pytest

import nengo_loihi


def test_cx_model_validate_notempty(Simulator):
    with nengo.Network() as model:
        nengo_loihi.add_params(model)

        a = nengo.Ensemble(10, 1)
        model.config[a].on_chip = False

    with pytest.raises(nengo.exceptions.BuildError):
        with Simulator(model):
            pass


@pytest.mark.parametrize("precompute", [True, False])
def test_probedict_fallbacks(precompute, Simulator):
    with nengo.Network() as net:
        nengo_loihi.add_params(net)
        node_a = nengo.Node(size_in=1)
        with nengo.Network():
            ens_b = nengo.Ensemble(10, 1)
            conn_ab = nengo.Connection(node_a, ens_b)
        ens_c = nengo.Ensemble(5, 1)
        net.config[ens_c].on_chip = False
        conn_bc = nengo.Connection(ens_b, ens_c)
        probe_a = nengo.Probe(node_a)
        probe_c = nengo.Probe(ens_c)

    with Simulator(net, precompute=precompute) as sim:
        sim.run(0.002)

    assert node_a in sim.data
    assert ens_b in sim.data
    assert ens_c in sim.data
    assert probe_a in sim.data
    assert probe_c in sim.data

    # TODO: connections are currently not probeable as they are
    #       replaced in the splitting process
    assert conn_ab  # in sim.data
    assert conn_bc  # in sim.data


@pytest.mark.parametrize('dt, a_on_chip',
                         [(2e-4, True),
                          (3e-4, False),
                          (4e-4, True)])
def test_dt(dt, a_on_chip, Simulator, seed, plt, allclose):
    function = lambda x: x**2

    # dt = 0.001
    # # dt = 0.002
    # a_on_chip = False
    # # a_on_chip = True

    # probe_synapse = nengo.Lowpass(0.01)
    probe_synapse = nengo.Alpha(0.01)

    ens_params = dict(
        intercepts=nengo.dists.Uniform(-0.9, 0.9),
        max_rates=nengo.dists.Uniform(100, 120))

    with nengo.Network(seed=seed) as model:
        nengo_loihi.add_params(model)

        u = nengo.Node(lambda t: -np.sin(2 * np.pi * t))
        u_p = nengo.Probe(u, synapse=probe_synapse)

        a = nengo.Ensemble(100, 1, **ens_params)
        model.config[a].on_chip = a_on_chip
        a_p = nengo.Probe(a, synapse=probe_synapse)

        b = nengo.Ensemble(101, 1, **ens_params)
        b_p = nengo.Probe(b, synapse=probe_synapse)

        nengo.Connection(u, a)
        nengo.Connection(a, b, function=function,
                         solver=nengo.solvers.LstsqL2(weights=True))

    with Simulator(model, dt=dt, precompute=False) as sim:
        sim.run(1.0)

    x = sim.data[u_p]
    y = function(x)
    plt.plot(sim.trange(), x, 'k--')
    plt.plot(sim.trange(), y, 'k--')
    plt.plot(sim.trange(), sim.data[a_p])
    plt.plot(sim.trange(), sim.data[b_p])

    assert allclose(sim.data[a_p], x, rtol=0.1, atol=0.1)
    assert allclose(sim.data[b_p], y, rtol=0.1, atol=0.1)
