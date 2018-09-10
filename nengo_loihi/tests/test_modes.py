import nengo
import numpy as np

def test_identical_outputs(Simulator):
    D = 1
    N = 100
    with nengo.Network(seed=1) as net:
        stim = nengo.Node([0]*D)
        a = nengo.Ensemble(n_neurons=N, dimensions=D)
        nengo.Connection(stim, a)
        b = nengo.Ensemble(n_neurons=N, dimensions=D)
        nengo.Connection(a, b)

        p = nengo.Probe(b)

    with Simulator(net, target='sim', precompute=False) as sim:
        sim.run(0.1)
    with Simulator(net, target='sim', precompute=True) as sim_pre:
        sim_pre.run(0.1)
    with Simulator(net, target='loihi', precompute=False) as loihi:
        loihi.run(0.1)
    with Simulator(net, target='loihi', precompute=True) as loihi_pre:
        loihi_pre.run(0.1)

    s = slice(15,25)
    print(sim.data[p][s,0])
    print(sim_pre.data[p][s,0])
    print(loihi.data[p][s,0])
    print(loihi_pre.data[p][s,0])

    assert np.allclose(sim.data[p], sim_pre.data[p], atol=1e-6)
    assert np.allclose(sim.data[p], loihi.data[p], atol=1e-6)
    assert np.allclose(sim.data[p], loihi_pre.data[p], atol=1e-6)
