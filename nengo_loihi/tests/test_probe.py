import pytest
import nengo
import numpy as np


def test_spike_units(Simulator, seed):
    with nengo.Network(seed=seed) as model:
        a = nengo.Ensemble(100, 1)
        p = nengo.Probe(a.neurons)
    with Simulator(model) as sim:
        sim.run(0.1)

    values = np.unique(sim.data[p])
    assert values[0] == 0
    assert values[1] == int(1.0 / sim.dt)
    assert len(values) == 2


@pytest.mark.parametrize('dim', [1, 3])
def test_voltage_decode(allclose, Simulator, seed, plt, dim):
    with nengo.Network(seed=seed) as model:
        stim = nengo.Node(
            lambda t: [np.sin(2 * np.pi * t) / np.sqrt(dim)] * dim)
        p_stim = nengo.Probe(stim, synapse=0.01)

        a = nengo.Ensemble(100 * 3, dim,
                           intercepts=nengo.dists.Uniform(-.95, .95))
        nengo.Connection(stim, a)

        p_a = nengo.Probe(a, synapse=0.01)

    with Simulator(model, precompute=True) as sim:
        sim.run(1.)

    plt.plot(sim.trange(), sim.data[p_a])
    plt.plot(sim.trange(), sim.data[p_stim])

    assert allclose(sim.data[p_stim], sim.data[p_a], atol=0.3)


def test_repeated_probes(Simulator):
    with nengo.Network() as net:
        ens = nengo.Ensemble(1024, 1)
        nengo.Probe(ens.neurons)

    for _ in range(5):
        with Simulator(net) as sim:
            sim.run(0.1)


@pytest.mark.parametrize('precompute', [True, False])
@pytest.mark.parametrize('probe_target', ['input', 'voltage'])
# @pytest.mark.parametrize('probe_target', ['input'])
# @pytest.mark.parametrize('probe_target', ['voltage'])
def test_neuron_probes(precompute, probe_target, Simulator, seed, plt,
                       allclose):
    with nengo.Network(seed=seed) as model:
        stim = nengo.Node(lambda t: [1.0*np.sin(t * 2 * np.pi)])

        a = nengo.Ensemble(1, 1,
                           neuron_type=nengo.LIF(min_voltage=-1),
                           encoders=nengo.dists.Choice([[1]]),
                           max_rates=nengo.dists.Choice([100]),
                           intercepts=nengo.dists.Choice([0.]))
        nengo.Connection(stim, a, synapse=None)

        p_stim = nengo.Probe(stim, synapse=0.005)
        p_neurons = nengo.Probe(a.neurons, probe_target)

        probe_synapse = nengo.Alpha(0.01)
        p_stim_f = nengo.Probe(
            stim, synapse=probe_synapse.combine(nengo.Lowpass(0.005)))
        p_neurons_f = nengo.Probe(a.neurons, probe_target,
                                  synapse=probe_synapse)

    with Simulator(model, precompute=precompute) as sim:
        sim.run(1.0)

    scale = float(sim.data[p_neurons].max())
    t = sim.trange()
    x = sim.data[p_stim]
    xf = sim.data[p_stim_f]
    y = sim.data[p_neurons] / scale
    yf = sim.data[p_neurons_f] / scale
    plt.plot(t, x, label='stim')
    plt.plot(t, xf, label='stim filt')
    plt.plot(t, y, label='loihi')
    plt.plot(t, yf, label='loihi filt')
    plt.legend()

    if probe_target == 'input':
        # shape of current input should roughly match stimulus
        assert allclose(y, x, atol=0.25, rtol=0)
        assert allclose(yf, xf, atol=0.05, rtol=0)
    elif probe_target == 'voltage':
        # check for voltage fluctuations (spiking) when stimulus is positive,
        # and negative voltage when stimulus is most negative
        spos = (t > 0.1) & (t < 0.4)
        assert allclose(yf[spos], 0.5, atol=0.1, rtol=0.1)
        assert y[spos].std() > 0.25

        sneg = (t > 0.7) & (t < 0.9)
        assert np.all(y[sneg] < 0)
