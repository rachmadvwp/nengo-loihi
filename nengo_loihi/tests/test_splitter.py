import nengo
import numpy as np
import pytest

import nengo_loihi


@pytest.mark.parametrize("pre_dims", [1, 3])
@pytest.mark.parametrize("post_dims", [1, 3])
@pytest.mark.parametrize("learn", [True, False])
@pytest.mark.parametrize("use_solver", [True, False])
def test_manual_decoders(
        seed, Simulator, pre_dims, post_dims, learn, use_solver):

    with nengo.Network(seed=seed) as model:
        pre = nengo.Ensemble(50, dimensions=pre_dims,
                             gain=np.ones(50),
                             bias=np.ones(50) * 5)
        post = nengo.Node(None, size_in=post_dims)

        learning_rule_type = nengo.PES() if learn else None
        weights = np.zeros((post_dims, 50))
        if use_solver:
            conn = nengo.Connection(pre, post,
                                    function=lambda x: np.zeros(post_dims),
                                    learning_rule_type=learning_rule_type,
                                    solver=nengo.solvers.NoSolver(weights.T))
        else:
            conn = nengo.Connection(pre.neurons, post,
                                    learning_rule_type=learning_rule_type,
                                    transform=weights)

        if learn:
            error = nengo.Node(np.zeros(post_dims))
            nengo.Connection(error, conn.learning_rule)

        pre_probe = nengo.Probe(pre.neurons, synapse=None)
        post_probe = nengo.Probe(post, synapse=None)

    with Simulator(model, precompute=False) as sim:
        sim.run(0.1)

    # Ensure pre population has a lot of activity
    assert np.mean(sim.data[pre_probe]) > 100
    # But that post has no activity due to the zero weights
    assert np.all(sim.data[post_probe] == 0)


@pytest.mark.parametrize("pre_onchip, post_onchip", [
    (True, True), (True, False), (False, True)
])
def test_n2n_transform_solver(
        allclose, plt, rng, Simulator, pre_onchip, post_onchip):
    with nengo.Network() as net:
        nengo_loihi.add_params(net)
        n_neurons = 10

        pre = nengo.Ensemble(n_neurons, 1)
        net.config[pre].on_chip = pre_onchip

        weights = rng.uniform(-0.001, 0.001, size=(n_neurons, n_neurons))

        # Neuron to neuron connection with NoSolver
        post_solver = nengo.Ensemble(n_neurons, 1, seed=10)
        net.config[post_solver].on_chip = post_onchip
        nengo.Connection(pre, post_solver.neurons,
                         function=lambda x: np.zeros(n_neurons),
                         solver=nengo.solvers.NoSolver(weights.T))

        # Neuron to neuron connection with transform
        post_transform = nengo.Ensemble(n_neurons, 1, seed=10)
        net.config[post_transform].on_chip = post_onchip
        nengo.Connection(pre.neurons, post_transform.neurons, transform=weights)

        probe_solver = nengo.Probe(post_solver, synapse=0.03)
        probe_transform = nengo.Probe(post_transform, synapse=0.03)

    with Simulator(net) as sim:
        sim.run(0.1)

    plt.plot(sim.trange(), sim.data[probe_solver] - sim.data[probe_transform])
    plt.ylabel("Difference between NoSolver and transform")
    plt.xlabel("Time (s)")

    # Ensure post_solver and post_transform have the same params
    assert np.all(sim.data[post_transform].gain == sim.data[post_solver].gain)
    assert np.all(sim.data[post_transform].bias == sim.data[post_solver].bias)

    assert allclose(sim.data[probe_solver], sim.data[probe_transform])
