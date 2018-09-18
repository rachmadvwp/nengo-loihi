import nengo

from nengo_loihi.utils import seed_network


def test_seed_network(seed):
    with nengo.Network(seed=seed) as model:
        e0 = nengo.Ensemble(1, 1)
        e1 = nengo.Ensemble(1, 1, seed=2)
        e2 = nengo.Ensemble(1, 1)
        nengo.Connection(e0, e1)
        nengo.Connection(e0, e2, seed=3)

        with nengo.Network():
            n = nengo.Node(0)
            e = nengo.Ensemble(1, 1)
            nengo.Node(1)
            nengo.Connection(n, e)
            nengo.Probe(e)

        with nengo.Network(seed=8):
            nengo.Ensemble(8, 1, seed=3)
            nengo.Node(1)

    seeds, seeded = seed_network(model)
    sim = nengo.Simulator(model)
    for obj in model.all_objects:
        assert seeds[obj] == sim.model.seeds[obj]
        assert seeded[obj] == sim.model.seeded[obj]
