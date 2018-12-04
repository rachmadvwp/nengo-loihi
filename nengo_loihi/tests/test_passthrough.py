import pytest
import nengo
import numpy as np

from nengo_loihi import splitter
import nengo_loihi


def test_passthrough_placement():
    with nengo.Network() as model:
        stim = nengo.Node(0)
        a = nengo.Node(None, size_in=1)   # should be off-chip
        b = nengo.Ensemble(10, 1)
        c = nengo.Node(None, size_in=1)   # should be removed
        d = nengo.Node(None, size_in=1)   # should be removed
        e = nengo.Node(None, size_in=1)   # should be removed
        f = nengo.Ensemble(10, 1)
        g = nengo.Node(None, size_in=1)   # should be off-chip
        nengo.Connection(stim, a)
        nengo.Connection(a, b)
        nengo.Connection(b, c)
        nengo.Connection(c, d)
        nengo.Connection(d, e)
        nengo.Connection(e, f)
        nengo.Connection(f, g)
        nengo.Probe(g)

    nengo_loihi.add_params(model)
    networks = splitter.split(model, precompute=False,
                              remove_passthrough=True,
                              max_rate=1000, inter_tau=0.005)
    chip = networks.chip
    host = networks.host

    assert a in host.nodes
    assert a not in chip.nodes
    assert c not in host.nodes
    assert c not in chip.nodes
    assert d not in host.nodes
    assert d not in chip.nodes
    assert e not in host.nodes
    assert e not in chip.nodes
    assert g in host.nodes
    assert g not in chip.nodes


@pytest.mark.parametrize("D1", [1, 3])
@pytest.mark.parametrize("D2", [1, 3])
@pytest.mark.parametrize("D3", [1, 3])
def test_transform_merging(D1, D2, D3):
    with nengo.Network() as model:
        a = nengo.Ensemble(10, D1)
        b = nengo.Node(None, size_in=D2)
        c = nengo.Ensemble(10, D3)

        t1 = np.random.uniform(-1, 1, (D2, D1))
        t2 = np.random.uniform(-1, 1, (D3, D2))

        nengo.Connection(a, b, transform=t1)
        nengo.Connection(b, c, transform=t2)

    nengo_loihi.add_params(model)
    networks = splitter.split(model, precompute=False,
                              remove_passthrough=True,
                              max_rate=1000, inter_tau=0.005)
    chip = networks.chip
    host = networks.host

    assert len(chip.connections) == 1
    conn = chip.connections[0]
    assert np.allclose(conn.transform, np.dot(t2, t1))


@pytest.mark.parametrize("n_ensembles", [1, 3])
@pytest.mark.parametrize("ens_dimensions", [1, 3])
def test_identity_array(n_ensembles, ens_dimensions):
    with nengo.Network() as model:
        a = nengo.networks.EnsembleArray(10, n_ensembles, ens_dimensions)
        b = nengo.networks.EnsembleArray(10, n_ensembles, ens_dimensions)
        nengo.Connection(a.output, b.input)

    nengo_loihi.add_params(model)
    networks = splitter.split(model, precompute=False,
                              remove_passthrough=True,
                              max_rate=1000, inter_tau=0.005)

    # ignore the a.input -> a.ensemble connections
    connections = [c for c in networks.chip.connections
                   if not (isinstance(c.pre_obj, splitter.ChipReceiveNode)
                           and c.post_obj in a.ensembles)]

    assert len(connections) == n_ensembles
    pre = set()
    post = set()
    for c in connections:
        assert c.pre in a.all_ensembles or c.pre_obj is a.input
        assert c.post in b.all_ensembles
        assert np.allclose(c.transform, np.eye(ens_dimensions))
        pre.add(c.pre)
        post.add(c.post)
    assert len(pre) == n_ensembles
    assert len(post) == n_ensembles


@pytest.mark.parametrize("n_ensembles", [1, 3])
@pytest.mark.parametrize("ens_dimensions", [1, 3])
def test_full_array(n_ensembles, ens_dimensions):
    with nengo.Network() as model:
        a = nengo.networks.EnsembleArray(10, n_ensembles, ens_dimensions)
        b = nengo.networks.EnsembleArray(10, n_ensembles, ens_dimensions)
        D = n_ensembles * ens_dimensions
        nengo.Connection(a.output, b.input, transform=np.ones((D, D)))

    nengo_loihi.add_params(model)
    networks = splitter.split(model, precompute=False,
                              remove_passthrough=True,
                              max_rate=1000, inter_tau=0.005)

    # ignore the a.input -> a.ensemble connections
    connections = [c for c in networks.chip.connections
                   if not (isinstance(c.pre_obj, splitter.ChipReceiveNode)
                           and c.post_obj in a.ensembles)]

    assert len(connections) == n_ensembles ** 2
    pairs = set()
    for c in connections:
        assert c.pre in a.all_ensembles
        assert c.post in b.all_ensembles
        assert np.allclose(c.transform, np.ones((ens_dimensions,
                                                 ens_dimensions)))
        pairs.add((c.pre, c.post))
    assert len(pairs) == n_ensembles ** 2


def test_synapse_merging():
    with nengo.Network() as model:
        a = nengo.networks.EnsembleArray(10, n_ensembles=2)
        b = nengo.Node(None, size_in=2)
        c = nengo.networks.EnsembleArray(10, n_ensembles=2)
        nengo.Connection(a.output[0], b[0], synapse=None)
        nengo.Connection(a.output[1], b[1], synapse=0.1)
        nengo.Connection(b[0], c.input[0], synapse=None)
        nengo.Connection(b[0], c.input[1], synapse=0.2)
        nengo.Connection(b[1], c.input[0], synapse=None)
        nengo.Connection(b[1], c.input[1], synapse=0.2)

    nengo_loihi.add_params(model)
    networks = splitter.split(model, precompute=False,
                              remove_passthrough=True,
                              max_rate=1000, inter_tau=0.005)

    # ignore the a.input -> a.ensemble connections
    connections = [c for c in networks.chip.connections
                   if not (isinstance(c.pre_obj, splitter.ChipReceiveNode)
                           and c.post_obj in a.ensembles)]

    assert len(connections) == 4
    desired_filters = {
        ('0', '0'): None,
        ('0', '1'): nengo.synapses.Lowpass(0.2),
        ('1', '0'): nengo.synapses.Lowpass(0.1),
        ('1', '1'): nengo.synapses.Lowpass(0.1).combine(
                        nengo.synapses.Lowpass(0.2)),
        }
    for c in connections:
        assert desired_filters[(c.pre.label, c.post.label)] == c.synapse
