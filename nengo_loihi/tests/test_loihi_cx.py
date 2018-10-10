from nengo.exceptions import SimulationError
import numpy as np
import pytest

from nengo_loihi.loihi_cx import (
    CxAxons, CxGroup, CxModel, CxProbe, CxSimulator, CxSpikeInput, CxSynapses)


def test_simulator_noise(request, plt, seed):
    target = request.config.getoption("--target")

    model = CxModel()
    group = CxGroup(10)
    group.configure_relu()

    group.bias[:] = np.linspace(0, 0.01, group.n)

    group.enableNoise[:] = 1
    group.noiseExp0 = -2
    group.noiseMantOffset0 = 0
    group.noiseAtDendOrVm = 1

    probe = CxProbe(target=group, key='v')
    group.add_probe(probe)
    model.add_group(group)

    model.discretize()

    if target == 'loihi':
        sim = model.get_loihi(seed=seed)
        sim.run_steps(1000)
        y = np.column_stack([
            p.timeSeries.data for p in sim.board.probe_map[probe]])
    else:
        sim = model.get_simulator(seed=seed)
        sim.run_steps(1000)
        y = sim.probe_outputs[probe]

    plt.plot(y)
    plt.yticks(())


def test_strict_mode():
    # Tests should be run in strict mode
    assert CxSimulator.strict

    with pytest.raises(SimulationError):
        CxSimulator.error("Error in emulator")
    CxSimulator.strict = False
    with pytest.warns(UserWarning):
        CxSimulator.error("Error in emulator")

    # Strict mode is a global setting so we set it back to True
    # for subsequent test runs.
    CxSimulator.strict = True


def test_population_input(request, allclose):
    target = request.config.getoption("--target")
    dt = 0.001

    n_inputs = 3
    n_axons = 1
    n_cx = 2

    model = CxModel()

    input = CxSpikeInput(n_inputs, dt)
    input.add_spikes(1, [1, 0, 0])
    input.add_spikes(2, [0, 0, 0])
    input.add_spikes(3, [0, 1, 0])
    input.add_spikes(4, [0, 0, 0])
    input.add_spikes(5, [0, 0, 1])
    input.add_spikes(6, [0, 0, 0])
    model.add_input(input)

    input_axons = CxAxons(n_axons)
    axon_map = np.zeros(n_inputs, dtype=int)
    atoms = np.arange(n_inputs)
    input_axons.set_axon_map(axon_map, atoms)
    input.add_axons(input_axons)

    group = CxGroup(n_cx)
    group.configure_lif(tau_rc=0., tau_ref=0., dt=dt)
    group.configure_filter(0, dt=dt)
    model.add_group(group)

    synapses = CxSynapses(n_axons)
    weights = 0.1 * np.array([[[1, 2], [2, 3], [4, 5]]], dtype=float)
    indices = np.array([[[0, 1], [0, 1], [0, 1]]], dtype=int)
    axon_to_weight_map = np.zeros(n_axons, dtype=int)
    cx_bases = np.zeros(n_axons, dtype=int)
    synapses.set_population_weights(
        weights, indices, axon_to_weight_map, cx_bases, pop_type=32)
    group.add_synapses(synapses)
    input_axons.target = synapses

    probe = CxProbe(target=group, key='v')
    group.add_probe(probe)

    model.discretize()

    if target == 'loihi':
        with model.get_loihi() as sim:
            sim.run_steps(6)
            y = np.column_stack([
                p.timeSeries.data for p in sim.board.probe_map[probe]])
    else:
        sim = model.get_simulator()
        sim.run_steps(6)
        y = np.array(sim.probe_outputs[probe])

    vth = group.vth[0]
    assert (group.vth == vth).all()
    z = y / vth
    assert allclose(z[[0, 2, 4]], weights[0], atol=4e-2, rtol=0)
