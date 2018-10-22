from nengo.exceptions import SimulationError
import numpy as np
import pytest

from nengo_loihi.loihi_api import BIAS_MAX, VTH_MAX
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
        with model.get_loihi(seed=seed) as sim:
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

    try:
        with pytest.raises(SimulationError):
            CxSimulator.error("Error in emulator")
        CxSimulator.strict = False
        with pytest.warns(UserWarning):
            CxSimulator.error("Error in emulator")
    finally:
        # Strict mode is a global setting so we set it back to True
        # for subsequent test runs.
        CxSimulator.strict = True


@pytest.mark.skipif(pytest.config.getoption("--target") != "loihi",
                    reason="need Loihi as comparison")
@pytest.mark.parametrize('n_axons', [200, 1000])
def test_uv_overflow(n_axons, Simulator, plt, allclose):
    # TODO: Currently this is not testing the V overflow, since it is higher
    #  and I haven't been able to figure out a way to make it overflow.
    nt = 15

    model = CxModel()

    # n_axons controls number of input spikes and thus amount of overflow
    input_spikes = np.ones((nt, n_axons), dtype=bool)
    input = CxSpikeInput(input_spikes)

    group = CxGroup(1)
    group.configure_relu()
    group.configure_filter(0.1)
    group.vmin = -2**22

    synapses = CxSynapses(n_axons)
    synapses.set_full_weights(np.ones((n_axons, 1)))

    axons = CxAxons(n_axons)
    axons.target = synapses
    input.add_axons(axons)
    group.add_synapses(synapses)

    probe_u = CxProbe(target=group, key='u')
    group.add_probe(probe_u)
    probe_v = CxProbe(target=group, key='v')
    group.add_probe(probe_v)
    probe_s = CxProbe(target=group, key='s')
    group.add_probe(probe_s)

    model.add_input(input)
    model.add_group(group)
    model.discretize()

    group.vth[:] = VTH_MAX  # must set after `discretize`

    assert CxSimulator.strict  # Tests should be run in strict mode
    CxSimulator.strict = False
    try:
        emu = model.get_simulator()
        with pytest.warns(UserWarning):
            emu.run_steps(nt)
    finally:
        CxSimulator.strict = True  # change back to True for subsequent tests

    emu_u = np.array(emu.probe_outputs[probe_u])
    emu_v = np.array(emu.probe_outputs[probe_v])
    emu_s = np.array(emu.probe_outputs[probe_s])

    with model.get_loihi() as sim:
        sim.run_steps(nt)
        sim_u = np.column_stack([
            p.timeSeries.data for p in sim.board.probe_map[probe_u]])
        sim_v = np.column_stack([
            p.timeSeries.data for p in sim.board.probe_map[probe_v]])
        sim_s = np.column_stack([
            p.timeSeries.data for p in sim.board.probe_map[probe_s]])
        sim_v[sim_s > 0] = 0  # since Loihi has placeholder voltage after spike

    plt.subplot(311)
    plt.plot(emu_u)
    plt.plot(sim_u)

    plt.subplot(312)
    plt.plot(emu_v)
    plt.plot(sim_v)

    plt.subplot(313)
    plt.plot(emu_s)
    plt.plot(sim_s)

    assert allclose(emu_u[:-1], sim_u[1:])
    assert allclose(emu_v[:-1], sim_v[1:])


@pytest.mark.parametrize('positive', [True, False])
def test_v_overflow(positive, Simulator, plt, allclose):
    n_axons = 10
    # nt = 15
    nt = 1000

    model = CxModel()

    # n_axons controls number of input spikes and thus amount of overflow
    input_spikes = np.zeros((nt, n_axons), dtype=bool)
    input_spikes[0, :] = 1  # spike on first timestep only
    input = CxSpikeInput(input_spikes)

    group = CxGroup(1)
    group.configure_nonspiking()
    group.configure_filter(0.1)
    group.vmin = -2**22

    synapses = CxSynapses(n_axons)
    synapses.set_full_weights(np.ones((n_axons, 1)))

    axons = CxAxons(n_axons)
    axons.target = synapses
    input.add_axons(axons)
    group.add_synapses(synapses)

    probe_u = CxProbe(target=group, key='u')
    group.add_probe(probe_u)
    probe_v = CxProbe(target=group, key='v')
    group.add_probe(probe_v)
    probe_s = CxProbe(target=group, key='s')
    group.add_probe(probe_s)

    model.add_input(input)
    model.add_group(group)
    model.discretize()

    group.vth[:] = VTH_MAX  # must set after `discretize`
    group.decayU[:] = 0
    group.decayV[:] = 0
    group.bias[:] = BIAS_MAX
    synapses.format(wgtExp=7)
    synapses.weights[0][:] = 254

    assert CxSimulator.strict  # Tests should be run in strict mode
    CxSimulator.strict = False
    try:
        emu = model.get_simulator()
        # with pytest.warns(UserWarning, match='V'):
        emu.run_steps(nt)
    finally:
        CxSimulator.strict = True  # change back to True for subsequent tests

    emu_u = np.array(emu.probe_outputs[probe_u])
    emu_v = np.array(emu.probe_outputs[probe_v])
    emu_s = np.array(emu.probe_outputs[probe_s])

    # with model.get_loihi() as sim:
    #     sim.run_steps(nt)
    #     sim_u = np.column_stack([
    #         p.timeSeries.data for p in sim.board.probe_map[probe_u]])
    #     sim_v = np.column_stack([
    #         p.timeSeries.data for p in sim.board.probe_map[probe_v]])
    #     sim_s = np.column_stack([
    #         p.timeSeries.data for p in sim.board.probe_map[probe_s]])
    #     sim_v[sim_s > 0] = 0  # since Loihi has placeholder voltage after spike

    plt.subplot(311)
    plt.plot(emu_u)
    # plt.plot(sim_u)

    plt.subplot(312)
    plt.plot(emu_v)
    # plt.plot(sim_v)

    plt.subplot(313)
    plt.plot(emu_s)
    # plt.plot(sim_s)

    # assert allclose(emu_u[:-1], sim_u[1:])
    # assert allclose(emu_v[:-1], sim_v[1:])
