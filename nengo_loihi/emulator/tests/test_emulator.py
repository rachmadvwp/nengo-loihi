import nengo
from nengo.exceptions import SimulationError
import numpy as np
import pytest

from nengo_loihi.axons import CxAxons
from nengo_loihi.builder import Model
from nengo_loihi.compartments import CxGroup
from nengo_loihi.discretize import VTH_MAX
from nengo_loihi.emulator import EmulatorInterface
from nengo_loihi.io_objects import CxSpikeInput
from nengo_loihi.hardware import HardwareInterface
from nengo_loihi.probes import CxProbe
from nengo_loihi.synapses import CxSynapses


def test_strict_mode():
    # Tests should be run in strict mode
    assert EmulatorInterface.strict

    try:
        with pytest.raises(SimulationError):
            EmulatorInterface.error("Error in emulator")
        EmulatorInterface.strict = False
        with pytest.warns(UserWarning):
            EmulatorInterface.error("Error in emulator")
    finally:
        # Strict mode is a global setting so we set it back to True
        # for subsequent test runs.
        EmulatorInterface.strict = True


@pytest.mark.skipif(pytest.config.getoption("--target") != "loihi",
                    reason="need Loihi as comparison")
@pytest.mark.parametrize('n_axons', [200, 1000])
def test_uv_overflow(n_axons, Simulator, plt, allclose):
    # TODO: Currently this is not testing the V overflow, since it is higher
    #  and I haven't been able to figure out a way to make it overflow.
    nt = 15

    model = Model()

    # n_axons controls number of input spikes and thus amount of overflow
    input = CxSpikeInput(n_axons)
    for t in np.arange(1, nt+1):
        input.add_spikes(t, np.arange(n_axons))  # send spikes to all axons
    model.add_input(input)

    group = CxGroup(1)
    group.configure_relu()
    group.configure_filter(0.1)
    group.vmin = -2**22

    synapses = CxSynapses(n_axons)
    synapses.set_full_weights(np.ones((n_axons, 1)))
    group.add_synapses(synapses)

    axons = CxAxons(n_axons)
    axons.target = synapses
    input.add_axons(axons)

    probe_u = CxProbe(target=group, key='u')
    group.add_probe(probe_u)
    probe_v = CxProbe(target=group, key='v')
    group.add_probe(probe_v)
    probe_s = CxProbe(target=group, key='s')
    group.add_probe(probe_s)

    model.add_group(group)
    model.discretize()

    group.vth[:] = VTH_MAX  # must set after `discretize`

    assert EmulatorInterface.strict  # Tests should be run in strict mode
    EmulatorInterface.strict = False
    try:
        with EmulatorInterface(model) as emu:
            with pytest.warns(UserWarning):
                emu.run_steps(nt)
            emu_u = emu.get_probe_output(probe_u)
            emu_v = emu.get_probe_output(probe_v)
            emu_s = emu.get_probe_output(probe_s)
    finally:
        EmulatorInterface.strict = True  # back to True for subsequent tests

    with HardwareInterface(model, use_snips=False) as sim:
        sim.run_steps(nt)
        sim_u = sim.get_probe_output(probe_u)
        sim_v = sim.get_probe_output(probe_v)
        sim_s = sim.get_probe_output(probe_s)
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

    assert allclose(emu_u, sim_u)
    assert allclose(emu_v, sim_v)
