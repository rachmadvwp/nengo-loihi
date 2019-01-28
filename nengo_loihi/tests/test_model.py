import numpy as np

from nengo_loihi.block import LoihiBlock
from nengo_loihi.builder.probe import Probe
from nengo_loihi.discretize import discretize_model
from nengo_loihi.emulator import EmulatorInterface
from nengo_loihi.hardware import HardwareInterface
from nengo_loihi.simulator import Model


def test_simulator_noise(request, plt, seed):
    target = request.config.getoption("--target")
    n_cx = 10

    model = Model()
    block = LoihiBlock(n_cx)
    block.compartments.configure_relu()

    block.compartments.bias[:] = np.linspace(0, 0.01, n_cx)

    block.compartments.enableNoise[:] = 1
    block.compartments.noiseExp0 = -2
    block.compartments.noiseMantOffset0 = 0
    block.compartments.noiseAtDendOrVm = 1

    probe = Probe(target=block, key='voltage')
    block.add_probe(probe)
    model.add_block(block)

    discretize_model(model)

    if target == 'loihi':
        with HardwareInterface(model, use_snips=False, seed=seed) as sim:
            sim.run_steps(1000)
            y = sim.get_probe_output(probe)
    else:
        with EmulatorInterface(model, seed=seed) as sim:
            sim.run_steps(1000)
            y = sim.get_probe_output(probe)

    plt.plot(y)
    plt.yticks(())
