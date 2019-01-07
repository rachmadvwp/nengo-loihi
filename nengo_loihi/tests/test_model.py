import numpy as np

from nengo_loihi.builder import Model
from nengo_loihi.emulator import Emulator
from nengo_loihi.hardware import LoihiSimulator
from nengo_loihi.neurongroup import NeuronGroup
from nengo_loihi.probes import Probe


def test_simulator_noise(request, plt, seed):
    target = request.config.getoption("--target")
    n_cx = 10

    model = Model()
    group = NeuronGroup(n_cx)
    group.compartments.configure_relu()

    group.compartments.bias[:] = np.linspace(0, 0.01, n_cx)

    group.compartments.enableNoise[:] = 1
    group.compartments.noiseExp0 = -2
    group.compartments.noiseMantOffset0 = 0
    group.compartments.noiseAtDendOrVm = 1

    probe = Probe(target=group, key='voltage')
    group.add_probe(probe)
    model.add_group(group)

    model.discretize()

    if target == 'loihi':
        with LoihiSimulator(model, use_snips=False, seed=seed) as sim:
            sim.run_steps(1000)
            y = sim.get_probe_output(probe)
    else:
        with Emulator(model, seed=seed) as sim:
            sim.run_steps(1000)
            y = sim.get_probe_output(probe)

    plt.plot(y)
    plt.yticks(())
