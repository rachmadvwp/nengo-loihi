import numpy as np

from nengo_loihi.builder import Model
from nengo_loihi.compartments import CxGroup
from nengo_loihi.emulator import CxSimulator
from nengo_loihi.hardware import LoihiSimulator
from nengo_loihi.probes import CxProbe


def test_simulator_noise(request, plt, seed):
    target = request.config.getoption("--target")

    model = Model()
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
        with LoihiSimulator(model, use_snips=False, seed=seed) as sim:
            sim.run_steps(1000)
            y = sim.get_probe_output(probe)
    else:
        with CxSimulator(model, seed=seed) as sim:
            sim.run_steps(1000)
            y = sim.get_probe_output(probe)

    plt.plot(y)
    plt.yticks(())
