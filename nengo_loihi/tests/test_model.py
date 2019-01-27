import numpy as np

from nengo_loihi.discretize import discretize_model
from nengo_loihi.emulator import EmulatorInterface
from nengo_loihi.hardware import HardwareInterface
from nengo_loihi.probe_builders import Probe
from nengo_loihi.segment import LoihiSegment
from nengo_loihi.simulator import Model


def test_simulator_noise(request, plt, seed):
    target = request.config.getoption("--target")
    n_cx = 10

    model = Model()
    segment = LoihiSegment(n_cx)
    segment.compartments.configure_relu()

    segment.compartments.bias[:] = np.linspace(0, 0.01, n_cx)

    segment.compartments.enableNoise[:] = 1
    segment.compartments.noiseExp0 = -2
    segment.compartments.noiseMantOffset0 = 0
    segment.compartments.noiseAtDendOrVm = 1

    probe = Probe(target=segment, key='voltage')
    segment.add_probe(probe)
    model.add_segment(segment)

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
