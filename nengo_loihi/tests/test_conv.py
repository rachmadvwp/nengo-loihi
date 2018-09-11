import os
import pickle

import numpy as np
import scipy.signal

import nengo_loihi
import nengo_loihi.loihi_cx as loihi_cx

from nengo_extras.matplotlib import tile, imshow
from nengo_extras.vision import Gabor

home_dir = os.path.dirname(nengo_loihi.__file__)
test_dir = os.path.join(home_dir, 'tests')


def test_conv_cxsynapses(plt, rng):
    with open(os.path.join(test_dir, 'mnist10.pkl'), 'rb') as f:
        test10 = pickle.load(f)

    test_x, test_y = test10[0][0].reshape(28, 28), test10[1][0]
    test_x = 2. * test_x - 1.
    # print(test_x.min(), test_x.max())

    filters = Gabor().generate(16, (7, 7), rng=rng)
    stride = 2
    tau_rc = 0.02
    tau_ref = 0.002
    tau_s = 0.005
    dt = 0.001

    pres_time = 1.0

    # --- compute ideal outputs
    ref_out = np.array([
        scipy.signal.convolve2d(test_x, kernel, mode='valid')[
            ::stride, ::stride]
        for kernel in filters])

    inp_radius = np.abs(test_x).max()
    # inp_radius = np.abs(outputs).max()

    # --- compute nengo_loihi outputs
    inp_biases = np.vstack([test_x, -test_x]) / inp_radius

    model = loihi_cx.CxModel()
    inp = loihi_cx.CxGroup(2 * 28 * 28)
    inp.configure_relu()
    inp.bias[:] = inp_biases.ravel()

    # inp_ax = loihi_cx.CxAxons(inp.n)
    # inp.add_axons(inp_ax)

    inp_probe = loihi_cx.CxProbe(target=inp, key='s')
    inp.add_probe(inp_probe)

    model.add_group(inp)

    # neurons = loihi_cx.CxGroup(outputs.size)
    # neurons.configure_lif(tau_rc=tau_rc, tau_ref=tau_ref, dt=dt)
    # neurons.configure_filter(tau_s, dt=dt)

    # synapses = ConvCxSynapses()
    # neurons.add_synapses(synapses)

    # inp_ax.target = synapses

    # model.add_group(neurons)

    sim = model.get_simulator()
    sim_inp = []
    for i in range(int(pres_time / dt)):
        sim.step()

    sim_inp = np.sum(sim.probe_outputs[inp_probe], axis=0)
    sim_inp.shape = (2 * 28, 28)
    print(sim_inp.min(), sim_inp.max())

    ax = plt.subplot(311)
    tile(filters, cols=8, ax=ax)

    ax = plt.subplot(312)
    tile(ref_out, cols=8, ax=ax)

    ax = plt.subplot(313)
    imshow(sim_inp, vmin=0, vmax=1./dt, ax=ax)
