import os
import pickle

import numpy as np
import scipy.signal

import nengo

import nengo_loihi
import nengo_loihi.loihi_cx as loihi_cx

from nengo_extras.matplotlib import tile, imshow
from nengo_extras.vision import Gabor

home_dir = os.path.dirname(nengo_loihi.__file__)
test_dir = os.path.join(home_dir, 'tests')


def test_conv2d_weights(plt, rng):
    with open(os.path.join(test_dir, 'mnist10.pkl'), 'rb') as f:
        test10 = pickle.load(f)

    test_x, test_y = test10[0][0].reshape(28, 28), test10[1][0]
    test_x = 2. * test_x - 1.
    # print(test_x.min(), test_x.max())

    filters = Gabor().generate(8, (7, 7), rng=rng)
    stride = 2
    tau_rc = 0.02
    tau_ref = 0.002
    tau_s = 0.005
    dt = 0.001

    neuron_type = nengo.LIF(tau_rc=tau_rc, tau_ref=tau_ref)
    gain = 1.
    # bias = 0.
    bias = 1.

    pres_time = 1.0

    # --- compute ideal outputs
    ref_out = np.array([
        scipy.signal.correlate2d(test_x, kernel, mode='valid')[
            ::stride, ::stride]
        for kernel in filters])
    ref_out = neuron_type.rates(ref_out, gain, bias)

    inp_radius = np.abs(test_x).max()
    # inp_radius = np.abs(outputs).max()

    # --- compute nengo_loihi outputs
    inp_biases = np.vstack([test_x, -test_x]) / inp_radius
    nk = 2  # number of channels (positive/negative)
    ni, nj = test_x.shape
    nij = ni * nj
    out_size = ref_out.size
    nf, nyi, nyj = ref_out.shape
    assert out_size <= 1024

    nxi = (nyi-1)*stride + 1
    nxj = (nyj-1)*stride + 1

    model = loihi_cx.CxModel()

    # input group
    inp = loihi_cx.CxGroup(ni * nj * nk)
    inp.configure_relu()
    inp.bias[:] = inp_biases.ravel()

    inp_ax = loihi_cx.CxAxons(nij)
    inp_ax.cx_to_axon_map = np.tile(np.arange(nij), nk)
    inp_ax.cx_atoms = np.concatenate([
        i * np.ones(nij, dtype=int) for i in range(nk)])
    inp.add_axons(inp_ax)

    inp_probe = loihi_cx.CxProbe(target=inp, key='s')
    inp.add_probe(inp_probe)

    model.add_group(inp)

    # conv group
    neurons = loihi_cx.CxGroup(out_size)
    neurons.configure_lif(tau_rc=tau_rc, tau_ref=tau_ref, dt=dt)
    neurons.configure_filter(tau_s, dt=dt)
    neurons.bias[:] = bias

    synapses = loihi_cx.CxSynapses(ni*nj)
    kernel = np.array([filters, -filters])  # two channels, pos and neg
    kernel = np.transpose(kernel, (0, 2, 3, 1))
    input_shape = (ni, nj, nk)
    print(kernel.shape)
    print(input_shape)
    print((nyi, nyj, nf))
    synapses.set_conv2d_weights(kernel, input_shape, strides=(stride, stride))
    neurons.add_synapses(synapses)

    out_probe = loihi_cx.CxProbe(target=neurons, key='s')
    neurons.add_probe(out_probe)

    inp_ax.target = synapses
    model.add_group(neurons)

    # simulation
    sim = model.get_simulator()
    sim_inp = []
    for i in range(int(pres_time / dt)):
        sim.step()

    sim_inp = np.mean(sim.probe_outputs[inp_probe], axis=0)
    sim_inp.shape = (2 * 28, 28)
    print(sim_inp.min(), sim_inp.max())

    sim_out = np.mean(sim.probe_outputs[out_probe], axis=0)
    sim_out.shape = (nyi, nyj, nf)
    sim_out = np.transpose(sim_out, (2, 0, 1))
    print(sim_out.max())

    # --- plot results
    rows = 4
    cols = 1

    ax = plt.subplot(rows, cols, 1)
    tile(filters, cols=8, ax=ax)

    ax = plt.subplot(rows, cols, 2)
    tile(ref_out, cols=8, ax=ax)

    ax = plt.subplot(rows, cols, 3)
    imshow(sim_inp, vmin=0, vmax=1, ax=ax)

    ax = plt.subplot(rows, cols, 4)
    # tile(sim_out, vmin=0, vmax=1, cols=8, ax=ax)
    tile(sim_out, vmin=0, vmax=sim_out.max(), cols=8, ax=ax)
