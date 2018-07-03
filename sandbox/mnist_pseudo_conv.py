import collections
import gzip
import os
import pickle
from urllib.request import urlretrieve
import zipfile

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import nengo
import nengo_dl
import nengo_extras.vision


def softmax_loss(x, y):
    return tf.nn.softmax_cross_entropy_with_logits_v2(logits=x, labels=y)


def classification_error(outputs, targets):
    return 100 * tf.reduce_mean(
        tf.cast(tf.not_equal(tf.argmax(outputs[:, -1], axis=-1),
                             tf.argmax(targets[:, -1], axis=-1)),
                tf.float32))


if not os.path.exists('mnist.pkl.gz'):
    urlretrieve('http://deeplearning.net/data/mnist/mnist.pkl.gz',
                'mnist.pkl.gz')

with gzip.open('mnist.pkl.gz') as f:
    train_data, _, test_data = pickle.load(f, encoding="latin1")
train_data = list(train_data)
test_data = list(test_data)
for data in (train_data, test_data):
    one_hot = np.zeros((data[0].shape[0], 10))
    one_hot[np.arange(data[0].shape[0]), data[1]] = 1
    data[1] = one_hot

# for i in range(3):
#     plt.figure()
#     plt.imshow(np.reshape(train_data[0][i], (28, 28)))
#     plt.axis('off')
#     plt.title(str(np.argmax(train_data[1][i])))

# lif parameters
test_neurons = nengo.LIF(amplitude=0.01)

# softlif parameters (lif parameters + sigma)
train_neurons = nengo_dl.SoftLIFRate(amplitude=0.01, sigma=0.01, tau_rc=0.05, tau_ref=0.001)
# train_neurons = nengo_dl.SoftLIFRate(sigma=0.01, tau_rc=0.05, tau_ref=0.001)
# train_neurons = nengo.neurons.RectifiedLinear()

# ensemble parameters
ens_params = dict(
    max_rates=nengo.dists.Choice([100]),
    intercepts=nengo.dists.Choice([0]))


def build_network(neuron_type, synapse=None, output_synapse=None):
    rng = np.random.RandomState(0)
    seed = rng.randint(2**31-1)

    input_shape = (28, 28)
    kernel_shape = (7, 7)
    kernel_stride = (3, 3)
    n_filters = 64

    amp = 1.0
    # amp = 0.01

    fully_connected_dist = nengo.dists.UniformHypersphere(surface=True)

    probes = collections.OrderedDict()
    with nengo.Network(seed=seed) as net:
        net.config[nengo.Connection].synapse = synapse

        inp = nengo.Node([0] * 28 * 28, label='in')

        layer_1 = []
        for i in range(0, input_shape[0] - kernel_shape[0], kernel_stride[0]):
            for j in range(0, input_shape[1] - kernel_shape[1], kernel_stride[1]):
                label = 'layer_1.%d.%d' % (i, j)
                encoders = nengo_extras.vision.Gabor().generate(
                    n_filters, kernel_shape, rng=rng).reshape(n_filters, -1)
                a = nengo.Ensemble(
                    n_filters, np.prod(kernel_shape), label=label,
                    neuron_type=neuron_type,
                    encoders=encoders,
                    **ens_params)
                layer_1.append(a)
                probes[label] = nengo.Probe(a.neurons)

                si, sj = kernel_shape
                input_inds = (i + np.arange(si)[:, None])*input_shape[1] + (j + np.arange(sj))
                input_inds = input_inds.ravel()
                nengo.Connection(inp[input_inds], a, synapse=None)

        n_neurons = 256
        layer_2 = nengo.Ensemble(n_neurons, n_filters, label='layer_2',
                                 neuron_type=neuron_type)
        probes['layer_2'] = nengo.Probe(layer_2.neurons)
        for a in layer_1:
            transform = fully_connected_dist.sample(n_neurons, n_filters, rng=rng)
            transform = 0.01 * amp * transform
            nengo.Connection(a.neurons, layer_2.neurons, transform=transform)

        out = nengo.Node(label='out', size_in=10)
        transform = fully_connected_dist.sample(10, n_neurons, rng=rng)
        transform = amp * transform
        nengo.Connection(layer_2.neurons, out, transform=transform)

        # out_p = nengo.Probe(out)
        out_p = nengo.Probe(out, synapse=output_synapse)
        probes['out'] = out_p

    return net, inp, out_p, probes

# construct the network
net, inp, out_p, probes = build_network(train_neurons)

# construct the simulator
minibatch_size = 200
sim = nengo_dl.Simulator(net, minibatch_size=minibatch_size)

# note that we need to add the time dimension (axis 1), which has length 1
# in this case. we're also going to reduce the number of test images, just to
# speed up this example.
train_inputs = {inp: train_data[0][:, None, :]}
train_targets = {out_p: train_data[1][:, None, :]}
test_inputs = {inp: test_data[0][:minibatch_size*2, None, :]}
test_targets = {out_p: test_data[1][:minibatch_size*2, None, :]}

if 0:
    # --- print forward activities
    demo_inputs = {inp: train_data[0][:minibatch_size, None, :]}
    with sim:
        sim.run_steps(1, input_feeds=demo_inputs)

    for label, probe in probes.items():
        y = sim.data[probe]
        print(y.shape)
        print("%s: min %s, max %s, e-std %s, u-std %s" % (
            label, y.min(), y.max(), y.std(0).mean(), y.std(2).mean()))

    assert 0


print("error before training: %.2f%%" %
      sim.loss(test_inputs, test_targets, classification_error))

do_training = True
if do_training:
    n_epochs = 10
    opt = tf.train.RMSPropOptimizer(learning_rate=0.001)

    # run training
    sim.train(train_inputs, train_targets, opt, objective=softmax_loss,
              n_epochs=n_epochs)

    # save the parameters to file
    sim.save_params("./mnist_params")
else:
    # load parameters
    sim.load_params("./mnist_params")

print("error after training: %.2f%%" %
      sim.loss(test_inputs, test_targets, classification_error))

sim.close()

# --- run in spiking neurons
net, inp, out_p, _ = build_network(
    test_neurons, synapse=0.005, output_synapse=0.03)

sim = nengo_dl.Simulator(
    net, minibatch_size=minibatch_size, unroll_simulation=10)
sim.load_params("./mnist_params")

n_steps = 30
test_inputs_time = {
    inp: np.tile(v, (1, n_steps, 1)) for v in test_inputs.values()}
test_targets_time = {
    out_p: np.tile(v, (1, n_steps, 1)) for v in test_targets.values()}

print("spiking neuron error: %.2f%%" %
      sim.loss(test_inputs_time, test_targets_time, classification_error))

sim.run_steps(n_steps,
              input_feeds={inp: test_inputs_time[inp][:minibatch_size]})
for i in range(5):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(np.reshape(test_data[0][i], (28, 28)))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.plot(sim.trange(), sim.data[out_p][i])
    plt.legend([str(i) for i in range(10)], loc="upper left")
    plt.xlabel("time")

sim.close()

plt.show()
