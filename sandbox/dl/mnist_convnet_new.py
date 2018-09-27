"""
NOTES:
- Occasionally, the training for the original network can fail to converge
  (the loss stops going down at the start of training, and remains around 500).
  I believe this is due to bad random weights chosen for the initial kernels.
  In this case, simply restart the training and it should work.
"""

import collections
from functools import partial
import gzip
import os
import pickle
from urllib.request import urlretrieve
import tempfile
import zipfile

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import nengo
import nengo_dl

import nengo_loihi
from nengo_loihi.conv import Conv2D, ImageShape, ImageSlice, split_transform


def crossentropy(outputs, targets):
    """Cross-entropy loss function (for training)."""
    # return tf.nn.softmax_cross_entropy_with_logits_v2(
    #     logits=outputs, labels=targets)
    return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=outputs, labels=targets))


def classification_error(outputs, targets):
    """Classification error function (for testing)."""
    return 100 * tf.reduce_mean(
        tf.cast(tf.not_equal(tf.argmax(outputs[:, -1], axis=-1),
                             tf.argmax(targets[:, -1], axis=-1)),
                tf.float32))


def percentile_rate_l2_loss(x, y, weight=1.0, target=0.0, percentile=99.):
    # x axes are (batch examples, time (==1), neurons)
    assert len(x.shape) == 3
    rates = tf.contrib.distributions.percentile(x, percentile, axis=(0, 1))
    return weight * tf.nn.l2_loss(rates - target)


def percentile_l2_loss_range(x, y, weight=1.0, min=0.0, max=np.inf,
                             percentile=99.):
    # x axes are (batch examples, time (==1), neurons)
    assert len(x.shape) == 3
    neuron_p = tf.contrib.distributions.percentile(x, percentile, axis=(0, 1))
    low_error = tf.maximum(0.0, min - neuron_p)
    high_error = tf.maximum(0.0, neuron_p - max)
    return weight * tf.nn.l2_loss(low_error + high_error)


def has_checkpoint(checkpoint_base):
    checkpoint_dir, checkpoint_name = os.path.split(checkpoint_base)
    if not os.path.exists(checkpoint_dir):
        return False

    files = os.listdir(checkpoint_dir)
    files = [f for f in files if f.startswith(checkpoint_name)]
    return len(files) > 0


def get_layer_rates(sim, input_data, probes, amplitude=1):
    """Collect firing rates on internal layers"""

    assert len(input_data) == 1
    in_p, in_x = next(iter(input_data.items()))
    assert in_x.ndim == 3
    n_steps = in_x.shape[1]

    tmpdir = tempfile.TemporaryDirectory()
    sim.save_params(os.path.join(tmpdir.name, "tmp"),
                    include_local=True, include_global=False)

    sim.run_steps(n_steps, input_feeds=input_data, progress_bar=False)

    output = [sim.data[p] / amplitude for p in probes]

    sim.load_params(os.path.join(tmpdir.name, "tmp"),
                    include_local=True, include_global=False)
    tmpdir.cleanup()

    return output


# load mnist dataset
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

checkpoint_base = './checkpoints/mnist_convnet'
RETRAIN = True
amp = 0.01
dt = 0.001
presentation_time = 0.2
rate_reg = 1e-2
max_rate = 100
rate_target = max_rate * amp  # must be in amplitude scaled units
channels_last = False
input_shape = ImageShape(28, 28, 1, channels_last=channels_last)
test_images = test_data[0].reshape(
    (-1,) + input_shape.shape(channels_last=True))
if not channels_last:
    test_images = np.transpose(test_images, (0, 3, 1, 2))

neuron_type = nengo.SpikingRectifiedLinear(amplitude=amp)
layer_dicts = [
    dict(transform=Conv2D(1, ImageShape(28, 28, 1, channels_last),
                          kernel_size=1, kernel=np.ones((1, 1, 1, 1))),
         neuron_type=neuron_type, on_chip=False, min_rate=False),
    dict(transform=Conv2D(6, ImageShape(28, 28, 1, channels_last), strides=2),
         neuron_type=neuron_type, parallel=1),
    dict(transform=Conv2D(24, ImageShape(13, 13, 6, channels_last), strides=2),
         neuron_type=neuron_type, parallel=1),
    # dict(transform=Conv2D(24, ImageShape(6, 6, 24, channels_last), strides=2),
    #      neuron_type=neuron_type, parallel=5),
    dict(transform=nengo_dl.dists.Glorot(), units=10),
]

# build the nengo_dl network
objective = {}
layer_probes = collections.OrderedDict()
with nengo.Network(seed=0) as net:
    nengo_loihi.add_params(net)
    net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([max_rate])
    net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    net.config[nengo.Connection].synapse = None
    # nengo_dl.configure_settings(lif_smoothing=0.01)
    # nengo_dl.configure_settings(trainable=None)
    # net.config[nengo.ensemble.Neurons].trainable = False

    # the input node that will be used to feed in input images
    inp = nengo.Node(
        nengo.processes.PresentInput(test_data[0], presentation_time),
        size_out=28 * 28)

    pres = [inp]
    for layer_dict in layer_dicts:
        print("LAYER")
        print(layer_dict)

        transform = layer_dict["transform"]
        neuron_type = layer_dict.get("neuron_type", None)
        on_chip = layer_dict.get("on_chip", True)
        min_rate = layer_dict.get("min_rate", True)
        posts = []

        for i in range(layer_dict.get("parallel", 1)):
            if "units" in layer_dict:
                post_size = layer_dict["units"]
            else:
                post_size = transform.output_shape.size

            if neuron_type is None:
                # node layer
                post = nengo.Node(size_in=post_size)
                assert on_chip not in layer_dict
            else:
                # ensemble layer
                post = nengo.Ensemble(
                    post_size, 1, neuron_type=neuron_type).neurons
                net.config[post.ensemble].on_chip = on_chip
            posts.append(post)

            for pre in pres:
                nengo.Connection(pre, post, transform=transform)

            post_probe = nengo.Probe(post)
            # if min_rate:
            #     objective[post_probe] = partial(
            #         percentile_l2_loss_range, weight=rate_reg,
            #         min=0.5 * rate_target, max=rate_target, percentile=99.9)
            # else:
            #     objective[post_probe] = partial(
            #         percentile_l2_loss_range, weight=10 * rate_reg,
            #         min=0, max=rate_target, percentile=99.9)

            layer_probes[post] = post_probe
        pres = posts

    assert len(posts) == 1
    out = posts[0]

    out_p = nengo.Probe(out)
    out_p_filt = nengo.Probe(out, synapse=nengo.Alpha(0.01))

    objective[out_p] = crossentropy

# set up training/test data
minibatch_size = 256
train_inputs = {inp: train_data[0][:, None, :]}
train_targets = {out_p: train_data[1][:, None, :]}
test_inputs = {inp: np.tile(test_data[0][:, None, :],
                            (1, int(presentation_time / dt), 1))}
test_targets = {out_p_filt: np.tile(test_data[1][:, None, :],
                                    (1, int(presentation_time / dt), 1))}
# test_inputs = {inp: test_data[0][:, None, :]}
# test_targets = {out_p: test_data[1][:, None, :]}
rate_inputs = {inp: np.tile(train_data[0][:minibatch_size, None, :],
                            (1, int(presentation_time / dt), 1))}

# train our network in NengoDL
with nengo_dl.Simulator(net, minibatch_size=minibatch_size, seed=0) as sim:
    if not RETRAIN and has_checkpoint(checkpoint_base):
        sim.load_params(checkpoint_base)

    else:
        print("error before training: %.2f%%" %
              sim.loss(test_inputs, test_targets, classification_error))

        # run training
        sim.train(train_inputs, train_targets,
                  tf.train.RMSPropOptimizer(learning_rate=0.001),
                  objective=objective, n_epochs=10)
        sim.save_params(checkpoint_base)

        print("error after training: %.2f%%" %
              sim.loss(test_inputs, test_targets, classification_error))

        spikes = get_layer_rates(sim, rate_inputs, layer_probes.values(),
                                 amplitude=amp)
        for layer, spikes in zip(layer_probes, spikes):
            print("%s rate: mean=%0.3f, 1st: %0.3f 99th: %0.3f" % (
                layer, spikes.mean(),
                np.percentile(np.mean(spikes, axis=(0, 1)), 1),
                np.percentile(np.mean(spikes, axis=(0, 1)), 99)))

    # store trained parameters back into the network
    sim.freeze_params(net)

for conn in net.all_connections:
    conn.synapse = 0.005

with nengo_dl.Simulator(net, minibatch_size=256) as sim:
    print("error w/ synapse: %.2f%%" %
          sim.loss(test_inputs, test_targets, classification_error))

n_presentations = 100
# with nengo.Simulator(net, dt=dt, optimize=False) as sim:
with nengo_loihi.Simulator(net, dt=dt, precompute=True) as sim:
    sim.run(n_presentations * presentation_time)

    step = int(presentation_time / dt)
    output = sim.data[out_p_filt][step - 1::step]

    correct = 100 * np.mean(np.argmax(output, axis=-1) !=
                            np.argmax(test_data[1][:n_presentations], axis=-1))

    print("loihi error: %.2f%%" % correct)

# --- fancy plots
n_plots = 10
plt.figure()

plt.subplot(2, 1, 1)
images = test_data[0].reshape(-1, 28, 28, 1)
ni, nj, nc = images[0].shape
allimage = np.zeros((ni, nj * n_plots, nc), dtype=images.dtype)
for i, image in enumerate(images[:n_plots]):
    allimage[:, i * nj:(i + 1) * nj] = image
if allimage.shape[-1] == 1:
    allimage = allimage[:, :, 0]
plt.imshow(allimage, aspect='auto', interpolation='none', cmap='gray')

plt.subplot(2, 1, 2)
plt.plot(sim.trange()[:n_plots * step], sim.data[out_p_filt][:n_plots * step])
plt.legend(['%d' % i for i in range(10)], loc='best')

# target = sim.target if isinstance(sim, nengo_loihi.Simulator) else 'nengo'
# plt.savefig('mnist_convnet_%s.png' % target)
plt.show()
