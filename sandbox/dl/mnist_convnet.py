"""
NOTES:
- Occasionally, the training for the original network can fail to converge
  (the loss stops going down at the start of training, and remains around 500).
  I believe this is due to bad random weights chosen for the initial kernels.
  In this case, simply restart the training and it should work.
"""

import gzip
import os
import pickle
from urllib.request import urlretrieve

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import nengo
import nengo_dl

import nengo_loihi
from nengo_loihi.conv import Conv2D, ImageShape, ImageSlice


def crossentropy(outputs, targets):
    """Cross-entropy loss function (for training)."""
    return tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=outputs, labels=targets)


def classification_error(outputs, targets):
    """Classification error function (for testing)."""
    return 100 * tf.reduce_mean(
        tf.cast(tf.not_equal(tf.argmax(outputs[:, -1], axis=-1),
                             tf.argmax(targets[:, -1], axis=-1)),
                tf.float32))


def has_checkpoint(checkpoint_base):
    checkpoint_dir, checkpoint_name = os.path.split(checkpoint_base)
    if not os.path.exists(checkpoint_dir):
        return False

    files = os.listdir(checkpoint_dir)
    files = [f for f in files if f.startswith(checkpoint_name)]
    return len(files) > 0


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
         neuron_type=neuron_type, on_chip=False),
    dict(transform=Conv2D(6, ImageShape(28, 28, 1, channels_last), strides=2),
         neuron_type=neuron_type, parallel=1),
    dict(transform=Conv2D(24, ImageShape(13, 13, 6, channels_last), strides=2),
         neuron_type=neuron_type, parallel=1),
    # dict(transform=Conv2D(24, ImageShape(6, 6, 24, channels_last), strides=2),
    #      neuron_type=neuron_type, parallel=5),
    dict(transform=nengo_dl.dists.Glorot(), units=10),
]

# build the network
with nengo.Network(seed=0) as net:
    nengo_loihi.add_params(net)
    net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
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
                conn = nengo.Connection(pre, post, transform=transform)

        pres = posts

    assert len(posts) == 1
    out = posts[0]

    out_p = nengo.Probe(out)
    out_p_filt = nengo.Probe(out, synapse=nengo.Alpha(0.01))

# set up training/test data
train_inputs = {inp: train_data[0][:, None, :]}
train_targets = {out_p: train_data[1][:, None, :]}
test_inputs = {inp: np.tile(test_data[0][:, None, :],
                            (1, int(presentation_time / dt), 1))}
test_targets = {out_p_filt: np.tile(test_data[1][:, None, :],
                                    (1, int(presentation_time / dt), 1))}
# test_inputs = {inp: test_data[0][:, None, :]}
# test_targets = {out_p: test_data[1][:, None, :]}

# train our network in NengoDL
with nengo_dl.Simulator(net, minibatch_size=256, seed=0) as sim:
    if not RETRAIN and has_checkpoint(checkpoint_base):
        sim.load_params(checkpoint_base)

    else:
        print("error before training: %.2f%%" %
              sim.loss(test_inputs, test_targets, classification_error))

        # run training
        sim.train(train_inputs, train_targets,
                  tf.train.RMSPropOptimizer(learning_rate=0.001),
                  objective=crossentropy, n_epochs=10)
        sim.save_params(checkpoint_base)

        print("error after training: %.2f%%" %
              sim.loss(test_inputs, test_targets, classification_error))

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

# plt.savefig('mnist_convnet_%s.png' % sim.target)
plt.show()
