"""
Use backprop to find the optimal tuning for population interneurons.
"""
import nengo
from nengo.utils.numpy import rms
import nengo_loihi
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.RandomState(3)


def function(x):
    return x


def plot_tuning(model):
    x = np.linspace(-1, 1, 101).reshape(-1, 1)
    y = function(x)

    with nengo.Simulator(model) as sim:
        _, acts = nengo.utils.ensemble.tuning_curves(model.a, sim, inputs=x)
        decoders = sim.data[model.c_out].weights
        yest = np.dot(acts, decoders.T)

    print("RMSE: %0.3f" % rms(yest - y))

    plt.figure(figsize=(6, 10))
    plt.subplot(211)
    plt.plot(x, acts)

    plt.subplot(212)
    plt.plot(x, y)
    plt.plot(x, yest)
    plt.xlim([-1, 1])
    plt.axis('equal')

# --- data
x = np.linspace(-1, 1, 1000)
y = function(x)

# --- parameters
uniform_gain = False
# uniform_gain = True

# n = 5
n = 10

neuron_type = nengo_loihi.neurons.LoihiSpikingRectifiedLinear()


if 0:
    encoders = np.vstack([np.ones((n, 1)), -np.ones((n, 1))])
    intercepts = np.tile(np.linspace(-0.8, 0.8, n), 2)
    if uniform_gain:
        max_rates = 100 * np.ones(2*n)
    else:
        # max_rates = np.tile(np.linspace(80, 120, n)[::-1], 2)
        # max_rates = np.tile(np.linspace(70, 120, n)[::-1], 2)
        max_rates = np.tile(np.linspace(70, 160, n)[::-1], 2)

    target_point = 1.0
    # target_point = 0.85
    gain, bias = neuron_type.gain_bias(max_rates, intercepts)
    target_rates = neuron_type.rates(target_point, gain, bias)
    print(target_rates)
    target_rate = target_rates.sum()
    out_gain = 2.*target_point / target_rate
    decoders = out_gain * encoders.T
else:
    # hyperopt
    import hyperopt
    from hyperopt import hp

    def args_to_params(args):
        i_max = args['intercept_max']
        i_min = i_max - args['intercept_range']
        r_min = args['rate_min']
        r_max = r_min + args['rate_range']
        encoders = np.vstack([np.ones((n, 1)), -np.ones((n, 1))])
        intercepts = np.tile(np.linspace(i_min, i_max, n), 2)
        max_rates = np.linspace(r_min, r_max, n)

        if args.get('flip_r', True):
            max_rates = max_rates[::-1]

        max_rates = np.tile(max_rates, 2)

        target_point = args.get('target_point', 1.0)
        gain, bias = neuron_type.gain_bias(max_rates, intercepts)
        target_rates = neuron_type.rates(target_point, gain, bias)
        target_rate = target_rates.sum()
        out_gain = 2.*target_point / target_rate if target_rate > 0 else 0
        decoders = out_gain * encoders
        # if np.isinf(decoders).any():
            # import pdb; pdb.set_trace()

        return encoders, intercepts, max_rates, gain, bias, decoders

    def objective(args):
        encoders, intercepts, max_rates, gain, bias, decoders = args_to_params(args)

        acts = neuron_type.rates(np.dot(x[:, None], encoders.T), gain, bias)
        yest = np.dot(acts, decoders)[:, 0]

        rmse = rms(yest - y)
        return rmse

    space = {
        'intercept_max': hp.uniform('intercept_max', 0, 0.9),
        'intercept_range': hp.uniform('intercept_range', 0.5, 2.),
        # 'rate_min': hp.uniform('rate_min', 50, 100),
        # 'rate_range': hp.uniform('rate_range', 0, 200),
        'rate_min': hp.uniform('rate_min', 70, 100),
        'rate_range': hp.uniform('rate_range', 30, 100),
        'target_point': hp.uniform('target_point', 0.1, 1.0),
        'flip_r': hp.quniform('flip_r', 0, 1, 1),
    }

    # best = hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, max_evals=1)
    best = hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, max_evals=100)
    print("Best: %s" % best)
    encoders, intercepts, max_rates, gain, bias, decoders = args_to_params(best)
    print("Intercepts: %0.3f %0.3f" % (intercepts[0], intercepts[-1]))
    print("Max rates: %0.3f %0.3f" % (max_rates[0], max_rates[-1]))
    print("Gain: %0.3f %0.3f" % (gain[0], gain[-1]))
    print("Bias: %0.3f %0.3f" % (bias[0], bias[-1]))
    print(best['target_point'])
    print(best['flip_r'])


with nengo.Network() as model:
    model.u = nengo.Node([0])
    model.a = nengo.Ensemble(2*n, 1,
                             neuron_type=neuron_type,
                             encoders=encoders,
                             max_rates=max_rates,
                             intercepts=intercepts)
    model.v = nengo.Node(size_in=1)

    model.c_in = nengo.Connection(model.u, model.a, synapse=None)
    model.c_out = nengo.Connection(model.a.neurons, model.v, synapse=None,
                                   transform=decoders.T)

    model.up = nengo.Probe(model.u)
    model.vp = nengo.Probe(model.v)

plot_tuning(model)
plt.show()
