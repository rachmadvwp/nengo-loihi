import os
import matplotlib as mpl
haveDisplay = "DISPLAY" in os.environ
if not haveDisplay:
    mpl.use('Agg')

import nengo
import nengo_loihi
import numpy as np
import matplotlib.pyplot as plt

import nxsdk
Simulator = nengo_loihi.Simulator


with nengo.Network(seed=1) as model:
    stim = nengo.Node(lambda t: 0.5)#np.sin(2*np.pi*t))

    a = nengo.Ensemble(100, 1, label='b',
                       max_rates=nengo.dists.Uniform(100, 120),
                       intercepts=nengo.dists.Uniform(-0.5, 0.5)
			)
    nengo.Connection(stim, a)

    #p = nengo.Probe(a)

import timeit
with Simulator(model) as sim:
    start = timeit.default_timer()
    sim.run(1.0)
    time_taken = timeit.default_timer() - start
print(time_taken)

#filt = nengo.Lowpass(0.1)
#print(filt.filt(sim.data[p]))
