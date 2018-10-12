import numpy as np

from nxsdk.arch.n2a.graph.graph import N2Board
from nxsdk.arch.n2a.graph.inputgen import BasicSpikeGenerator


def decay_int(x, decay, a=12, b=0):
    r = (2**a - b - np.asarray(decay)).astype(np.int64)
    x = np.sign(x) * np.right_shift(np.abs(x) * r, a)  # round to zero
    return x


WGT = 200
WGT_EXP = 5
DECAY_U = 50


def setupNetwork():

    w_in = [WGT]
    i_in = list(range(len(w_in)))
    n0 = len(w_in)

    # --- board
    boardId = 1
    numChips = 1
    # Number of cores per chip
    numCoresPerChip = [1]
    # Number of synapses per core
    numSynapsesPerCore = [[n0]]
    # Initialize the board
    board = N2Board(boardId, numChips, numCoresPerChip, numSynapsesPerCore)

    chip = board.n2Chips[0]
    core0 = chip.n2Cores[0]

    # --- core0
    decayU = DECAY_U
    decayV = 0

    vth = 2**16

    core0.cxProfileCfg[0].configure(decayV=decayV, decayU=decayU)
    core0.vthProfileCfg[0].staticCfg.configure(vth=vth)
    core0.dendriteSharedCfg.configure(posVmLimit=7, negVmLimit=0)

    core0.synapseFmt[1].wgtExp = WGT_EXP
    core0.synapseFmt[1].wgtBits = 7
    core0.synapseFmt[1].numSynapses = 63
    core0.synapseFmt[1].idxBits = 7
    core0.synapseFmt[1].compression = 3
    core0.synapseFmt[1].fanoutType = 1

    for i, (w, idx) in enumerate(zip(w_in, i_in)):
        core0.synapses[i].CIdx = idx
        core0.synapses[i].Wgt = w
        core0.synapses[i].synFmtId = 1

    core0.synapseMap[0].synapsePtr = 0
    core0.synapseMap[0].synapseLen = len(w_in)
    core0.synapseMap[0].discreteMapEntry.configure()

    for i in range(n0):
        core0.cxCfg[i].configure(bias=0, biasExp=0, vthProfile=0, cxProfile=0)

    core0.numUpdates.configure(numUpdates=1 + n0//4)

    return board


if __name__ == '__main__':
    board = setupNetwork()
    chip = board.n2Chips[0]
    core0 = chip.n2Cores[0]

    mon = board.monitor
    u0p = mon.probe(core0.cxState, [0], 'u')
    v0p = mon.probe(core0.cxState, [0], 'v')
    tsteps = 10

    sgen = BasicSpikeGenerator(board)
    sgen.addSpike(1, chip.id, core0.id, axonId=0)

    board.run(tsteps)

    print("Cx[%d] U: %s" % (0, u0p[0].timeSeries.data))
    # print("Cx[%d] V: %s" % (0, v0p[0].timeSeries.data))

    y = WGT * 2**(6 + WGT_EXP)
    ys = [0]
    for i in range(1, tsteps):
        ys.append(y)
        y = decay_int(y, DECAY_U, b=1)
    print("Predicted: %s" % (ys,))

    y = WGT * 2**(6 + WGT_EXP)
    ys = [0]
    for i in range(1, tsteps):
        ys.append(y)
        y = decay_int(y, DECAY_U, b=0)
    print("Desired: %s" % (ys,))
