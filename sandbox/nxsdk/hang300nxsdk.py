from nxsdk.arch.n2a.graph.graph import N2Board
from nxsdk.arch.n2a.graph.inputgen import BasicSpikeGenerator

n_synapses = 300
n_cx = 350

boardId = 1
numChips = 1
numCoresPerChip = [1]
numSynapsesPerCore = [[n_synapses]]
board = N2Board(boardId,numChips,numCoresPerChip,numSynapsesPerCore)
core = board.n2Chips[0].n2Cores[0]

core.cxProfileCfg[0].configure(decayU=0, decayV=0)
core.vthProfileCfg[0].staticCfg.configure(vth=40)

for i in range(n_cx):
    core.cxCfg[i].configure(
        bias=0, biasExp=0, vthProfile=0, cxProfile=0)

for i in range(n_synapses):
    core.synapses[i].configure(CIdx=i % n_cx, Wgt=0, synFmtId=1)

core.synapseMap[0].synapsePtr = 0
core.synapseMap[0].synapseLen = n_synapses
core.synapseMap[0].discreteMapEntry.configure()

core.synapseFmt[1].wgtBits = 7
core.synapseFmt[1].numSynapses = 63
core.synapseFmt[1].idxBits = 4
core.synapseFmt[1].compression = 3
core.synapseFmt[1].fanoutType = 1

spikegen = BasicSpikeGenerator(board)
spikegen.addSpike(0, 0, core.id, 0)
spikegen.addSpike(1, 0, core.id, 0)

board.run(100)
