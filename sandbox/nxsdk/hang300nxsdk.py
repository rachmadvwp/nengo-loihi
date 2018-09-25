from nxsdk.arch.n2a.graph.graph import N2Board
from nxsdk.arch.n2a.graph.inputgen import BasicSpikeGenerator

#n = 301
n = 300
#n = 299

boardId = 1
numChips = 1
numCoresPerChip = [1]
numSynapsesPerCore = [[2*n]]
board = N2Board(boardId,numChips,numCoresPerChip,numSynapsesPerCore)
core = board.n2Chips[0].n2Cores[0]

for i in range(n):
    core.cxCfg[i].configure(
        bias=0, biasExp=0, vthProfile=0, cxProfile=0)

s0 = 0
for a in range(2):
    for i in range(n):
        core.synapses[s0 + i].configure(CIdx=0, Wgt=0, synFmtId=1)

    core.synapseMap[a].synapsePtr = s0
    core.synapseMap[a].synapseLen = n
    core.synapseMap[a].discreteMapEntry.configure()
    s0 += n

core.synapseFmt[1].wgtBits = 7
core.synapseFmt[1].numSynapses = 63
core.synapseFmt[1].idxBits = 4
core.synapseFmt[1].compression = 3
core.synapseFmt[1].fanoutType = 1

spikegen = BasicSpikeGenerator(board)
spikegen.addSpike(0, 0, core.id, 0)
spikegen.addSpike(1, 0, core.id, 0)

board.run(100)
