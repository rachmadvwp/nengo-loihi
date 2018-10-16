import os

from nxsdk.arch.n2a.compiler.tracecfggen.tracecfggen import TraceCfgGen
from nxsdk.arch.n2a.graph.graph import N2Board
from nxsdk.arch.n2a.graph.inputgen import BasicSpikeGenerator
from nxsdk.arch.n2a.graph.probes import N2SpikeProbe

board_id = 1
n_chips = 1
cores_per_chip = [1]
synapses_per_core = [[10]]
board = N2Board(board_id, n_chips, cores_per_chip, synapses_per_core)
chip = board.n2Chips[0]
core = chip.n2Cores[0]

# --- set up network
weights = [[100, 110, 120],
           [100, 110, 120]]
n_axons = len(weights)
n_cx = len(weights[0])
numStdp = n_cx
n_errors = n_cx

core.cxProfileCfg[0].configure(decayU=2**12-1, decayV=2**12-1)
core.vthProfileCfg[0].staticCfg.configure(vth=100)

# Configure a synapseFormat
core.synapseFmt[1].wgtBits = 7
core.synapseFmt[1].numSynapses = 63
core.synapseFmt[1].idxBits = 1
core.synapseFmt[1].compression = 3
core.synapseFmt[1].fanoutType = 1

k = 0
for i, wrow in enumerate(weights):
    core.synapseMap[2*i].synapsePtr = 0
    core.synapseMap[2*i].synapseLen = n_cx
    core.synapseMap[2*i].discreteMapEntry.configure(cxBase=0)

    core.synapseMap[2*i+1].singleTraceEntry.configure(preProfile=0, tcs=0)

    for j, w in enumerate(wrow):
        core.synapses[k].configure(
            CIdx=j,
            Wgt=w,
            synFmtId=1,
            LrnEn=1)
        k += 1

for i in range(n_cx):
    core.cxCfg[i].configure(bias=0, biasExp=0)
    phasex = 'phase%d' % (i % 4,)
    core.cxMetaState[i//4].configure(**{phasex: 2})
    core.stdpPostState[i].configure(
        stdpProfile=0,
        traceProfile=3,  # TODO: why this value
    )

core.stdpCfg.configure(
    firstLearningIndex=0,
    numRewardAxons=0,
)

core.stdpPreProfileCfg[0].configure(
    updateAlways=1,
    numTraces=0,
    numTraceHist=0,
    stdpProfile=0,
)

# stdpProfileCfg positive error
core.stdpProfileCfg[0].configure(
    uCodePtr=0,
    decimateExp=0,
    numProducts=1,
    requireY=1,
    usesXepoch=1,
)
core.stdpUcodeMem[0].word = 0x00102108  # 2^-7 learn rate

# stdpProfileCfg negative error
core.stdpProfileCfg[1].configure(
    uCodePtr=1,
    decimateExp=0,
    numProducts=1,
    requireY=1,
    usesXepoch=1,
)
core.stdpUcodeMem[1].word = 0x00f02108  # 2^-7 learn rate

tcg = TraceCfgGen()
tc = tcg.genTraceCfg(
    tau=0,
    spikeLevelInt=0,
    spikeLevelFrac=0,
)
tc.writeToRegister(core.stdpPostCfg[0])

core.numUpdates.configure(
    numUpdates=n_cx // 4 + 1,
    numStdp=numStdp,
)


# --- set up snips
snips_dir = os.path.dirname(os.path.realpath(__file__))

learn_process = board.createProcess(
    name="learn_nengo",
    cFilePath=os.path.join(snips_dir, "learn_nengo.c"),
    includeDir=snips_dir,
    funcName="learn_nengo",
    guardName="guard_learn",
    phase="preLearnMgmt",
)

inputChannel = board.createChannel(b'inputChannel', "int", n_errors)
# recvChannel = board.createChannel(b'recvChannel', "int", n_errors)
inputChannel.connect(None, learn_process)
# recvChannel.connect(learn_process, None)

# --- run the simulation
spike_gen = BasicSpikeGenerator(board)
for t in (2, 5, 7):
    spike_gen.addSpike(time=t, chipId=chip.id, coreId=core.id, axonId=0)
    spike_gen.addSpike(time=t, chipId=chip.id, coreId=core.id, axonId=1)

board.startDriver()

inputChannel.write(n_errors, [0] * n_errors)

n_steps = 10
board.run(n_steps, async=True)
for i in range(n_steps):
    inputChannel.write(n_cx, [0] * n_errors)

print("Waiting for run to finish")
board.finishRun()
