from __future__ import division

import logging

from nengo.utils.stdlib import groupby
import numpy as np

from nengo_loihi.discretize import bias_to_manexp
from nengo_loihi.hardware.nxsdk_objects import (
    CX_PROFILES_MAX,
    LoihiSpikeInput,
    VTH_PROFILES_MAX,
)
from nengo_loihi.hardware.nxsdk_shim import (
    BasicSpikeGenerator,
    microcodegen_uci,
    N2Board,
    TraceCfgGen,
)
from nengo_loihi.node_builders import SpikeInput

logger = logging.getLogger(__name__)


def build_board(board):
    n_chips = board.n_chips()
    n_cores_per_chip = board.n_cores_per_chip()
    n_synapses_per_core = board.n_synapses_per_core()
    n2board = N2Board(
        board.board_id, n_chips, n_cores_per_chip, n_synapses_per_core)

    # add our own attribute for storing our spike generator
    assert not hasattr(n2board, 'global_spike_generator')
    n2board.global_spike_generator = BasicSpikeGenerator(n2board)

    # custom attr for storing SpikeInputs (filled in build_input)
    assert not hasattr(n2board, 'spike_inputs')
    n2board.spike_inputs = {}

    # build all chips
    assert len(board.chips) == len(n2board.n2Chips)
    for chip, n2chip in zip(board.chips, n2board.n2Chips):
        logger.debug("Building chip %s", chip)
        build_chip(n2chip, chip)

    return n2board


def build_chip(n2chip, chip):
    assert len(chip.cores) == len(n2chip.n2Cores)
    for core, n2core in zip(chip.cores, n2chip.n2Cores):
        logger.debug("Building core %s", core)
        build_core(n2core, core)


def build_core(n2core, core):  # noqa: C901
    assert len(core.cxProfiles) < CX_PROFILES_MAX
    assert len(core.vthProfiles) < VTH_PROFILES_MAX

    logger.debug("- Configuring cxProfiles")
    for i, cxProfile in enumerate(core.cxProfiles):
        n2core.cxProfileCfg[i].configure(
            decayV=cxProfile.decayV,
            decayU=cxProfile.decayU,
            refractDelay=cxProfile.refractDelay,
            enableNoise=cxProfile.enableNoise,
            bapAction=1,
        )

    logger.debug("- Configuring vthProfiles")
    for i, vthProfile in enumerate(core.vthProfiles):
        n2core.vthProfileCfg[i].staticCfg.configure(
            vth=vthProfile.vth,
        )

    logger.debug("- Configuring synapseFmts")
    for i, synapseFmt in enumerate(core.synapseFmts):
        if synapseFmt is None:
            continue

        n2core.synapseFmt[i].wgtLimitMant = synapseFmt.wgtLimitMant
        n2core.synapseFmt[i].wgtLimitExp = synapseFmt.wgtLimitExp
        n2core.synapseFmt[i].wgtExp = synapseFmt.wgtExp
        n2core.synapseFmt[i].discMaxWgt = synapseFmt.discMaxWgt
        n2core.synapseFmt[i].learningCfg = synapseFmt.learningCfg
        n2core.synapseFmt[i].tagBits = synapseFmt.tagBits
        n2core.synapseFmt[i].dlyBits = synapseFmt.dlyBits
        n2core.synapseFmt[i].wgtBits = synapseFmt.wgtBits
        n2core.synapseFmt[i].reuseSynData = synapseFmt.reuseSynData
        n2core.synapseFmt[i].numSynapses = synapseFmt.numSynapses
        n2core.synapseFmt[i].cIdxOffset = synapseFmt.cIdxOffset
        n2core.synapseFmt[i].cIdxMult = synapseFmt.cIdxMult
        n2core.synapseFmt[i].skipBits = synapseFmt.skipBits
        n2core.synapseFmt[i].idxBits = synapseFmt.idxBits
        n2core.synapseFmt[i].synType = synapseFmt.synType
        n2core.synapseFmt[i].fanoutType = synapseFmt.fanoutType
        n2core.synapseFmt[i].compression = synapseFmt.compression
        n2core.synapseFmt[i].stdpProfile = synapseFmt.stdpProfile
        n2core.synapseFmt[i].ignoreDly = synapseFmt.ignoreDly

    logger.debug("- Configuring stdpPreCfgs")
    for i, traceCfg in enumerate(core.stdpPreCfgs):
        tcg = TraceCfgGen()
        tc = tcg.genTraceCfg(
            tau=traceCfg.tau,
            spikeLevelInt=traceCfg.spikeLevelInt,
            spikeLevelFrac=traceCfg.spikeLevelFrac,
        )
        tc.writeToRegister(n2core.stdpPreCfg[i])

    # --- learning
    firstLearningIndex = None
    for synapses in core.iterate_synapses():
        if synapses.learning and firstLearningIndex is None:
            firstLearningIndex = core.synapse_axons[synapses][0]
            core.learning_coreid = n2core.id
            break

    numStdp = 0
    if firstLearningIndex is not None:
        for synapses in core.iterate_synapses():
            axons = np.array(core.synapse_axons[synapses])
            if synapses.learning:
                numStdp += len(axons)
                assert np.all(axons >= firstLearningIndex)
            else:
                assert np.all(axons < firstLearningIndex)

    if numStdp > 0:
        logger.debug("- Configuring PES learning")
        # add configurations tailored to PES learning
        n2core.stdpCfg.configure(
            firstLearningIndex=firstLearningIndex,
            numRewardAxons=0,
        )

        assert core.stdp_pre_profile_idx is None
        assert core.stdp_profile_idx is None
        core.stdp_pre_profile_idx = 0  # hard-code for now
        core.stdp_profile_idx = 0  # hard-code for now (also in synapse_fmt)
        n2core.stdpPreProfileCfg[0].configure(
            updateAlways=1,
            numTraces=0,
            numTraceHist=0,
            stdpProfile=0,
        )

        # stdpProfileCfg positive error
        n2core.stdpProfileCfg[0].configure(
            uCodePtr=0,
            decimateExp=0,
            numProducts=1,
            requireY=1,
            usesXepoch=1,
        )

        # Microcode for the learning rule. `u1` evaluates the learning rule
        # every 2**1 timesteps, `x1` is the pre-trace, `y1` is the post-trace,
        # and 2^-7 is the learning rate. See `help(ruleToUCode)` for more info.
        ucode = microcodegen_uci.ruleToUCode(
            ['dw = u1*x1*y1*(2^-7)'], doOptimize=False)
        assert ucode.numUCodes == 1
        n2core.stdpUcodeMem[0].word = ucode.uCodes[0]

        # stdpProfileCfg negative error
        n2core.stdpProfileCfg[1].configure(
            uCodePtr=1,
            decimateExp=0,
            numProducts=1,
            requireY=1,
            usesXepoch=1,
        )
        # use negative version of above microcode rule
        ucode = microcodegen_uci.ruleToUCode(
            ['dw = -u1*x1*y1*(2^-7)'], doOptimize=False)
        assert ucode.numUCodes == 1
        n2core.stdpUcodeMem[1].word = ucode.uCodes[0]

        tcg = TraceCfgGen()
        tc = tcg.genTraceCfg(
            tau=0,
            spikeLevelInt=0,
            spikeLevelFrac=0,
        )
        tc.writeToRegister(n2core.stdpPostCfg[0])

    # TODO: allocator should be checking that vmin, vmax are the same
    #   for all groups on a core
    n_cx = 0
    if len(core.groups) > 0:
        group0 = core.groups[0]
        vmin, vmax = group0.compartments.vmin, group0.compartments.vmax
        assert all(group.compartments.vmin == vmin for group in core.groups)
        assert all(group.compartments.vmax == vmax for group in core.groups)
        negVmLimit = np.log2(-vmin + 1)
        posVmLimit = (np.log2(vmax + 1) - 9) * 0.5
        assert int(negVmLimit) == negVmLimit
        assert int(posVmLimit) == posVmLimit

        noiseExp0 = group0.compartments.noiseExp0
        noiseMantOffset0 = group0.compartments.noiseMantOffset0
        noiseAtDendOrVm = group0.compartments.noiseAtDendOrVm
        assert all(group.compartments.noiseExp0 == noiseExp0
                   for group in core.groups)
        assert all(group.compartments.noiseMantOffset0 == noiseMantOffset0
                   for group in core.groups)
        assert all(group.compartments.noiseAtDendOrVm == noiseAtDendOrVm
                   for group in core.groups)

        n2core.dendriteSharedCfg.configure(
            posVmLimit=int(posVmLimit),
            negVmLimit=int(negVmLimit),
            noiseExp0=noiseExp0,
            noiseMantOffset0=noiseMantOffset0,
            noiseAtDendOrVm=noiseAtDendOrVm,
        )

        n2core.dendriteAccumCfg.configure(
            delayBits=3)
        # ^ DelayBits=3 allows 1024 Cxs per core

        for group, cx_idxs, ax_range in core.iterate_groups():
            build_group(n2core, core, group, cx_idxs, ax_range)
            n_cx = max(max(cx_idxs) + 1, n_cx)

    for inp, cx_idxs in core.iterate_inputs():
        build_input(n2core, core, inp, cx_idxs)

    logger.debug("- Configuring numUpdates=%d", n_cx // 4 + 1)
    n2core.numUpdates.configure(
        numUpdates=n_cx // 4 + 1,
        numStdp=numStdp,
    )

    n2core.dendriteTimeState[0].tepoch = 2
    n2core.timeState[0].tepoch = 2


def build_group(n2core, core, group, cx_idxs, ax_range):
    assert group.compartments.scaleU is False
    assert group.compartments.scaleV is False

    logger.debug("Building %s on core.id=%d", group, n2core.id)

    for i, bias in enumerate(group.compartments.bias):
        bman, bexp = bias_to_manexp(bias)
        icx = core.cx_profile_idxs[group][i]
        ivth = core.vth_profile_idxs[group][i]

        ii = cx_idxs[i]
        n2core.cxCfg[ii].configure(
            bias=bman, biasExp=bexp, vthProfile=ivth, cxProfile=icx)

        phasex = 'phase%d' % (ii % 4,)
        n2core.cxMetaState[ii // 4].configure(**{phasex: 2})

    logger.debug("- Building %d synapses", len(group.synapses))
    for synapses in group.synapses:
        build_synapses(n2core, core, group, synapses, cx_idxs)

    logger.debug("- Building %d axons", len(group.axons))
    all_axons = []  # (cx, atom, type, tchip_id, tcore_id, taxon_id)
    for axons in group.axons:
        all_axons.extend(collect_axons(n2core, core, group, axons, cx_idxs))

    build_axons(n2core, core, group, all_axons)

    logger.debug("- Building %d probes", len(group.probes))
    for probe in group.probes:
        build_probe(n2core, core, group, probe, cx_idxs)


def build_input(n2core, core, spike_input, cx_idxs):
    assert len(spike_input.axons) > 0

    for probe in spike_input.probes:
        build_probe(n2core, core, spike_input, probe, cx_idxs)

    n2board = n2core.parent.parent

    assert isinstance(spike_input, SpikeInput)
    loihi_input = LoihiSpikeInput()
    loihi_input.set_axons(core.board, n2board, spike_input)
    assert spike_input not in n2board.spike_inputs
    n2board.spike_inputs[spike_input] = loihi_input

    # add any pre-existing spikes to spikegen
    for t in spike_input.spike_times():
        spikes = spike_input.spike_idxs(t)
        for spike in loihi_input.spikes_to_loihi(t, spikes):
            assert spike.axon.atom == 0, (
                "Cannot send population spikes through spike generator")
            n2board.global_spike_generator.addSpike(
                time=spike.time, chipId=spike.axon.chip_id,
                coreId=spike.axon.core_id, axonId=spike.axon.axon_id)


def build_synapses(n2core, core, group, synapses, cx_idxs):  # noqa C901
    axon_ids = core.synapse_axons[synapses]

    synapse_fmt_idx = core.synapse_fmt_idxs[synapses]
    stdp_pre_cfg_idx = core.stdp_pre_cfg_idxs[synapses]

    atom_bits = synapses.atom_bits()
    axon_bits = synapses.axon_bits()
    atom_bits_extra = synapses.atom_bits_extra()

    target_cxs = set()
    synapse_map = {}  # map weight_idx to (ptr, pop_size, len)
    total_synapse_ptr = int(core.synapse_entries[synapses][0])
    for axon_idx, axon_id in enumerate(axon_ids):
        assert axon_id <= 2**axon_bits

        weight_idx = int(synapses.axon_weight_idx(axon_idx))
        cx_base = synapses.axon_cx_base(axon_idx)

        if weight_idx not in synapse_map:
            weights = synapses.weights[weight_idx]
            indices = synapses.indices[weight_idx]
            weights = weights // synapses.synapse_fmt.scale
            assert weights.ndim == 2
            assert weights.shape == indices.shape
            assert np.all(weights <= 255) and np.all(weights >= -256), str(
                weights)
            n_populations, n_cxs = weights.shape

            synapse_map[weight_idx] = (
                total_synapse_ptr, n_populations, n_cxs)

            for p in range(n_populations):
                for q in range(n_cxs):
                    cx_idx = cx_idxs[indices[p, q]]
                    n2core.synapses[total_synapse_ptr].configure(
                        CIdx=cx_idx,
                        Wgt=weights[p, q],
                        synFmtId=synapse_fmt_idx,
                        LrnEn=int(synapses.learning),
                    )
                    target_cxs.add(cx_idx)
                    total_synapse_ptr += 1

        synapse_ptr, n_populations, n_cxs = synapse_map[weight_idx]
        assert n_populations <= 2**atom_bits

        if cx_base is None:
            # this is a dummy axon with no weights, so set n_cxs to 0
            synapse_ptr = 0
            n_cxs = 0
            cx_base = 0
        else:
            cx_base = int(cx_base)

        assert cx_base <= 256, "Currently limited by hardware"
        n2core.synapseMap[axon_id].synapsePtr = synapse_ptr
        n2core.synapseMap[axon_id].synapseLen = n_cxs
        if synapses.pop_type == 0:  # discrete
            assert n_populations == 1
            n2core.synapseMap[axon_id].discreteMapEntry.configure(
                cxBase=cx_base)
        elif synapses.pop_type == 16:  # pop16
            n2core.synapseMap[axon_id].popSize = n_populations
            assert cx_base % 4 == 0
            n2core.synapseMap[axon_id].population16MapEntry.configure(
                cxBase=cx_base//4, atomBits=atom_bits_extra)
        elif synapses.pop_type == 32:  # pop32
            n2core.synapseMap[axon_id].popSize = n_populations
            n2core.synapseMap[axon_id].population32MapEntry.configure(
                cxBase=cx_base)
        else:
            raise ValueError("Unrecognized pop_type: %d" % (synapses.pop_type))

        if synapses.learning:
            assert core.stdp_pre_profile_idx is not None
            assert stdp_pre_cfg_idx is not None
            n2core.synapseMap[axon_id+1].singleTraceEntry.configure(
                preProfile=core.stdp_pre_profile_idx, tcs=stdp_pre_cfg_idx)

    assert total_synapse_ptr == core.synapse_entries[synapses][1], (
        "Synapse pointer did not align with precomputed synapses length")

    if synapses.learning:
        assert core.stdp_profile_idx is not None
        for target_cx in target_cxs:
            # TODO: check that no cx gets configured by multiple synapses
            n2core.stdpPostState[target_cx].configure(
                stdpProfile=core.stdp_profile_idx,
                traceProfile=3,  # TODO: why this value
            )


def collect_axons(n2core, core, group, axons, cx_ids):
    synapses = axons.target
    tchip_idx, tcore_idx, tsyn_idxs = core.board.find_synapses(synapses)
    n2board = n2core.parent.parent
    tchip_id = n2board.n2Chips[tchip_idx].id
    tcore_id = n2board.n2Chips[tchip_idx].n2Cores[tcore_idx].id

    cx_idxs = np.arange(len(cx_ids))
    spikes = axons.map_cx_spikes(cx_idxs)

    all_axons = []  # (cx, atom, type, tchip_id, tcore_id, taxon_id)
    for cx_id, spike in zip(cx_ids, spikes):
        taxon_idx = int(spike.axon_id)
        taxon_id = int(tsyn_idxs[taxon_idx])
        atom = int(spike.atom)
        n_populations = synapses.axon_populations(taxon_idx)
        all_axons.append((cx_id, atom, synapses.pop_type,
                          tchip_id, tcore_id, taxon_id))
        if synapses.pop_type == 0:  # discrete
            assert atom == 0
            assert n_populations == 1
        elif synapses.pop_type == 16 or synapses.pop_type == 32:
            assert (len(core.groups) == 0
                    or (len(core.groups) == 1 and group is core.groups[0]))
            assert len(group.probes) == 0
        else:
            raise ValueError("Unrecognized pop_type: %d" % (synapses.pop_type))

    return all_axons


def build_axons(n2core, core, group, all_axons):  # noqa C901
    if len(all_axons) == 0:
        return

    pop_type0 = all_axons[0][2]
    if pop_type0 == 0:
        for cx_id, atom, pop_type, tchip_id, tcore_id, taxon_id in all_axons:
            assert pop_type == 0, "All axons must be discrete, or none"
            assert atom == 0
            n2core.createDiscreteAxon(
                srcCxId=cx_id,
                dstChipId=tchip_id, dstCoreId=tcore_id, dstSynMapId=taxon_id)

        return
    else:
        assert all(axon[2] != 0 for axon in all_axons), (
            "All axons must be discrete, or none")

    axons_by_cx = groupby(all_axons, key=lambda x: x[0])  # group by cx_id

    axon_id = 0
    axon_map = {}
    for cx_id, cx_axons in axons_by_cx:
        if len(cx_axons) == 0:
            continue

        # cx_axon -> (cx, atom, type, tchip_id, tcore_id, taxon_id)
        assert all(cx_axon[0] == cx_id for cx_axon in cx_axons)
        atom = cx_axons[0][1]
        assert all(cx_axon[1] == atom for cx_axon in cx_axons), (
            "cx atom must be the same for all axons")

        cx_axons = sorted(cx_axons, key=lambda a: a[2:])
        key = tuple(cx_axon[2:] for cx_axon in cx_axons)
        if key not in axon_map:
            axon_id0 = axon_id
            axon_len = 0

            for cx_axon in cx_axons:
                pop_type, tchip_id, tcore_id, taxon_id = cx_axon[2:]
                if pop_type == 0:  # discrete
                    assert False, "Should have been handled in code above"
                elif pop_type == 16:  # pop16
                    n2core.axonCfg[axon_id].pop16.configure(
                        coreId=tcore_id, axonId=taxon_id)
                    axon_id += 1
                    axon_len += 1
                elif pop_type == 32:  # pop32
                    n2core.axonCfg[axon_id].pop32_0.configure(
                        coreId=tcore_id, axonId=taxon_id)
                    n2core.axonCfg[axon_id+1].pop32_1.configure()
                    axon_id += 2
                    axon_len += 2
                else:
                    raise ValueError("Unrecognized pop_type: %d" % (pop_type,))

            axon_map[key] = (axon_id0, axon_len)

        axon_ptr, axon_len = axon_map[key]
        n2core.axonMap[cx_id].configure(ptr=axon_ptr, len=axon_len, atom=atom)


def build_probe(n2core, core, group, probe, cx_idxs):
    key_map = {'current': 'u', 'voltage': 'v', 'spiked': 'spike'}
    assert probe.key in key_map, "probe key not found"
    key = key_map[probe.key]

    n2board = n2core.parent.parent
    r = cx_idxs[probe.slice]

    if probe.use_snip:
        probe.snip_info = dict(coreid=n2core.id, cxs=r, key=key)
    else:
        p = n2board.monitor.probe(n2core.cxState, r, key)
        core.board.map_probe(probe, p)
