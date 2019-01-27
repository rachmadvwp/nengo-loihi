from nengo.exceptions import ValidationError
import numpy as np

from nengo_loihi.discretize import (
    tracing_mag_int_frac,
    vth_to_manexp,
)
from nengo_loihi.hardware.nxsdk_objects import (
    Board,
    CxProfile,
    TraceCfg,
    VthProfile,
)
from nengo_loihi.hardware.validate import validate_board


def compute_profiles(core, list_profiles):
    profile_lists = []
    for segment in core.segments:
        profile_lists.append(list_profiles(segment))

    profiles = list(set(p for plist in profile_lists for p in plist))
    profile_idxs = {}
    for segment, plist in zip(core.segments, profile_lists):
        profile_idxs[segment] = np.zeros(len(plist), dtype=int)
        for k, profile in enumerate(profiles):
            profile_idxs[segment][[p == profile for p in plist]] = k

    return profiles, profile_idxs


def core_cx_profiles(core):
    """Compute all cxProfiles needed for a core"""
    def list_cx_profiles(segment):
        profiles = []
        for i in range(segment.compartments.n_compartments):
            profiles.append(CxProfile(
                decayU=segment.compartments.decayU[i],
                decayV=segment.compartments.decayV[i],
                refractDelay=segment.compartments.refractDelay[i],
                enableNoise=segment.compartments.enableNoise[i],
            ))

        return profiles

    return compute_profiles(core, list_cx_profiles)


def core_vth_profiles(core):
    """Compute all vthProfiles needed for a core"""
    def list_vth_profiles(segment):
        profiles = []
        vth, _ = vth_to_manexp(segment.compartments.vth)
        for i in range(segment.compartments.n_compartments):
            profiles.append(VthProfile(
                vth=vth[i],
            ))

        return profiles

    return compute_profiles(core, list_vth_profiles)


def core_stdp_pre_cfgs(core):
    profiles = []
    profile_idxs = {}
    for synapses in core.synapses:
        if synapses.learning:
            mag_int, mag_frac = tracing_mag_int_frac(synapses.tracing_mag)
            tracecfg = TraceCfg(
                tau=synapses.tracing_tau,
                spikeLevelInt=mag_int,
                spikeLevelFrac=mag_frac,
            )

            if tracecfg in profiles:
                profile_idxs[synapses] = profiles.index(tracecfg)
            else:
                profile_idxs[synapses] = len(profiles)
                profiles.append(tracecfg)
        else:
            profile_idxs[synapses] = None

    return profiles, profile_idxs


def one_to_one_allocator(model):
    board = Board()
    chip = board.new_chip()

    for segment in model.segments:
        if segment.compartments.n_compartments > 1024:
            raise ValidationError("Segment does not fit on one chip",
                                  "n_neurons")

        core = chip.new_core()
        core.add_segment(segment)

        cx_profiles, cx_profile_idxs = core_cx_profiles(core)
        [core.add_cx_profile(cx_profile) for cx_profile in cx_profiles]
        core.cx_profile_idxs = cx_profile_idxs

        vth_profiles, vth_profile_idxs = core_vth_profiles(core)
        [core.add_vth_profile(vth_profile) for vth_profile in vth_profiles]
        core.vth_profile_idxs = vth_profile_idxs

        for syn in segment.synapses:
            core.add_synapses(syn)

        for axons in segment.axons:
            core.add_axons(axons)

        stdp_pre_cfgs, stdp_pre_cfg_idxs = core_stdp_pre_cfgs(core)
        [core.add_stdp_pre_cfg(stdp_pre_cfg) for stdp_pre_cfg in stdp_pre_cfgs]
        core.stdp_pre_cfg_idxs = stdp_pre_cfg_idxs

        core.stdp_pre_profile_idx = None  # hardware.builder will set
        core.stdp_profile_idx = None  # hardware.builder will set

    for input in model.inputs:
        # TODO: how to allocate inputs?
        core = chip.new_core()
        core.add_input(input)
        for axons in input.axons:
            core.add_axons(axons)

    validate_board(board)
    return board
