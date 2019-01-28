from nengo.exceptions import BuildError
import numpy as np

from nengo_loihi.block import Synapse


def validate_model(model):
    if len(model.blocks) == 0:
        raise BuildError("No neurons marked for execution on-chip. "
                         "Please mark some ensembles as on-chip.")

    for block in model.blocks:
        validate_block(block)


def validate_block(block):
    # -- Compartments
    validate_compartments(block.compartments)

    # -- Axons
    OUT_AXONS_MAX = 4096
    n_axons = sum(a.axon_slots() for a in block.axons)
    if n_axons > OUT_AXONS_MAX:
        raise BuildError("Output axons (%d) exceeded max (%d)" % (
            n_axons, OUT_AXONS_MAX))

    for axon in block.axons:
        validate_axon(axon)

    # -- Synapses
    IN_AXONS_MAX = 4096
    n_axons = sum(s.n_axons for s in block.synapses)
    if n_axons > IN_AXONS_MAX:
        raise BuildError("Input axons (%d) exceeded max (%d)" % (
            n_axons, IN_AXONS_MAX))

    MAX_SYNAPSE_BITS = 16384*64
    synapse_bits = sum(s.bits() for s in block.synapses)
    if synapse_bits > MAX_SYNAPSE_BITS:
        raise BuildError("Total synapse bits (%d) exceeded max (%d)" % (
            synapse_bits, MAX_SYNAPSE_BITS))

    for synapses in block.synapses:
        validate_synapses(synapses)

    # -- Probes
    for probe in block.probes:
        validate_probe(probe)


def validate_compartments(comp):
    N_CX_MAX = 1024
    if comp.n_compartments > N_CX_MAX:
        raise BuildError("Number of compartments (%d) exceeded max (%d)" %
                         (comp.n_compartments, N_CX_MAX))


def validate_axon(axon):
    if isinstance(axon.target, Synapse):
        if axon.cx_atoms is not None:
            cx_idxs = np.arange(len(axon.cx_atoms))
            axon_ids = axon.map_cx_axon(cx_idxs)
            for atom, axon_id in zip(axon.cx_atoms, axon_ids):
                n_populations = axon.target.axon_populations(axon_id)
                assert 0 <= atom < n_populations


def validate_synapses(synapses):
    validate_synapse_fmt(synapses.synapse_fmt)
    if synapses.axon_cx_bases is not None:
        assert np.all(synapses.axon_cx_bases < 256), "CxBase cannot be > 256"
    if synapses.pop_type == 16:
        if synapses.axon_cx_bases is not None:
            assert np.all(synapses.axon_cx_bases % 4 == 0)


def validate_synapse_fmt(synapse_fmt):
    assert -7 <= synapse_fmt.wgtExp <= 7
    assert 0 <= synapse_fmt.tagBits < 4
    assert 0 <= synapse_fmt.dlyBits < 8
    assert 1 <= synapse_fmt.wgtBits < 8
    assert 0 <= synapse_fmt.cIdxOffset < 16
    assert 0 <= synapse_fmt.cIdxMult < 16
    assert 0 <= synapse_fmt.idxBits < 8
    assert 1 <= synapse_fmt.fanoutType < 4


def validate_probe(probe):
    pass
