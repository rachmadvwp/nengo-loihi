from nengo import Node

from nengo_loihi.builder.builder import Builder
from nengo_loihi.io_objects import (
    ChipReceiveNode,
    SpikeInput,
)


@Builder.register(Node)
def build_node(model, node):
    if isinstance(node, ChipReceiveNode):
        spike_input = SpikeInput(node.raw_dimensions, label=node.label)
        model.add_input(spike_input)
        model.objs[node]['out'] = spike_input
        node.cx_spike_input = spike_input
    else:
        raise NotImplementedError()
