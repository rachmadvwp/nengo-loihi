from collections import defaultdict

import numpy as np


class SpikeGenerator(object):

    def __init__(self):
        self.core_ids = []
        self.axon_ids = defaultdict(list)
        self.spikes = {}
        self.bytes_per_step = None

    @property
    def tiled_core_ids(self):
        core_ids = []
        for core_id in self.core_ids:
            core_ids.extend([
                core_id for _ in range(len(self.axon_ids[core_id]))
            ])
        return core_ids

    def add_spike(self, time, chip_id, core_id, axon_id):
        # NOTE: chip_id ignored for now
        if core_id not in self.core_ids:
            self.core_ids.append(core_id)
            self.spikes[core_id] = {}
        if axon_id not in self.spikes[core_id]:
            self.axon_ids[core_id].append(axon_id)
            self.spikes[core_id][axon_id] = []
        self.spikes[core_id][axon_id].append(time)

    def write_spikes(self, path):
        # Sort spike lists, just in case, though they should be sorted
        for per_axon in self.spikes.values():
            for axon_id in per_axon:
                per_axon[axon_id] = list(sorted(per_axon[axon_id]))

        # Get the number of steps and neurons to construct rasters
        n_steps = 0
        n_neurons = 0
        for per_axon in self.spikes.values():
            for spikelist in per_axon.values():
                n_neurons += 1
                n_steps = max(n_steps, spikelist[-1])
        self.bytes_per_step = int(np.ceil(n_neurons / 8.))

        # Construct and compress the raster
        spikes = np.zeros((n_steps + 1, n_neurons), dtype=bool)
        ix = 0
        for core_id in self.core_ids:
            for axon_id in self.axon_ids[core_id]:
                spikes[:, ix][self.spikes[core_id][axon_id]] = True
                ix += 1
        spikes = np.packbits(spikes[1:], axis=-1)  # Ignore time 0
        assert spikes.shape == (n_steps, self.bytes_per_step)

        # Write to a binary file
        with open(path, "wb") as f:
            for row in spikes:
                f.write(row.tobytes())
