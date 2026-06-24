#!/usr/bin/env python
"""Convert sw_tomography.npz to HDF5 with clean observation data.

Output:
    /observations    (15661,) - slowness in s/km
    /stations        (1122, 2) - unique [lat, lon]
    /station_pairs   (15661, 2) - [sta1_idx, sta2_idx] per observation
"""

from pathlib import Path

import h5py
import numpy as np

script_dir = Path(__file__).parent
data = np.load(script_dir / "sw_tomography.npz", allow_pickle=True)

# Extract unique stations
sta1 = data["station_coords"][:, :2]
sta2 = data["station_coords"][:, 2:]
stations, inv = np.unique(np.vstack([sta1, sta2]), axis=0, return_inverse=True)
station_pairs = np.column_stack([inv[:len(sta1)], inv[len(sta1):]]).astype(np.int32)

# Write HDF5
with h5py.File(script_dir / "sw_tomography.h5", "w") as f:
    f.create_dataset("observations", data=data["slowness"])
    f.create_dataset("stations", data=stations)
    f.create_dataset("station_pairs", data=station_pairs)

print(f"observations:  {data['slowness'].shape}")
print(f"stations:      {stations.shape}")
print(f"station_pairs: {station_pairs.shape}")
