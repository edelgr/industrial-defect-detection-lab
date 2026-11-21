## How 3D inspection works (educational baseline)

1. **Normal point cloud.** One or more defect-free scans are the reference. They are
   down-sampled on a voxel grid (default: 1/150 of the bounding-box diagonal) and surface normals
   are estimated.
2. **Test point cloud.** The scan to inspect, in its own coordinate frame.
3. **Alignment.** *Coarse*: FPFH local shape descriptors (Rusu et al., 2009) matched with
   RANSAC. *Fine*: ICP point-to-plane (Chen & Medioni, 1992) — or point-to-point (Besl & McKay,
   1992). Check the **fitness** (fraction of inlier points) and **RMSE**: if alignment fails, every
   deviation is meaningless.
4. **Geometric comparison.** For each aligned test point, find the nearest reference point and
   its normal.
5. **Deviation.** Point-to-surface distance `|n·(p − q)|` or point-to-point distance, in the units
   stored in the file (MulSen-AD: mm per the paper). Object score = 99th percentile.
6. **Coloured 3D visualisation.** Colour the cloud by deviation and compare with ground truth.

**Limitations.** Symmetric objects can be aligned upside-down by RANSAC; normal shape variation
appears as anomaly; partial scans reduce overlap. Published methods (Reg3D-AD, M3DM, MulSen-TripleAD)
add learned point features and memory banks on top of registration.

References: Besl & McKay 1992 · Chen & Medioni 1992 · Rusu et al. 2009 · Zhou et al. 2018 (Open3D)
— open them from the Methods Library entry *Registration + geometric deviation*.
