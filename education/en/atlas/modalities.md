## Sensor modalities

| Modality | What is physically measured | Typical defects revealed | Caveats |
|---|---|---|---|
| **RGB / grey camera** | Light reflected by the surface (depends on illumination, material, viewing angle) | Scratches, stains, discoloration, print errors, missing parts | Lighting changes alter appearance; shiny or transparent materials are hard |
| **Infrared — lock-in thermography** (MulSen-AD) | Surface temperature response to periodically modulated heating; phase/amplitude images reveal how heat diffuses | Sub-surface cracks, voids, delamination, internal breaks | Stored images are usually *normalised visualisations*, **not calibrated temperatures**; lower resolution |
| **3D laser / structured-light scanning** | Geometry: 3D coordinates of surface points (mm), sometimes meshes | Dents, bulges, deformation, missing material, height defects | Needs registration; units, sensor noise and occlusions matter |
| **Photometric-stereo pseudo-3D** (Real-IAD D³) | Surface normals/shading recovered from images under several light directions | Fine relief (scratches, embossing) invisible in a single photo | Not true metric depth; assumptions about reflectance |
| **Depth images (xyz TIFF)** (3D-ADAM, MVTec 3D-AD) | Per-pixel 3D coordinates aligned with an image | Geometric defects with image alignment | Missing values, sensor-specific noise |

## Anomaly types

- **Structural / surface** — a local change of appearance: scratch, crack, stain, contamination.
- **Logical** — every part looks fine, but the *configuration* is wrong: missing, extra,
  misplaced or mismatched components (MVTec LOCO AD).
- **Geometric** — the shape deviates: dents, bulges, sinks, deformation (Real3D-AD, MulSen-AD 3D).
- **Internal** — defects below the surface, visible to thermography or X-ray but not to cameras
  (MulSen-AD infrared).

A single product can exhibit all four. A method designed for one type may be blind to another:
PatchCore excels at structural anomalies but struggles with logical ones; RGB-only methods cannot
see internal defects.
