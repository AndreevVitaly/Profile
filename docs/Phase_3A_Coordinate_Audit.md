# Phase 3A Coordinate Audit

## Source PFR

- `mesh.vertices[x, y]` are image pixels.
- MediaPipe `z` is detector-relative depth multiplied by image width. It is not metric depth.
- PFR canonicalization currently removes 2D translation, roll, and scale. Yaw and pitch may be diagnostic proxies rather than calibrated camera angles.
- PFR stores mesh schema, schema version, source topology, vertex count, and a deterministic semantic landmark map.
- Source PFR does not store triangle faces or calibrated camera parameters. OBJ export is therefore a vertex cloud, not a watertight surface.
- Per-vertex detector confidence and true visibility/occlusion are unavailable. Phase 3A reports observation support and robust fusion confidence without presenting them as detector confidence.

## Reconstruction Contract

- Only frames with identical topology signatures are fused. Incompatible meshes fail with an explicit remapping requirement; no silent interpolation is performed.
- Coarse normalization removes eye-centered translation, selected relative scale, and roll.
- A rigid Kabsch transform aligns stable semantic anchors. Non-rigid warping is not used.
- Fusion is vertex-wise median by default, with an optional weighted mean.
- Coordinate system: right-handed canonical relative-depth coordinates.
- Scale modes: `unit_ipd` and `unit_face_width`. Neither represents millimeters.
- Every source-frame normalization and alignment matrix is retained in the output.

## Scientific Boundary

The result is a canonical monocular pseudo-3D representation suitable for within-dataset stability experiments and standardized projections. It is not a calibrated physical 3D scan, does not provide metric depth, and does not establish biometric identity.
