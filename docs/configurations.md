# Experiment and deployment configurations

`paper.json` defines the controlled-comparison reconstruction and feedback
settings. Coverage uses valid native points and adjacent native triangles with
confidence above 1.5 and relative depth jump at most 0.05. Local quality uses
native confidence below 3.2 and connected regions of at least 20 assessment pixels.

Surface-history cells span 16 native pixels per side. Strong highlighting
requires two consecutive qualifying observations. Each observation window spans
at least two distinct processed images and two seconds; three completed windows
establish the recorded acquisition status. Geometric revisions and incompatible
surface associations invalidate the corresponding history.

The controlled comparisons use native objectives with diagonal regularization
`lm=0.0001`, `ep=0.1` in the calibrated update. The CUDA source in `csrc/gn_kernels.cu`
implements this solve. Numerical status is retained with the output.

`office.json` uses the same structured coverage and reminder mechanism, with
the current depth-compatible observation evidence in `surface_evidence.py`.
The primary calibrated solve uses native numerical settings, with the regularized
solve as a retry on reported failure. Its source is `csrc/gn_kernels_native.cu`.
The profile permits a longer smartphone session, a 256-keyframe buffer and
up to six processed inputs per second. Camera transport and processing rates
are recorded separately.

The image-sequence command's `--fps` defines capture timestamps for files without
timestamp sidecars. Use the source sequence's acquisition frequency. The
smartphone path uses the receiver timestamps of the actual input frames.

