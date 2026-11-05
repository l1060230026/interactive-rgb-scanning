# Evaluation and supplied measurements

The controlled evaluation separates spatial assessment, visual-feedback
evolution, and reconstruction after additional guided acquisition.

## Spatial assessment

Five evaluation scenes provide 240 frozen model-query pairs for coverage and
360 pairs for local quality. Coverage labels use reference-surface visibility
before the model cutoff. Quality labels identify projected model points whose
distance from the reference mesh exceeds 5 cm. Counts are pooled within each
scene, followed by equal-scene averaging of precision, recall and MCC.
`results/spatial.csv` contains scene metrics and their macro means.

## Observation-driven feedback

Direct highlighting and observation-driven feedback share the same RGB images,
models and raw cues. Fifteen 180 s trajectories provide 361 states each, including
both endpoints. Valid states before the endpoint define mean strong area,
cumulative and longest half-screen durations, and retained region-state area.
`results/feedback.csv` contains the five scene means and their equal-scene mean.

## Guided acquisition

Room0, office0 and room1 are development scenes. Room2 and office1–office4 are
evaluation scenes with fixed settings. Each initial state supports spatial
guidance and three spatially randomized controls, using seeds 101, 202 and 303.
The 180 s budget begins after common initial acquisition. All camera motion,
waiting and recovery consume this additional budget.

Randomized seeds are averaged within initial states, then initial states within
scenes. Development and evaluation means weight their scenes equally. Geometry
uses 14 evaluable initial-state pairs in the evaluation group; all 15 trajectories
are retained for the feedback comparison. The second office1 initial state has
no usable initial-only alignment correspondences.

Completeness is the percentage of the fixed task surface within 5 cm of the
aligned reconstruction. Mean error is the distance of retained, 1 cm voxel-sampled
model points to the reference mesh, expressed in centimeters. The similarity
alignment estimated from the initial scan is fixed across branches and time.

`results/acquisition.csv` contains all eight scene endpoints. `results/trajectories.csv`
contains the three metric curves at 0, 30, 60, 90, 120, 150 and 180 s. The
summary script recomputes the published scene-group means from these files.
Replica input images, scene meshes, model weights and recorded physical-room
data are obtained separately under their respective terms.

