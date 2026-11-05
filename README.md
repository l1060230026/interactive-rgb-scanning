# Interactive RGB Scanning

Continuous handheld indoor reconstruction with spatially registered coverage
and quality feedback. The system connects incoming RGB images, an evolving
MASt3R-SLAM model, current-view assessment, and observation-driven visual updates.

Associated manuscript: **Online Reconstruction Deficiency Detection and
Interactive Scanning Guidance for Handheld Indoor Mapping**.

## Method

1. **Online geometry:** calibrated RGB images update native keyframe pointmaps
   and camera poses through MASt3R-SLAM.
2. **Spatial assessment:** valid adjacent surfaces complement projected points
   when identifying missing coverage. Native confidence supplies local quality
   cues on represented geometry.
3. **Observation history:** surface cells accumulate distinct processed
   observations and transition between strong and light visual emphasis.
4. **AR feedback:** the processed RGB image and its corresponding pose define
   the returned overlay. The operator adjusts the camera while scanning continues.

## Quick start

Python 3.11 is required. Install the lightweight package and exercise the
observation-history mechanism:

```bash
python -m pip install -e .
python examples/feedback_states.py
```

The example uses synthetic masks and runs on CPU. For reconstruction, install
the pinned GPU backend using [the installation instructions](docs/installation.md).

Process a calibrated RGB sequence from the repository root:

```bash
rgb-scan --images /path/to/rgb --intrinsics /path/to/intrinsics.json \
  --config configs/paper.json --output runs/sequence
```

PNG and JPEG files are read in natural filename order. Supply the camera's
calibrated image size, intrinsic matrix and distortion coefficients using the
schema in [intrinsics.example.json](configs/intrinsics.example.json). Its numbers
illustrate the format; substitute the calibration for the images being processed.

Outputs include `model.ply`, native model states, `frames.jsonl`, the final RGB
and spatial-feedback images, the selected configuration, and solver status.

## Smartphone deployment

The browser interface captures RGB and displays the corresponding processed
view. A workstation runs reconstruction and assessment. The media bridge retains
one in-flight frame and the newest pending frame, and records the input video,
returned feedback, cue states and model checkpoints.

Install the media dependencies and build the model viewer:

```bash
python -m pip install -e '.[mobile,recording]'
cd web/model-viewer
npm ci
node build.mjs
cd ../..
```

Configure LiveKit and HTTPS using [the deployment guide](docs/mobile.md). Camera
calibration is available in the phone interface. Use the accepted calibration
with the same camera, orientation and lens setting during scanning.

## Configurations

| Configuration | Purpose |
| --- | --- |
| `configs/paper.json` | Controlled comparisons: 128 keyframes, 2 Hz sampling, surface-supported coverage and native-confidence quality cues |
| `configs/office.json` | Smartphone office deployment: recorded continuous acquisition, selected native surfaces and current observation evidence |

Both profiles retain the observation-window reminder mechanism. Their camera
scheduling, surface evidence and native-solver settings follow their respective
experimental and deployment roles; details are in [the configuration guide](docs/configurations.md).

## Paper results

The supplied measurements cover three development scenes and five evaluation
scenes. Under matched 180 s additional-acquisition budgets, the five evaluation
scenes achieve mean completeness of **61.48% versus 52.13%**, a gain of **9.34
percentage points**, with mean geometric error of **5.43 versus 5.89 cm**.

```bash
python scripts/summarize_results.py
```

Scene endpoints, time curves, spatial-assessment metrics and feedback metrics
are provided as CSV files in [`results/`](results). [The evaluation guide](docs/evaluation.md)
describes the paired controls, metric domains and aggregation.

## Code organization

| Module | Function |
| --- | --- |
| `backend/` | Native reconstruction adapter, calibrated solver and graph-event records |
| `geometry.py`, `projection.py` | Native pointmap geometry and valid-surface rasterization |
| `assessment.py` | Current-view coverage and represented-surface quality masks |
| `feedback.py`, `overlay.py` | Surface-associated observation histories and registered visual updates |
| `surface_evidence.py` | Observation evidence used by the office deployment profile |
| `mobile/`, `recording.py` | Camera calibration, browser media bridge and session recording |
| `policy.py` | Shared image-region scoring and randomized spatial-score controls |
| `web/` | Phone interface and workstation model viewer |

## Tests

```bash
python -m pip install -e '.[recording]'
python -m unittest discover -s tests -v
```

The tests cover spatial annotation metrics, surface-history transitions,
camera calibration and session recording. The recording tests use the
`recording` dependency group.

## License and attribution

This research code is distributed under **CC BY-NC-SA 4.0**. MASt3R-SLAM and
MASt3R retain their upstream licenses. The calibrated CUDA solver preserves
the DROID-SLAM and MASt3R-SLAM copyright notices. Third-party components and
source revisions are listed in [THIRD-PARTY.md](THIRD-PARTY.md).

