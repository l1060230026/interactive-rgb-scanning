# Smartphone and workstation deployment

The media service uses LiveKit for camera and feedback transport. Provide a
LiveKit endpoint reachable from both phone and workstation, and expose the
scanner page through an HTTPS reverse proxy. Browser camera capture requires
a secure origin.

Copy `configs/livekit.example.env` to `configs/livekit.env` and supply the
endpoint, API key and API secret for your LiveKit server. This local file is
excluded from Git. The API secret remains on the workstation; browser tokens
are issued for each session.

After installing the GPU backend and the `mobile` and `recording` dependency
groups, build the model viewer as described in the main README. Start the
service from the repository root:

```bash
python scripts/serve_phone.py --credentials configs/livekit.env \
  --output runs/media --scan-output runs/scans \
  --scan-config configs/office.json --port 8775
```

The app listens on `127.0.0.1:8775`. Route the HTTPS scanner origin to this
address. The generated `runs/media/access-key` identifies the phone entry URL:
open `https://YOUR_SCANNER_HOST/join/KEY` with that value in place of `KEY`.

Use the interface's calibration mode before the first scan. Print the supplied
A4 board at its original scale, verify the 25 mm squares, and capture varied
sharp views of the planar board. A successful calibration is saved under the
media output directory. Retain the calibrated camera, 1× lens, orientation and
video dimensions during scanning.

The phone acquires images and displays returned feedback. Reconstruction and
spatial assessment run in an isolated worker. For a separate GPU environment,
pass its Python executable with `--scan-python`. `RGB_SCAN_UPSTREAM` and
`RGB_SCAN_CHECKPOINTS` can point to existing backbone and checkpoint directories.

Ending a scan saves the reconstructed point cloud and its camera records. With
recording enabled, the session also retains camera and feedback videos, their
frame indices, processed-input associations, per-frame cue states and native
model checkpoints. The workstation model page supports point-cloud inspection
and PLY export.

