"""Exercise exact video replay, irregular capture times and input/output matching."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import av
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from interactive_rgb_scanning.recording import ScanRecording


class ScanRecordingTests(unittest.TestCase):
    def test_lossless_rgb_timing_and_selected_input_correspondence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            camera = dict(
                outgoing=dict(width=80, height=64), settings=dict(frameRate=15)
            )
            recorder = ScanRecording(root, camera, {})
            frames = np.random.default_rng(6).integers(
                0, 256, (3, 64, 80, 3), dtype=np.uint8
            )
            for seq, elapsed, frame in zip([10, 12, 17], [0.02, 0.17, 0.91], frames):
                recorder.camera_frame(
                    frame.tobytes(), seq, recorder.started + elapsed, 1000 * seq
                )
            recorder.processed_frame(17)
            recorder.feedback_frame(
                frames[2].tobytes(),
                dict(
                    input_seq=17, render_input_seq=17, output_kind="observation_overlay"
                ),
            )
            recorder.close()
            with av.open(str(root / "recording/camera-lossless.mkv")) as container:
                decoded = list(container.decode(video=0))
            np.testing.assert_array_equal(
                np.stack([f.to_ndarray(format="rgb24") for f in decoded]), frames
            )
            np.testing.assert_allclose(
                [float(f.pts * f.time_base) for f in decoded],
                [0.02, 0.17, 0.91],
                atol=0.001,
            )
            rows = [
                json.loads(s)
                for s in (root / "recording/camera-frames.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual([r["input_seq"] for r in rows], [10, 12, 17])
            selected = json.loads(
                (root / "recording/processed-inputs.jsonl").read_text()
            )
            self.assertEqual(
                (selected["input_seq"], selected["camera_video_index"]), (17, 2)
            )
            returned = json.loads(
                (root / "recording/feedback-frames.jsonl").read_text()
            )
            self.assertEqual(
                (returned["input_seq"], returned["render_input_seq"]), (17, 17)
            )
            manifest = json.loads((root / "recording/manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["encoded_camera_frames"], 3)
            with av.open(str(root / "recording/camera.mp4")) as container:
                self.assertEqual(len(list(container.decode(video=0))), 3)

    def test_empty_scan_closes_recorders_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            camera = dict(
                outgoing=dict(width=80, height=64), settings=dict(frameRate=15)
            )
            recorder = ScanRecording(temporary, camera, {})
            recorder.close()
            recorder.close()
            self.assertEqual(recorder.status()["status"], "complete")
            self.assertEqual(recorder.status()["camera_frames"], 0)


if __name__ == "__main__":
    unittest.main()
