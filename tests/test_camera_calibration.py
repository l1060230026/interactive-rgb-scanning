"""Recover known camera parameters; reject degenerate and mismatched captures."""

from pathlib import Path
import json
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from interactive_rgb_scanning.mobile.calibration import (
    CalibrationCapture,
    object_points,
    solve_views,
)


def synthetic_views(count=30):
    rng = np.random.default_rng(31)
    matrix = np.array([[500.0, 0.0, 239.0], [0.0, 505.0, 321.0], [0.0, 0.0, 1.0]])
    distortion = np.array([-0.07, 0.015, 0.0005, -0.0008, 0.0])
    points = object_points(25)
    result = []
    while len(result) < count:
        rotation = rng.uniform([-0.65, -0.65, -0.3], [0.65, 0.65, 0.3])
        translation = rng.uniform([-230.0, -230.0, 400.0], [50.0, 140.0, 950.0])
        projected, _ = cv2.projectPoints(
            points, rotation, translation, matrix, distortion
        )
        xy = projected.reshape(-1, 2)
        if (xy.min(0) < 15).any() or (xy.max(0) > [465, 625]).any():
            continue
        result.append(
            (projected + rng.normal(0, 0.07, projected.shape)).astype(np.float32)
        )
    return matrix, distortion, result


class CameraCalibrationTests(unittest.TestCase):
    def test_recovers_known_intrinsics_and_validates_withheld_views(self):
        expected, _, corners = synthetic_views()
        result = solve_views(corners, (480, 640), 25)
        self.assertEqual(json.loads(json.dumps(result))["status"], result["status"])
        self.assertEqual(result["status"], "passed", result["checks"])
        actual = np.asarray(result["camera_matrix"])
        np.testing.assert_allclose(
            np.diag(actual)[:2], np.diag(expected)[:2], rtol=0.015
        )
        np.testing.assert_allclose(actual[:2, 2], expected[:2, 2], atol=3.0)
        self.assertEqual(result["validation"]["withheld_views"], 6)
        self.assertLess(max(result["validation"]["per_view_rms_px"]), 0.3)

    def test_repeated_front_views_do_not_pass_diversity_checks(self):
        _, _, corners = synthetic_views(1)
        with self.assertRaisesRegex(ValueError, "位置变化"):
            solve_views(corners * 30, (480, 640), 25)

    def test_selected_images_are_detected_duplicates_rejected_and_size_changes_stop(
        self,
    ):
        with tempfile.TemporaryDirectory() as folder:
            capture = CalibrationCapture(
                folder, 25, {"outgoing": {"width": 480, "height": 640}}
            )
            rgb = np.full((640, 480, 3), 255, np.uint8)
            for row in range(7):
                for column in range(10):
                    if (row + column) % 2 == 0:
                        rgb[
                            220 + 20 * row : 240 + 20 * row,
                            140 + 20 * column : 160 + 20 * column,
                        ] = 0
            capture.process_rgb(rgb)
            self.assertEqual(len(capture.corners), 1, capture.hint)
            self.assertTrue((capture.folder / "frames/frame-000.png").exists())
            capture.process_rgb(rgb)
            self.assertEqual(len(capture.corners), 1)
            capture.process_rgb(np.full((480, 640, 3), 255, np.uint8))
            self.assertEqual(capture.phase, "invalid")

    def test_too_few_frames_cannot_produce_a_calibration(self):
        _, _, corners = synthetic_views(4)
        with self.assertRaisesRegex(ValueError, "至少需要"):
            solve_views(corners, (480, 640), 25)


if __name__ == "__main__":
    unittest.main()
