import json
from pathlib import Path
import unittest

import numpy as np
from interactive_rgb_scanning.feedback import ScanReminder


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.shape = (64, 64)
        self.owner = np.arange(4096).reshape(self.shape)
        self.red = np.zeros(self.shape, bool)
        self.red[16:32, 16:32] = True
        self.blue = np.zeros(self.shape, bool)
        self.blue[:, :8] = True
        config = json.loads(
            (Path(__file__).resolve().parents[1] / "configs/paper.json").read_text()
        )
        self.memory = ScanReminder(self.shape, config["feedback"])

    def update(self, sequence, timestamp=None, **values):
        return self.memory.update(
            frame_seq=sequence,
            capture_time=float(sequence if timestamp is None else timestamp),
            tracking_valid=values.pop("tracking_valid", True),
            raw_red=self.red,
            raw_blue=self.blue,
            native_owner=self.owner,
            render_seq=values.pop("render_seq", sequence),
            **values,
        )

    def test_new_observations_acknowledge_and_retain_the_region(self):
        for index in range(6):
            display = self.update(index)
        self.assertEqual(int(display["normal_red"].sum()), 256)
        display = self.update(6)
        self.assertEqual(int(display["normal_red"].sum()), 0)
        self.assertEqual(int(display["light_red"].sum()), 256)
        np.testing.assert_array_equal(
            self.red, display["normal_red"] | display["light_red"]
        )

    def test_repeated_processing_does_not_advance_observation_history(self):
        self.update(0)
        for timestamp in range(1, 30):
            display = self.update(0, timestamp)
        self.assertFalse(display["reminder_statistics"]["new_processed_observation"])
        self.assertEqual(int(display["light_red"].sum()), 0)
        self.assertEqual(len(self.memory.acknowledged_ids()), 0)

    def test_invalid_tracking_or_frame_alignment_pauses_the_overlay(self):
        self.update(0)
        self.update(1)
        for values in [dict(tracking_valid=False), dict(render_seq=99)]:
            display = self.update(2, **values)
            self.assertFalse(
                any(
                    display[key].any()
                    for key in ["normal_red", "normal_blue", "light_red"]
                )
            )

    def test_changed_surface_identity_discards_its_acquisition_state(self):
        for index in range(7):
            self.update(index)
        ids = self.memory.acknowledged_ids()
        self.assertTrue(len(ids))
        self.memory.invalidate(ids)
        self.assertEqual(len(self.memory.acknowledged_ids()), 0)
        self.assertFalse(self.update(7)["light_red"].any())


if __name__ == "__main__":
    unittest.main()
