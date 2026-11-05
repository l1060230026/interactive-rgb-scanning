import unittest
import numpy as np
from interactive_rgb_scanning.metrics import evaluate_states, clearance, metrics


class ColorMetrics(unittest.TestCase):
    def test_unknown_is_scored_against_truth_not_declared_bad(self):
        red = np.array([1, 0, 0, 0], bool)
        defined = np.array([1, 0, 1, 0], bool)
        bad = np.array([1, 0, 1, 1], bool)
        valid = np.array([1, 1, 1, 0], bool)
        r = evaluate_states(red, defined, bad, valid, metrics)
        self.assertEqual(r["conditions"]["red_or_blue"]["tp"], 1)
        self.assertEqual(r["conditions"]["red_or_blue"]["fp"], 1)
        self.assertEqual(r["conditions"]["red_or_blue"]["fn"], 1)
        self.assertEqual(r["conditions"]["blue_unknown"]["precision"], 0.0)
        self.assertEqual(r["uncoloured_deficiency_rate"], 1.0)

    def test_all_blue_gets_prevalence_precision_and_zero_clear_coverage(self):
        r = evaluate_states(
            np.zeros(4, bool),
            np.zeros(4, bool),
            np.array([1, 0, 0, 0], bool),
            np.ones(4, bool),
            metrics,
        )
        self.assertEqual(r["conditions"]["red_or_blue"]["precision"], 0.25)
        self.assertEqual(r["conditions"]["red_or_blue"]["recall"], 1.0)
        self.assertEqual(r["uncoloured_coverage"], 0.0)
        self.assertIsNone(r["uncoloured_deficiency_rate"])

    def test_red_to_blue_never_counts_as_clear(self):
        previous = {
            k: np.array(v, bool)
            for k, v in dict(
                red=[1, 1, 0], defined=[1, 1, 0], bad=[1, 1, 1], valid=[1, 1, 1]
            ).items()
        }
        current = {
            k: np.array(v, bool)
            for k, v in dict(
                red=[0, 0, 0], defined=[0, 1, 1], bad=[1, 1, 0], valid=[1, 1, 1]
            ).items()
        }
        r = clearance(previous, current)
        self.assertEqual(r["cleared_pixels"], 2)
        self.assertEqual(r["still_deficient_when_cleared"], 1)
        self.assertEqual(r["false_clearance_rate"], 0.5)
        self.assertEqual(r["red_to_blue_excluded_from_clearance"], 1)

    def test_absent_denominator_remains_undefined(self):
        x = np.zeros(3, bool)
        r = evaluate_states(x, np.ones(3, bool), x, np.ones(3, bool), metrics)
        self.assertIsNone(r["conditions"]["red"]["precision"])
        self.assertIsNone(r["conditions"]["red"]["recall"])
        p = dict(red=x, defined=np.ones(3, bool), bad=x, valid=np.ones(3, bool))
        self.assertIsNone(clearance(p, p)["false_clearance_rate"])


if __name__ == "__main__":
    unittest.main()
