"""Project colour-state reporting around the existing binary metric function."""

import numpy as np


def evaluate_states(red, defined, bad, valid, metrics):
    red, defined, bad, valid = [
        np.asarray(v, dtype=bool) for v in (red, defined, bad, valid)
    ]
    assert red.shape == defined.shape == bad.shape == valid.shape
    assert not np.any(red & ~defined), "Red must be a defined surface cue"
    blue = ~defined
    clear = defined & ~red
    count = int(valid.sum())
    bad_count = int((bad & valid).sum())
    masks = dict(
        red=red,
        blue_unknown=blue,
        red_or_blue=red | blue,
        all_alert=np.ones_like(red),
        no_alert=np.zeros_like(red),
    )
    results = {}
    for name, mask in masks.items():
        row = metrics(mask, bad, valid)
        denominator = 2 * row["tp"] + row["fp"] + row["fn"]
        row.update(
            f1=2 * row["tp"] / denominator if denominator else None,
            predicted_fraction=row["predicted_pixels"] / count if count else None,
        )
        results[name] = row
    colors = {}
    for name, mask in dict(
        red=red, blue_unknown=blue, defined_uncoloured=clear
    ).items():
        total = int((mask & valid).sum())
        wrong = int((mask & bad & valid).sum())
        colors[name] = dict(
            pixels=total,
            deficient_pixels=wrong,
            sufficient_pixels=total - wrong,
            fraction_of_valid=total / count if count else None,
            deficient_fraction_within_color=wrong / total if total else None,
            fraction_of_all_deficient=wrong / bad_count if bad_count else None,
        )
    assert sum(r["pixels"] for r in colors.values()) == count
    return dict(
        valid_pixels=count,
        deficient_pixels=bad_count,
        deficient_fraction=bad_count / count if count else None,
        conditions=results,
        color_counts=colors,
        uncoloured_coverage=colors["defined_uncoloured"]["fraction_of_valid"],
        uncoloured_deficiency_rate=colors["defined_uncoloured"][
            "deficient_fraction_within_color"
        ],
    )


def clearance(previous, current):
    """Fixed reference pixels only; entering unknown is never clearance."""
    for k in ("red", "defined", "bad", "valid"):
        assert previous[k].shape == current[k].shape
    valid = previous["valid"] & current["valid"]
    initial_bad_alert = (
        previous["bad"] & (previous["red"] | ~previous["defined"]) & valid
    )
    now_clear = current["defined"] & ~current["red"]
    denominator = initial_bad_alert & now_clear
    numerator = denominator & current["bad"]
    n = int(denominator.sum())
    wrong = int(numerator.sum())
    return dict(
        previous_deficient_alerted_pixels=int(initial_bad_alert.sum()),
        cleared_pixels=n,
        still_deficient_when_cleared=wrong,
        false_clearance_rate=wrong / n if n else None,
        red_to_blue_excluded_from_clearance=int(
            (previous["red"] & ~current["defined"] & valid).sum()
        ),
        initial_deficient_alerted_now_sufficient=int(
            (initial_bad_alert & ~current["bad"]).sum()
        ),
    )


def metrics(predicted, label, valid):
    p, g = predicted[valid], label[valid]
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    return dict(
        valid_pixels=len(g),
        deficient_pixels=int(g.sum()),
        predicted_pixels=int(p.sum()),
        tp=tp,
        fp=fp,
        fn=fn,
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
        iou=tp / (tp + fp + fn) if tp + fp + fn else None,
    )
