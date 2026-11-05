#!/usr/bin/env python3
"""Read the published scene-level measurements and report group means."""
import csv
import json
from pathlib import Path
from statistics import mean

root = Path(__file__).resolve().parents[1] / "results"
with (root / "acquisition.csv").open() as stream:
    rows = list(csv.DictReader(stream))
for group in ["development", "evaluation"]:
    scenes = [
        r for r in rows if r["group"] == group and not r["scene"].endswith("mean")
    ]
    values = {
        key: mean(float(r[key]) for r in scenes)
        for key in [
            "spatial_completeness_pct",
            "randomized_completeness_pct",
            "gain_pp",
            "spatial_error_cm",
            "randomized_error_cm",
        ]
    }
    print(json.dumps(dict(group=group, scenes=len(scenes), **values), indent=2))
