"""Exercise observation-driven visual states with small synthetic masks."""

import json
from pathlib import Path

import numpy as np
from interactive_rgb_scanning.feedback import ScanReminder

configuration = json.loads(
    (Path(__file__).resolve().parents[1] / "configs/paper.json").read_text()
)
shape = (64, 64)
owner = np.arange(np.prod(shape)).reshape(shape)
red = np.zeros(shape, dtype=bool)
red[16:32, 16:32] = True
blue = np.zeros(shape, dtype=bool)
blue[:, :8] = True
memory = ScanReminder(shape, configuration["feedback"])

for sequence in range(9):
    display = memory.update(
        frame_seq=sequence,
        capture_time=float(sequence),
        tracking_valid=True,
        render_seq=sequence,
        raw_blue=blue,
        raw_red=red,
        native_owner=owner,
    )
    print(
        json.dumps(
            {
                "seconds": sequence,
                "strong_pixels": int(display["normal_red"].sum()),
                "light_region_pixels": int(display["light_red"].sum()),
            }
        )
    )
