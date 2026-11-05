#!/usr/bin/env python3
"""Register the pinned upstream Python modules in the active environment."""
from pathlib import Path
import sysconfig

root = Path(__file__).resolve().parents[1]
source = root / "third_party/mast3r-slam"
paths = [
    source,
    root / "third_party/lietorch",
    source / "thirdparty/mast3r",
    source / "thirdparty/in3d",
    source / "thirdparty/mast3r/dust3r/croco/models/curope",
]
if not all(path.is_dir() for path in paths):
    raise FileNotFoundError(
        "Clone and initialize the upstream sources before registering them."
    )
target = Path(sysconfig.get_path("purelib")) / "interactive-rgb-backbone.pth"
target.write_text("".join(str(path) + "\n" for path in paths))
print(target)
