"""Process a calibrated RGB image sequence with online spatial feedback."""

import argparse
import json
from pathlib import Path
import re


def natural_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/paper.json"))
    parser.add_argument(
        "--upstream", type=Path, default=Path("third_party/mast3r-slam")
    )
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--solver-build", type=Path)
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    import cv2
    import numpy as np
    import torch.multiprocessing as mp
    from .pipeline import ScanningPipeline

    mp.set_start_method("spawn", force=True)
    config = json.loads(args.config.read_text())
    calibration = json.loads(args.intrinsics.read_text())
    K = np.asarray(calibration["camera_matrix"], dtype=np.float32)
    width, height = map(int, calibration["image_size"])
    files = sorted(
        [
            p
            for p in args.images.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        ],
        key=natural_key,
    )
    if not files:
        parser.error("The image directory contains no PNG or JPEG images.")
    pipeline = ScanningPipeline(
        args.upstream,
        args.checkpoints,
        config,
        (width, height),
        K,
        args.output,
        solver_build=args.solver_build,
    )
    try:
        for sequence, path in enumerate(files):
            image = cv2.imread(str(path))
            if image is None or image.shape[:2] != (height, width):
                raise ValueError(f"Image size differs from calibration: {path.name}")
            if calibration.get("distortion_coefficients"):
                image = cv2.undistort(
                    image, K, np.asarray(calibration["distortion_coefficients"])
                )
            row, _, _ = pipeline.process(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB), sequence, sequence / args.fps
            )
            print(
                json.dumps(
                    {k: row[k] for k in ["input_seq", "keyframes", "tracking_ok"]}
                ),
                flush=True,
            )
            if len(pipeline.runtime.keyframes) >= config["buffer"]:
                break
        pipeline.export()
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
