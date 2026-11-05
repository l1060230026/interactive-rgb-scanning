"""Calibrated RGB worker for the smartphone media bridge."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from .bridge import CalibratedCamera, atomic_json, packet_bytes, read_packet

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["calibration", "config", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    wire = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    import torch.multiprocessing as mp
    from ..overlay import fit_display
    from ..pipeline import ScanningPipeline
    from ..recording import NativeMapRecording

    mp.set_start_method("spawn", force=True)
    args.output.mkdir(parents=True, exist_ok=True)
    status = dict(
        backend="mast3r-slam",
        model_kind="point_cloud",
        phase="loading_models",
        frames_processed=0,
        keyframes=0,
        points=0,
        model_saved=False,
    )
    pipeline = recording = None

    def save(**values):
        status.update(values)
        atomic_json(args.output / "worker-status.json", status)

    def send(header, pixels=b""):
        wire.write(packet_bytes(header, pixels))
        wire.flush()

    try:
        save()
        config = json.loads(args.config.read_text())
        camera = CalibratedCamera(args.calibration)
        upstream = Path(
            os.environ.get("RGB_SCAN_UPSTREAM", ROOT / "third_party/mast3r-slam")
        )
        weights = Path(os.environ.get("RGB_SCAN_CHECKPOINTS", ROOT / "checkpoints"))
        pipeline = ScanningPipeline(
            upstream,
            weights,
            config,
            camera.processed_size,
            camera.processed_K,
            args.output,
        )
        if config.get("input_video_recording"):
            recording = NativeMapRecording(
                args.output, config["recording_checkpoint_seconds"]
            )
        save(phase="waiting_frames")
        send(dict(kind="ready"))
        started = None
        while True:
            try:
                header, pixels = read_packet(sys.stdin.buffer)
            except EOFError:
                break
            if header.get("kind") == "stop":
                break
            if header.get("kind") != "frame":
                raise ValueError("Expected a camera frame.")
            if (header["width"], header["height"]) != camera.size or len(
                pixels
            ) != np.prod(camera.size) * 3:
                raise ValueError("Camera dimensions differ from the calibration.")
            rgb = np.frombuffer(pixels, np.uint8).reshape(
                camera.size[1], camera.size[0], 3
            )
            started = started or time.monotonic()
            row, _, overlay = pipeline.process(
                camera.prepare(rgb), header["seq"], header["received_at"]
            )
            reason = (
                "keyframe_capacity"
                if row["keyframes"] >= config["buffer"]
                else (
                    "duration_limit"
                    if time.monotonic() - started >= config["max_seconds"]
                    else None
                )
            )
            if recording:
                recording.record_cues(pipeline.feedback, row)
                recording.record(pipeline.runtime, pipeline.feedback, row)
            kind = (
                "observation_overlay"
                if row.get("overlay_available")
                else "observation_unknown"
            )
            summary = {
                k: v
                for k, v in row.items()
                if k
                not in (
                    "tracking_pose_sim3",
                    "display_pose_sim3",
                    "pending_backend_tasks",
                )
            }
            summary.update(
                phase="mapping" if row["tracking_ok"] else "initializing",
                frames_processed=len(pipeline.runtime.frames),
                output_kind=kind,
                stop_reason=reason,
            )
            save(**summary)
            send(
                dict(
                    kind="frame",
                    width=camera.size[0],
                    height=camera.size[1],
                    output_kind=kind,
                    input_seq=header["seq"],
                    render_input_seq=header["seq"],
                    processing_seconds=row["processing_seconds"],
                    terminal=bool(reason),
                ),
                fit_display(overlay, camera.size).tobytes(),
            )
            if reason:
                break
        save(phase="saving")
        if recording and pipeline.runtime.frames:
            recording.record(pipeline.runtime, pipeline.feedback, row, final=True)
            recording.close()
            recording = None
        pipeline.export()
        save(phase="complete", model_saved=(args.output / "model.ply").is_file())
        send(dict(kind="complete", model_saved=status["model_saved"]))
    except Exception as error:
        save(phase="failed", error=str(error))
        send(dict(kind="error", message=str(error)))
        raise
    finally:
        if recording:
            recording.close()
        if pipeline:
            pipeline.close()
        wire.close()


if __name__ == "__main__":
    main()
