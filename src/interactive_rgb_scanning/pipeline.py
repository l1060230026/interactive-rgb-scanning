"""RGB reconstruction, spatial assessment and observation-history updates."""

import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np


class ScanningPipeline:
    def __init__(
        self,
        upstream,
        checkpoints,
        configuration,
        image_size,
        K,
        output,
        *,
        solver_build=None,
    ):
        import torch
        from .backend.runtime import Mast3rRuntime
        from .backend.solver import RegularizedSolver, load_solver
        from .overlay import NativeReminderOverlay

        self.config = dict(configuration)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        upstream = Path(upstream).resolve()
        commit = (
            subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"])
            .decode()
            .strip()
        )
        if commit != configuration["upstream_commit"]:
            raise ValueError(
                "Check out the MASt3R-SLAM revision specified in the selected configuration."
            )
        random.seed(configuration["seed"])
        np.random.seed(configuration["seed"])
        torch.manual_seed(configuration["seed"])
        if (configuration["solver"]["lm"], configuration["solver"]["ep"]) != (
            0.0001,
            0.1,
        ):
            raise ValueError(
                "The supplied regularized kernel implements lm=0.0001 and ep=0.1."
            )
        regularized = load_solver(upstream, solver_build)
        primary = (
            regularized
            if configuration["solver"]["primary"] == "regularized"
            else load_solver(upstream, solver_build, regularized=False)
        )
        runtime = Mast3rRuntime(
            upstream,
            Path(checkpoints),
            upstream / configuration["native_config"],
            *image_size,
            [K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
            buffer=configuration["buffer"],
            cpu_threads=configuration["cpu_threads"],
            max_allocated_gib=configuration["max_allocated_gib"],
        )
        self.runtime = runtime
        self.solver = RegularizedSolver(runtime, regularized, primary)
        feedback = dict(
            configuration["feedback"], blue_projection=configuration["coverage"]
        )
        if configuration.get("surface_evidence"):
            feedback["surface_evidence"] = configuration["surface_evidence"]
        self.feedback = NativeReminderOverlay(runtime, feedback)
        self.events = (self.output / "frames.jsonl").open("w")
        self.last_overlay = self.last_rgb = None
        (self.output / "configuration.json").write_text(
            json.dumps(configuration, indent=2) + "\n"
        )

    def process(self, rgb, sequence, capture_time):
        import torch

        start = time.perf_counter()
        with torch.inference_mode():
            row = self.runtime.consume(rgb, int(sequence))
            row["capture_time"] = float(capture_time)
            native_rgb, overlay, details = self.feedback.process(row)
        torch.cuda.synchronize()
        row.update(
            details,
            input_seq=int(sequence),
            render_input_seq=int(sequence),
            slam_seconds=row["processing_seconds"],
            processing_seconds=time.perf_counter() - start,
        )
        self.last_rgb, self.last_overlay = native_rgb, overlay
        self.events.write(json.dumps(row, allow_nan=False) + "\n")
        self.events.flush()
        return row, native_rgb, overlay

    def export(self):
        import cv2

        if not self.runtime.frames:
            return
        self.runtime.export(self.output, 1.5)
        self.feedback.export(self.output)
        cv2.imwrite(
            str(self.output / "last-rgb.png"),
            cv2.cvtColor(self.last_rgb, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            str(self.output / "last-feedback.png"),
            cv2.cvtColor(self.last_overlay, cv2.COLOR_RGB2BGR),
        )
        (self.output / "solver-status.json").write_text(
            json.dumps(self.solver.records, indent=2) + "\n"
        )

    def close(self):
        self.events.close()
        self.solver.close()
        self.runtime.close()
