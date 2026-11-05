"""Thin frame/lifecycle adapter around the official WSL MASt3R-SLAM components.

Matching, pose solving, fusion, retrieval, relocalization and graph optimization
are upstream calls. This module supplies RGB, calibration, capacity and timings.
"""

import importlib.util
from pathlib import Path
import sys
import time


class Mast3rRuntime:
    def __init__(
        self,
        source,
        checkpoints,
        config_path,
        width,
        height,
        calibration,
        buffer=128,
        cpu_threads=4,
        max_allocated_gib=9,
    ):
        import numpy as np
        import torch
        import torch.multiprocessing as mp

        self.torch = torch
        self.np = np
        self.source = Path(source).resolve()
        sys.path.insert(0, str(self.source))
        torch.set_num_threads(cpu_threads)
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        if max_allocated_gib:
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(
                min(0.95, max_allocated_gib * 2**30 / total)
            )
        spec = importlib.util.spec_from_file_location(
            "topologybim_official_mast3r", self.source / "main.py"
        )
        self.native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.native)
        self.native.load_config(str(config_path))
        self.native.config["use_calib"] = True
        self.native.config["dataset"]["subsample"] = 1
        self.intrinsics = self.native.Intrinsics.from_calib(
            512, width, height, calibration
        )
        self.K = torch.from_numpy(self.intrinsics.K_frame).to(
            "cuda", dtype=torch.float32
        )
        # Shape is obtained from upstream resizing, including its central crop.
        probe = self.native.create_frame(
            0,
            np.zeros((height, width, 3), dtype=np.float32),
            self.native.lietorch.Sim3.Identity(1, device="cuda"),
        )
        self.h, self.w = map(int, probe.img_shape.flatten())
        self.manager = mp.get_context("spawn").Manager()
        self.buffer = buffer
        self.keyframes = self.native.SharedKeyframes(
            self.manager, self.h, self.w, buffer=buffer
        )
        self.states = self.native.SharedStates(self.manager, self.h, self.w)
        self.keyframes.set_intrinsics(self.K)
        checkpoints = Path(checkpoints)
        self.model = self.native.load_mast3r(
            str(checkpoints / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
        )
        self.tracker = self.native.FrameTracker(self.model, self.keyframes, "cuda")
        # The official WSL backend function reads these module globals.
        self.native.factor_graph = self.native.FactorGraph(
            self.model, self.keyframes, self.K, "cuda"
        )
        self.native.retrieval_database = self.native.load_retriever(
            self.model,
            str(
                checkpoints
                / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
            ),
        )
        self.frames = []
        self.last_frame = None
        self.last_output_pose = None
        self.last_valid = False
        from interactive_rgb_scanning.backend.diagnostics import NativeGraphDiagnostics

        self.graph_diagnostics = NativeGraphDiagnostics(self)
        torch.cuda.synchronize()

    def consume(self, rgb, seq):
        torch = self.torch
        n = self.native
        torch.cuda.synchronize()
        started = time.perf_counter()
        self.graph_diagnostics.seq = seq
        diagnostic_start = self.graph_diagnostics.seconds
        diagnostic_event_start = len(self.graph_diagnostics.events)
        image = self.intrinsics.remap(rgb).astype(self.np.float32) / 255.0
        pose = (
            n.lietorch.Sim3.Identity(1, device="cuda")
            if not self.frames
            else self.states.get_frame().T_WC
        )
        frame = n.create_frame(seq, image, pose, device="cuda")
        mode = self.states.get_mode()
        new_kf = False
        tracking_ok = False
        frontend_started = time.perf_counter()
        if mode == n.Mode.INIT:
            X, C = n.mast3r_inference_mono(self.model, frame)
            frame.update_pointmap(X, C)
            self.keyframes.append(frame)
            self.states.queue_global_optimization(len(self.keyframes) - 1)
            self.states.set_mode(n.Mode.TRACKING)
            self.states.set_frame(frame)
            # Matches upstream's first-frame continue, without a backend call.
            new_kf = True
        else:
            if mode == n.Mode.TRACKING:
                new_kf, _, try_reloc = self.tracker.track(frame)
                tracking_ok = not try_reloc
                if try_reloc:
                    self.states.set_mode(n.Mode.RELOC)
                self.states.set_frame(frame)
            elif mode == n.Mode.RELOC:
                X, C = n.mast3r_inference_mono(self.model, frame)
                frame.update_pointmap(X, C)
                self.states.set_frame(frame)
                self.states.queue_reloc()
            else:
                raise RuntimeError("Unexpected upstream mode")
            if new_kf:
                if len(self.keyframes) >= self.buffer:
                    raise RuntimeError("Registered keyframe capacity reached")
                self.keyframes.append(frame)
                self.states.queue_global_optimization(len(self.keyframes) - 1)
        torch.cuda.synchronize()
        frontend_seconds = time.perf_counter() - frontend_started
        anchor_before = n.lietorch.Sim3(
            self.keyframes.last_keyframe().T_WC.data.clone()
        )
        backend_mode = self.states.get_mode()
        backend_started = time.perf_counter()
        if mode != n.Mode.INIT:
            n.run_backend(self.states, self.keyframes)
        torch.cuda.synchronize()
        backend_seconds = time.perf_counter() - backend_started
        total_seconds = time.perf_counter() - started
        # Display-only pose rebase after graph corrections; never fed to tracking.
        anchor_after = self.keyframes.last_keyframe()
        output_pose = anchor_after.T_WC * anchor_before.inv() * frame.T_WC
        if backend_mode == n.Mode.RELOC and self.states.get_mode() == n.Mode.TRACKING:
            if anchor_after.frame_id == seq:
                # Hand the accepted current-frame pose and reference change
                # back to tracking as well as to the scanner interface.
                frame.T_WC = n.lietorch.Sim3(anchor_after.T_WC.data.clone())
                self.states.set_frame(frame)
                self.tracker.reset_idx_f2k()
                output_pose = n.lietorch.Sim3(frame.T_WC.data.clone())
                tracking_ok = True
        self.last_frame = frame
        self.last_output_pose = output_pose
        self.last_valid = tracking_ok
        record = dict(
            seq=seq,
            mode_before=mode.name,
            mode_after=self.states.get_mode().name,
            tracking_ok=tracking_ok,
            initial_prior=mode == n.Mode.INIT,
            new_keyframe=bool(new_kf),
            keyframes=len(self.keyframes),
            frontend_seconds=frontend_seconds,
            backend_seconds=backend_seconds,
            processing_seconds=total_seconds,
            tracking_pose_sim3=frame.T_WC.data.detach().cpu().tolist(),
            display_pose_sim3=output_pose.data.detach().cpu().tolist(),
            pending_backend_tasks=len(self.states.global_optimizer_tasks),
        )
        record["graph_diagnostic_seconds"] = (
            self.graph_diagnostics.seconds - diagnostic_start
        )
        record["graph_events"] = [
            dict(kind=e["kind"], accepted=e.get("accepted"), rejected=e.get("rejected"))
            for e in self.graph_diagnostics.events[diagnostic_event_start:]
        ]
        self.frames.append(record)
        return record

    def export(self, folder, confidence_threshold=1.5):
        import json

        folder = Path(folder)
        folder.mkdir(exist_ok=True, parents=True)
        self.native.eval.save_reconstruction(
            folder, "model.ply", self.keyframes, confidence_threshold
        )
        poses = []
        for index in range(len(self.keyframes)):
            frame = self.keyframes[index]
            poses.append(
                dict(
                    index=index,
                    input_seq=frame.frame_id,
                    sim3=frame.T_WC.data.detach().cpu().tolist(),
                )
            )
        (folder / "keyframes.json").write_text(json.dumps(poses, indent=2) + "\n")
        self.graph_diagnostics.export(folder)
        # Final native pointmap outputs permit exact surface-state inspection;
        # this is not an input-video recording or a replay of earlier states.
        count = len(self.keyframes)
        self.np.savez_compressed(
            folder / "native-keyframe-state.npz",
            depth=self.keyframes.X[:count, :, 2].cpu().numpy(),
            confidence=(
                self.keyframes.C[:count, :, 0]
                / self.keyframes.N[:count, None].clamp_min(1)
            )
            .cpu()
            .numpy(),
            sim3=self.keyframes.T_WC[:count].cpu().numpy(),
            K=self.K.cpu().numpy(),
            shape=self.np.array([self.h, self.w]),
        )
        # These are model outputs only; no reference data or input video.
        frame = self.last_frame
        self.np.savez_compressed(
            folder / "last-frame.npz",
            rgb=frame.uimg.cpu().numpy(),
            pointmap=frame.X_canon.cpu().numpy(),
            confidence=frame.get_average_conf().cpu().numpy(),
            world_to_camera=self.last_output_pose.inv().matrix().cpu().numpy(),
            K=self.K.cpu().numpy(),
            pose_valid=self.last_valid,
        )

    def close(self):
        self.states.set_mode(self.native.Mode.TERMINATED)
        self.manager.shutdown()
