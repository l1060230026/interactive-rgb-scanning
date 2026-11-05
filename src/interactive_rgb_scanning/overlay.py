"""Spatially registered overlays and surface-associated observation updates."""

import time
import numpy as np
import torch
import cv2
from .assessment import native_cues_from_view


def fit_display(image, size):
    """Letterbox the native crop, preserving the RGB / overlay pixel geometry."""
    width, height = size
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    target = (round(w * scale), round(h * scale))
    result = np.zeros((height, width, 3), dtype=np.uint8)
    x, y = (width - target[0]) // 2, (height - target[1]) // 2
    result[y : y + target[1], x : x + target[0]] = cv2.resize(image, target)
    return result


class NativeReminderOverlay:
    """Current native surfaces, observation evidence and lightweight reminders."""

    def __init__(self, runtime, config):
        from types import SimpleNamespace
        from interactive_rgb_scanning.feedback import ScanReminder
        from interactive_rgb_scanning.geometry import view_from_runtime

        self.runtime = runtime
        self.config = dict(config)
        self.view_from_runtime = view_from_runtime
        self.surface_evidence_options = config.get("surface_evidence")
        self.last_evidence = self.last_view = None
        self.surface_projector = None
        surface_config = config.get("blue_projection", {})
        if surface_config.get("mode") == "structured_surface":
            from interactive_rgb_scanning.projection import SurfaceProjector

            self.surface_projector = SurfaceProjector(
                **{k: v for k, v in surface_config.items() if k != "mode"}
            )
        self.reminder = ScanReminder((runtime.h, runtime.w), config)
        from interactive_rgb_scanning.geometry import make_rays

        K = runtime.K.clone()
        K[:2] /= 2
        K[0, 2] -= 0.25
        K[1, 2] -= 0.25
        view = SimpleNamespace(
            K=K,
            size=(runtime.w // 2, runtime.h // 2),
            native_rays=make_rays(runtime.K, (runtime.w, runtime.h)),
        )
        self.K = view.K.cpu().numpy()
        self.size = view.size
        self.scorer = SimpleNamespace(
            provenance={
                "method": "native C/N<3.2 and projection holes",
                "projection": "FixedMap.project",
                "states": "evaluate_mast3r_score_comparison.states",
                "reminder": "display-only processed-observation windows",
            }
        )
        if self.surface_projector is not None:
            self.scorer.provenance["blue_projection"] = (
                self.surface_projector.provenance
            )
        if self.surface_evidence_options:
            self.scorer.provenance.update(
                method="native surfaces with directional observation evidence",
                surface_selection="Two separated contradicting views without two separated supporting views",
                confidence_update="Newest depth-compatible native observation; increases and decreases retained",
                unresolved_display="Light outline for retained surfaces with conflicting observations",
                configuration=self.surface_evidence_options,
            )
        self.updates = 0
        self.last_packet = self.last_display = None
        self.last_red = self.last_defined = None
        self.geometry_refs = {}
        self.invalidated = 0
        count = runtime.h * runtime.w
        index = torch.arange(count, device="cuda")
        self.tile = (index // runtime.w // self.reminder.cell) * self.reminder.nx + (
            index % runtime.w // self.reminder.cell
        )
        self.geometry = SimpleNamespace(
            rt=runtime,
            native_rays=view.native_rays,
            K=view.K,
            size=view.size,
            projection=None,
            projection_pose=None,
            projection_poses=None,
            projection_revisions=None,
        )
        from interactive_rgb_scanning.geometry import SurfaceGeometry

        self.geometry._native_batch = lambda *args: SurfaceGeometry._native_batch(
            self.geometry, *args
        )
        self.project_native = lambda pose, poses: SurfaceGeometry.project_native(
            self.geometry, pose, poses
        )

    def _geometry_changes(self):
        import torch.nn.functional as F

        rt = self.runtime
        cell = self.reminder.cell
        revisions = rt.keyframes.N_updates[: len(rt.keyframes)].cpu().tolist()
        for i, revision in enumerate(revisions):
            previous = self.geometry_refs.get(i)
            if previous is not None and previous[0] == revision:
                continue
            z = rt.keyframes.X[i, :, 2].reshape(rt.h, rt.w)
            valid = torch.isfinite(z) & (z > 0)
            pad = (0, (-rt.w) % cell, 0, (-rt.h) % cell)
            sums = F.avg_pool2d(F.pad(torch.where(valid, z, 0)[None, None], pad), cell)[
                0, 0
            ]
            counts = F.avg_pool2d(F.pad(valid.float()[None, None], pad), cell)[0, 0]
            mean = (
                torch.where(counts > 0, sums / counts, float("nan"))
                .flatten()
                .cpu()
                .numpy()
            )
            if previous is not None:
                old = previous[1]
                tol = float(self.config["surface_alias_absolute_native"]) + float(
                    self.config["surface_alias_relative"]
                ) * np.abs(mean)
                changed = ~(
                    np.isfinite(old) & np.isfinite(mean) & (np.abs(old - mean) <= tol)
                )
                ids = i * self.reminder.cells_per_anchor + np.flatnonzero(changed)
                self.reminder.invalidate(ids)
                self.invalidated += len(ids)
            self.geometry_refs[i] = (revision, mean)

    def _compatible_batches(self, view, pose, depth):
        for start, stop, pixels, z, inside, _ in self.project_native(pose, view.poses):
            target = depth.flatten()[pixels]
            tolerance = float(self.config["surface_alias_absolute_native"]) + float(
                self.config["surface_alias_relative"]
            ) * z.clamp_min(0)
            good = inside & torch.isfinite(target) & ((z - target).abs() <= tolerance)
            cells = (
                torch.arange(start, stop, device="cuda")[:, None]
                * self.reminder.cells_per_anchor
                + self.tile[None]
            )
            yield pixels, good, cells

    def _inherit(self, view, pose, packet):
        ids = self.reminder.acknowledged_ids()
        if not len(ids):
            return
        total = self.runtime.buffer * self.reminder.cells_per_anchor
        ack = torch.zeros(total, device="cuda", dtype=torch.bool)
        ack[torch.as_tensor(ids, device="cuda")] = True
        depth = torch.as_tensor(packet["depth_native"], device="cuda")
        inherited = torch.zeros(depth.numel(), device="cuda", dtype=torch.int32)
        for pixels, good, cells in self._compatible_batches(view, pose, depth):
            inherited.scatter_reduce_(
                0,
                pixels.flatten(),
                (good & ack[cells]).int().flatten(),
                reduce="amax",
                include_self=True,
            )
        mask = (inherited.reshape(view.shape) > 0).cpu().numpy() & packet["raw_red"]
        self.reminder.acknowledge(self.reminder.cell_ids(packet["native_owner"])[mask])

    def _transfer(self, view, pose, packet, display):
        if not display["light_red"].any():
            return
        mask = torch.as_tensor(display["light_red"], device="cuda").flatten()
        depth = torch.as_tensor(packet["depth_native"], device="cuda")
        found = []
        for pixels, good, cells in self._compatible_batches(view, pose, depth):
            found.append(torch.unique(cells[good & mask[pixels]]))
        if found:
            self.reminder.acknowledge(torch.unique(torch.cat(found)).cpu().numpy())

    @torch.inference_mode()
    def process(self, row):
        from interactive_rgb_scanning.feedback import compose_reminders

        rt = self.runtime
        tick = time.perf_counter()
        view = self.view_from_runtime(rt)
        evidence = None
        if self.surface_evidence_options:
            from interactive_rgb_scanning.surface_evidence import surface_evidence

            evidence = surface_evidence(
                view, rt.K, (rt.w, rt.h), self.surface_evidence_options
            )
            view.confidence = torch.where(evidence["retained"], view.confidence, 0)
            view.surface_evidence = evidence
            self.last_evidence = evidence
            self.last_view = view
        pose = rt.last_output_pose.matrix().reshape(4, 4)
        packet = native_cues_from_view(view, pose, self.surface_projector)
        packet.update(
            frame_seq=int(row["seq"]),
            capture_time=float(row.get("capture_time", time.monotonic())),
            tracking_valid=bool(row["tracking_ok"]),
            render_seq=int(row["seq"]),
        )
        self._geometry_changes()
        if row["tracking_ok"] and evidence is not None:
            unresolved = packet["geometry_unresolved"] & packet["raw_red"]
            self.reminder.acknowledge(
                self.reminder.cell_ids(packet["native_owner"])[unresolved]
            )
        if row["tracking_ok"]:
            self._inherit(view, pose, packet)
        display = self.reminder.update(
            **{
                k: packet[k]
                for k in [
                    "frame_seq",
                    "capture_time",
                    "tracking_valid",
                    "render_seq",
                    "raw_blue",
                    "raw_red",
                    "native_owner",
                ]
            }
        )
        if row["tracking_ok"]:
            self._transfer(view, pose, packet, display)
            if display["light_red"].any():
                # Read this frame's final alias memory without another observation.
                fresh = display["reminder_statistics"]["new_processed_observation"]
                display = self.reminder.update(
                    has_new_observation=False,
                    **{
                        k: packet[k]
                        for k in [
                            "frame_seq",
                            "capture_time",
                            "tracking_valid",
                            "render_seq",
                            "raw_blue",
                            "raw_red",
                            "native_owner",
                        ]
                    },
                )
                display["reminder_statistics"]["new_processed_observation"] = fresh
        rgb = (
            np.clip(rt.last_frame.uimg.cpu().numpy() * 255, 0, 255).round().astype("u1")
        )
        overlay = compose_reminders(rgb, display, self.config)
        self.updates += 1
        self.last_packet = packet
        self.last_display = display
        self.last_red = cv2.resize(
            packet["raw_red"].astype("u1"),
            (rt.w, rt.h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        self.last_defined = cv2.resize(
            packet["defined"].astype("u1"),
            (rt.w, rt.h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        details = dict(
            quality_updated=True,
            quality_updates=self.updates,
            quality_age_seconds=0.0,
            quality_next_interval_seconds=0.0,
            score_seconds=0.0,
            sampling_seconds=0.0,
            projection_composite_seconds=time.perf_counter() - tick,
            red_fraction=float(packet["raw_red"].mean()),
            unknown_fraction=float(packet["raw_blue"].mean()),
            defined_fraction=float(packet["defined"].mean()),
            pose_valid=bool(row["tracking_ok"]),
            overlay_available=bool(row["tracking_ok"]),
            feedback_representation="native_confidence_with_reminder",
            support_points=int(np.prod(view.shape)),
            points=int(
                (
                    torch.isfinite(view.depth)
                    & (view.depth > 1e-6)
                    & (view.confidence > 1.5)
                ).sum()
            ),
            hint=display["status_text"],
            geometry_invalidated_surface_cells=self.invalidated,
            **display["reminder_statistics"],
        )
        if evidence is not None:
            details.update(
                feedback_representation="native_surfaces_with_observation_evidence",
                surface_evidence_seconds=evidence["seconds"],
                surface_removed_points=evidence["removed_points"],
                surface_input_points=evidence["valid_points"],
                unresolved_geometry_fraction=float(
                    packet["geometry_unresolved"].mean()
                ),
            )
        return rgb, overlay, details

    def export(self, folder):
        import json
        from pathlib import Path

        folder = Path(folder)
        packet = self.last_packet
        display = self.last_display
        np.savez_compressed(
            folder / "last-native-cues-and-reminders.npz",
            **{
                k: packet[k]
                for k in [
                    "raw_blue",
                    "raw_red",
                    "native_owner",
                    "confidence",
                    "depth_native",
                    "K",
                ]
            },
            **{k: display[k] for k in ["normal_blue", "normal_red", "light_red"]},
        )
        (folder / "reminder-summary.json").write_text(
            json.dumps(
                dict(
                    states={
                        str(i): dict(acknowledged=s.acknowledged, windows=s.windows)
                        for i, s in self.reminder.surfaces.items()
                    },
                    quality_changed_by_reminder=False,
                    config=self.config,
                ),
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        if self.last_evidence is not None:
            fields = [
                "retained",
                "support",
                "conflict",
                "red",
                "unresolved",
                "latest_confidence",
            ]
            np.savez_compressed(
                folder / "surface-evidence.npz",
                **{k: self.last_evidence[k].cpu().numpy() for k in fields},
            )
            np.savez_compressed(
                folder / "last-surface-evidence-query.npz",
                geometry_unresolved=packet["geometry_unresolved"],
                latest_confidence=packet["latest_confidence"],
            )
            view = self.last_view
            rt = self.runtime
            points = []
            colours = []
            for i in range(len(view.depth)):
                take = self.last_evidence["retained"][i]
                local = view.native_rays[take] * view.depth[i, take, None]
                pose = view.poses[i]
                world = (
                    local[:, 0, None] * pose[:3, 0]
                    + local[:, 1, None] * pose[:3, 1]
                    + local[:, 2, None] * pose[:3, 2]
                    + pose[:3, 3]
                )
                points.append(world.cpu().numpy())
                colours.append(
                    (
                        rt.keyframes.uimg[i].numpy().reshape(-1, 3)[take.cpu().numpy()]
                        * 255
                    ).astype("u1")
                )
            (folder / "model.ply").rename(folder / "model-native.ply")
            rt.native.eval.save_ply(
                folder / "model.ply", np.concatenate(points), np.concatenate(colours)
            )
