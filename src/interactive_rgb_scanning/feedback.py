"""Observation histories and visual states for reconstructed surface cells."""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class _Surface:
    acknowledged: bool = False
    red_streak: int = 0
    windows: int = 0
    window_start: float | None = None
    samples: int = 0
    last_seen_seq: int = -1


class ScanReminder:
    """Track acquisition progress for native reconstructed surfaces."""

    def __init__(self, native_shape, config):
        self.shape = tuple(map(int, native_shape))
        self.cfg = dict(config)
        self.cell = int(config["native_cell_size"])
        self.nx = (self.shape[1] + self.cell - 1) // self.cell
        self.ny = (self.shape[0] + self.cell - 1) // self.cell
        self.cells_per_anchor = self.nx * self.ny
        self.surfaces = {}
        self.last_seq = None
        self.last_time = None

    def cell_ids(self, owner):
        owner = np.asarray(owner, dtype=np.int64)
        pixels = self.shape[0] * self.shape[1]
        native = owner % pixels
        anchor = owner // pixels
        ids = (
            anchor * self.cells_per_anchor
            + (native // self.shape[1] // self.cell) * self.nx
            + (native % self.shape[1] // self.cell)
        )
        return np.where(owner >= 0, ids, -1)

    def acknowledge(self, ids):
        for i in np.unique(np.asarray(ids, dtype=np.int64)):
            if i >= 0:
                self.surfaces.setdefault(int(i), _Surface()).acknowledged = True

    def invalidate(self, ids):
        for i in np.asarray(ids, dtype=np.int64):
            self.surfaces.pop(int(i), None)

    def acknowledged_ids(self):
        return np.fromiter(
            (i for i, s in self.surfaces.items() if s.acknowledged), dtype=np.int64
        )

    def update(
        self,
        *,
        frame_seq,
        capture_time,
        tracking_valid,
        raw_blue,
        raw_red,
        native_owner,
        render_seq=None,
        has_new_observation=True,
    ):
        blue = np.asarray(raw_blue, dtype=bool)
        red = np.asarray(raw_red, dtype=bool)
        owner = np.asarray(native_owner, dtype=np.int64)
        if blue.shape != red.shape or blue.shape != owner.shape:
            raise ValueError("Reminder input shapes differ")
        if (blue & red).any():
            raise ValueError("Raw red and blue overlap")
        if (red & (owner < 0)).any():
            raise ValueError("A red native surface needs an owner")
        seq = int(frame_seq)
        t = float(capture_time)
        if not np.isfinite(t):
            raise ValueError("Invalid capture timestamp")
        if self.last_seq is not None and seq < self.last_seq:
            raise ValueError("Out-of-order reminder input")
        fresh = bool(has_new_observation) and seq != self.last_seq
        if fresh and self.last_time is not None and t < self.last_time:
            raise ValueError("Capture time moved backwards")
        aligned = render_seq is None or int(render_seq) == seq
        valid = bool(tracking_valid) and aligned
        normal = np.zeros_like(red)
        light = np.zeros_like(red)
        cells = self.cell_ids(owner)
        if valid:
            visible_ids = np.unique(cells[cells >= 0])
            red_ids = set(np.unique(cells[red]).tolist())
            previous = self.last_seq
            for i in visible_ids:
                i = int(i)
                s = self.surfaces.get(i)
                bad = i in red_ids
                if s is None:
                    if not bad:
                        continue
                    s = self.surfaces.setdefault(i, _Surface())
                if fresh:
                    continuous = s.last_seen_seq == previous
                    if not continuous:
                        s.window_start = None
                        s.samples = 0
                        s.red_streak = 0
                    if bad:
                        s.red_streak += 1
                        if s.window_start is None:
                            s.window_start = t
                            s.samples = 0
                        s.samples += 1
                        if s.samples >= 2 and t - s.window_start >= float(
                            self.cfg["observation_window_seconds"]
                        ):
                            s.windows += 1
                            s.window_start = t
                            s.samples = 1
                        if s.windows >= int(self.cfg["windows_before_light"]):
                            s.acknowledged = True
                    else:
                        s.red_streak = 0
                        s.window_start = None
                        s.samples = 0
                    s.last_seen_seq = seq
            ack = np.fromiter(
                (i for i in red_ids if self.surfaces[i].acknowledged), dtype=np.int64
            )
            strong = np.fromiter(
                (
                    i
                    for i in red_ids
                    if not self.surfaces[i].acknowledged
                    and self.surfaces[i].red_streak
                    >= int(self.cfg["confirmation_updates"])
                ),
                dtype=np.int64,
            )
            light = red & np.isin(cells, ack)
            normal = red & np.isin(cells, strong)
        if fresh:
            self.last_seq = seq
            self.last_time = t
        if not valid:
            status = (
                "Tracking or frame synchronization is unavailable; feedback is paused."
            )
        elif light.any():
            status = "Recorded observations are acknowledged; light outlines retain unresolved regions."
        elif normal.any():
            status = "Acquire overlapping views of the highlighted regions."
        elif blue.any():
            status = "Blue regions indicate missing model coverage."
        else:
            status = None
        return dict(
            normal_blue=blue.copy() if valid else np.zeros_like(blue),
            normal_red=normal,
            light_red=light,
            status_text=status,
            reminder_statistics=dict(
                raw_red_fraction=float(red.mean()),
                raw_blue_fraction=float(blue.mean()),
                normal_red_fraction=float(normal.mean()),
                light_red_fraction=float(light.mean()),
                remembered_surface_cells=len(self.surfaces),
                acknowledged_surface_cells=sum(
                    s.acknowledged for s in self.surfaces.values()
                ),
                new_processed_observation=fresh,
                display_pose_valid=valid,
            ),
        )


def compose_reminders(rgb, display, config):
    """Use the original blue hatch; replace acknowledged red fill with an outline."""
    import cv2

    rgb = np.asarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    resize = lambda m: cv2.resize(
        np.asarray(m, dtype=np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    red = resize(display["normal_red"])
    blue = resize(display["normal_blue"])
    light = resize(display["light_red"])
    out = rgb.copy()
    a = float(config["normal_opacity"])
    out[red] = np.rint((1 - a) * rgb[red] + a * np.array([245, 72, 67])).astype("u1")
    yy, xx = np.indices((h, w))
    hatch = blue & ((xx + yy) % 16 < 3)
    out[hatch] = np.rint(0.55 * rgb[hatch] + 0.45 * np.array([56, 154, 220])).astype(
        "u1"
    )
    contours, _ = cv2.findContours(
        light.astype("u1"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    outline = np.zeros((h, w), np.uint8)
    cv2.drawContours(outline, contours, -1, 1, 1)
    faint = (outline > 0) & ~red & ~hatch
    out[faint] = np.rint(0.65 * rgb[faint] + 0.35 * np.array([225, 145, 135])).astype(
        "u1"
    )
    return out
