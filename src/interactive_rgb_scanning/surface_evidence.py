"""Native-surface selection from independent directional depth observations."""

import time

import numpy as np
import torch

from interactive_rgb_scanning.geometry import projected_pixels, transform_points


@torch.inference_mode()
def surface_evidence(view, native_K, native_size, config):
    """Keep native identities while rejecting surfaces contradicted across views."""
    tick = time.perf_counter()
    valid = torch.isfinite(view.depth) & (view.depth > 1e-6) & (view.confidence > 1.5)
    ids = torch.where(valid.flatten())[0]
    per_frame = view.depth.shape[1]
    support = torch.zeros_like(view.depth, dtype=torch.int16)
    conflict = torch.zeros_like(support)
    latest_confidence = view.confidence.clone()
    inverse = torch.linalg.inv(view.poses)
    observations = torch.arange(len(view.poses), device=view.depth.device)[:, None]
    cosine_limit = float(np.cos(np.deg2rad(config["minimum_angle_degrees"])))
    for start in range(0, len(ids), 32768):
        chosen = ids[start : start + 32768]
        anchor, pixel = chosen // per_frame, chosen % per_frame
        local = view.native_rays[pixel] * view.depth[anchor, pixel, None]
        pose = view.poses[anchor]
        world = (
            local[:, 0, None] * pose[:, :3, 0]
            + local[:, 1, None] * pose[:, :3, 1]
            + local[:, 2, None] * pose[:, :3, 2]
            + pose[:, :3, 3]
        )
        pixels, z, inside = projected_pixels(
            transform_points(world[None], inverse), native_K, native_size
        )
        depth = view.depth.gather(1, pixels)
        confidence = view.confidence.gather(1, pixels)
        observed = inside & torch.isfinite(depth) & (depth > 1e-6) & (confidence > 1.5)
        tolerance = config["visibility_absolute_native"] + config[
            "visibility_relative"
        ] * depth.clamp_min(0)
        agree = observed & ((depth - z).abs() <= tolerance)
        free = observed & (depth - z > tolerance)
        latest = torch.where(agree, observations, -1).amax(0)
        matched_confidence = confidence.gather(0, latest.clamp_min(0)[None])[0]
        latest_confidence.flatten()[chosen] = torch.where(
            latest >= 0, matched_confidence, view.confidence.flatten()[chosen]
        )
        # A measured foreground surface occludes the tested point and supplies
        # no evidence that the point behind it is incorrect.
        directions = view.poses[:, None, :3, 3] - world[None]
        directions /= directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        for mask, output in [(agree, support), (free, conflict)]:
            latest = torch.where(mask, observations, -1).amax(0)
            reference = directions[
                latest.clamp_min(0), torch.arange(len(chosen), device=view.depth.device)
            ]
            separate = (directions * reference[None]).sum(-1) <= cosine_limit
            count = (latest >= 0).to(torch.int16) + (mask & separate).any(0).to(
                torch.int16
            )
            output.flatten()[chosen] = count
    removed = valid & (conflict >= config["minimum_free_views"]) & (conflict > support)
    confirmed = (support >= config["minimum_support_views"]) & (support > conflict)
    unresolved = valid & (conflict > 0) & (conflict >= support)
    red = valid & (unresolved | ((latest_confidence < 3.2) & ~confirmed))
    torch.cuda.synchronize()
    return dict(
        retained=valid & ~removed,
        support=support,
        conflict=conflict,
        red=red,
        latest_confidence=latest_confidence,
        unresolved=unresolved,
        seconds=time.perf_counter() - tick,
        removed_points=int(removed.sum()),
        valid_points=int(valid.sum()),
    )
