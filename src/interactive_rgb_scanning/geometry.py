"""Calibrated native pointmap geometry and surface associations."""

from types import SimpleNamespace
import numpy as np
import torch


def transform_points(xyz, T):
    # Elementwise transform avoids TF32 rounding in projection's pixel test.
    return (
        xyz[..., 0:1] * T[..., None, :3, 0]
        + xyz[..., 1:2] * T[..., None, :3, 1]
        + xyz[..., 2:3] * T[..., None, :3, 2]
        + T[..., None, :3, 3]
    )


def projected_pixels(camera, K, size):
    width, height = size
    z = camera[..., 2]
    uv = torch.round(
        camera[..., :2] / z.clamp_min(1e-8)[..., None] * torch.stack([K[0, 0], K[1, 1]])
        + torch.stack([K[0, 2], K[1, 2]])
    )
    valid = torch.isfinite(camera).all(-1) & (z > 1e-6)
    valid &= (
        (uv[..., 0] >= 0)
        & (uv[..., 0] < width)
        & (uv[..., 1] >= 0)
        & (uv[..., 1] < height)
    )
    uv = torch.nan_to_num(uv, nan=0, posinf=0, neginf=0)
    pixels = uv[..., 1].long().clamp(0, height - 1) * width + uv[..., 0].long().clamp(
        0, width - 1
    )
    return pixels, z, valid


def make_rays(K, size):
    width, height = size
    yy, xx = torch.meshgrid(
        torch.arange(height, device=K.device),
        torch.arange(width, device=K.device),
        indexing="ij",
    )
    return (
        torch.stack(
            [(xx - K[0, 2]) / K[0, 0], (yy - K[1, 2]) / K[1, 1], torch.ones_like(xx)],
            -1,
        )
        .float()
        .reshape(-1, 3)
    )


def view_from_runtime(rt):
    """Read-only runtime view with the same projection arithmetic as snapshots."""
    n = len(rt.keyframes)
    view = SimpleNamespace(
        depth=rt.keyframes.X[:n, :, 2],
        confidence=rt.keyframes.C[:n, :, 0] / rt.keyframes.N[:n, None].clamp_min(1),
        poses=rt.native.lietorch.Sim3(rt.keyframes.T_WC[:n].reshape(n, 8))
        .matrix()
        .reshape(n, 4, 4),
        native_rays=make_rays(rt.K, (rt.w, rt.h)),
        K=rt.K.clone(),
        shape=(rt.h // 2, rt.w // 2),
        native_shape=(rt.h, rt.w),
    )
    view.K[:2] /= 2
    view.K[0, 2] -= 0.25
    view.K[1, 2] -= 0.25
    view.size = view.shape[::-1]
    return view


def project_points(self, pose, need_owner=False):
    count = int(np.prod(self.shape))
    depth = torch.full((count,), float("inf"), device="cuda")
    batches = []
    inverse = torch.linalg.inv(pose)
    for start in range(0, len(self.poses), 16):
        stop = min(start + 16, len(self.poses))
        z = self.depth[start:stop]
        good = torch.isfinite(z) & (z > 1e-6) & (self.confidence[start:stop] > 1.5)
        local = self.native_rays[None] * torch.where(good, z, 0)[..., None]
        # Keep calibrated pixel coordinates independent of neural TF32 settings.
        relative = (inverse[None, :, :, None] * self.poses[start:stop, None, :, :]).sum(
            2
        )
        pixels, z, inside = projected_pixels(
            transform_points(local, relative), self.K, self.size
        )
        inside &= good
        depth.scatter_reduce_(
            0,
            pixels.flatten(),
            torch.where(inside, z, float("inf")).flatten(),
            reduce="amin",
            include_self=True,
        )
        batches.append((start, pixels, z, inside))
    if not need_owner:
        return depth.reshape(self.shape), None
    owner = torch.full((count,), self.depth.numel(), dtype=torch.long, device="cuda")
    for start, pixels, z, inside in batches:
        ids = torch.arange(
            start * self.depth.shape[1],
            start * self.depth.shape[1] + z.numel(),
            device="cuda",
        ).reshape_as(z)
        nearest = inside & (z == depth[pixels])
        owner.scatter_reduce_(
            0,
            pixels.flatten(),
            torch.where(nearest, ids, self.depth.numel()).flatten(),
            reduce="amin",
            include_self=True,
        )
    return depth.reshape(self.shape), owner.clamp_max(self.depth.numel() - 1)


class SurfaceGeometry:
    """Projection cache indexed by native surface owners and revisions."""

    def _native_batch(self, start, stop, pose, poses):
        rt = self.rt
        kfs = rt.keyframes
        z = kfs.X[start:stop, :, 2]
        valid = (
            torch.isfinite(z)
            & (z > 1e-6)
            & ((kfs.C[start:stop, :, 0] / kfs.N[start:stop, None].clamp_min(1)) > 1.5)
        )
        xyz = self.native_rays[None] * torch.where(valid, z, 0)[..., None]
        inverse = torch.linalg.inv(pose)
        relative = (inverse[None, :, :, None] * poses[start:stop, None, :, :]).sum(2)
        return (
            *projected_pixels(transform_points(xyz, relative), self.K, self.size),
            valid,
        )

    def project_native(self, pose, poses):
        revisions = self.rt.keyframes.N_updates[: len(poses)]
        if (
            self.projection is not None
            and torch.equal(pose, self.projection_pose)
            and torch.equal(poses, self.projection_poses)
            and torch.equal(revisions, self.projection_revisions)
        ):
            return self.projection
        self.projection = None
        projected = []
        for start in range(0, len(poses), 16):
            stop = min(len(poses), start + 16)
            pixels, z, inside, valid = self._native_batch(start, stop, pose, poses)
            projected.append((start, stop, pixels, z, inside & valid, valid.sum()))
        self.projection_pose = pose.clone()
        self.projection_poses = poses.clone()
        self.projection_revisions = revisions.clone()
        self.projection = projected
        return projected
