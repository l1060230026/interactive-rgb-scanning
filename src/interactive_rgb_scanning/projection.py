"""Rasterize the calibrated structured pointmap, without changing native SLAM.

Neighbour topology follows MASt3R-SLAM's trianglemap viewer (TL/BL/TR,
TR/BL/BR). Rasterization and z-buffering call NVIDIA nvdiffrast directly.
Unlike the viewer, every triangle vertex must have valid depth/confidence;
triangles crossing a depth discontinuity are excluded. No image-space closing
or dilation is used. Returned depth is in the query camera's native units.
"""

from importlib.metadata import version

import torch


class SurfaceProjector:
    def __init__(
        self, relative_depth_jump=0.05, confidence_threshold=1.5, keyframes_per_batch=8
    ):
        import nvdiffrast.torch as dr

        self.dr = dr
        self.context = dr.RasterizeCudaContext()
        self.relative_depth_jump = float(relative_depth_jump)
        self.confidence_threshold = float(confidence_threshold)
        self.batch = int(keyframes_per_batch)
        self.topologies = {}
        self.provenance = dict(
            rasterizer="nvdiffrast.torch.RasterizeCudaContext/rasterize/interpolate",
            rasterizer_version=version("nvdiffrast"),
            upstream="https://github.com/NVlabs/nvdiffrast",
            topology="MASt3R-SLAM resources/programs/trianglemap.glsl neighbour triangles",
            relative_depth_jump=self.relative_depth_jump,
            confidence_threshold=self.confidence_threshold,
            image_morphology=False,
            reference_geometry_read=False,
        )

    def _triangles(self, h, w, device):
        key = (h, w, str(device))
        if key not in self.topologies:
            yy, xx = torch.meshgrid(
                torch.arange(h - 1, device=device),
                torch.arange(w - 1, device=device),
                indexing="ij",
            )
            tl = (yy * w + xx).flatten()
            self.topologies[key] = torch.stack(
                (
                    torch.stack((tl, tl + w, tl + 1), -1),
                    torch.stack((tl + 1, tl + w, tl + w + 1), -1),
                ),
                1,
            ).reshape(-1, 3)
        return self.topologies[key]

    @torch.inference_mode()
    def project(self, view, pose):
        """Same view/pose contract as FixedMap; no GT, future image or scale fit."""
        width, height = view.size
        # Source pointmaps retain native resolution; query is usually half size.
        native_h, native_w = getattr(
            view, "native_shape", (view.shape[0] * 2, view.shape[1] * 2)
        )
        per_frame = native_h * native_w
        if view.depth.shape[1] != per_frame:
            raise ValueError(
                "Native pointmap dimensions do not match calibrated topology"
            )
        triangles = self._triangles(native_h, native_w, view.depth.device)
        result = torch.full(view.shape, float("inf"), device=view.depth.device)
        inverse = torch.linalg.inv(pose)
        K = view.K
        for start in range(0, len(view.poses), self.batch):
            stop = min(start + self.batch, len(view.poses))
            depth = view.depth[start:stop]
            valid = (
                torch.isfinite(depth)
                & (depth > 0)
                & torch.isfinite(view.confidence[start:stop])
                & (view.confidence[start:stop] > self.confidence_threshold)
            )
            ztri = depth[:, triangles]
            good = valid[:, triangles].all(-1) & (
                (ztri.amax(-1) - ztri.amin(-1))
                <= self.relative_depth_jump * ztri.amin(-1)
            )
            offsets = (
                torch.arange(stop - start, device=depth.device)[:, None, None]
                * per_frame
            )
            tri = (triangles[None] + offsets)[good].to(torch.int32).contiguous()
            if not tri.numel():
                continue
            local = view.native_rays[None] * torch.where(valid, depth, 0)[..., None]
            transform = (
                inverse[None, :, :, None] * view.poses[start:stop, None, :, :]
            ).sum(2)
            # Three scalar products avoid TF32 rounding of coordinates when
            # native neural inference has globally enabled TF32 matmul.
            xyz = (
                local[..., 0, None] * transform[:, None, :3, 0]
                + local[..., 1, None] * transform[:, None, :3, 1]
                + local[..., 2, None] * transform[:, None, :3, 2]
                + transform[:, None, :3, 3]
            ).reshape(-1, 3)
            z = xyz[:, 2]
            # Infinite far plane; near clipping is relative to native depth,
            # so no metre-scale assumption enters monocular projection.
            near = (
                torch.where(valid, depth, 0).sum() / valid.sum().clamp_min(1)
            ) * 1e-4
            clip = torch.stack(
                (
                    (2 * K[0, 0] / width) * xyz[:, 0]
                    + (2 * (K[0, 2] + 0.5) / width - 1) * z,
                    (2 * K[1, 1] / height) * xyz[:, 1]
                    + (2 * (K[1, 2] + 0.5) / height - 1) * z,
                    z - 2 * near,
                    z,
                ),
                -1,
            ).contiguous()
            # nvdiffrast's row zero is the -Y edge. Positive image Y above is
            # deliberate: tensor row zero directly matches the CV image.
            ranges = torch.tensor([[0, len(tri)]], dtype=torch.int32)
            raster, _ = self.dr.rasterize(
                self.context, clip, tri, view.shape, ranges=ranges, grad_db=False
            )
            projected, _ = self.dr.interpolate(z[:, None].contiguous(), raster, tri)
            surface = torch.where(
                raster[0, ..., 3] > 0, projected[0, ..., 0], float("inf")
            )
            result = torch.minimum(result, surface)
        return result
