"""Current-view coverage and represented-surface quality assessment."""

import cv2
import numpy as np
import torch
from .geometry import project_points


def connected(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype("u1"), connectivity=8
    )
    keep = stats[:, cv2.CC_STAT_AREA] >= 20
    keep[0] = False
    return keep[labels]


def states(risk, threshold):
    defined = np.isfinite(risk)
    return connected(defined & (risk > threshold)), defined


def native_cues_from_view(view, pose, surface_projector=None):
    """Native red scores, with optional continuous-surface blue coverage.

    Red confidence/owner/depth are preserved exactly for display-memory
    compatibility. Surface coverage only adds support between valid adjacent
    samples; it neither synthesizes confidence nor certifies geometric repair.
    """
    depth, owner = project_points(view, pose, need_owner=True)
    owner = owner.reshape(view.shape)
    finite = torch.isfinite(depth)
    confidence = view.confidence.flatten()[owner]
    risk = torch.where(finite, -confidence, float("nan")).cpu().numpy()
    red, defined = states(risk, -3.2)
    coverage = finite
    if surface_projector is not None:
        coverage = coverage | torch.isfinite(surface_projector.project(view, pose))
    packet = dict(
        raw_blue=(~coverage).cpu().numpy(),
        raw_red=red,
        defined=defined,
        native_owner=torch.where(finite, owner, -1).cpu().numpy().astype("i4"),
        confidence=-risk,
        depth_native=depth.cpu().numpy(),
        K=view.K.cpu().numpy(),
    )
    if hasattr(view, "surface_evidence"):
        evidence = view.surface_evidence
        flag = evidence["red"].flatten()[owner]
        red_risk = torch.where(finite, flag.float(), float("nan")).cpu().numpy()
        packet["raw_red"], _ = states(red_risk, 0.5)
        packet["geometry_unresolved"] = (
            (finite & evidence["unresolved"].flatten()[owner]).cpu().numpy()
        )
        packet["latest_confidence"] = (
            torch.where(
                finite, evidence["latest_confidence"].flatten()[owner], float("nan")
            )
            .cpu()
            .numpy()
        )
    return packet
