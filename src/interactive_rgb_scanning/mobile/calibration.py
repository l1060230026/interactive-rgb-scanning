"""Calibrate the decoded phone stream; never reuse the packaged room0 intrinsics."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import secrets
import time

import cv2
import numpy as np

PATTERN = (9, 6)
MIN_VIEWS = 20
TARGET_VIEWS = 30
MAX_VIEWS = 40
cv2.setNumThreads(2)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def object_points(square_mm):
    points = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    points[:, :2] = np.mgrid[: PATTERN[0], : PATTERN[1]].T.reshape(-1, 2) * square_mm
    return points


def view_metrics(corners, size):
    width, height = size
    points = corners.reshape(-1, 2)
    quad = points[[0, PATTERN[0] - 1, -1, -PATTERN[0]]]
    edges = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)
    return dict(
        center=(points.mean(0) / [width, height]).tolist(),
        area=float(cv2.contourArea(quad.astype(np.float32)) / (width * height)),
        perspective=float(
            max(abs(np.log(edges[0] / edges[2])), abs(np.log(edges[1] / edges[3])))
        ),
    )


def coverage(corners, size):
    if not corners:
        return dict(hull_fraction=0.0, center_span=[0.0, 0.0], tilted_views=0)
    metrics = [view_metrics(p, size) for p in corners]
    centers = np.asarray([m["center"] for m in metrics])
    hull = cv2.convexHull(np.concatenate(corners).reshape(-1, 2).astype(np.float32))
    return dict(
        hull_fraction=float(cv2.contourArea(hull) / np.prod(size)),
        center_span=np.ptp(centers, axis=0).tolist(),
        tilted_views=sum(m["perspective"] > 0.08 for m in metrics),
    )


def coverage_problems(corners, size):
    c = coverage(corners, size)
    issues = []
    if c["hull_fraction"] < 0.30:
        issues.append("棋盘覆盖范围偏小，请靠近一些，并拍摄画面边缘的位置。")
    if min(c["center_span"]) < 0.15:
        issues.append("位置变化不够，请让棋盘分别出现在画面左、右、上、下。")
    if c["tilted_views"] < 3:
        issues.append("倾斜视角不足，请增加左右和上下倾斜约 20–40 度的画面。")
    return issues


def fit_camera(corners, size, square_mm):
    objects = [object_points(square_mm) for _ in corners]
    # A low-resolution stream does not justify a freely fitted sixth-order radial term.
    rms, matrix, distortion, rotations, translations = cv2.calibrateCamera(
        objects, corners, size, None, None, flags=cv2.CALIB_FIX_K3
    )
    errors = []
    for obj, observed, rotation, translation in zip(
        objects, corners, rotations, translations
    ):
        projected, _ = cv2.projectPoints(obj, rotation, translation, matrix, distortion)
        errors.append(
            float(np.sqrt(np.mean(np.sum((projected - observed) ** 2, axis=2))))
        )
    return float(rms), matrix, distortion, errors


def solve_views(corners, size, square_mm):
    """Return quality checks, a fitted model, and validation on withheld images."""
    if len(corners) < MIN_VIEWS:
        raise ValueError(f"至少需要 {MIN_VIEWS} 张有效画面，目前 {len(corners)} 张。")
    issues = coverage_problems(corners, size)
    if issues:
        raise ValueError("\n".join(issues))
    rms, matrix, distortion, errors = fit_camera(corners, size, square_mm)
    cutoff = max(0.8, 2.5 * float(np.median(errors)))
    inliers = [i for i, error in enumerate(errors) if error <= cutoff]
    if len(inliers) < MIN_VIEWS:
        raise ValueError("模糊或异常画面偏多，请再采集几张清晰画面。")
    kept = [corners[i] for i in inliers]
    issues = coverage_problems(kept, size)
    if issues:
        raise ValueError("\n".join(issues))
    rms, matrix, distortion, fit_errors = fit_camera(kept, size, square_mm)
    # Pose is estimated separately for validation images; their corners do not fit K/D.
    training = [p for i, p in enumerate(kept) if i % 5 != 4]
    validation = [p for i, p in enumerate(kept) if i % 5 == 4]
    _, train_matrix, train_distortion, _ = fit_camera(training, size, square_mm)
    validation_errors = []
    for observed in validation:
        ok, rotation, translation = cv2.solvePnP(
            object_points(square_mm), observed, train_matrix, train_distortion
        )
        if not ok:
            raise ValueError("验证画面位姿求解失败，请补充不同角度的清晰画面。")
        projected, _ = cv2.projectPoints(
            object_points(square_mm),
            rotation,
            translation,
            train_matrix,
            train_distortion,
        )
        validation_errors.append(
            float(np.sqrt(np.mean(np.sum((projected - observed) ** 2, axis=2))))
        )
    width, height = size
    focal_change = float(
        np.max(np.abs(np.diag(train_matrix)[:2] / np.diag(matrix)[:2] - 1))
    )
    principal_change = float(
        np.linalg.norm(train_matrix[:2, 2] - matrix[:2, 2]) / np.hypot(width, height)
    )
    checks = dict(
        fit_rms_below_0_8_px=rms < 0.8,
        validation_rms_below_1_px=bool(
            validation_errors and max(validation_errors) < 1.0
        ),
        focal_stability_within_10_percent=focal_change < 0.10,
        principal_stability_within_5_percent=principal_change < 0.05,
        principal_point_inside_image=bool(
            0 < matrix[0, 2] < width and 0 < matrix[1, 2] < height
        ),
        positive_plausible_focal_lengths=bool(
            0.15 * width < matrix[0, 0] < 8 * width
            and 0.15 * height < matrix[1, 1] < 8 * height
        ),
        finite_parameters=bool(
            np.isfinite(matrix).all() and np.isfinite(distortion).all()
        ),
    )
    if not checks["finite_parameters"]:
        raise ValueError("参数求解不稳定，请重新采集。")
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, size, 0, size)
    return dict(
        status="passed" if all(checks.values()) else "needs_review",
        image_size=list(size),
        square_mm=square_mm,
        inner_corners=list(PATTERN),
        camera_matrix=matrix.tolist(),
        distortion_coefficients=distortion.ravel().tolist(),
        distortion_order=["k1", "k2", "p1", "p2", "k3"],
        k3_fixed_to_zero=True,
        fit_rms_px=rms,
        per_view_rms_px=fit_errors,
        accepted_indices=inliers,
        outlier_indices=[i for i in range(len(corners)) if i not in inliers],
        validation=dict(
            training_views=len(training),
            withheld_views=len(validation),
            per_view_rms_px=validation_errors,
            focal_change_fraction=focal_change,
            principal_change_image_diagonal=principal_change,
        ),
        checks=checks,
        coverage=coverage(kept, size),
        rectified_camera_matrix=new_matrix.tolist(),
        rectified_roi=list(map(int, roi)),
        rectification="cv2.undistort(frame, K, D, None, rectified_K); keep full image, no ROI crop",
        limitations="Empirical quality checks; valid only for the recorded camera, crop, orientation and stream settings. Does not establish SLAM metric scale.",
    )


class CalibrationCapture:
    def __init__(self, root, square_mm, metadata):
        if not 10 <= square_mm <= 50:
            raise ValueError("单格边长应在 10–50 mm 之间。")
        self.id = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ-"
        ) + secrets.token_hex(3)
        self.folder = Path(root) / self.id
        (self.folder / "frames").mkdir(parents=True)
        self.square_mm = square_mm
        self.metadata = metadata
        self.software = dict(
            opencv=cv2.__version__,
            numpy=np.__version__,
            adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        self.corners = []
        self.rows = []
        self.size = None
        self.phase = "collecting"
        self.hint = "请让整个棋盘进入画面，移动后停稳一下。"
        self.busy = False
        self.last_attempt = 0.0
        self.result = None
        self.save()

    def status(self):
        return dict(
            id=self.id,
            phase=self.phase,
            accepted=len(self.corners),
            target=TARGET_VIEWS,
            maximum=MAX_VIEWS,
            minimum=MIN_VIEWS,
            image_size=list(self.size) if self.size else None,
            hint=self.hint,
            coverage=(
                coverage(self.corners, self.size) if self.size else coverage([], (1, 1))
            ),
            result=self.result,
            saves_selected_images=True,
        )

    def save(self):
        write_json(
            self.folder / "capture.json",
            dict(
                **self.status(),
                square_mm=self.square_mm,
                inner_corners=list(PATTERN),
                camera=self.metadata,
                software=self.software,
                selected_frames=self.rows,
            ),
        )

    def process_rgb(self, rgb):
        if self.phase != "collecting":
            return
        height, width = rgb.shape[:2]
        if self.size is not None and self.size != (width, height):
            self.phase = "invalid"
            self.hint = "画面尺寸发生变化，请结束连接、固定方向后重新标定。"
            self.save()
            return
        self.size = (width, height)
        expected = self.metadata.get("outgoing", {})
        if expected and (width, height) != (
            expected.get("width"),
            expected.get("height"),
        ):
            self.phase = "invalid"
            self.hint = "收到的尺寸与发送设置不一致，请重新连接。"
            self.save()
            return
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        found, corners = cv2.findChessboardCornersSB(
            gray, PATTERN, cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if self.phase != "collecting":
            return
        if not found:
            self.hint = "未识别完整棋盘：请露出全部 10×7 格，避免反光、遮挡或过小。"
            return
        points = corners.reshape(-1, 2)
        if (points.min(0) < 8).any() or (points.max(0) > [width - 9, height - 9]).any():
            self.hint = "棋盘太靠近画面边缘，请稍微移回一点。"
            return
        x, y, w, h = cv2.boundingRect(points)
        sharpness = float(cv2.Laplacian(gray[y : y + h, x : x + w], cv2.CV_64F).var())
        if sharpness < 50 or view_metrics(corners, self.size)["area"] < 0.035:
            self.hint = "棋盘偏小或不够清晰，请靠近、停稳并增加照明。"
            return
        diagonal = np.hypot(width, height)
        if any(
            min(
                float(np.sqrt(np.mean(np.sum((corners - old) ** 2, axis=2)))),
                float(np.sqrt(np.mean(np.sum((corners - old[::-1]) ** 2, axis=2)))),
            )
            / diagonal
            < 0.025
            for old in self.corners
        ):
            self.hint = "这个角度已经采集，请改变位置、距离或倾斜角度。"
            return
        index = len(self.corners)
        name = f"frame-{index:03d}.png"
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(self.folder / "frames" / name), bgr):
            raise OSError("Unable to save calibration frame")
        self.corners.append(corners.copy())
        self.rows.append(
            dict(
                file="frames/" + name,
                sharpness=sharpness,
                **view_metrics(corners, self.size),
            )
        )
        annotated = bgr.copy()
        cv2.drawChessboardCorners(annotated, PATTERN, corners, True)
        cv2.imwrite(str(self.folder / "latest-corners.jpg"), annotated)
        self.hint = f"已保存 {len(self.corners)} 张。请继续改变位置，并增加倾斜视角。"
        if len(self.corners) >= TARGET_VIEWS:
            self.hint = f"已保存 {len(self.corners)} 张，可以点「计算标定」检查结果。"
        if len(self.corners) >= MAX_VIEWS:
            self.phase = "paused"
            self.hint = "已采满 40 张，请计算标定。"
        self.save()

    def solve(self):
        self.phase = "solving"
        try:
            result = solve_views(self.corners, self.size, self.square_mm)
            result.update(
                id=self.id,
                camera=self.metadata,
                software=self.software,
                created_utc=datetime.now(timezone.utc).isoformat(),
                selected_frames=self.rows,
            )
            self.result = result
            write_json(self.folder / "calibration.json", result)
            if result["status"] == "passed":
                K = np.asarray(result["rectified_camera_matrix"])
                np.savetxt(
                    self.folder / "calib-rectified.txt",
                    [[K[0, 0], K[1, 1], K[0, 2], K[1, 2]]],
                    fmt="%.9f",
                )
                raw = np.asarray(result["camera_matrix"])
                np.savetxt(
                    self.folder / "calib-raw-with-distortion.txt",
                    [
                        [
                            raw[0, 0],
                            raw[1, 1],
                            raw[0, 2],
                            raw[1, 2],
                            *result["distortion_coefficients"],
                        ]
                    ],
                    fmt="%.9f",
                )
                first = cv2.imread(
                    str(self.folder / self.rows[result["accepted_indices"][0]]["file"])
                )
                rectified = cv2.undistort(
                    first, raw, np.asarray(result["distortion_coefficients"]), None, K
                )
                cv2.imwrite(
                    str(self.folder / "undistortion-preview.jpg"),
                    np.concatenate([first, rectified], axis=1),
                )
                write_json(
                    self.folder.parent / "latest-passed.json",
                    dict(
                        id=self.id,
                        calibration=str(self.folder / "calibration.json"),
                        scope="Last calibration that passed empirical checks; verify camera settings before reuse.",
                    ),
                )
                self.phase = "complete"
                self.hint = "标定检查通过，参数和去畸变预览已保存。请保持相同镜头、方向和采集设置。"
            else:
                self.phase = "paused"
                self.hint = "误差或参数稳定性尚未通过，请继续补拍边缘、远近和倾斜视角后重新计算。"
        except ValueError as error:
            self.phase = "paused"
            self.hint = str(error)
        except Exception:
            self.result = None
            self.phase = "paused"
            self.hint = "计算未完成，请稍后重试。"
            self.save()
            raise
        self.save()
        return self.status()
