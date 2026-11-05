"""Calibrated geometry and a bounded bridge to the isolated CUDA worker."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import struct
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MAX_HEADER = 65536
MAX_PAYLOAD = 16 * 1024 * 1024


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def report_initial_mapping(mapper, callback):
    """Observe the expensive native initialization; invoke the original exactly once."""
    original = mapper.initialize_map

    def monitored(*args, **kwargs):
        callback()
        return original(*args, **kwargs)

    mapper.initialize_map = monitored


def packet_bytes(header, payload=b""):
    if len(payload) > MAX_PAYLOAD:
        raise ValueError("Oversized frame payload")
    data = json.dumps(
        dict(header, payload_bytes=len(payload)), allow_nan=False
    ).encode()
    if len(data) > MAX_HEADER:
        raise ValueError("Oversized frame header")
    return struct.pack("!I", len(data)) + data + payload


def parse_header(data):
    header = json.loads(data)
    size = header.get("payload_bytes")
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or not 0 <= size <= MAX_PAYLOAD
    ):
        raise ValueError("Invalid payload length")
    return header


def read_packet(stream):
    def exact(size):
        data = bytearray()
        while len(data) < size:
            chunk = stream.read(size - len(data))
            if not chunk:
                raise EOFError("Frame pipe closed")
            data.extend(chunk)
        return bytes(data)

    length = struct.unpack("!I", exact(4))[0]
    if not 0 < length <= MAX_HEADER:
        raise ValueError("Invalid header length")
    header = parse_header(exact(length))
    return header, exact(header["payload_bytes"])


async def read_packet_async(reader):
    length = struct.unpack("!I", await reader.readexactly(4))[0]
    if not 0 < length <= MAX_HEADER:
        raise ValueError("Invalid header length")
    header = parse_header(await reader.readexactly(length))
    return header, await reader.readexactly(header["payload_bytes"])


def latest_profile(calibration_root):
    root = Path(calibration_root).resolve()
    try:
        pointer = json.loads((root / "latest-passed.json").read_text())
        path = Path(pointer["calibration"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError()
        profile = CalibratedCamera(path)
        if not profile.record.get("camera", {}).get("settings", {}).get("deviceId"):
            raise ValueError()
        return profile
    except (OSError, KeyError, TypeError, ValueError):
        raise ValueError("尚未找到通过检查的手机标定，请先完成相机标定。") from None


class CalibratedCamera:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.record = record = json.loads(self.path.read_text())
        if record.get("status") != "passed":
            raise ValueError("相机标定尚未通过检查。")
        self.size = tuple(record["image_size"])
        if len(self.size) != 2 or any(
            type(v) is not int or not 64 <= v <= 4096 for v in self.size
        ):
            raise ValueError("Invalid calibration image size")
        self.K = np.asarray(record["camera_matrix"], np.float64)
        self.D = np.asarray(record["distortion_coefficients"], np.float64).ravel()
        self.rectified_K = np.asarray(record["rectified_camera_matrix"], np.float64)
        if (
            self.K.shape != (3, 3)
            or self.rectified_K.shape != (3, 3)
            or len(self.D) not in (4, 5, 8, 12, 14)
        ):
            raise ValueError("Invalid calibration matrix")
        if not all(np.isfinite(v).all() for v in (self.K, self.D, self.rectified_K)):
            raise ValueError("Non-finite calibration")
        if any(K[0, 0] <= 0 or K[1, 1] <= 0 for K in (self.K, self.rectified_K)):
            raise ValueError("Invalid calibrated focal length")
        width, height = self.size
        scale = math.sqrt((341 * 640) / (width * height))
        self.processed_size = (
            int(width * scale) // 8 * 8,
            int(height * scale) // 8 * 8,
        )
        self.processed_K = self.rectified_K.copy()
        self.processed_K[0] *= self.processed_size[0] / width
        self.processed_K[1] *= self.processed_size[1] / height
        self.maps = None

    def public(self):
        camera = self.record.get("camera", {})
        return dict(
            ready=True,
            id=self.record.get("id"),
            label=camera.get("label"),
            device_id=camera.get("settings", {}).get("deviceId"),
            image_size=list(self.size),
            zoom=camera.get("settings", {}).get("zoom", 1),
            orientation=camera.get("orientation"),
            fit_rms_px=self.record.get("fit_rms_px"),
        )

    def validate_camera(self, camera):
        expected = self.record["camera"]
        settings = camera.get("settings", {})
        original = expected.get("settings", {})
        if (
            not original.get("deviceId")
            or settings.get("deviceId") != original["deviceId"]
        ):
            raise ValueError("镜头与标定记录不匹配，请选择已经标定的「后置相机」。")
        if camera.get("label") != expected.get("label"):
            raise ValueError("相机名称与标定记录不匹配，请重新选择已标定镜头。")
        if (
            tuple(camera.get("outgoing", {}).get(k) for k in ("width", "height"))
            != self.size
        ):
            raise ValueError("传输尺寸与标定记录不一致，请保持标定时的竖屏和分辨率。")
        for key in ("width", "height", "resizeMode", "facingMode"):
            if key in original and settings.get(key) != original[key]:
                raise ValueError("相机采集设置与标定不一致，请使用相同镜头和拍摄方向。")
        for key in ("zoom", "frameRate"):
            if key in original:
                try:
                    value = float(settings[key])
                    if not math.isfinite(value) or abs(value - float(original[key])) > (
                        0.1 if key == "frameRate" else 0.001
                    ):
                        raise ValueError()
                except (KeyError, TypeError, ValueError):
                    raise ValueError("请保持标定时的 1× 倍率和采集帧率。") from None
        if (
            expected.get("orientation")
            and camera.get("orientation") != expected["orientation"]
        ):
            raise ValueError("手机方向与标定时不一致，请恢复原来的竖屏方向。")

    def prepare(self, rgb):
        if rgb.dtype != np.uint8 or rgb.shape != (self.size[1], self.size[0], 3):
            raise ValueError("视频画面尺寸发生变化，已停止本次扫描。")
        if self.maps is None:
            self.maps = cv2.initUndistortRectifyMap(
                self.K, self.D, None, self.rectified_K, self.size, cv2.CV_32FC1
            )
        corrected = cv2.remap(rgb, *self.maps, interpolation=cv2.INTER_LINEAR)
        # Match demo.mono_stream's resize and independently scale both image axes.
        prepared = cv2.resize(corrected, self.processed_size)
        return prepared


class LiveScanBridge:
    """One in-flight frame plus one pending frame, with configured scan recording."""

    def __init__(self, profile, camera, output, on_output, *, python=None, config=None):
        profile.validate_camera(camera)
        self.profile = profile
        self.camera = camera
        self.folder = Path(output)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.config = Path(config or ROOT / "configs/office.json")
        settings = json.loads(self.config.read_text())
        self.backend = settings.get("backend", "mast3r-slam")
        workers = {"mast3r-slam": "run_mobile_worker.py"}
        if self.backend not in workers:
            raise ValueError("Unknown local reconstruction backend")
        environment = "mast3r-slam" if self.backend == "mast3r-slam" else "local-scan"
        self.python = Path(python or sys.executable)
        if not self.python.is_file():
            raise ValueError("本机重建环境不可用。")
        self.model_filename = (
            "model.ply" if self.backend == "mast3r-slam" else "online-gaussians.ply"
        )
        self.worker_script = ROOT / "scripts" / workers[self.backend]
        self.max_fps = float(settings["max_input_fps"])
        self.max_seconds = float(settings.get("max_seconds", 120))
        if not 0 < self.max_fps <= 15:
            raise ValueError("Invalid live frame rate")
        if not 0 < self.max_seconds <= 3600:
            raise ValueError("Invalid live duration")
        self.on_output = on_output
        self.queue = asyncio.Queue(maxsize=1)
        self.phase = "starting"
        self.proc = None
        self.task = None
        self.log = None
        self.stopping = False
        self.offered = 0
        self.dropped = 0
        self.inflight = 0
        self.output_frames = 0
        self.model_frames = 0
        self.overlay_frames = 0
        self.last_reply = {}
        self.error = None
        record = dict(profile.record)
        record.pop("selected_frames", None)
        atomic_json(self.folder / "calibration.json", record)
        atomic_json(self.folder / "camera.json", camera)
        self.recording = None
        if settings.get("input_video_recording"):
            from interactive_rgb_scanning.recording import ScanRecording

            self.recording = ScanRecording(self.folder, camera, settings)
            snapshots = self.folder / "recording/configs"
            snapshots.mkdir()
            replay_settings = dict(settings)
            for key in (
                "runtime_config",
                "support_config",
                "reminder_config",
                "solver_retry_config",
                "blue_projection_config",
                "surface_evidence_config",
            ):
                if settings.get(key):
                    source = ROOT / settings[key]
                    target = snapshots / source.name
                    shutil.copyfile(source, target)
                    replay_settings[key] = str(target)
            atomic_json(self.folder / "recording/replay-config.json", replay_settings)
            shutil.copyfile(self.config, snapshots / self.config.name)
            sources = self.folder / "recording/source"
            sources.mkdir()
            for source in (
                Path(__file__),
                self.worker_script,
                ROOT / "src/interactive_rgb_scanning/backend/runtime.py",
                ROOT / "src/interactive_rgb_scanning/overlay.py",
                ROOT / "src/interactive_rgb_scanning/feedback.py",
            ):
                shutil.copyfile(source, sources / source.name)

    async def start(self):
        env = dict(os.environ)
        env.update(
            PYTHONUNBUFFERED="1",
            PYTHONDONTWRITEBYTECODE="1",
            MPLBACKEND="Agg",
            PYTHONPATH=str(ROOT / "src"),
            OMP_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="4",
        )
        self.log = (self.folder / "worker.log").open("ab")
        try:
            self.proc = await asyncio.create_subprocess_exec(
                str(self.python),
                str(self.worker_script),
                "--calibration",
                str(self.folder / "calibration.json"),
                "--config",
                str(self.config),
                "--output",
                str(self.folder),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=self.log,
                env=env,
                limit=MAX_PAYLOAD + MAX_HEADER,
            )
        except Exception:
            self.log.close()
            self.phase = "failed"
            if self.recording:
                await asyncio.to_thread(self.recording.close)
            raise
        self.task = asyncio.create_task(self.pump())

    def offer(self, frame, seq, received_at, source_timestamp_us=None):
        if self.stopping or self.phase not in ("starting", "running"):
            return
        if self.recording:
            from livekit import rtc

            rgb = frame.convert(rtc.VideoBufferType.RGB24)
            try:
                self.recording.camera_frame(
                    bytes(rgb.data), seq, received_at, source_timestamp_us
                )
            except RuntimeError as error:
                self.error = str(error)
                self.recording.error = self.error
                asyncio.create_task(self.stop())
                return
        self.offered += 1
        if self.queue.full():
            self.queue.get_nowait()
            self.dropped += 1
        self.queue.put_nowait((frame, seq, received_at))

    def status(self):
        worker = {}
        try:
            worker = json.loads((self.folder / "worker-status.json").read_text())
        except (OSError, ValueError):
            pass
        phase = (
            self.phase
            if self.phase in ("complete", "failed", "stopping")
            else worker.get("phase", self.phase)
        )
        return dict(
            worker,
            backend=self.backend,
            phase=phase,
            lifecycle=self.phase,
            id=self.folder.name,
            frames_offered=self.offered,
            frames_dropped=self.dropped,
            pending=self.queue.qsize(),
            in_flight=self.inflight,
            returned_frames=self.output_frames,
            returned_model_frames=self.model_frames,
            returned_overlay_frames=self.overlay_frames,
            model_saved=self.phase == "complete"
            and bool(worker.get("model_saved"))
            and (self.folder / self.model_filename).is_file(),
            error=self.error or worker.get("error"),
            calibration_id=self.profile.record.get("id"),
            recording=(
                self.recording.status() if self.recording else dict(enabled=False)
            ),
        )

    async def send(self, header, payload=b""):
        self.proc.stdin.write(packet_bytes(header, payload))
        await self.proc.stdin.drain()

    async def receive(self):
        header, payload = await asyncio.wait_for(
            read_packet_async(self.proc.stdout), 120
        )
        if header.get("kind") == "error":
            raise RuntimeError(header.get("message", "重建进程未能完成处理。"))
        if payload:
            width, height = header.get("width"), header.get("height")
            if (
                not isinstance(width, int)
                or not isinstance(height, int)
                or len(payload) != width * height * 3
            ):
                raise ValueError("Invalid worker RGB payload")
            if self.recording:
                self.recording.feedback_frame(payload, header)
            published = await self.on_output(width, height, payload, header)
            if published is not False:
                self.output_frames += 1
                self.model_frames += int(header.get("output_kind") == "gaussian_render")
                self.overlay_frames += int(
                    header.get("output_kind") == "observation_overlay"
                )
        self.last_reply = header
        return header

    async def pump(self):
        try:
            ready = await self.receive()
            if ready.get("kind") != "ready":
                raise ValueError("Invalid worker handshake")
            self.phase = "running"
            due = 0.0
            deadline = None
            stop_reason = None
            terminal = False
            while not self.stopping:
                await asyncio.sleep(max(0.0, due - time.monotonic()))
                if deadline is not None and time.monotonic() >= deadline:
                    stop_reason = "duration_limit"
                    break
                try:
                    item = (
                        await asyncio.wait_for(
                            self.queue.get(), max(0.001, deadline - time.monotonic())
                        )
                        if deadline is not None
                        else await self.queue.get()
                    )
                except asyncio.TimeoutError:
                    stop_reason = "duration_limit"
                    break
                if item is None or self.stopping:
                    break
                if deadline is None:
                    deadline = time.monotonic() + self.max_seconds
                frame, seq, received_at = item
                self.inflight = 1
                from livekit import rtc

                rgb = frame.convert(rtc.VideoBufferType.RGB24)
                if self.recording:
                    self.recording.processed_frame(seq)
                await self.send(
                    dict(
                        kind="frame",
                        seq=seq,
                        width=frame.width,
                        height=frame.height,
                        received_at=received_at,
                    ),
                    bytes(rgb.data),
                )
                reply = await self.receive()
                self.inflight = 0
                due = time.monotonic() + max(
                    0.0, 1 / self.max_fps - reply.get("processing_seconds", 0.0)
                )
                if reply.get("terminal"):
                    terminal = True
                    break
            self.phase = "stopping"
            if not terminal:
                await self.send(dict(kind="stop", reason=stop_reason))
            final = await self.receive()
            if final.get("kind") != "complete":
                raise ValueError("Invalid completion response")
            await asyncio.wait_for(self.proc.wait(), 15)
            if self.proc.returncode:
                raise RuntimeError("重建进程异常退出，已保留本机日志。")
            self.phase = "complete"
        except asyncio.CancelledError:
            self.phase = "failed"
            self.error = "重建已中断。"
            raise
        except Exception as error:
            self.phase = "failed"
            # No credentials enter the CUDA process or its protocol.
            self.error = (
                str(error)[:300]
                if isinstance(error, (ValueError, RuntimeError))
                else "重建连接中断，请结束本轮后重试。"
            )
        finally:
            self.inflight = 0
            while not self.queue.empty():
                if self.queue.get_nowait() is not None:
                    self.dropped += 1
            if self.proc and self.proc.returncode is None:
                self.proc.terminate()
                try:
                    await asyncio.wait_for(self.proc.wait(), 10)
                except asyncio.TimeoutError:
                    self.proc.kill()
                    await self.proc.wait()
            if self.log:
                self.log.close()
            if self.recording:
                try:
                    await asyncio.to_thread(self.recording.close)
                except Exception as error:
                    self.phase = "failed"
                    self.error = str(error)[:300]
            atomic_json(self.folder / "bridge-summary.json", self.status())

    async def stop(self):
        if not self.task:
            return
        if not self.task.done():
            self.stopping = True
            if self.phase != "failed":
                self.phase = "stopping"
            if self.queue.full():
                self.queue.get_nowait()
                self.dropped += 1
            self.queue.put_nowait(None)
            try:
                await asyncio.wait_for(asyncio.shield(self.task), 90)
            except asyncio.TimeoutError:
                self.task.cancel()
                await asyncio.gather(self.task, return_exceptions=True)
        else:
            await self.task
