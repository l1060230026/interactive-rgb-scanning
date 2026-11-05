#!/usr/bin/env python3
"""LiveKit video echo, camera calibration and an isolated native RGB SLAM worker.

Only this dedicated app is exposed by the HTTPS gateway. API credentials remain
in a mode-0600 file. Browser tokens are room-scoped and camera-only. Scan video
recording follows the deployment configuration. Calibration saves selected stills.
Run artifacts and private access state live under home.

LIVEKIT_URL is returned to the browser. Optional LIVEKIT_INTERNAL_URL lets the
local receiver connect over loopback while the browser uses the HTTPS gateway.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
from datetime import datetime, timedelta, timezone
import hmac
import json
import logging
from pathlib import Path
import re
import secrets
import sys
import time

from aiohttp import web
from livekit import api, rtc

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def read_credentials(path: Path) -> dict:
    values = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("\"'")
    for key in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        if not values.get(key):
            raise ValueError(f"Missing {key}")
    return values


class EchoSession:
    def __init__(self, credentials: dict, output: Path, *, scan_options=None):
        self.credentials = credentials
        self.output = output
        self.id = secrets.token_hex(10)
        self.name = "rgb-scanning-phone-" + self.id
        self.room = rtc.Room()
        self.tasks = set()
        self.source = None
        self.track = None
        self.started = time.monotonic()
        self.last_activity = self.started
        self.closed = False
        self.closing = False
        self.close_complete = asyncio.Event()
        self.closing_task = None
        self.scan_lock = asyncio.Lock()
        self.camera = None
        self.calibration = None
        self.calibration_task = None
        self.scan_options = scan_options or {}
        self.scan_profile = None
        self.scan = None
        self.stats = {
            "session": self.id,
            "mode": "raw_video_echo",
            "frames_received": 0,
            "frames_returned": 0,
            "phone_seen": False,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "reconstruction_enabled": False,
            "video_recorded": False,
        }

    def token(self, identity: str, phone: bool = False) -> str:
        grants = api.VideoGrants(
            room_join=True,
            room=self.name,
            can_publish=True,
            can_subscribe=True,
            can_publish_data=True,
            can_publish_sources=["camera"] if phone else None,
        )
        return (
            api.AccessToken(
                self.credentials["LIVEKIT_API_KEY"],
                self.credentials["LIVEKIT_API_SECRET"],
            )
            .with_identity(identity)
            .with_ttl(timedelta(minutes=30))
            .with_grants(grants)
            .to_jwt()
        )

    def launch(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)

        def done(completed):
            self.tasks.discard(completed)
            if not completed.cancelled() and completed.exception():
                self.stats["last_error_type"] = type(completed.exception()).__name__
                self.save()

        task.add_done_callback(done)
        return task

    async def start(self):
        @self.room.on("participant_connected")
        def connected(participant):
            if participant.identity == "phone":
                self.stats["phone_seen"] = True
                self.last_activity = time.monotonic()

        @self.room.on("track_subscribed")
        def subscribed(track, publication, participant):
            if (
                participant.identity == "phone"
                and track.kind == rtc.TrackKind.KIND_VIDEO
            ):
                self.launch(self.echo(track))

        @self.room.on("participant_disconnected")
        def disconnected(participant):
            if participant.identity == "phone" and not self.closed:
                self.launch(self.close())

        @self.room.on("data_received")
        def data(packet):
            if (
                packet.participant
                and packet.participant.identity == "phone"
                and packet.topic == "ping"
            ):
                self.launch(
                    self.room.local_participant.publish_data(
                        packet.data,
                        reliable=False,
                        destination_identities=["phone"],
                        topic="pong",
                    )
                )

        await asyncio.wait_for(
            self.room.connect(
                self.credentials.get(
                    "LIVEKIT_INTERNAL_URL", self.credentials["LIVEKIT_URL"]
                ),
                self.token("server"),
            ),
            30,
        )
        self.stats["server_connected"] = True
        self.save()

    async def echo(self, track):
        stream = rtc.VideoStream(track, capacity=1, format=rtc.VideoBufferType.I420)
        try:
            async for event in stream:
                if self.closed:
                    break
                frame = event.frame
                # WebRTC may insert 8x8 placeholders as a track is muted/stopped.
                if frame.width < 64 or frame.height < 64:
                    self.stats["placeholder_frames_ignored"] = (
                        self.stats.get("placeholder_frames_ignored", 0) + 1
                    )
                    continue
                self.last_activity = time.monotonic()
                self.stats["frames_received"] += 1
                self.stats["width"], self.stats["height"] = frame.width, frame.height
                if self.camera and self.camera["mode"] == "scan" and self.scan:
                    self.scan.offer(
                        frame,
                        self.stats["frames_received"],
                        time.monotonic(),
                        getattr(event, "timestamp_us", None),
                    )
                    continue
                capture = self.calibration
                if (
                    capture
                    and capture.phase == "collecting"
                    and not capture.busy
                    and time.monotonic() - capture.last_attempt >= 1
                ):
                    capture.busy = True
                    capture.last_attempt = time.monotonic()
                    self.calibration_task = self.launch(
                        self.sample_calibration(capture, frame)
                    )
                await self.return_frame(frame)
        finally:
            await stream.aclose()

    async def return_frame(self, frame):
        if self.closed:
            return False
        if self.source is None:
            self.source = rtc.VideoSource(frame.width, frame.height)
            self.track = rtc.LocalVideoTrack.create_video_track(
                "server-return", self.source
            )
            options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
            options.video_encoding.max_bitrate = 1_200_000
            options.video_encoding.max_framerate = 15
            options.simulcast = False
            await self.room.local_participant.publish_track(self.track, options)
        self.source.capture_frame(frame)
        self.stats["frames_returned"] += 1
        if self.stats["frames_returned"] % 30 == 0:
            self.save()
        return True

    async def scan_output(self, width, height, payload, header):
        return await self.return_frame(
            rtc.VideoFrame(width, height, rtc.VideoBufferType.RGB24, payload)
        )

    async def start_scan(self):
        from interactive_rgb_scanning.mobile.bridge import LiveScanBridge

        async with self.scan_lock:
            if self.closed:
                raise ValueError("手机连接已结束，请重新连接。")
            self.scan_profile.validate_camera(self.camera)
            folder = Path(self.scan_options["output"]) / (
                "live-phone-"
                + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
                + self.id[:10]
                + "-"
                + secrets.token_hex(3)
            )
            self.scan = LiveScanBridge(
                self.scan_profile,
                self.camera,
                folder,
                self.scan_output,
                python=self.scan_options.get("python"),
                config=self.scan_options.get("config"),
            )
            await self.scan.start()
            self.stats.update(
                mode="live_rgb_scan",
                reconstruction_enabled=True,
                video_recorded=self.scan.recording is not None,
            )
            self.save()

    async def sample_calibration(self, capture, frame):
        def process():
            import numpy as np

            rgb_frame = frame.convert(rtc.VideoBufferType.RGB24)
            rgb = np.frombuffer(rgb_frame.data, dtype=np.uint8).reshape(
                frame.height, frame.width, 3
            )
            capture.process_rgb(rgb)

        try:
            await asyncio.to_thread(process)
        except Exception as error:
            capture.phase = "paused"
            capture.hint = "采集未完成，请暂停后重试。"
            self.stats["calibration_error_type"] = type(error).__name__
        finally:
            capture.busy = False

    def status(self):
        return {
            **self.stats,
            "closed": self.closed,
            "closing": self.closing,
            "camera": self.camera,
            "calibration": self.calibration.status() if self.calibration else None,
            "scan": self.scan.status() if self.scan else None,
            "elapsed_seconds": round(time.monotonic() - self.started, 1),
            "last_frame_age_seconds": round(time.monotonic() - self.last_activity, 1),
        }

    def save(self):
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / (self.id + ".json")).write_text(
            json.dumps(self.status(), indent=2) + "\n"
        )

    async def close(self):
        if self.closed:
            if self.closing and self.closing_task is not asyncio.current_task():
                await self.close_complete.wait()
            return
        self.closed = True
        self.closing = True
        self.closing_task = asyncio.current_task()
        try:
            async with self.scan_lock:
                if self.scan:
                    await self.scan.stop()
            if self.calibration and self.calibration.phase == "collecting":
                self.calibration.phase = "paused"
            # A CPU OpenCV call already running in to_thread cannot be cancelled.
            if self.calibration_task and not self.calibration_task.done():
                await asyncio.shield(self.calibration_task)
            if self.calibration:
                self.calibration.save()
            pending = [task for task in self.tasks if task is not self.closing_task]
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            await self.room.disconnect()
            if self.source and hasattr(self.source, "aclose"):
                with suppress(Exception):
                    await self.source.aclose()
        finally:
            self.closing = False
            try:
                self.save()
            finally:
                self.close_complete.set()


async def create_app(args):
    credentials = read_credentials(args.credentials)
    access_file = args.output / "access-key"
    args.output.mkdir(parents=True, exist_ok=True)
    if not access_file.exists():
        access_file.write_text(secrets.token_urlsafe(24))
        access_file.chmod(0o600)
    access_key = access_file.read_text().strip()
    session = None
    lock = asyncio.Lock()
    calibration_root = Path(
        getattr(args, "calibration_root", None) or args.output / "calibrations"
    )
    scan_options = dict(
        output=Path(
            getattr(args, "scan_output", None) or args.output.parent / "local-runs"
        ),
        python=getattr(args, "scan_python", None),
        config=getattr(args, "scan_config", None),
    )
    viewer_assets = ROOT / "web/model-viewer/dist"

    def local_model_read(request):
        # Desktop convenience is scoped to saved models, with a literal loopback
        # Host and no proxy headers. Caddy/phone requests still require the key.
        return (
            request.method in ("GET", "HEAD")
            and (
                request.path == "/api/models" or request.path.startswith("/api/models/")
            )
            and request.remote == "127.0.0.1"
            and re.fullmatch(r"127\.0\.0\.1(?::\d+)?", request.host) is not None
            and "X-Forwarded-For" not in request.headers
            and request.headers.get("Sec-Fetch-Site", "none") in ("none", "same-origin")
        )

    @web.middleware
    async def headers(request, handler):
        if request.path.startswith("/api/"):
            supplied = request.headers.get("Authorization", "")
            if not hmac.compare_digest(
                supplied, "Bearer " + access_key
            ) and not local_model_read(request):
                raise web.HTTPUnauthorized(text="Access key required")
        response = await handler(request)
        response.headers.update(
            {
                "Cache-Control": "no-store",
                "Referrer-Policy": "no-referrer",
                "X-Content-Type-Options": "nosniff",
                "Permissions-Policy": "camera=(self), microphone=()",
            }
        )
        return response

    app = web.Application(middlewares=[headers], client_max_size=16 * 1024)

    async def index(request):
        path_key = request.match_info.get("access_key")
        if path_key is not None and not hmac.compare_digest(path_key, access_key):
            raise web.HTTPNotFound(text="Invalid test link")
        return web.FileResponse(args.page)

    async def sdk(request):
        return web.FileResponse(args.sdk)

    async def model_page(request):
        config_path = scan_options["config"] or ROOT / "configs/office.json"
        backend = json.loads(Path(config_path).read_text()).get("backend", "hi-slam2")
        if request.query.get("kind") == "points" or (
            backend == "mast3r-slam" and request.query.get("kind") != "gaussians"
        ):
            return web.FileResponse(ROOT / "web/model.html")
        return web.FileResponse(viewer_assets / "index.html")

    async def model_asset(request):
        name = request.match_info["name"]
        if name not in (
            "viewer.js",
            "viewer.css",
            "phone-points.js",
            "THIRD-PARTY.txt",
        ):
            raise web.HTTPNotFound()
        return web.FileResponse(viewer_assets / name)

    def saved_model(run_id):
        if re.fullmatch(r"live-phone-[A-Za-z0-9-]+", run_id) is None:
            raise web.HTTPNotFound()
        root = scan_options["output"].resolve()
        folder = (root / run_id).resolve()
        if folder.parent != root:
            raise web.HTTPNotFound()
        manifest = folder / "web-view/manifest.json"
        if not manifest.is_file() or not manifest.resolve().is_relative_to(folder):
            raise web.HTTPNotFound(text="该扫描尚未准备网页预览。")
        return folder, json.loads(manifest.read_text())

    async def model_list(request):
        def collect():
            runs = []
            for manifest in sorted(
                scan_options["output"].glob("live-phone-*/web-view/manifest.json"),
                reverse=True,
            ):
                try:
                    _, data = saved_model(manifest.parent.parent.name)
                except (web.HTTPNotFound, ValueError, OSError):
                    continue
                runs.append(
                    {
                        key: data[key]
                        for key in ("id", "title", "backend", "vertex_count")
                    }
                )
            return runs

        return web.json_response(await asyncio.to_thread(collect))

    async def model_file(request):
        folder, data = await asyncio.to_thread(
            saved_model, request.match_info["run_id"]
        )
        kind = request.match_info["kind"]
        if kind == "manifest":
            # Private filesystem paths are not part of the browser contract.
            data["previews"] = [
                {k: p[k] for k in ("camera_id", "label")} for p in data["previews"]
            ]
            return web.json_response(data)
        if kind == "model.ply":
            path = folder / (
                "model.ply"
                if data.get("model_kind") == "point_cloud"
                else "online-gaussians.ply"
            )
        elif kind == "preview":
            preview = next(
                (
                    p
                    for p in data["previews"]
                    if str(p["camera_id"]) == request.match_info.get("camera_id")
                ),
                None,
            )
            if not preview:
                raise web.HTTPNotFound()
            path = folder / preview["image"]
        else:
            raise web.HTTPNotFound()
        if not path.resolve().is_relative_to(folder) or not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path)

    async def health(request):
        return web.json_response(
            {
                "ready": True,
                "mode": "video_echo",
                "camera_calibration": True,
                "live_reconstruction": True,
            }
        )

    async def start(request):
        nonlocal session
        async with lock:
            if session and (not session.closed or session.closing):
                raise web.HTTPConflict(
                    text="已有连接，请先点结束；断开后也会自动释放。"
                )
            session = EchoSession(
                credentials, args.output / "sessions", scan_options=scan_options
            )
            try:
                await session.start()
            except Exception as error:
                await session.close()
                # Never expose SDK messages: they can contain credentialed URLs.
                raise web.HTTPBadGateway(text="服务器连接失败：" + type(error).__name__)
            return web.json_response(
                {
                    "session": session.id,
                    "url": credentials["LIVEKIT_URL"],
                    "token": session.token("phone", phone=True),
                }
            )

    async def stop(request):
        async with lock:
            if session:
                await session.close()
        return web.json_response({"stopped": True})

    async def status(request):
        return web.json_response(session.status() if session else {"closed": True})

    async def measurements(request):
        data = await request.json()
        if session and data.get("session") == session.id:
            allowed = (
                "video_rtt_ms",
                "data_rtt_ms",
                "returned_video_frames",
                "video_rtt_samples",
            )
            session.stats["browser_measurements"] = {
                k: data[k]
                for k in allowed
                if k in data and isinstance(data[k], (int, float))
            }
            session.save()
        return web.json_response({"saved": True})

    async def camera_metadata(request):
        data = await request.json()
        if not session or session.closed or data.get("session") != session.id:
            raise web.HTTPConflict(text="请先连接手机相机。")
        if session.calibration or session.scan:
            raise web.HTTPConflict(text="采集开始后不能更改相机设置，请重新连接。")
        mode = data.get("mode")
        if mode not in ("echo", "calibration", "scan"):
            raise web.HTTPBadRequest(text="无效的采集模式。")
        allowed = (
            "width",
            "height",
            "frameRate",
            "aspectRatio",
            "facingMode",
            "deviceId",
            "resizeMode",
            "zoom",
            "focusMode",
            "focusDistance",
        )
        settings = data.get("settings", {})
        outgoing = data.get("outgoing", {})
        if not isinstance(settings, dict) or not isinstance(outgoing, dict):
            raise web.HTTPBadRequest(text="相机设置格式不正确。")
        for dimension in ("width", "height"):
            value = outgoing.get(dimension)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 64 <= value <= 4096
            ):
                raise web.HTTPBadRequest(text="画面尺寸无效。")
        if not settings.get("deviceId"):
            raise web.HTTPBadRequest(
                text="没有读取到相机标识，请重新连接并允许相机权限。"
            )
        camera = dict(
            mode=mode,
            label=str(data.get("label", ""))[:200],
            settings={
                k: v
                for k, v in settings.items()
                if k in allowed and isinstance(v, (str, int, float, bool))
            },
            outgoing={k: outgoing[k] for k in ("width", "height")},
            orientation=str(data.get("orientation", ""))[:80],
            barcode_enabled=mode == "echo",
        )
        if mode == "scan":
            from interactive_rgb_scanning.mobile.bridge import latest_profile

            try:
                profile = latest_profile(calibration_root)
                profile.validate_camera(camera)
            except ValueError as error:
                raise web.HTTPConflict(text=str(error))
            session.scan_profile = profile
        session.camera = camera
        session.stats["mode"] = {
            "calibration": "camera_calibration",
            "echo": "raw_video_echo",
            "scan": "live_rgb_scan_preview",
        }[mode]
        session.save()
        return web.json_response({"saved": True})

    async def scan_profile(request):
        from interactive_rgb_scanning.mobile.bridge import latest_profile

        try:
            profile = latest_profile(calibration_root).public()
            config_path = scan_options["config"] or ROOT / "configs/office.json"
            settings = json.loads(Path(config_path).read_text())
            backend = settings.get("backend", "hi-slam2")
            profile.update(
                backend=backend,
                backend_label={
                    "hi-slam2": "HI-SLAM2",
                    "photo-slam": "Photo-SLAM",
                    "mast3r-slam": "MASt3R-SLAM",
                }.get(backend, backend),
                feedback_enabled=backend == "mast3r-slam",
                model_kind=(
                    "point_cloud" if backend == "mast3r-slam" else "gaussian_splats"
                ),
                feedback_representation=settings.get("feedback_representation"),
                input_video_recording=bool(settings.get("input_video_recording")),
                max_seconds=settings.get("max_seconds", 120),
            )
            return web.json_response(profile)
        except ValueError as error:
            return web.json_response(dict(ready=False, message=str(error)))

    async def scan_action(request):
        data = await request.json()
        async with lock:
            if not session or session.closed or data.get("session") != session.id:
                raise web.HTTPConflict(text="请先连接手机相机。")
            if (
                not session.camera
                or session.camera["mode"] != "scan"
                or not session.scan_profile
            ):
                raise web.HTTPConflict(text="请选择「实时重建」并连接已标定镜头。")
            action = data.get("action")
            if action == "start":
                if session.scan and session.scan.phase not in ("complete", "failed"):
                    raise web.HTTPConflict(
                        text="本轮扫描仍在进行或保存中，请先结束本轮。"
                    )
                if session.scan:
                    await session.scan.stop()
                try:
                    await session.start_scan()
                except ValueError as error:
                    raise web.HTTPConflict(text=str(error))
                except Exception as error:
                    raise web.HTTPBadGateway(
                        text="无法启动本机重建：" + type(error).__name__
                    )
            elif action == "stop":
                if not session.scan:
                    raise web.HTTPConflict(text="尚未开始扫描。")
                await session.scan.stop()
                session.save()
            else:
                raise web.HTTPBadRequest(text="无效的扫描操作。")
            return web.json_response(session.scan.status())

    async def scan_file(request):
        if not session or not session.scan:
            raise web.HTTPNotFound()
        names = {
            "model": session.scan.model_filename,
            "summary": "summary.json",
            "render": "reconstruction.png",
            "camera-video": "recording/camera.mp4",
            "feedback-video": "recording/feedback.mp4",
        }
        name = names.get(request.match_info["kind"])
        if not name:
            raise web.HTTPNotFound()
        if session.scan.phase != "complete":
            raise web.HTTPConflict(text="本轮尚未完成保存。")
        path = session.scan.folder / name
        if not path.is_file():
            raise web.HTTPNotFound(
                text="本轮尚未生成模型，请重新开始并缓慢横向移动手机。"
            )
        response = web.FileResponse(path)
        response.headers["Content-Disposition"] = f'attachment; filename="{path.name}"'
        return response

    async def calibration_action(request):
        data = await request.json()
        async with lock:
            if not session or session.closed or data.get("session") != session.id:
                raise web.HTTPConflict(text="请先连接手机相机。")
            if not session.camera or session.camera["mode"] != "calibration":
                raise web.HTTPConflict(text="请先选择「相机标定」模式再连接。")
            action = data.get("action")
            capture = session.calibration
            if action == "start":
                if capture:
                    raise web.HTTPConflict(
                        text="已有标定采集。可继续采集，或结束连接后重新开始。"
                    )
                if data.get("print_verified") is not True:
                    raise web.HTTPBadRequest(
                        text="请先用尺检查打印尺寸，并确认纸面平整。"
                    )
                try:
                    from interactive_rgb_scanning.mobile.calibration import (
                        CalibrationCapture,
                    )

                    capture = CalibrationCapture(
                        args.output / "calibrations",
                        float(data.get("square_mm", 25)),
                        session.camera,
                    )
                except (TypeError, ValueError):
                    raise web.HTTPBadRequest(text="单格边长应在 10–50 mm 之间。")
                session.calibration = capture
            elif action in ("pause", "resume", "solve"):
                if not capture:
                    raise web.HTTPConflict(text="请先开始采集。")
                if capture.phase in ("complete", "invalid", "solving"):
                    raise web.HTTPConflict(
                        text="本次采集已经结束或正在计算，请查看结果。"
                    )
                capture.phase = "paused"
                if session.calibration_task and not session.calibration_task.done():
                    await asyncio.shield(session.calibration_task)
                if action == "resume":
                    from interactive_rgb_scanning.mobile.calibration import MAX_VIEWS

                    if len(capture.corners) >= MAX_VIEWS:
                        raise web.HTTPConflict(
                            text="已满 40 张，请计算；如未通过，请结束连接后重新采集。"
                        )
                    capture.phase = "collecting"
                    capture.hint = "继续采集，请补充缺少的位置和倾斜视角。"
                elif action == "solve":
                    await asyncio.to_thread(capture.solve)
                else:
                    capture.hint = "采集已暂停。可以继续采集或计算标定。"
                capture.save()
            else:
                raise web.HTTPBadRequest(text="无效的标定操作。")
            session.save()
            return web.json_response(capture.status())

    async def calibration_file(request):
        if not session or not session.calibration:
            raise web.HTTPNotFound()
        files = {
            "corners": "latest-corners.jpg",
            "result": "calibration.json",
            "preview": "undistortion-preview.jpg",
            "intrinsics": "calib-rectified.txt",
        }
        name = files.get(request.match_info["kind"])
        if name is None or not (session.calibration.folder / name).is_file():
            raise web.HTTPNotFound()
        response = web.FileResponse(session.calibration.folder / name)
        if request.match_info["kind"] in ("result", "intrinsics"):
            response.headers["Content-Disposition"] = f'attachment; filename="{name}"'
        return response

    async def calibration_material(request):
        files = {
            "board.pdf": "checkerboard-a4-25mm.pdf",
            "instructions.txt": "instructions-en.txt",
        }
        name = files.get(request.match_info["name"])
        if name is None:
            raise web.HTTPNotFound()
        return web.FileResponse(ROOT / "web/calibration" / name)

    async def diagnostic(request):
        # Startup errors can occur before a room exists. Retain small diagnostics
        # independently of a session, with credentials and URLs removed.
        data = await request.json()
        record = {"received_utc": datetime.now(timezone.utc).isoformat()}
        for key in ("event", "stage", "name", "message", "browser", "visibility"):
            value = str(data.get(key, ""))[:400]
            for secret in (*credentials.values(), access_key):
                value = value.replace(secret, "[redacted]")
            value = re.sub(
                r"eyJ[A-Za-z0-9_.-]+|(?:https?|wss?)://\S+", "[redacted]", value
            )
            record[key] = value
        local = data.get("local")
        if isinstance(local, dict):
            record["local"] = {
                k: v
                for k, v in local.items()
                if k in ("width", "height", "readyState", "paused", "trackState")
                and isinstance(v, (str, int, float, bool))
            }
        with (args.output / "client-diagnostics.jsonl").open("a") as log:
            log.write(json.dumps(record) + "\n")
        return web.json_response({"received": True})

    async def watchdog():
        while True:
            await asyncio.sleep(10)
            if session and not session.closed:
                if (
                    time.monotonic() - session.last_activity > 120
                    or time.monotonic() - session.started > 1800
                ):
                    async with lock:
                        await session.close()

    async def cleanup(application):
        task = asyncio.create_task(watchdog())
        yield
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        if session:
            await session.close()

    app.cleanup_ctx.append(cleanup)
    app.router.add_get("/", index)
    app.router.add_get("/join/{access_key}", index)
    app.router.add_get("/sdk.js", sdk)
    app.router.add_get("/model", model_page)
    app.router.add_get("/model-assets/{name}", model_asset)
    app.router.add_get("/api/models", model_list)
    app.router.add_get("/api/models/{run_id}/{kind}", model_file)
    app.router.add_get("/api/models/{run_id}/{kind:preview}/{camera_id}", model_file)
    app.router.add_get("/health", health)
    app.router.add_post("/api/session", start)
    app.router.add_delete("/api/session", stop)
    app.router.add_get("/api/status", status)
    app.router.add_post("/api/measurements", measurements)
    app.router.add_post("/api/diagnostic", diagnostic)
    app.router.add_post("/api/camera", camera_metadata)
    app.router.add_get("/api/scan/profile", scan_profile)
    app.router.add_post("/api/scan", scan_action)
    app.router.add_get("/api/scan/{kind}", scan_file)
    app.router.add_post("/api/calibration", calibration_action)
    app.router.add_get("/api/calibration/{kind}", calibration_file)
    app.router.add_get("/calibration/{name}", calibration_material)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--page", type=Path, default=ROOT / "web/phone.html")
    parser.add_argument(
        "--sdk", type=Path, default=ROOT / "web/vendor/livekit-client.umd.min.js"
    )
    parser.add_argument("--port", type=int, default=8775)
    parser.add_argument("--scan-python", type=Path)
    parser.add_argument(
        "--scan-config", type=Path, default=ROOT / "configs/office.json"
    )
    parser.add_argument("--scan-output", type=Path)
    parser.add_argument("--calibration-root", type=Path)
    args = parser.parse_args()
    # SDK URLs and JWTs must not be written to general service logs.
    logging.disable(logging.CRITICAL)
    web.run_app(
        create_app(args),
        host="127.0.0.1",
        port=args.port,
        access_log=None,
        print=lambda _: None,
    )


if __name__ == "__main__":
    main()
