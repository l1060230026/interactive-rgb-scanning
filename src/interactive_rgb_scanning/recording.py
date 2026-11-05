"""Timestamped scan video, exact RGB replay inputs and sparse native map snapshots."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import queue
import threading
import time
import zipfile

import numpy as np


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


class TimestampedVideo:
    """Encode off the media loop, retaining each submitted RGB frame and its clock."""

    def __init__(self, folder, name, width, height, fps, *, lossless=False):
        import av

        self.av = av
        self.folder = Path(folder)
        self.name, self.width, self.height = name, width, height
        self.fps, self.lossless = fps, lossless
        self.queue = queue.Queue(maxsize=64)
        self.frames = 0
        self.error = None
        self.closed = False
        self.thread = threading.Thread(
            target=self._run, name="scan-video-" + name, daemon=True
        )
        self.thread.start()

    def append(self, payload, elapsed, metadata):
        if self.error:
            raise RuntimeError("录像写入失败，请结束扫描并保留本轮记录：" + self.error)
        if self.closed:
            raise RuntimeError("Recording is already closed")
        if len(payload) != self.width * self.height * 3:
            raise ValueError("Recording frame dimensions do not match the camera")
        try:
            self.queue.put_nowait((bytes(payload), float(elapsed), dict(metadata)))
        except queue.Full:
            raise RuntimeError(
                "录像写入速度不足，扫描已暂停保存，已收到的画面会保留。"
            ) from None

    def _run(self):
        containers = []
        try:
            preview = self.av.open(
                str(self.folder / (self.name + ".mp4")),
                "w",
                options={"movflags": "frag_keyframe+empty_moov+default_base_moof"},
            )
            containers.append(preview)
            stream = preview.add_stream("libx264", rate=Fraction(str(self.fps)))
            stream.width, stream.height = self.width, self.height
            stream.pix_fmt = "yuv420p"
            stream.time_base = Fraction(1, 1_000_000)
            stream.codec_context.time_base = Fraction(1, 1_000_000)
            stream.codec_context.thread_count = 1
            stream.options = {"preset": "veryfast", "crf": "19", "tune": "zerolatency"}
            encoders = [(preview, stream)]
            if self.lossless:
                exact = self.av.open(
                    str(self.folder / (self.name + "-lossless.mkv")), "w"
                )
                containers.append(exact)
                exact_stream = exact.add_stream("ffv1", rate=Fraction(str(self.fps)))
                exact_stream.width, exact_stream.height = self.width, self.height
                exact_stream.pix_fmt = "bgr0"
                exact_stream.time_base = Fraction(1, 1_000_000)
                exact_stream.codec_context.time_base = Fraction(1, 1_000_000)
                exact_stream.codec_context.thread_count = 1
                encoders.append((exact, exact_stream))
            previous_pts = -1
            with (self.folder / (self.name + "-frames.jsonl")).open("w") as index:
                while True:
                    item = self.queue.get()
                    if item is None:
                        break
                    payload, elapsed, metadata = item
                    rgb = np.frombuffer(payload, np.uint8).reshape(
                        self.height, self.width, 3
                    )
                    pts = max(previous_pts + 1, round(elapsed * 1_000_000))
                    for container, encoder in encoders:
                        frame = self.av.VideoFrame.from_ndarray(rgb, format="rgb24")
                        frame.pts, frame.time_base = pts, Fraction(1, 1_000_000)
                        for packet in encoder.encode(frame):
                            container.mux(packet)
                    index.write(
                        json.dumps(
                            dict(
                                metadata,
                                video_index=self.frames,
                                elapsed_seconds=elapsed,
                                video_pts_us=pts,
                            ),
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    index.flush()
                    self.frames += 1
                    previous_pts = pts
            for container, encoder in encoders:
                for packet in encoder.encode():
                    container.mux(packet)
        except Exception as error:
            self.error = str(error)[:300]
        finally:
            for container in containers:
                try:
                    container.close()
                except Exception as error:
                    self.error = self.error or str(error)[:300]

    def close(self):
        if not self.closed:
            self.closed = True
            while self.thread.is_alive():
                try:
                    self.queue.put(None, timeout=0.2)
                    break
                except queue.Full:
                    continue
            self.thread.join()
        if self.error:
            raise RuntimeError("Recording encoder failed: " + self.error)


class ScanRecording:
    def __init__(self, folder, camera, config):
        self.folder = Path(folder) / "recording"
        self.folder.mkdir()
        self.started = time.monotonic()
        self.created_utc = datetime.now(timezone.utc).isoformat()
        width, height = (camera["outgoing"][k] for k in ("width", "height"))
        fps = float(camera["settings"].get("frameRate", 15))
        self.camera_video = TimestampedVideo(
            self.folder, "camera", width, height, fps, lossless=True
        )
        self.feedback_video = TimestampedVideo(
            self.folder, "feedback", width, height, fps
        )
        self.inputs = {}
        self.feedback_frames = 0
        self.closed, self.error = False, None
        self.processed = (self.folder / "processed-inputs.jsonl").open("w")
        self.config = config
        self.save()

    def camera_frame(self, payload, seq, received_at, source_timestamp_us=None):
        index = len(self.inputs)
        self.camera_video.append(
            payload,
            received_at - self.started,
            dict(
                input_seq=int(seq),
                received_at=float(received_at),
                source_timestamp_us=source_timestamp_us,
            ),
        )
        self.inputs[int(seq)] = index

    def processed_frame(self, seq):
        self.processed.write(
            json.dumps(
                dict(
                    input_seq=int(seq),
                    camera_video_index=self.inputs[int(seq)],
                    sent_at=time.monotonic(),
                ),
                allow_nan=False,
            )
            + "\n"
        )
        self.processed.flush()

    def feedback_frame(self, payload, header):
        self.feedback_video.append(
            payload,
            time.monotonic() - self.started,
            dict(
                input_seq=header.get("input_seq"),
                render_input_seq=header.get("render_input_seq"),
                output_kind=header.get("output_kind"),
            ),
        )
        self.feedback_frames += 1

    def status(self):
        error = self.error or self.camera_video.error or self.feedback_video.error
        return dict(
            enabled=True,
            status="failed" if error else "complete" if self.closed else "recording",
            camera_frames=len(self.inputs),
            encoded_camera_frames=self.camera_video.frames,
            feedback_frames=self.feedback_frames,
            error=error,
            exact_replay_video="recording/camera-lossless.mkv",
            input_preview="recording/camera.mp4",
            feedback_preview="recording/feedback.mp4",
        )

    def save(self):
        write_json(
            self.folder / "manifest.json",
            dict(
                self.status(),
                created_utc=self.created_utc,
                clock="receiver monotonic seconds; video PTS in the accompanying frame indices",
                input_scope="Every decoded RGB offered during this scan, before latest-frame queue selection",
                exact_replay="Lossless FFV1 RGB frames selected by processed-inputs.jsonl and worker frames.jsonl",
                preview_scope="H264 viewing copies; use camera-lossless.mkv for numerical replay",
            ),
        )

    def close(self):
        if self.closed:
            return
        for video in (self.camera_video, self.feedback_video):
            try:
                video.close()
            except Exception as error:
                self.error = self.error or str(error)[:300]
        self.processed.close()
        self.closed = True
        self.save()
        if self.error:
            raise RuntimeError(self.error)


class NativeMapRecording:
    """Retain a few complete native states; arbitrary other states can be replayed."""

    def __init__(self, folder, checkpoint_seconds):
        self.folder = Path(folder) / "recording"
        self.folder.mkdir(exist_ok=True)
        self.checkpoint_seconds = list(checkpoint_seconds)
        self.next_checkpoint = 0
        self.origin = None
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="scan-map-save"
        )
        self.pending = []
        self.checkpoints = []
        self.cues = zipfile.ZipFile(
            self.folder / "cue-history.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=1
        )
        self.cue_index = (self.folder / "cue-frames.jsonl").open("w")

    def record_cues(self, feedback, row):
        packet, display = getattr(feedback, "last_packet", None), getattr(
            feedback, "last_display", None
        )
        if packet is None or display is None:
            return
        fields = (
            "raw_blue",
            "raw_red",
            "defined",
            "normal_blue",
            "normal_red",
            "light_red",
        )
        masks = [
            np.asarray((packet if name in packet else display)[name], bool)
            for name in fields
        ]
        shape = masks[0].shape
        name = "frame-%09d.bin" % int(row["seq"])
        self.cues.writestr(name, np.packbits(np.stack(masks).reshape(-1)).tobytes())
        self.cue_index.write(
            json.dumps(
                dict(
                    file=name,
                    fields=fields,
                    shape=shape,
                    input_seq=int(row["seq"]),
                    capture_time=float(row["capture_time"]),
                    tracking_ok=bool(row["tracking_ok"]),
                ),
                allow_nan=False,
            )
            + "\n"
        )
        self.cue_index.flush()

    def record(self, runtime, feedback, row, *, final=False):
        stamp = float(row["capture_time"])
        if self.origin is None:
            self.origin = stamp
        elapsed = stamp - self.origin
        due = (
            self.next_checkpoint < len(self.checkpoint_seconds)
            and elapsed >= self.checkpoint_seconds[self.next_checkpoint]
        )
        if not final and not due:
            return
        if due:
            while (
                self.next_checkpoint < len(self.checkpoint_seconds)
                and elapsed >= self.checkpoint_seconds[self.next_checkpoint]
            ):
                self.next_checkpoint += 1
        n = len(runtime.keyframes)
        if not n:
            return
        k = runtime.keyframes
        data = dict(
            points=k.X[:n].detach().cpu().numpy().copy(),
            confidence=(k.C[:n, :, 0] / k.N[:n, None].clamp_min(1))
            .detach()
            .cpu()
            .numpy()
            .copy(),
            sim3=k.T_WC[:n].detach().cpu().numpy().copy(),
            rgb=np.clip(k.uimg[:n].numpy() * 255, 0, 255).astype("u1"),
            input_seq=k.dataset_idx[:n].cpu().numpy().copy(),
            K=runtime.K.cpu().numpy().copy(),
            shape=np.array([runtime.h, runtime.w]),
            query_rgb=np.clip(runtime.last_frame.uimg.numpy() * 255, 0, 255).astype(
                "u1"
            ),
            query_pose=runtime.last_output_pose.matrix().cpu().numpy().copy(),
        )
        packet, display = getattr(feedback, "last_packet", None), getattr(
            feedback, "last_display", None
        )
        if packet is not None:
            data.update(
                {
                    key: np.asarray(packet[key]).copy()
                    for key in ("raw_blue", "raw_red", "native_owner", "depth_native")
                }
            )
        if display is not None:
            data.update(
                {
                    key: np.asarray(display[key]).copy()
                    for key in ("normal_blue", "normal_red", "light_red")
                }
            )
        evidence = getattr(feedback, "last_evidence", None)
        if evidence is not None:
            data["retained"] = evidence["retained"].detach().cpu().numpy().copy()
        filename = ("final" if final else "map-%06d" % round(elapsed)) + ".npz"
        entry = dict(
            file=filename,
            input_seq=int(row["seq"]),
            capture_time=stamp,
            elapsed_seconds=elapsed,
            keyframes=n,
            final=final,
        )
        self.checkpoints.append(entry)
        self.pending.append(
            self.executor.submit(np.savez_compressed, self.folder / filename, **data)
        )
        write_json(self.folder / "checkpoints.json", self.checkpoints)

    def close(self):
        self.executor.shutdown(wait=True)
        self.cues.close()
        self.cue_index.close()
        for pending in self.pending:
            pending.result()
