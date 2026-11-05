#!/usr/bin/env python3
"""Fetch the three author-published MASt3R-SLAM checkpoint files, with provenance."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

EXPECTED_SHA256 = {
    "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth": "e28f91b488554653e2b46ddae9c78c1143e0bcb2e27d3e26cdb0b717f1568eb2",
    "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth": "000a324f06f9369ab3e32d03dd4b8a838d48e98e3f3bf9d7dd034dce1bfd4c50",
    "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl": "700af9e31ad0ca6159955dc80b4fd6e14234780b52f5c91884f6553991933c1a",
}

BASE = "https://download.europe.naverlabs.com/ComputerVision/MASt3R/"
FILES = [
    "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
    "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth",
    "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl",
]


def download_ranges(url, partial, expected, workers=8):
    chunk = 32 * 1024 * 1024
    pieces = partial.with_suffix(partial.suffix + ".ranges")
    pieces.mkdir(exist_ok=True)
    # Preserve completed prefix blocks from a previous ordinary resumable transfer.
    prefix = partial.stat().st_size if partial.exists() else 0
    if prefix:
        with partial.open("rb") as old:
            for start in range(0, min(prefix, expected), chunk):
                length = min(chunk, expected - start)
                if start + length > prefix:
                    break
                piece = pieces / str(start)
                if not piece.exists() or piece.stat().st_size != length:
                    piece.write_bytes(old.read(length))
                else:
                    old.seek(length, 1)

    def one(start):
        length = min(chunk, expected - start)
        piece = pieces / str(start)
        if piece.exists() and piece.stat().st_size == length:
            return
        end = start + length - 1
        for attempt in range(4):
            try:
                request = urllib.request.Request(
                    url, headers={"Range": f"bytes={start}-{end}"}
                )
                with urllib.request.urlopen(request, timeout=90) as response:
                    if (
                        response.status != 206
                        or response.headers.get("Content-Range")
                        != f"bytes {start}-{end}/{expected}"
                    ):
                        raise RuntimeError("Server returned an incorrect byte range")
                    with piece.open("wb") as out:
                        shutil.copyfileobj(response, out, 1024 * 1024)
                if piece.stat().st_size != length:
                    raise RuntimeError("Incomplete byte range")
                return
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)

    starts = list(range(0, expected, chunk))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, start) for start in starts]
        for done, future in enumerate(as_completed(futures), 1):
            future.result()
            if done % 8 == 0 or done == len(starts):
                print(
                    json.dumps(
                        {
                            "file": partial.name,
                            "blocks_complete": done,
                            "blocks_total": len(starts),
                        }
                    ),
                    flush=True,
                )
    with partial.open("wb") as out:
        for start in starts:
            with (pieces / str(start)).open("rb") as src:
                shutil.copyfileobj(src, out, 1024 * 1024)


def fetch(folder, name):
    url = BASE + name
    with urllib.request.urlopen(
        urllib.request.Request(url, method="HEAD"), timeout=40
    ) as r:
        expected = int(r.headers["Content-Length"])
        modified = r.headers.get("Last-Modified")
    target = folder / name
    if not target.exists() or target.stat().st_size != expected:
        partial = target.with_suffix(target.suffix + ".part")
        if expected > 64 * 1024 * 1024:
            download_ranges(url, partial, expected)
        else:
            subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--retry",
                    "4",
                    "--retry-delay",
                    "2",
                    "--continue-at",
                    "-",
                    "--silent",
                    "--show-error",
                    "--output",
                    str(partial),
                    url,
                ],
                check=True,
            )
        if partial.stat().st_size != expected:
            raise RuntimeError("Incomplete checkpoint: " + name)
        partial.replace(target)
    digest = hashlib.file_digest(target.open("rb"), "sha256").hexdigest()
    if digest != EXPECTED_SHA256[name]:
        raise RuntimeError("Checkpoint does not match the registered model: " + name)
    print(json.dumps({"file": name, "bytes": expected, "verified": True}), flush=True)
    return dict(
        file=name,
        url=url,
        bytes=expected,
        sha256=digest,
        last_modified=modified,
        hash_scope="locally computed download fingerprint; not an author-supplied checksum",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(lambda n: fetch(args.output, n), FILES))
    (args.output / "provenance.json").write_text(
        json.dumps(
            dict(source="Official MASt3R-SLAM README checkpoint URLs", files=records),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
