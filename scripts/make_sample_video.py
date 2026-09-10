#!/usr/bin/env python3
"""Generate a synthetic landscape source video for local testing.

Produces a 16:9 clip with a moving marker, burned-in second counter and an
audio track, so the ingest -> analyse -> clip -> QC pipeline can be exercised
without copyrighted material.

Usage::

    python scripts/make_sample_video.py --seconds 90 --out data/samples/sample.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))

from app.video.ffmpeg import ffmpeg_path, probe, run_ffmpeg  # noqa: E402


def build(out: Path, seconds: int, width: int, height: int, fps: int) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    # The moving box gives the scene detector and the crop tracker real
    # horizontal motion to follow; testsrc2 supplies periodic scene changes.
    video_src = f"testsrc2=size={width}x{height}:rate={fps}:duration={seconds}"
    audio_src = f"sine=frequency=220:sample_rate=48000:duration={seconds}"
    box = (
        "drawbox=x='(w-260)/2 + (w/3)*sin(2*PI*t/12)':y='h/2-130':"
        "w=260:h=260:color=orange@0.85:t=fill"
    )
    args = [
        "-y",
        "-f", "lavfi", "-i", video_src,
        "-f", "lavfi", "-i", audio_src,
        "-filter_complex", f"[0:v]{box}[v]",
        "-map", "[v]", "-map", "1:a",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        str(out),
    ]
    run_ffmpeg(args, timeout=600)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=90)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--out", type=Path, default=Path("data/samples/sample_landscape.mp4")
    )
    args = parser.parse_args()

    print(f"ffmpeg: {ffmpeg_path()}")
    out = build(args.out, args.seconds, args.width, args.height, args.fps)
    info = probe(out)
    print(f"wrote {out} -> {info.to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
