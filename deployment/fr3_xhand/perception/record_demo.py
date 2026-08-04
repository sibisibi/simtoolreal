"""Record an aligned RGB-D human demo from the D435i in FoundationPose layout.

Output, <out>/rgb/%06d.png, <out>/depth/%06d.png (uint16 mm, aligned to
color), cam_K.txt, timestamps.txt. Frames buffer in RAM during capture so the
30 Hz rate never waits on disk, then write out afterward.

The first frames must show the object alone and unoccluded, the offline
pipeline takes its segmentation mask from frame 0 before any hand enters.

    python deployment/fr3_xhand/perception/record_demo.py --seconds 35 --lead 8
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="")
    parser.add_argument("--seconds", type=float, default=35.0)
    parser.add_argument("--lead", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    out = Path(args.out) if args.out else (
        REPO_ROOT / "deployment/fr3_xhand/demos"
        / f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    (out / "rgb").mkdir(parents=True)
    (out / "depth").mkdir(parents=True)

    ctx = rs.context()
    ctx.query_devices()[0].hardware_reset()
    time.sleep(6)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, args.fps)
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, args.fps)
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)

    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    k = np.array([[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]])
    np.savetxt(out / "cam_K.txt", k)

    for _ in range(int(args.lead * args.fps)):
        pipe.wait_for_frames()
        remaining = args.lead - _ / args.fps
        if _ % args.fps == 0:
            print(f"[record] starting in {remaining:.0f} s, keep hands away from the object")

    n = int(args.seconds * args.fps)
    colors, depths, stamps = [], [], []
    print(f"[record] RECORDING {args.seconds:.0f} s NOW")
    for i in range(n):
        frames = align.process(pipe.wait_for_frames())
        colors.append(np.asanyarray(frames.get_color_frame().get_data()).copy())
        depths.append(np.asanyarray(frames.get_depth_frame().get_data()).copy())
        stamps.append(frames.get_timestamp())
        if i % (5 * args.fps) == 0:
            print(f"[record] {i // args.fps} s of {args.seconds:.0f}")
    pipe.stop()
    print("[record] DONE, writing to disk")

    for i, (c, d) in enumerate(zip(colors, depths)):
        cv2.imwrite(str(out / "rgb" / f"{i:06d}.png"), c)
        cv2.imwrite(str(out / "depth" / f"{i:06d}.png"), d)
    dt = np.diff(np.array(stamps))
    (out / "timestamps.txt").write_text("\n".join(f"{s:.3f}" for s in stamps) + "\n")
    print(f"[record] wrote {len(colors)} frames to {out}")
    print(f"[record] frame interval ms, mean {dt.mean():.1f}, max {dt.max():.1f}")


if __name__ == "__main__":
    main()
