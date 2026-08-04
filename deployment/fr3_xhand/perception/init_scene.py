"""Freeze one camera frame and segment the object with SAM 2 for registration.

FoundationPose needs a mask on the frame it registers against. The demo mask
was made with SAM 2, so the live one is too, rather than something weaker.

Click points on the object, left button for the object and right button for
background, then ENTER to segment. The result is shown so it can be redone
before anything is written.

Leave the object where it is after running this, the registration is only
valid while the object has not moved.

Runs in the sam3d conda env, which carries torch 2.5.1 for SAM 2. The fp env
is a version behind and upgrading it would break FoundationPose.

    /home/davian/anaconda3/envs/sam3d/bin/python \
        deployment/fr3_xhand/perception/init_scene.py \
        --out deployment/fr3_xhand/init/davian_handle_eraser
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

SAM2_ROOT = Path("/home/davian/sibeenkim/sam2")
SAM2_CHECKPOINT = SAM2_ROOT / "checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

CAPTURE_HZ = 30
SETTLE_FRAMES = 30  # auto exposure needs a moment before the frame is usable
RESET_SETTLE_S = 6.0
MIN_MASK_PX = 2000


def capture_frame():
    """One aligned RGB-D frame, the same stream config record_demo.py used."""
    # Without the reset the first wait_for_frames times out whenever another
    # session has held the device.
    rs.context().query_devices()[0].hardware_reset()
    time.sleep(RESET_SETTLE_S)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, CAPTURE_HZ)
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, CAPTURE_HZ)
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    for _ in range(SETTLE_FRAMES):
        pipe.wait_for_frames()
    frames = align.process(pipe.wait_for_frames())
    rgb = np.asanyarray(frames.get_color_frame().get_data()).copy()
    depth = np.asanyarray(frames.get_depth_frame().get_data()).copy()
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    k = np.array([[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]])
    pipe.stop()
    return rgb, depth, k


def build_predictor():
    sys.path.insert(0, str(SAM2_ROOT))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    assert SAM2_CHECKPOINT.exists(), f"missing checkpoint {SAM2_CHECKPOINT}"
    model = build_sam2(SAM2_CONFIG, str(SAM2_CHECKPOINT), device="cuda")
    return SAM2ImagePredictor(model)


def prompt_and_segment(rgb_bgr: np.ndarray, predictor, window: str) -> np.ndarray:
    predictor.set_image(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB))
    points: list[tuple[int, int]] = []
    labels: list[int] = []
    mask = np.zeros(rgb_bgr.shape[:2], np.uint8)

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            labels.append(1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append((x, y))
            labels.append(0)

    cv2.setMouseCallback(window, on_mouse)
    while True:
        canvas = rgb_bgr.copy()
        if mask.any():
            canvas[mask > 0] = (0.45 * canvas[mask > 0] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
        for (px, py), lab in zip(points, labels):
            cv2.circle(canvas, (px, py), 6, (0, 255, 0) if lab else (0, 0, 255), -1)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 10) and points:
            masks, scores, _ = predictor.predict(
                point_coords=np.array(points, dtype=np.float32),
                point_labels=np.array(labels, dtype=np.int32),
                multimask_output=True,
            )
            best = int(np.argmax(scores))
            mask = (masks[best] > 0).astype(np.uint8) * 255
            print(f"[init] segmented {int((mask > 0).sum())} px, score {scores[best]:.3f}")
        elif key == ord("u") and points:
            points.pop()
            labels.pop()
        elif key == ord("y") and mask.any():
            break
    cv2.destroyAllWindows()
    return mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rgb, depth, k = capture_frame()

    # The window has to exist before SAM 2 is built. Building the model first
    # leaves the later namedWindow spinning and no window ever maps, which
    # looks like a hang with nothing on screen. Isolated by elimination, torch
    # on its own and a full RealSense capture on its own both keep windows
    # working, and only build_sam2 ahead of the window reproduces it.
    window = "left click object, right click background, ENTER segment, u undo, y accept"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.imshow(window, rgb)
    cv2.waitKey(50)

    predictor = build_predictor()
    mask = prompt_and_segment(rgb, predictor, window)

    covered = int((mask > 0).sum())
    assert covered > MIN_MASK_PX, f"mask covers only {covered} pixels, prompt the object again"

    cv2.imwrite(str(out / "rgb.png"), rgb)
    cv2.imwrite(str(out / "depth.png"), depth)
    cv2.imwrite(str(out / "mask.png"), mask)
    np.savetxt(out / "cam_K.txt", k)
    (out / "captured_at.txt").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    print(f"[init] wrote rgb, depth, mask over {covered} pixels, and cam_K to {out}")


if __name__ == "__main__":
    main()
