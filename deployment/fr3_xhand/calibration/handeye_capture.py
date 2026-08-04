"""Eye-to-hand calibration, ChArUco on the hand, solved with cv2.calibrateHandEye.

The camera is fixed to the world and the board rides on the arm, so this is the
eye-to-hand case. OpenCV's calibrateHandEye is written for eye-in-hand, and the
standard way to reuse it is to hand it the inverted robot poses, base in gripper
rather than gripper in base. The loop it solves is then

    T_cam_board = T_cam_base . T_base_ee . T_ee_board

and what comes back is the camera in the base frame, which is what the
perception node loads.

The board's placement on the hand never appears in the answer. It only has to
stay put between poses, so tape is fine and no fixture or measurement is needed.

Guide the arm by hand. Vary orientation, not just position, because two poses
whose rotation axes are parallel leave the rotation undetermined.

Runs on the fp env, which is the only one here carrying pyrealsense2, rclpy and
an OpenCV that still has calibrateHandEye. .venv_deploy's cv2 5.0.0 dropped it.

    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/davian/anaconda3/envs/fp/lib \
        /home/davian/anaconda3/envs/fp/bin/python \
        deployment/fr3_xhand/calibration/handeye_capture.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[3]

# The board actually in the lab, measured. Not the one in byungkunlee's tree,
# which is DICT_4X4_250 with 33 mm squares. This one decodes as 5x5 and its
# squares measure 47 mm, confirmed two ways, by ruler and by solving for the
# size that puts it flat on the table.
ARUCO_DICT = cv2.aruco.DICT_5X5_250
SQUARES_X, SQUARES_Y = 5, 7
SQUARE_M = 0.047
MARKER_RATIO = 0.75

CAPTURE_HZ = 30
RESET_SETTLE_S = 6.0
SETTLE_FRAMES = 30

# A pose is only worth keeping if the board is seen well enough to trust.
MIN_CORNERS = 8
MAX_REPROJ_PX = 1.5
MIN_POSES = 8


def se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).ravel()
    return T


class ArmPose(Node):
    """Latest end effector pose in the base frame, straight from the robot.

    franka_robot_state_broadcaster publishes this at 1 kHz using Franka's own
    kinematics, so there is no URDF to load and no forward kinematics of ours
    to be wrong. Which frame on the arm it refers to does not matter, the board
    is rigid with respect to all of them and the offset is solved out.
    """

    TOPIC = "/franka_robot_state_broadcaster/current_pose"

    def __init__(self) -> None:
        super().__init__("handeye_arm_pose")
        self.T: np.ndarray | None = None
        self.create_subscription(PoseStamped, self.TOPIC, self._on_pose, QoSProfile(depth=1))

    def _on_pose(self, msg: PoseStamped) -> None:
        p, o = msg.pose.position, msg.pose.orientation
        self.T = se3(
            Rotation.from_quat([o.x, o.y, o.z, o.w]).as_matrix(), [p.x, p.y, p.z]
        )


def board_pose(gray, board, detector, k, d):
    """Board in camera frame, plus corner count and reprojection error."""
    corners, ids, _, _ = detector.detectBoard(gray)
    if corners is None or len(corners) < MIN_CORNERS:
        return None, 0 if corners is None else len(corners), np.inf
    obj, img = board.matchImagePoints(corners, ids)
    ok, rvec, tvec = cv2.solvePnP(obj, img, k, d)
    if not ok:
        return None, len(corners), np.inf
    proj, _ = cv2.projectPoints(obj, rvec, tvec, k, d)
    err = float(np.sqrt(((proj.reshape(-1, 2) - img.reshape(-1, 2)) ** 2).sum(axis=1)).mean())
    return se3(cv2.Rodrigues(rvec)[0], tvec), len(corners), err


def solve(t_base_ee: list, t_cam_board: list):
    """Eye-to-hand via the inverted robot poses, every method OpenCV offers."""
    inv = [np.linalg.inv(T) for T in t_base_ee]
    R_b2e = [T[:3, :3] for T in inv]
    t_b2e = [T[:3, 3] for T in inv]
    R_t2c = [T[:3, :3] for T in t_cam_board]
    t_t2c = [T[:3, 3] for T in t_cam_board]

    methods = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    out = {}
    for name, m in methods.items():
        R, t = cv2.calibrateHandEye(R_b2e, t_b2e, R_t2c, t_t2c, method=m)
        out[name] = se3(R, t)
    return out


def residuals(T_base_cam, t_base_ee, t_cam_board):
    """Spread of the recovered board-on-hand transform.

    T_ee_board is a fixed but unknown quantity. Every pose implies its own
    estimate, and if the calibration is right they all agree. The scatter is
    therefore a self-check that needs no ground truth.
    """
    ests = [
        np.linalg.inv(T_be) @ T_base_cam @ T_cb
        for T_be, T_cb in zip(t_base_ee, t_cam_board)
    ]
    pos = np.array([T[:3, 3] for T in ests])
    rot = np.array([Rotation.from_matrix(T[:3, :3]).as_rotvec() for T in ests])
    return (
        1000 * np.linalg.norm(pos - pos.mean(axis=0), axis=1).mean(),
        np.rad2deg(np.linalg.norm(rot - rot.mean(axis=0), axis=1)).mean(),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = ArmPose()
    while node.T is None:
        node.get_logger().warn(f"waiting for {ArmPose.TOPIC}", throttle_duration_sec=2.0)
        rclpy.spin_once(node, timeout_sec=0.2)

    aruco = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_M, SQUARE_M * MARKER_RATIO, aruco
    )
    detector = cv2.aruco.CharucoDetector(board)

    rs.context().query_devices()[0].hardware_reset()
    time.sleep(RESET_SETTLE_S)
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, CAPTURE_HZ)
    profile = pipe.start(cfg)
    for _ in range(SETTLE_FRAMES):
        pipe.wait_for_frames()
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    k = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
    d = np.array(intr.coeffs)

    window = "s capture, q solve, ESC abort"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    t_base_ee: list[np.ndarray] = []
    t_cam_board: list[np.ndarray] = []
    try:
        while True:
            rclpy.spin_once(node, timeout_sec=0.0)
            bgr = np.asanyarray(pipe.wait_for_frames().get_color_frame().get_data())
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            T_cb, n_corners, err = board_pose(gray, board, detector, k, d)

            usable = T_cb is not None and err <= MAX_REPROJ_PX
            canvas = bgr.copy()
            colour = (0, 200, 0) if usable else (0, 0, 255)
            cv2.putText(
                canvas,
                f"corners {n_corners}  reproj {err:.2f}px  captured {len(t_base_ee)}",
                (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2,
            )
            cv2.imshow(window, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("s"):
                if not usable:
                    print(f"[skip] corners {n_corners}, reproj {err:.2f} px")
                    continue
                t_base_ee.append(node.T.copy())
                t_cam_board.append(T_cb)
                print(f"[keep] pose {len(t_base_ee)}, corners {n_corners}, reproj {err:.2f} px")
            elif key == ord("q"):
                break
            elif key == 27:
                print("aborted, nothing written")
                return
    finally:
        pipe.stop()
        cv2.destroyAllWindows()

    assert len(t_base_ee) >= MIN_POSES, f"only {len(t_base_ee)} poses, need {MIN_POSES}"

    sols = solve(t_base_ee, t_cam_board)
    print(f"\n{len(t_base_ee)} poses\n")
    print(f"{'method':<12}{'cam xyz in base':>34}{'board-on-hand spread':>24}")
    best, best_score = None, np.inf
    for name, T in sols.items():
        pos_mm, rot_deg = residuals(T, t_base_ee, t_cam_board)
        print(f"{name:<12}{np.round(T[:3,3],4)!s:>34}{pos_mm:>14.2f} mm{rot_deg:>7.2f} deg")
        if pos_mm < best_score:
            best, best_score, best_name = T, pos_mm, name

    print(f"\nbest by spread: {best_name}")
    print("camera in base frame")
    print(np.round(best, 5))
    print("\nMay solve for comparison")
    may = np.load(
        "/home/davian/byungkunlee/davian_robotics_real3d/deploy/calibration/results/"
        "extrinsics_347622076599_latest.npz"
    )["T_base_cam"]
    print(np.round(may, 5))
    dt = 1000 * np.linalg.norm(best[:3, 3] - may[:3, 3])
    dr = np.rad2deg(
        np.linalg.norm(Rotation.from_matrix(best[:3, :3] @ may[:3, :3].T).as_rotvec())
    )
    print(f"\ndifference from May: {dt:.1f} mm, {dr:.2f} deg")

    np.savez(
        out / "extrinsics_347622076599_latest.npz",
        T_base_cam=best,
        T_cam_base=np.linalg.inv(best),
        intrinsics=k,
        num_poses=len(t_base_ee),
        camera_serial="347622076599",
        method=best_name,
        square_m=SQUARE_M,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    np.savez(out / "poses.npz", t_base_ee=np.array(t_base_ee), t_cam_board=np.array(t_cam_board))
    print(f"\nwrote {out}/extrinsics_347622076599_latest.npz")
    print("nothing downstream changes until this file is copied over the one in use,")
    print("and the goals are rebuilt with recompose_goals.py against it.")


if __name__ == "__main__":
    main()
