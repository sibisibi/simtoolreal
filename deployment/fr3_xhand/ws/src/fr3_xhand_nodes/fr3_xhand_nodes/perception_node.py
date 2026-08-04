"""Track the object live with FoundationPose and publish its base-frame pose.

Replaces fake_perception_node on the real stack. The node streams aligned
RGB-D from the D435i, registers once against the init frames written by
init_scene.py, then tracks at the camera rate and publishes PoseStamped on
/robot_frame/current_object_pose.

Registration runs on the stored init frames rather than a live one, which
mirrors the demo pipeline, where the mask was drawn on one frame and applied
to another with the object at rest. Move the object after init and the
registration is void, so run init_scene.py again.

The mesh comes from the object spec and is the canonical one, so FoundationPose
returns the canonical frame directly and the goal trajectory needs no further
transform.

Runs in the fp conda env, which is python 3.10 and resolves system rclpy
directly, so FoundationPose and ROS share one process with no bridge.

    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/davian/anaconda3/envs/fp/lib \
        /home/davian/anaconda3/envs/fp/bin/python -m \
        fr3_xhand_nodes.perception_node --ros-args -p init_dir:=<dir>
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile
from scipy.spatial.transform import Rotation

from fr3_xhand_nodes.object_spec import load_object_spec, repo_root_of

# Our own solve now, not the lab's May one, which read a flat board as 3.79 deg
# tilted across two placements. The May file is kept beside this as
# extrinsics_347622076599_may_backup.npz.
LAB_EXTRINSICS = (
    "/home/davian/sibeenkim/project/simtoolreal/deployment/fr3_xhand/calibration/"
    "results/extrinsics_347622076599_latest.npz"
)
CAPTURE_HZ = 30
RESET_SETTLE_S = 6.0
SETTLE_FRAMES = 30  # auto exposure needs a moment before the frames are usable
# A 30 Hz frame that travels 15 cm has jumped 4.5 m/s, far past anything a
# human demo produces, so the track has locked onto something else.
MAX_STEP_M = 0.15
REPORT_EVERY = 150

# Frames go to disk on a writer thread rather than over a topic. Depth is
# aligned to colour before it gets here, so both streams are 1280x720 and the
# pair is 4.61 MB a frame, 138 MB/s at 30 Hz, or 8.3 GB a minute. That is fine
# for the NVMe but not something to put through DDS inside the loop that owes a
# pose every 33 ms. Two seconds of buffer absorbs a stalled write, and a full
# queue means the disk genuinely cannot keep up, which is a failure worth
# hearing about rather than papering over by dropping frames.
#
# Whatever tears this node down has to let it finish. The writer drains on the
# way out and holds its file handles until it does, so killing the process
# early leaves the space allocated to files nobody can see.
FRAME_QUEUE_DEPTH = 60


def ms_to_stamp(ms: float) -> Time:
    return Time(sec=int(ms // 1000), nanosec=int((ms % 1000) * 1e6))


class FrameRecorder:
    """Append raw RGB-D to disk from a writer thread, fixed frame size.

    Raw rather than encoded, because PNG at 1280x720 costs tens of milliseconds
    a frame and the loop owes a pose every 33 ms. Fixed-size records mean a
    reader can np.memmap the file with the shape from meta.json.
    """

    def __init__(self, out_dir: Path, rgb_shape, depth_shape, k: np.ndarray) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir
        self.rgb_file = open(out_dir / "rgb.raw", "wb")
        self.depth_file = open(out_dir / "depth.raw", "wb")
        self.stamp_file = open(out_dir / "stamps_ms.raw", "wb")
        self.queue: queue.Queue = queue.Queue(maxsize=FRAME_QUEUE_DEPTH)
        self.count = 0
        self.dropped = 0
        (out_dir / "meta.json").write_text(
            json.dumps(
                {
                    "rgb_shape": list(rgb_shape),
                    "rgb_dtype": "uint8",
                    "depth_shape": list(depth_shape),
                    "depth_dtype": "uint16",
                    "stamp_dtype": "float64",
                    "stamp_units": "ms, host clock, shutter close",
                    "cam_K": k.tolist(),
                },
                indent=2,
            )
        )
        self.thread = threading.Thread(target=self._drain, daemon=True)
        self.thread.start()

    def _drain(self) -> None:
        while True:
            item = self.queue.get()
            if item is None:
                return
            rgb, depth, stamp_ms = item
            self.rgb_file.write(rgb)
            self.depth_file.write(depth)
            self.stamp_file.write(np.float64(stamp_ms).tobytes())
            self.count += 1

    def put(self, rgb: np.ndarray, depth: np.ndarray, stamp_ms: float) -> None:
        # tobytes copies, which is what lets the tracker reuse the frame buffers.
        self.queue.put_nowait((rgb.tobytes(), depth.tobytes(), stamp_ms))

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join(timeout=30.0)
        for f in (self.rgb_file, self.depth_file, self.stamp_file):
            f.flush()
            f.close()


def pose_to_msg(pose_base: np.ndarray, stamp) -> PoseStamped:
    msg = PoseStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = "robot_frame"
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pose_base[:3, 3]
    x, y, z, w = Rotation.from_matrix(pose_base[:3, :3]).as_quat()
    msg.pose.orientation.x = float(x)
    msg.pose.orientation.y = float(y)
    msg.pose.orientation.z = float(z)
    msg.pose.orientation.w = float(w)
    return msg


class PerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("perception_node")

        self.declare_parameter("object_spec", "")
        self.declare_parameter("init_dir", "")
        self.declare_parameter("extrinsics", LAB_EXTRINSICS)
        # Empty means do not record. The rollout is only replayable with the
        # frames, and they cannot be recovered afterwards.
        self.declare_parameter("frame_dir", "")
        spec_path = self.get_parameter("object_spec").value
        init_dir = self.get_parameter("init_dir").value
        assert spec_path, "object_spec parameter is required"
        assert init_dir, "init_dir parameter is required"

        spec = load_object_spec(spec_path)
        repo_root = repo_root_of(Path(spec_path))
        self.init_dir = Path(init_dir)
        self.t_base_cam = np.load(self.get_parameter("extrinsics").value)["T_base_cam"]

        sys.path.insert(0, str(repo_root / "deployment/fr3_xhand/perception"))
        from tracker import ObjectTracker

        k = np.loadtxt(self.init_dir / "cam_K.txt")
        self.tracker = ObjectTracker(spec["mesh"], k)

        qos = QoSProfile(depth=1)
        self.object_pose_pub = self.create_publisher(
            PoseStamped, "/robot_frame/current_object_pose", qos
        )
        self.get_logger().info(f"tracking {spec['object_name']} from {self.init_dir}")

    def register_from_init(self) -> np.ndarray:
        rgb = cv2.imread(str(self.init_dir / "rgb.png"))
        depth = cv2.imread(str(self.init_dir / "depth.png"), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(self.init_dir / "mask.png"), cv2.IMREAD_GRAYSCALE)
        assert rgb is not None and depth is not None and mask is not None, (
            f"{self.init_dir} needs rgb.png, depth.png and mask.png, run init_scene.py"
        )
        pose_base = self.t_base_cam @ self.tracker.register(rgb, depth, mask)
        self.get_logger().info(f"registered at base xyz {np.round(pose_base[:3, 3], 4)}")
        return pose_base

    def run(self) -> None:
        import pyrealsense2 as rs

        # Without the reset the first wait_for_frames times out whenever another
        # session has held the device, the same reset record_demo.py needs.
        rs.context().query_devices()[0].hardware_reset()
        time.sleep(RESET_SETTLE_S)

        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, CAPTURE_HZ)
        cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, CAPTURE_HZ)
        profile = pipe.start(cfg)
        align = rs.align(rs.stream.color)
        # Global time puts the frame stamps on the host clock, so the capture to
        # arrival term is measurable rather than guessed.
        for sensor in profile.get_device().query_sensors():
            if sensor.supports(rs.option.global_time_enabled):
                sensor.set_option(rs.option.global_time_enabled, 1)
        for _ in range(SETTLE_FRAMES):
            pipe.wait_for_frames()

        prev = self.register_from_init()

        recorder = None
        frame_dir = self.get_parameter("frame_dir").value
        if frame_dir:
            probe = align.process(pipe.wait_for_frames())
            recorder = FrameRecorder(
                Path(frame_dir),
                np.asanyarray(probe.get_color_frame().get_data()).shape,
                np.asanyarray(probe.get_depth_frame().get_data()).shape,
                np.loadtxt(self.init_dir / "cam_K.txt"),
            )
            self.get_logger().info(f"recording frames to {frame_dir}")

        latencies, count = [], 0
        try:
            while rclpy.ok():
                frames = align.process(pipe.wait_for_frames())
                color = frames.get_color_frame()
                # Global time puts this on the host clock, so the stamp is when
                # the shutter closed and downstream staleness counts real age.
                captured_ms = color.get_timestamp()
                rgb = np.asanyarray(color.get_data())
                depth = np.asanyarray(frames.get_depth_frame().get_data())

                pose_base = self.t_base_cam @ self.tracker.track(rgb, depth)
                step = float(np.linalg.norm(pose_base[:3, 3] - prev[:3, 3]))
                assert step < MAX_STEP_M, (
                    f"object jumped {step:.3f} m in one frame, the track is lost"
                )
                prev = pose_base

                self.object_pose_pub.publish(pose_to_msg(pose_base, ms_to_stamp(captured_ms)))
                if recorder is not None:
                    recorder.put(rgb, depth, captured_ms)
                latencies.append(time.time() * 1e3 - captured_ms)
                rclpy.spin_once(self, timeout_sec=0.0)

                count += 1
                if count % REPORT_EVERY == 0:
                    window = np.array(latencies[-REPORT_EVERY:])
                    backlog = "" if recorder is None else f" queue {recorder.queue.qsize()}"
                    self.get_logger().info(
                        f"{count} frames, capture to publish ms "
                        f"p50 {np.percentile(window, 50):.1f} "
                        f"p95 {np.percentile(window, 95):.1f} "
                        f"max {window.max():.1f}{backlog}"
                    )
        finally:
            pipe.stop()
            if recorder is not None:
                recorder.close()
                self.get_logger().info(f"wrote {recorder.count} frames to {frame_dir}")


def main() -> None:
    rclpy.init()
    PerceptionNode().run()


if __name__ == "__main__":
    main()
