"""Pull the arm target stream and its measured response out of a session bag.

rosbag2_py belongs to the system ROS python and the Isaac Sim venv is a minor
version ahead of it, so the bag cannot be read from the replay script. This
writes the two arrays to an npz, which is the boundary the replay reads, the
same way the contract JSON carries the sim's orderings out to the deploy nodes.

Measured positions and velocities are interpolated onto the target stamps, so
every array shares one time base and the replay can difference them directly.
Velocity is carried because the trajectory error in Tune to Learn, appendix
equation 8, sums a position and a velocity term.

    /usr/bin/python3 deployment/fr3_xhand/bag_to_targets.py \
        --bag <rosbag dir> --out arm_targets.npz
"""

from __future__ import annotations

import argparse

import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from sensor_msgs.msg import JointState

ARM_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    reader = SequentialReader()
    reader.open(StorageOptions(uri=args.bag, storage_id="sqlite3"), ConverterOptions("", ""))
    state_t, state_q, state_v, target_t, target_q = [], [], [], [], []
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        msg = deserialize_message(data, JointState)
        if topic == "/joint_states" and all(n in msg.name for n in ARM_NAMES):
            idx = [list(msg.name).index(n) for n in ARM_NAMES]
            state_t.append(stamp / 1e9)
            state_q.append([msg.position[i] for i in idx])
            state_v.append([msg.velocity[i] for i in idx] if len(msg.velocity) else [0.0] * len(idx))
        elif topic == "/fr3/joint_target":
            target_t.append(stamp / 1e9)
            target_q.append(list(msg.position))

    state_t, state_q, state_v = np.array(state_t), np.array(state_q), np.array(state_v)
    target_t, target_q = np.array(target_t), np.array(target_q)
    assert len(target_t) > 0, f"no /fr3/joint_target in {args.bag}"
    assert len(state_t) > 0, f"no /joint_states carrying the arm joints in {args.bag}"

    measured = np.stack(
        [np.interp(target_t, state_t, state_q[:, j]) for j in range(len(ARM_NAMES))], axis=1
    )
    measured_vel = np.stack(
        [np.interp(target_t, state_t, state_v[:, j]) for j in range(len(ARM_NAMES))], axis=1
    )
    np.savez(args.out, targets=target_q, measured=measured,
             measured_vel=measured_vel, stamps=target_t)
    print(f"[bag] {len(target_q)} targets and the matching response -> {args.out}")


if __name__ == "__main__":
    main()
