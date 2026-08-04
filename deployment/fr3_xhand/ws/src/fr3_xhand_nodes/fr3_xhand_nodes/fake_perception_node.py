"""rclpy port of deployment/fake/fake_perception_node.py.

Publishes a fixed object pose in place of a real perception pipeline.
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile

PERCEPTION_RATE_HZ = 30.0


class FakePerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("fake_perception_node")

        self.declare_parameter("pose_xyz", [0.45, 0.0, 0.55])
        self.declare_parameter("pose_xyzw", [0.0, 0.0, 0.0, 1.0])
        pose_xyz = self.get_parameter("pose_xyz").value
        pose_xyzw = self.get_parameter("pose_xyzw").value
        assert len(pose_xyz) == 3, f"pose_xyz has length {len(pose_xyz)}, expected 3"
        assert len(pose_xyzw) == 4, f"pose_xyzw has length {len(pose_xyzw)}, expected 4"
        self.pose_xyz = pose_xyz
        self.pose_xyzw = pose_xyzw

        qos = QoSProfile(depth=1)
        self.object_pose_pub = self.create_publisher(
            PoseStamped, "/robot_frame/current_object_pose", qos
        )
        self.create_timer(1.0 / PERCEPTION_RATE_HZ, self._publish_object_pose)

    def _publish_object_pose(self) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "robot_frame"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = self.pose_xyz
        x, y, z, w = self.pose_xyzw
        msg.pose.orientation.x = x
        msg.pose.orientation.y = y
        msg.pose.orientation.z = z
        msg.pose.orientation.w = w
        self.object_pose_pub.publish(msg)


def main() -> None:
    rclpy.init()
    rclpy.spin(FakePerceptionNode())


if __name__ == "__main__":
    main()
