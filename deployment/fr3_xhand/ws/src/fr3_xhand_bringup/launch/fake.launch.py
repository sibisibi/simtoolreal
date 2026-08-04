"""Fake-hardware bringup for the fr3-xhand deploy stack.

Runs fake_robot_node, fake_perception_node, goal_node, and policy_node
against the fake robot loop instead of real hardware, recording the six
contract topics with rosbag2.

fr3_xhand_nodes needs numpy, scipy, and yourdfpy from the .venv_deploy venv,
plus rclpy, which resolves through .venv_deploy because that venv was built
with --system-site-packages against /opt/ros/humble. Each Node action below
sets prefix to the venv's python3, so launch_ros invokes the installed
console script as `<venv>/bin/python3 <script>` instead of through the
script's own shebang, which would point at whatever python colcon build was
run with. `ros2 bag record` does not need the venv, it runs under the sourced
ROS environment directly.
"""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

DEPLOY_PYTHON = "/home/davian/sibeenkim/project/simtoolreal/.venv_deploy/bin/python3"
DEFAULT_OBJECT_SPEC = (
    "/home/davian/sibeenkim/project/simtoolreal/deployment/fr3_xhand/objects/"
    "claw_hammer_swing_down.json"
)

BAG_TOPICS = [
    "/fr3/joint_states",
    "/fr3/joint_target",
    "/xhand/joint_states",
    "/xhand/joint_target",
    "/robot_frame/current_object_pose",
    "/robot_frame/goal_object_pose",
]


def generate_launch_description() -> LaunchDescription:
    object_spec_arg = DeclareLaunchArgument(
        "object_spec", default_value=DEFAULT_OBJECT_SPEC
    )
    object_spec = LaunchConfiguration("object_spec")

    fake_robot_node = Node(
        package="fr3_xhand_nodes",
        executable="fake_robot_node",
        prefix=DEPLOY_PYTHON,
    )
    fake_perception_node = Node(
        package="fr3_xhand_nodes",
        executable="fake_perception_node",
        prefix=DEPLOY_PYTHON,
    )
    goal_node = Node(
        package="fr3_xhand_nodes",
        executable="goal_node",
        prefix=DEPLOY_PYTHON,
        parameters=[{"object_spec": object_spec}],
    )
    policy_node = Node(
        package="fr3_xhand_nodes",
        executable="policy_node",
        prefix=DEPLOY_PYTHON,
        parameters=[{"object_spec": object_spec}],
    )
    bag_record = ExecuteProcess(
        cmd=["ros2", "bag", "record"] + BAG_TOPICS,
        output="screen",
    )

    return LaunchDescription(
        [
            object_spec_arg,
            fake_robot_node,
            fake_perception_node,
            goal_node,
            policy_node,
            bag_record,
        ]
    )
