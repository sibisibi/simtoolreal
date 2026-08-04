"""Real-hardware bringup for the fr3-xhand deploy stack.

Includes franka_ros2's own franka.launch.py for the core ros2_control stack
(robot_state_publisher, ros2_control_node, joint_state_broadcaster), spawns
our impedance controller with gains read from the deploy contract at launch
time, pushes payload and collision behavior through arm_bringup, then brings
up hand_node, goal_node, policy_node, a bag recorder, and rviz2.

fr3_xhand_nodes and arm_bringup need numpy, scipy, yourdfpy, torch from the
.venv_deploy venv, plus rclpy, which resolves through .venv_deploy because
that venv was built with --system-site-packages against /opt/ros/humble.
Each of our Node actions sets prefix to the venv's python3 so launch_ros
invokes the installed console script through that interpreter, matching
fake.launch.py. franka.launch.py's own nodes and our controller spawner run
under the plain sourced ROS environment, they only need rclpy and franka_msgs.
"""

from __future__ import annotations

import json
import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

DEPLOY_PYTHON = "/home/davian/sibeenkim/project/simtoolreal/.venv_deploy/bin/python3"
CONTRACT_PATH = (
    "/home/davian/sibeenkim/project/simtoolreal/deployment/fr3_xhand/contract/a4h1.json"
)
DEFAULT_OBJECT_SPEC = (
    "/home/davian/sibeenkim/project/simtoolreal/deployment/fr3_xhand/objects/"
    "claw_hammer_swing_down.json"
)

ARM_HARDWARE_YAML = os.path.join(
    get_package_share_directory("fr3_xhand_bringup"), "config", "arm_hardware.yaml"
)

IMPEDANCE_CONTROLLER_NAME = "fr3_joint_impedance_controller"
IMPEDANCE_CONTROLLER_TYPE = "fr3_joint_impedance_controller/JointImpedanceController"

BAG_TOPICS = [
    # The real arm state arrives on /joint_states from the franka bringup
    # broadcaster, policy_node is remapped onto it below. /fr3/joint_states
    # stays recorded so fake-loop bags and real bags share a topic list.
    "/joint_states",
    "/fr3/joint_states",
    "/fr3/joint_target",
    "/xhand/joint_states",
    "/xhand/joint_target",
    "/robot_frame/current_object_pose",
    "/robot_frame/goal_object_pose",
]


def _arm_hardware_config() -> dict:
    with open(ARM_HARDWARE_YAML) as f:
        return yaml.safe_load(f)


def _write_impedance_params_file() -> str:
    # k_gains/d_gains are the contract's single source of truth, spawner's
    # -p needs a real file, so one is generated here instead of keeping a
    # second copy of the gains in a tracked yaml.
    with open(CONTRACT_PATH) as f:
        contract = json.load(f)
    arm_names = contract["joint_order"][: contract["num_arm_joints"]]
    k_gains = [contract["arm_stiffness"][name] for name in arm_names]
    d_gains = [contract["arm_damping"][name] for name in arm_names]

    params = {
        "/**": {
            IMPEDANCE_CONTROLLER_NAME: {
                "ros__parameters": {
                    "robot_type": "fr3",
                    "arm_prefix": "",
                    "k_gains": k_gains,
                    "d_gains": d_gains,
                }
            }
        }
    }
    fd, path = tempfile.mkstemp(prefix="fr3_joint_impedance_controller_params_", suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(params, f)
    return path


def generate_launch_description() -> LaunchDescription:
    arm_hardware = _arm_hardware_config()

    robot_ip_arg = DeclareLaunchArgument(
        "robot_ip", default_value=str(arm_hardware["robot_ip"])
    )
    use_fake_hardware_arg = DeclareLaunchArgument("use_fake_hardware", default_value="false")
    object_spec_arg = DeclareLaunchArgument("object_spec", default_value=DEFAULT_OBJECT_SPEC)
    # hand_node itself requires this and fails loud if it is empty, declaring
    # it here only makes it overridable as `device:=...` instead of the
    # `--ros-args -p` node-scoped override syntax.
    device_arg = DeclareLaunchArgument("device", default_value="")

    robot_ip = LaunchConfiguration("robot_ip")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    object_spec = LaunchConfiguration("object_spec")
    device = LaunchConfiguration("device")

    franka_core = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("franka_bringup"), "launch", "franka.launch.py"])
        ),
        launch_arguments={
            "robot_type": "fr3",
            "robot_ip": robot_ip,
            "load_gripper": "false",
            "use_fake_hardware": use_fake_hardware,
        }.items(),
    )

    # Not declared in franka_bringup's controllers.yaml, so type and params
    # are pushed explicitly. controller-manager-timeout matches franka_ros2's
    # own example.launch.py, ros2_control_node has not necessarily finished
    # starting when the spawner process comes up.
    impedance_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            IMPEDANCE_CONTROLLER_NAME,
            "-t",
            IMPEDANCE_CONTROLLER_TYPE,
            "-p",
            _write_impedance_params_file(),
            "--controller-manager-timeout",
            "30",
        ],
        output="screen",
    )

    # franka_hardware's param services only exist against real hardware,
    # arm_bringup waits for them itself so no launch-time delay is needed.
    arm_bringup = Node(
        package="fr3_xhand_nodes",
        executable="arm_bringup",
        prefix=DEPLOY_PYTHON,
        parameters=[{"config_path": ARM_HARDWARE_YAML}],
        condition=UnlessCondition(use_fake_hardware),
        output="screen",
    )

    hand_node = Node(
        package="fr3_xhand_nodes",
        executable="hand_node",
        prefix=DEPLOY_PYTHON,
        parameters=[{"device": device}],
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
        # franka_bringup publishes arm state on /joint_states, the node maps
        # joints by name so the remap is the whole bridge.
        remappings=[("/fr3/joint_states", "/joint_states")],
    )
    bag_record = ExecuteProcess(
        cmd=["ros2", "bag", "record"] + BAG_TOPICS,
        output="screen",
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=[
            "--display-config",
            PathJoinSubstitution([FindPackageShare("franka_description"), "rviz", "visualize_franka.rviz"]),
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            robot_ip_arg,
            use_fake_hardware_arg,
            object_spec_arg,
            device_arg,
            franka_core,
            impedance_controller_spawner,
            arm_bringup,
            hand_node,
            goal_node,
            policy_node,
            bag_record,
            rviz,
        ]
    )
