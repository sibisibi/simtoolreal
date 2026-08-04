"""Real-hardware bringup for the fr3-xhand deploy stack.

Includes franka_ros2's own franka.launch.py for the core ros2_control stack
(robot_state_publisher, ros2_control_node, joint_state_broadcaster), spawns
our impedance controller with gains read from the deploy contract at launch
time. Payload and collision behaviour go through arm_bringup first, and the
controller spawns only once that has exited, because SetLoad is rejected once
an active torque controller has put the robot in Move mode. Then it brings
up hand_node, goal_node, perception_node, policy_node, a bag recorder, and
rviz2.

perception_node is the exception to the venv rule, it imports FoundationPose
and so runs on the fp conda python, which is also 3.10 and finds system rclpy.
Its registration comes from the init dir that init_scene.py writes, so run
that first and leave the object where it stands.

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
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    AndSubstitution,
    LaunchConfiguration,
    NotSubstitution,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

DEPLOY_PYTHON = "/home/davian/sibeenkim/project/simtoolreal/.venv_deploy/bin/python3"
# perception_node imports FoundationPose, so it runs on the fp env's python,
# which is 3.10 and resolves system rclpy on its own.
FP_PYTHON = "/home/davian/anaconda3/envs/fp/bin/python"
FP_LIB = "/home/davian/anaconda3/envs/fp/lib"
CONTRACT_PATH = (
    "/home/davian/sibeenkim/project/simtoolreal/deployment/fr3_xhand/contract/a4h1.json"
)
DEFAULT_OBJECT_SPEC = (
    "/home/davian/sibeenkim/project/simtoolreal/deployment/fr3_xhand/objects/"
    "davian_handle_eraser.json"
)
DEFAULT_INIT_DIR = "/home/davian/sibeenkim/project/simtoolreal/deployment/fr3_xhand/init/davian_handle_eraser"

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
    # The hand reads its five fingertip sensors in the same RS485 exchange the
    # joints ride on, so recording them costs nothing the loop was not paying.
    "/xhand/tactile",
    # The real arm state at 1 kHz, straight off joint_state_broadcaster, and
    # the policy's actual input. /joint_states above is the 60 Hz republish and
    # stays recorded only so the two can be compared.
    "/franka/joint_states",
    # franka_robot_state_broadcaster runs already and its output was going
    # nowhere. Its robot_state topic is advertised but never published, so the
    # constituent topics are taken instead, all of them at 1 kHz and small.
    # external_joint_torques is the only view of what the arm actually felt.
    "/franka_robot_state_broadcaster/measured_joint_states",
    "/franka_robot_state_broadcaster/desired_joint_states",
    "/franka_robot_state_broadcaster/external_joint_torques",
    "/franka_robot_state_broadcaster/external_wrench_in_base_frame",
    "/franka_robot_state_broadcaster/current_pose",
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
    init_dir_arg = DeclareLaunchArgument("init_dir", default_value=DEFAULT_INIT_DIR)
    # The arm smoke test wants the controller holding position and nothing else.
    # Starting the hand with no power, or the policy before the arm is trusted,
    # are both worse than a second launch file argument.
    arm_only_arg = DeclareLaunchArgument("arm_only", default_value="false")
    # home_robot publishes to /fr3/joint_target too, so it needs the arm and the
    # hand up with the policy absent, otherwise both drive the same topic.
    no_policy_arg = DeclareLaunchArgument("no_policy", default_value="false")
    # hand_node itself requires this and fails loud if it is empty, declaring
    # it here only makes it overridable as `device:=...` instead of the
    # `--ros-args -p` node-scoped override syntax.
    device_arg = DeclareLaunchArgument("device", default_value="")
    # Empty means the rollout keeps poses but not the frames behind them.
    frame_dir_arg = DeclareLaunchArgument("frame_dir", default_value="")

    robot_ip = LaunchConfiguration("robot_ip")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    object_spec = LaunchConfiguration("object_spec")
    init_dir = LaunchConfiguration("init_dir")
    device = LaunchConfiguration("device")
    # Everything downstream of the arm is gated on this.
    loop_nodes = UnlessCondition(LaunchConfiguration("arm_only"))
    # The policy additionally sits out while the robot is being homed.
    policy_only = IfCondition(
        AndSubstitution(
            NotSubstitution(LaunchConfiguration("arm_only")),
            NotSubstitution(LaunchConfiguration("no_policy")),
        )
    )

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
    #
    # Built twice because the two hardware paths sequence it differently, and a
    # single action cannot carry two conditions.
    impedance_params_file = _write_impedance_params_file()

    def make_spawner(condition=None):
        return Node(
            package="controller_manager",
            executable="spawner",
            arguments=[
                IMPEDANCE_CONTROLLER_NAME,
                "-t",
                IMPEDANCE_CONTROLLER_TYPE,
                "-p",
                impedance_params_file,
                "--controller-manager-timeout",
                "30",
            ],
            condition=condition,
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

    # SetLoad is rejected once the robot is in Move mode, which is what an
    # active torque controller puts it in. Spawning the controller alongside
    # arm_bringup is a race, and losing it means the payload is never declared
    # and the arm runs its dynamics as though no hand were attached. The
    # controller therefore waits for arm_bringup to exit, which it does as soon
    # as the payload and the collision behaviour are set.
    spawn_after_bringup = RegisterEventHandler(
        OnProcessExit(
            target_action=arm_bringup,
            on_exit=[make_spawner()],
        ),
        condition=UnlessCondition(use_fake_hardware),
    )
    # Fake hardware has no arm_bringup to wait for.
    spawn_for_fake = make_spawner(condition=IfCondition(use_fake_hardware))

    hand_node = Node(
        package="fr3_xhand_nodes",
        executable="hand_node",
        prefix=DEPLOY_PYTHON,
        parameters=[{"device": device}],
        condition=loop_nodes,
    )
    goal_node = Node(
        package="fr3_xhand_nodes",
        executable="goal_node",
        prefix=DEPLOY_PYTHON,
        parameters=[{"object_spec": object_spec}],
        condition=loop_nodes,
    )
    # The CUDA runtime libs go after the ROS entries, which are what let rclpy
    # import under this interpreter.
    perception_node = Node(
        package="fr3_xhand_nodes",
        executable="perception_node",
        prefix=FP_PYTHON,
        parameters=[
            {
                "object_spec": object_spec,
                "init_dir": init_dir,
                "frame_dir": LaunchConfiguration("frame_dir"),
            }
        ],
        additional_env={"LD_LIBRARY_PATH": os.environ["LD_LIBRARY_PATH"] + ":" + FP_LIB},
        condition=loop_nodes,
        output="screen",
    )
    policy_node = Node(
        package="fr3_xhand_nodes",
        executable="policy_node",
        prefix=DEPLOY_PYTHON,
        parameters=[{"object_spec": object_spec}],
        # joint_state_broadcaster's own output, 1 kHz, rather than /joint_states,
        # which joint_state_publisher republishes at 60 Hz with
        # publish_default_positions true. That republisher exists to feed TF and
        # will emit URDF defaults for any joint it has not heard from, which is
        # what put the URDF default into the first samples of the arm test. A
        # default reaching the policy is a pose that was never measured.
        #
        # The broadcaster orders its joints 1, 3, 6, 7, 2, 4, 5. The node maps
        # by name, so the order does not matter and the remap is the whole
        # bridge. Subscribing at 1 kHz was measured against the 60 Hz tick and
        # costs nothing, 60.00 Hz achieved either way with lower jitter here.
        remappings=[("/fr3/joint_states", "/franka/joint_states")],
        condition=policy_only,
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
            init_dir_arg,
            arm_only_arg,
            no_policy_arg,
            device_arg,
            frame_dir_arg,
            franka_core,
            arm_bringup,
            spawn_after_bringup,
            spawn_for_fake,
            hand_node,
            goal_node,
            perception_node,
            policy_node,
            bag_record,
            rviz,
        ]
    )
