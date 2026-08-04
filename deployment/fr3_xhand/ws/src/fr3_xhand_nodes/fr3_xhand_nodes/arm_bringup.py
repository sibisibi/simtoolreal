"""One-shot node that pushes payload and collision behavior to the arm.

Reads arm_hardware.yaml, calls the franka_hardware SetLoad and
SetFullCollisionBehavior services once each, asserts both succeeded, exits 0.
Only meaningful against real hardware, franka_hardware's param services do
not exist under use_fake_hardware, real.launch.py skips this node there.
"""

from __future__ import annotations

import sys

import rclpy
import yaml
from franka_msgs.srv import SetFullCollisionBehavior, SetLoad
from rclpy.node import Node

SERVICE_WAIT_S = 30.0
SERVICE_CALL_S = 10.0


class ArmBringupNode(Node):
    def __init__(self) -> None:
        super().__init__("arm_bringup")
        self.declare_parameter("config_path", "")
        config_path = self.get_parameter("config_path").value
        assert config_path, "config_path parameter is required"
        config = yaml.safe_load(open(config_path).read())
        self.payload = config["payload"]
        self.collision = config["collision"]

        self.load_client = self.create_client(SetLoad, "/service_server/set_load")
        self.collision_client = self.create_client(
            SetFullCollisionBehavior, "/service_server/set_full_collision_behavior"
        )

    def run(self) -> None:
        assert self.load_client.wait_for_service(timeout_sec=SERVICE_WAIT_S), (
            "set_load service not available"
        )
        assert self.collision_client.wait_for_service(timeout_sec=SERVICE_WAIT_S), (
            "set_full_collision_behavior service not available"
        )

        load_req = SetLoad.Request()
        load_req.mass = float(self.payload["mass"])
        load_req.center_of_mass = [float(v) for v in self.payload["center_of_mass"]]
        load_req.load_inertia = [float(v) for v in self.payload["inertia"]]
        load_resp = self._call(self.load_client, load_req)
        assert load_resp.success, f"set_load failed, {load_resp.error}"
        self.get_logger().info("payload set")

        torque = [float(v) for v in self.collision["torque_thresholds"]]
        force = [float(v) for v in self.collision["force_thresholds"]]
        collision_req = SetFullCollisionBehavior.Request()
        collision_req.lower_torque_thresholds_acceleration = torque
        collision_req.upper_torque_thresholds_acceleration = torque
        collision_req.lower_torque_thresholds_nominal = torque
        collision_req.upper_torque_thresholds_nominal = torque
        collision_req.lower_force_thresholds_acceleration = force
        collision_req.upper_force_thresholds_acceleration = force
        collision_req.lower_force_thresholds_nominal = force
        collision_req.upper_force_thresholds_nominal = force
        collision_resp = self._call(self.collision_client, collision_req)
        assert collision_resp.success, (
            f"set_full_collision_behavior failed, {collision_resp.error}"
        )
        self.get_logger().info("collision behavior set")

    def _call(self, client, request):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=SERVICE_CALL_S)
        assert future.done(), "service call timed out"
        return future.result()


def main() -> None:
    rclpy.init()
    node = ArmBringupNode()
    node.run()
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0)


if __name__ == "__main__":
    main()
