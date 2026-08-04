// Joint impedance controller tracking policy targets with the SimToolReal
// trained gains. Adapted from franka_ros2 joint_impedance_example_controller.
//
// tau = k (q_target - q) - d dq. No coriolis term, the sim actuator applies
// none and the plant supplies coriolis physically in both worlds. Gravity is
// compensated by the robot firmware beneath commanded torques, matching the
// sim robot baked with gravity disabled.

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <string>

#include <Eigen/Eigen>
#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <realtime_tools/realtime_buffer.h>
#include <sensor_msgs/msg/joint_state.hpp>

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace fr3_joint_impedance_controller {

class JointImpedanceController : public controller_interface::ControllerInterface {
 public:
  using Vector7d = Eigen::Matrix<double, 7, 1>;
  [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration()
      const override;
  [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration()
      const override;
  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;
  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;

 private:
  struct Target {
    std::array<double, 7> q{};
    std::chrono::steady_clock::time_point stamp{};
    bool valid{false};
  };

  void updateJointStates();

  std::string robot_type_;
  std::string arm_prefix_;
  static constexpr int kNumJoints = 7;
  Vector7d q_;
  Vector7d initial_q_;
  Vector7d dq_;
  Vector7d dq_filtered_;
  Vector7d k_gains_;
  Vector7d d_gains_;
  double dq_filter_alpha_{1.0};
  std::chrono::milliseconds target_stale_{200};

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr target_sub_;
  realtime_tools::RealtimeBuffer<Target> target_buffer_;
  std::atomic<bool> fault_{false};
};

}  // namespace fr3_joint_impedance_controller
