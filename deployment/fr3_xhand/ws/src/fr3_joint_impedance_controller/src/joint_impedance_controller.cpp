#include <fr3_joint_impedance_controller/joint_impedance_controller.hpp>

#include <cassert>
#include <string>
#include <vector>

namespace fr3_joint_impedance_controller {

controller_interface::InterfaceConfiguration
JointImpedanceController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= kNumJoints; ++i) {
    config.names.push_back(arm_prefix_ + robot_type_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
JointImpedanceController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= kNumJoints; ++i) {
    config.names.push_back(arm_prefix_ + robot_type_ + "_joint" + std::to_string(i) + "/position");
    config.names.push_back(arm_prefix_ + robot_type_ + "_joint" + std::to_string(i) + "/velocity");
  }
  return config;
}

controller_interface::return_type JointImpedanceController::update(
    const rclcpp::Time& /*time*/,
    const rclcpp::Duration& /*period*/) {
  if (fault_.load()) {
    RCLCPP_FATAL(get_node()->get_logger(), "target topic fault, stopping controller");
    return controller_interface::return_type::ERROR;
  }
  updateJointStates();

  dq_filtered_ = dq_filter_alpha_ * dq_ + (1.0 - dq_filter_alpha_) * dq_filtered_;

  const Target target = *target_buffer_.readFromRT();
  Vector7d q_goal;
  if (!target.valid) {
    // No policy target yet, hold the activation pose.
    q_goal = initial_q_;
  } else if (std::chrono::steady_clock::now() - target.stamp > target_stale_) {
    // Stale policy, brake at the current pose. Firmware gravity compensation
    // holds the arm, the PD only damps residual motion.
    q_goal = q_;
    RCLCPP_ERROR_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000,
                          "joint target stale, holding current pose");
  } else {
    q_goal = Eigen::Map<const Vector7d>(target.q.data());
  }

  const Vector7d tau_d = k_gains_.cwiseProduct(q_goal - q_) - d_gains_.cwiseProduct(dq_filtered_);
  for (int i = 0; i < kNumJoints; ++i) {
    command_interfaces_[i].set_value(tau_d(i));
  }
  return controller_interface::return_type::OK;
}

CallbackReturn JointImpedanceController::on_init() {
  auto_declare<std::string>("robot_type", "fr3");
  auto_declare<std::string>("arm_prefix", "");
  auto_declare<std::vector<double>>("k_gains", {});
  auto_declare<std::vector<double>>("d_gains", {});
  auto_declare<double>("dq_filter_alpha", 1.0);
  auto_declare<int>("target_stale_ms", 200);
  return CallbackReturn::SUCCESS;
}

CallbackReturn JointImpedanceController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  robot_type_ = get_node()->get_parameter("robot_type").as_string();
  arm_prefix_ = get_node()->get_parameter("arm_prefix").as_string();
  arm_prefix_ = arm_prefix_.empty() ? "" : arm_prefix_ + "_";
  const auto k_gains = get_node()->get_parameter("k_gains").as_double_array();
  const auto d_gains = get_node()->get_parameter("d_gains").as_double_array();
  if (k_gains.size() != kNumJoints || d_gains.size() != kNumJoints) {
    RCLCPP_FATAL(get_node()->get_logger(), "k_gains and d_gains must each have %d entries",
                 kNumJoints);
    return CallbackReturn::FAILURE;
  }
  for (int i = 0; i < kNumJoints; ++i) {
    k_gains_(i) = k_gains.at(i);
    d_gains_(i) = d_gains.at(i);
  }
  dq_filter_alpha_ = get_node()->get_parameter("dq_filter_alpha").as_double();
  target_stale_ = std::chrono::milliseconds(get_node()->get_parameter("target_stale_ms").as_int());
  dq_filtered_.setZero();

  std::vector<std::string> expected_names;
  for (int i = 1; i <= kNumJoints; ++i) {
    expected_names.push_back(robot_type_ + "_joint" + std::to_string(i));
  }
  target_sub_ = get_node()->create_subscription<sensor_msgs::msg::JointState>(
      "/fr3/joint_target", rclcpp::QoS(1),
      [this, expected_names](const sensor_msgs::msg::JointState::SharedPtr msg) {
        if (msg->name != expected_names || msg->position.size() != kNumJoints) {
          RCLCPP_FATAL(get_node()->get_logger(), "joint target names or size mismatch");
          fault_.store(true);
          return;
        }
        Target t;
        for (int i = 0; i < kNumJoints; ++i) {
          t.q[i] = msg->position[i];
        }
        t.stamp = std::chrono::steady_clock::now();
        t.valid = true;
        target_buffer_.writeFromNonRT(t);
      });
  return CallbackReturn::SUCCESS;
}

CallbackReturn JointImpedanceController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  updateJointStates();
  dq_filtered_.setZero();
  initial_q_ = q_;
  target_buffer_.writeFromNonRT(Target{});
  return CallbackReturn::SUCCESS;
}

void JointImpedanceController::updateJointStates() {
  for (int i = 0; i < kNumJoints; ++i) {
    const auto& position_interface = state_interfaces_.at(2 * i);
    const auto& velocity_interface = state_interfaces_.at(2 * i + 1);
    assert(position_interface.get_interface_name() == "position");
    assert(velocity_interface.get_interface_name() == "velocity");
    q_(i) = position_interface.get_value();
    dq_(i) = velocity_interface.get_value();
  }
}

}  // namespace fr3_joint_impedance_controller

#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(fr3_joint_impedance_controller::JointImpedanceController,
                       controller_interface::ControllerInterface)
