# Deploy the SimToolReal checkpoint on FR3 plus XHand

## 1. Context

### 1.1 The goal

The user needs a real-world baseline for their research, and SimToolReal is the strongest candidate. They ported the lab robot, a Franka FR3 v2 arm with a RobotEra XHand, into the SimToolReal codebase and training is ongoing. The checkpoint at `runs/a4h1_20260731_085255/0_simtoolreal_sapg/last/model.pth` exists now. The deliverable is deployment infrastructure, not peak performance, so any future checkpoint deploys at ease. One RealSense D435i is mounted, fixed third person. Everything runs on this machine, Ubuntu 22.04.5, PREEMPT_RT kernel, RTX 4090.

Two principles govern every choice. Real must match sim, the actuator law, the gains, the observation math. Our stack must match the SimToolReal authors' stack, deviating only where the robot itself differs, they ran a KUKA iiwa 14 with a 22 DoF Sharpa hand, we run FR3 with a 12 DoF XHand.

### 1.2 What exploration established

**The policy contract.** The checkpoint is a state policy, no images, no point clouds. Verified from the weight tensors, obs 110, actions 19. The obs vector concatenates joint_pos 19 normalized to [-1,1] by URDF limits, joint_vel 19 raw with trained noise 0.1 rad/s, prev_action_targets 19 in radians, palm_pos 3 from body `fr3_link7` plus local offset (0,0,0.16), palm_rot 4 in xyzw, object_rot 4 in xyzw, fingertip_pos_rel_palm 15, keypoints_rel_palm 12, keypoints_rel_goal 12, object_scales 3, then clamps to plus minus 10. Fingertip order follows Isaac Lab `find_bodies()` order, not the config tuple, so it must be captured from a live env. Actions run at 60 Hz. The arm channel integrates, prev plus 1.5 times dt times action, clamp, EMA 0.1, clamp again. The hand channel maps [-1,1] affinely into limits, EMA 0.1, clamp. The sim actuator is implicit PD with arm kp 400 and kd [31.1307, 31.1307, 27.2029, 27.2029, 18.1328, 18.1328, 18.1328], hand kp 3.0 kd 0.1, and the robot links are baked with gravity disabled (`scene_utils.py:1632`). Training randomized delays, obs and action up to 3 steps, object pose up to 10 steps which is about 167 ms, and pose noise 1 cm and 5 degrees, so the policy tolerates realistic latency. The rl_games loader `deployment/rl_player.py` appends a constant 50.0 column for SAPG and needs a config with a top level `train:` key, the hydra run config stores that block under `agent:`. The LSTM needs `init_rnn()` per episode.

**What the repo deploy code lacks.** `deployment/` is ROS1 and hardwired to the authors' 29 DoF robot, obs dim 140, Sharpa joint literals, iiwa topic names. The arm driver was never in the repo, the authors drove the KUKA through an external package over two JointState topics. Perception is an external FoundationPose fork whose ROS1 node publishes object pose already in robot frame, extrinsics come from a calibration text file, and the paper never documents the calibration procedure. No Isaac Sim sim2sim harness exists, the existing one targets Isaac Gym. Known upstream warts to drop, live `breakpoint()` calls on control paths, an object scale mismatch between the goal node and the policy node, discarded message timestamps, no staleness checks.

**Why the stack choices are faithful.** Franka firmware always adds gravity compensation under external torque commands, and sim disabled gravity on the robot links, so raw PD torque on hardware reproduces the sim actuator. Coriolis stays uncompensated in both, sim leaves it in the dynamics, so the official examples' added coriolis term must be omitted. Only the external torque path exposes both kp and kd on the FR3, the internal impedance controller takes stiffness only.

### 1.3 Decisions taken with the user

- **ROS2 Humble with franka_ros2.** A custom C++ joint impedance controller adapted from the official `joint_impedance_example_controller`, which already computes tau equals kp times error minus kd times dq without coriolis. The 1 kHz loop stays in C++ inside controller_manager, no Python in the real-time path. The user chose the widely adopted vendor path over a hand-rolled bus after an explicit options review.
- **Full paper perception pipeline.** Record a human demo with the D435i, SAM 2 mask, SAM 3D metric mesh from real depth, handle and head split, grasp box extents, FoundationPose on the demo, goals downsampled to 3 Hz with lift-off truncation. Live tracking runs FoundationPose at 30 Hz. We design the eye-to-hand calibration ourselves since the paper omits it.
- **House style governs everything.** `vault/coding-style.md` for code, fail loud, no fallback handling, smallest change. `vault/writing-style.md` and the humanizer for all prose including comments and docs.

## 2. The plan

**Step zero.** `git pull --ff-only` in the repo, remote main is a verified descendant of local main. Then confirm `ROBOT_PROFILES["fr3-xhand-a4h1"]` exists in `isaacsimenvs/tasks/simtoolreal/robots.py` with the kd vector above. The user confirmed the pulled registry holds the exact checkpoint setup.

**New code layout.** All new code lives under `deployment/fr3_xhand/`, following the repo convention that deployment code sits in `deployment/`.

```
deployment/fr3_xhand/
  export_contract.py               Isaac Sim venv, run dir -> contract JSON + train yaml
  contract/a4h1.json               generated, committed, the single cross-env artifact
  contract/a4h1_train.yaml         generated, {train: <agent section>} shim for RlPlayer
  sim2sim_parity.py                Isaac Sim venv, obs/action parity + deploy-in-loop rollout
  calibration/collect_handeye.py   eye-to-hand data collection
  calibration/solve_handeye.py     cv2.calibrateHandEye -> T_camera_base.json
  perception/make_object_spec.py   paper pipeline, demo -> object spec + goal trajectory
  ws/                              colcon workspace, build dirs gitignored
    fr3_xhand.repos                pins franka_ros2 + libfranka versions
    src/fr3_xhand_nodes/           ament_python, contract.py, obs_action.py, policy_node.py,
                                   goal_node.py, hand_node.py, fake_robot_node.py,
                                   fake_perception_node.py, home_robot.py
    src/fr3_joint_impedance_controller/   ament_cmake C++, adapted vendor example
    src/fr3_xhand_bringup/         fake.launch.py, real.launch.py, config, rviz
```

**Python environments.** Three, with the contract JSON as the only boundary. The existing Isaac Sim venv runs export and parity. A deploy venv on python3.10 with system site packages picks up Humble rclpy and runs every node, `rl_player.py` imports there since rl_games is vendored and `isaacgymenvs/__init__.py` pulls only hydra and omegaconf. The FoundationPose conda env runs perception only.

**Topics.** Mirror the upstream contract with fr3 and xhand names. `/fr3/joint_states` 1 kHz from the franka_ros2 broadcaster, `/fr3/joint_target` 60 Hz, `/xhand/joint_states` at 60 Hz or better, `/xhand/joint_target` 60 Hz, `/robot_frame/current_object_pose` PoseStamped 30 Hz, `/robot_frame/goal_object_pose` PoseStamped 60 Hz, upgraded from Pose so staleness is checkable.

### Phase 1, contract and sim2sim parity, no hardware

- **1.1 Contract export.** Write `export_contract.py`, a headless Isaac Lab script patterned on `isaacsimenvs/tests/test_pretrained_rollout.py`, building the env from the run's hydra config and dumping live state, joint order and limits, fingertip body names in Lab order with per-tip offsets, palm body and offset, robot base pose (0, 0.48, 0.53) with the minus 90 degree z yaw as T_world_base, table pose, action constants, obs list with keypoint constants, gains, default arm pose, checkpoint path, dims. Also write the `{train: ...}` yaml shim. Add `isaacsimenvs/tests/test_deploy_contract.py` asserting byte equality between a fresh export and the committed JSON.
- **1.2 Deploy math and player smoke.** Write `obs_action.py`, pure numpy, no ROS imports. Obs builder mirrors `obs_utils.build_observations` for the deployed subset, FK through the contract URDF for palm and fingertips, poses mapped through T_world_base, quats xyzw, clamp plus minus 10. Target math mirrors `action_utils.apply_action_pipeline` exactly, including the arm post-EMA clamp the legacy module skips. Player smoke loads the shim yaml and checkpoint in the deploy venv and asserts one (1,19) forward pass.
- **1.3 ROS2 packages and fake loop.** Port the policy node loop skeleton, wait for all inputs, warmup of about 100 steps publishing measured q then `player.reset()`, 60 Hz loop, keep the 10 degree target versus measured arm check but abort instead of breakpoint, fail loud when perception is staler than 250 ms. Port the goal node, both nodes read object scales from one object spec file which removes the upstream mismatch. Port the fake robot and fake perception nodes and `home_robot.py`, and write `fake.launch.py` including rosbag2.

### Phase 2, hardware bring-up

- **2.1 Arm stack and controller.** Confirm the FR3 system image is at least 5.9.1 in Desk, pin matching libfranka at least 0.18 and franka_ros2 in `fr3_xhand.repos`, colcon build with RT scheduling for controller_manager. Adapt the vendor example controller, subscribe `/fr3/joint_target` through a realtime-safe buffer, load gains from ROS params that the launch file fills from the contract JSON, hold current q when targets go stale past 200 ms, set conservative collision thresholds. The example filters dq at 0.99 which changes effective damping, make the coefficient a parameter defaulting to none and decide from a step-response test. Smoke test arm alone, hold default pose, small per-joint sines at 60 Hz, overlay tracking against the same target sequence replayed in sim, error envelope within about twice sim.
- **2.2 Hand driver.** Thin rclpy wrapper over the RobotEra `xhand_controller` SDK, LeFranX bundles wheel v1.1.7 as reference usage. Assert the canonical order to SDK id mapping against SDK-reported names at startup. Publish joint states at 60 Hz or better with finite-difference velocity, acceptable since joint_vel trained with 0.1 rad/s noise. Smoke test per-finger sines within limits.
- **2.3 Camera and calibration.** ChArUco board rigid on the EE, about 20 arm poses, record FK base-to-EE and OpenCV camera-to-board with factory intrinsics, solve eye-to-hand with `cv2.calibrateHandEye`, write `T_camera_base.json`, the same role as the fork's T_RC file. Verify against a board at a measured base-frame location, residual under 5 mm and 1 degree, inside the trained 1 cm and 5 degree noise budget.

### Phase 3, perception and closed loop

- **3.1 FoundationPose ROS2 port.** Port the fork's publisher to rclpy inside its conda env, stamp poses at capture time, apply `T_camera_base.json`. Timebox the env bridge to half a day, try RoboStack rclpy in the conda env first, fall back to a small ZMQ republisher in the deploy venv if pins conflict. Measure capture-to-publish latency, gate p95 under 100 ms against the trained 167 ms tolerance.
- **3.2 Per-object pipeline.** `make_object_spec.py` runs the paper pipeline end to end and emits one object spec JSON, mesh path, object_scales, grasp box, plus a goal trajectory in the `dextoolbench/trajectories` format, the single artifact both nodes load.
- **3.3 Closed-loop rollout.** `real.launch.py` replaces the authors' six terminals, controller_manager with the impedance controller, hand node, perception, goal node, policy node, rosbag2 on every contract topic, rviz2 with robot model and pose displays.

## 3. Verification, risks, open items

### 3.1 Gates, one per phase

- **Gate 1, parity.** `test_deploy_contract.py` green. `sim2sim_parity.py` per-step obs diffs, exact fields under 1e-5, FK-derived fields under 2 mm, quats under 1e-3, target math under 1e-6, then a deploy-in-the-loop rollout reaching at least 80 percent of the `eval_isaacsim.py` baseline over 20 episodes. `fake.launch.py` sustains 60 Hz and the staleness abort fires when the fake perception node dies.
- **Gate 2, hardware.** Real arm and hand with fake perception at a fixed pose, warmup holds, 60 s of free-air policy motion with no reflex triggers and no missed deadlines, watchdogs verified by killing nodes, calibration residual passed, measured table height against the FR3 base matches the sim geometry from the contract.
- **Gate 3, closed loop.** Static object tracked within 1 cm and 5 degrees at a sustained 30 Hz, no track loss over 60 s of hand-moved motion and the staleness watchdog fires when the camera is covered, then grasp, lift, and follow of a demo trajectory on the pipeline object, repeatable in at least 3 of 5 attempts, full rosbag replayable.

### 3.2 Risk register

- **Actuator mismatch.** The dq filter, real joint friction, and torque rate limits differ from sim, and kp 400 can trip Franka reflexes. Mitigation, the step-response overlay in 2.1 before any object work, the filter coefficient as a parameter, conservative collision thresholds, free-air gate before contact, trained DR absorbs the residual.
- **Contract drift.** Lab body ordering and three separate Python environments invite silent skew. Mitigation, the contract is exported from a live env, byte-equality test on every export, runtime name asserts in every node, the JSON is the only cross-env artifact.
- **Perception and geometry.** Latency beyond the trained tolerance, calibration error, or a table mismatch breaks the policy silently. Mitigation, latency measured explicitly with a p95 gate, staleness aborts at 250 ms policy-side and 200 ms controller-side, calibration residual gate, explicit table geometry check, policy on CPU so the 4090 serves FoundationPose alone.

### 3.3 Open items needing user hardware info

- FR3 system image version, robot IP, and Desk admin access.
- XHand connection type, RS485 or EtherCAT, device permissions, SDK wheel compatibility with Python 3.10.
- Physical setup, camera mount rigidity and final pose, measured table height and lateral offset from the FR3 base plate to reproduce the sim base-over-table geometry.
