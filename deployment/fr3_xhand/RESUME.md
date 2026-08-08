# Resuming this work after a long gap

Written 2026-08-08, assuming you remember none of it. Read this first, then
`RUNBOOK.md` for the commands.

## What this is

Deploying a sim-trained SimToolReal checkpoint on the lab's Franka FR3 v2 with a
RobotEra XHand and one fixed RealSense D435i. The deliverable was always the
deployment infrastructure rather than a performance number, the stated bar being
that any future checkpoint deploys at ease.

Two principles governed every choice. Real must match sim in the actuator law,
the gains and the observation math. Our stack must match the SimToolReal
authors', deviating only where the robot differs, since they ran a KUKA iiwa
with a 22 DoF Sharpa hand and we run an FR3 with a 12 DoF XHand.

## Where it got to

The stack works. The policy grasps the tool on hardware. Four closed loops ran,
the longest 18.83 s, and in each the arm reached, the hand closed on the eraser,
the object moved up to 40 mm, and the thumb tactile went from 0 to 22 over the
grasp window. The contact is recorded, not inferred.

It never lifted the way sim does, **and the stack is not the reason**. The same
checkpoint in sim, with perfect state and no camera or robot involved, grasps,
lifts 11.4 cm, and reaches none of the 42 goals. That is the finding that
matters and it is the one to start from.

## The two open problems

### 1. The goal trajectory sits outside the trained regime

Every rehearsal episode starts **0.301 m** from the first goal in keypoint
distance. Training sampled goals at `delta_goal_distance: 0.1` and scored them
at `success_tolerance: 0.075`, tightening to `target_success_tolerance: 0.01`.
The best approach across eight episodes is 0.015 m.

So the policy is asked to move the object three times further than anything it
trained on, and judged at a tolerance seven times tighter than training used.
It grasps and lifts and then has nowhere to go.

Nothing in the deployment stack reaches this. The candidates are resampling the
trajectory near 0.1 m spacing, or judging rollouts at the training tolerance
rather than the tightened one, or retraining against the trajectory as it is.
That is a research decision, not an engineering one, which is why it was left
open rather than guessed at.

### 2. The eye-to-hand calibration is wrong and unresolved

Evidence it is wrong, both independent of each other:

- A physically flat board on the table reads **3.79 deg tilted** through the May
  extrinsics. Repeated at a second placement 219 mm away, it reads 3.41 deg. A
  warped board or a bad patch of table would not survive being moved.
- The tool's corner, measured by hand at 42.5 cm from the far table edge and
  44.5 cm from the left edge, sits **54.8 mm** from where the calibration puts
  it. That is the size of the grasp miss.

Evidence the obvious fix is also wrong. A replacement was solved from the board,
installed, then measured against that same tool corner and found **145.1 mm**
off, nearly three times worse than what it replaced. It was reverted.

The two sides disagree in a specific way:

| | board tilt | board height, expect ~20 mm | tool corner |
|---|---|---|---|
| May, live now | 3.4 to 3.8 deg | 14 and 18 mm | 54.8 mm |
| candidate_v2 | 0.20 deg | 34 and 32 mm | 0.0 mm |

One input is wrong, and **the board stack thickness decides which**. It was
guessed at 20 mm, being a 10 mm whiteboard plus an unmeasured backing, and never
measured. Measure it with a caliper and the contradiction resolves in seconds
from images already on disk.

Second suspect if calibration is exonerated: `palm_center_offset` in the
contract, `(0, 0, 0.16)`. It enters `keypoints_rel_palm` exactly as an
extrinsics error does and has never been checked against the real hand mount.

Why calibration matters at all, given the policy is goal-conditioned: of the ten
observation terms, only `keypoints_rel_palm` is corrupted by an extrinsics
error. `keypoints_rel_goal` cancels it, because the goals were tracked through
the same camera and the same solve. So calibration does not affect where the
policy tries to take the object, only whether it can pick it up. If you ever
replace the extrinsics you must rebuild the goals too, or you break the
cancellation that currently works and turn one bad term into two.

## The code

- Repo `simtoolreal`, remote `sibisibi/simtoolreal`, which redirects to
  `DAVIAN-Robotics/simtoolreal`.
- Branch `fr3-xhand-deploy-session`, commit `b49674b`, pushed.
- Everything deployment lives under `deployment/fr3_xhand/`, kept out of
  `.output/` deliberately because the deliverable is permanent infrastructure.
- Session bookkeeping in `.goals/`, `.plans/` and `.output/`, all under
  `002-real-world-deploy`, following `template/CLAUDE.md` and the per-zone
  `CLAUDE.md` files. Read those before adding to them.

## Environments, and which is for what

Three pythons plus one, and they cannot share objects, which is why the contract
JSON exists as the only cross-environment artifact.

| env | python | for |
|---|---|---|
| `.venv_isaacsim` | 3.11.14 | Isaac Lab, contract export, sim rehearsal, viser |
| `.venv_deploy` | 3.10.12 | every ROS node. Built with system site packages so it resolves Humble's rclpy |
| `fp` conda | 3.10.20 | FoundationPose, and it is also the only env with `pyrealsense2`, `rclpy` and an OpenCV that still has `calibrateHandEye` |
| `sam3d` conda | 3.11.0 | SAM 2, which needs torch 2.5.1. Upgrading `fp` to match would break FoundationPose |

`.venv_deploy` ships OpenCV 5.0.0, which **dropped `calibrateHandEye`**. Do not
write calibration code against it.

Append to `LD_LIBRARY_PATH` and `PYTHONPATH`, never replace. The conda python
shadows the system one, which is why `colcon build` is always run as
`PATH=/usr/bin:$PATH colcon build`.

## Hardware

- Arm at **172.16.0.2** on the dedicated link. FCI enabled in Desk, joints
  unlocked. An earlier config carried 192.168.18.1, stale from a robot that was
  swapped out, and nothing on this host was ever on that subnet.
- Hand on `/dev/ttyUSB0`, RS485 at 3 Mbaud.
- Camera serial **347622076599**.
- Checkpoint `runs/a4h1_20260731_085255/0_simtoolreal_sapg/last/model.pth`,
  episode 33000. Episode 55000 exists and reaches further when it works but is
  less reliable. Obs 110, actions 19.
- Object `davian_handle_eraser`, category eraser, task `wipe_up_down`, 58 g,
  weighed.

## Things that cost a run, so they are worth knowing

**Before every rollout**, home the arm and redo scene init. A rollout moves the
tool and leaves the arm where it stopped. Skipping the homing once put four of
seven joints outside the reset distribution the policy trained on and tripped a
torque discontinuity reflex 2.1 s in. Skipping the registration once had
perception silently reporting the object at a height below the table it was
sitting on. Neither fails loudly.

**The hand's bus.** One RS485 exchange carries all twelve joints and all five
tactile sensors and takes 12 ms, so the hand caps at 83 Hz. `send_command` is
one exchange and `read_state` with `force_update` is another, so doing both
every cycle at 60 Hz is 1.44 times over the bus and shows up as a CRC error. The
vendor's own test commands, then reads with `force_update` false.

**Recoverable device errors.** The hand's overcurrent warning and a corrupt
exchange are both survivable, measured: after the first overcurrent, 90 of 92
sends and 92 of 92 reads succeeded and the hand opened again on 148 of 148
commands. Treating either as fatal ended rollouts. They now skip the cycle, and
the node gives up only if the hand goes quiet past the policy's own staleness
bound, so no new threshold was invented.

**Payload before controller.** `SetLoad` is refused once an active torque
controller has the robot in Move mode. Spawning them together is a race, and
losing it means the payload is never declared and the arm computes its dynamics
as though no hand were bolted on. Watch for `payload set` before
`Configured and activated fr3_joint_impedance_controller`, in that order.

**Never trust a message's joint order.** `joint_state_broadcaster` reports the
arm as 1, 3, 6, 7, 2, 4, 5. Taking `msg.position` as it arrives commanded joint
6 to -131 degrees while it sat at 110, which against kp 400 is a power limit
reflex and a very loud noise. Map by name. `/joint_states` is a 30 Hz republish
from `joint_state_publisher` with `publish_default_positions` true, so it emits
URDF defaults for anything it has not heard from. Use `/franka/joint_states`,
the broadcaster's own 1 kHz output.

**Hand guiding and FCI both abort the stack.** Pressing the guiding buttons ends
libfranka's active control loop, so with a torque controller running the process
aborts and launch tears everything down. That is expected. Deactivate the
controller first if you want to guide. Relaunch afterwards rather than resuming,
and check the arm pose before commanding again.

**One mesh everywhere.** FoundationPose answers differently per mesh file. Goal
extraction on the raw reconstruction and live tracking on the canonical asset
put the two frames 8 mm and 7.25 deg apart on a stationary object. Everything
tracks the canonical asset now.

**Recorded rollouts are replayable.** Bags carry the arm at 1 kHz, external
joint torques, the hand's five fingertip sensors at 60 Hz, and per joint torque
in the JointState effort field. Frames go to disk raw on a writer thread,
costing 0.17 ms a frame inside the tracking loop, at 138 MB/s and 8.3 GB a
minute. Watch the disk.

## What I would do first, in order

1. **Measure the board stack thickness.** Ten seconds with a caliper. It decides
   which of two contradictory calibrations is right, and everything downstream
   of the camera waits on it.
2. **Re-run the sim rehearsal before touching hardware.** Eight minutes, no
   robot. It is what showed the goals unreachable, and it would have said so at
   the start of the session that found it at the end.
3. **Decide what to do about the goal trajectory.** Resample, re-score, or
   retrain. Until then a rollout cannot succeed by its own definition, and
   further stack work is not the constraint.

## Where everything is

- Code, committed and pushed: `DAVIAN-Robotics/simtoolreal`, branch
  `fr3-xhand-deploy-session`, commit `b49674b`.
- Local: `/home/davian/sibeenkim/project/simtoolreal`.
- NAS: `kaist143_sb:/home/nas4_user/sibeenkim/work/0805/`. The alias is
  `kaist143_sb`, not `143_sb`, at 143.248.159.143 port 8022.
- Session log, a standalone page that opens in a browser, at
  `0805/docs/session-log.html`, mirrored at
  https://claude.ai/code/artifact/66b35bf6-4917-4e1c-af47-ebf2bcad68b0

Robot data is gitignored and lives only on the local disk and the NAS. Seven
sessions, the largest being `20260805_loop5` at 19 GB, which is the longest
rollout and the best one to look at. `20260805_loop1` has its bag but its frames
were deleted by mistake, and it was the first successful grasp.
