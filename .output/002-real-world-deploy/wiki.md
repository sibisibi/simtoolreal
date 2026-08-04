# Real-world deploy on FR3 plus XHand, distilled output

## Summary

The deploy stack works and the policy grasps the tool on hardware. The contract
reproduces the sim's own math to 1e-7, live perception publishes base-frame
object poses at 30 Hz with 51 ms from shutter to publish, the arm tracks sines
within 0.25 degrees mean against a 0.15 degree sim gap, and the hand answers on
all twelve ids at 60 Hz. Four closed loops ran. The policy reached for the
eraser, closed on it, and moved it 40 mm while the thumb tactile went from 0 to
22, so the grasp is real.

It never lifted the tool the way sim does, and the reason is not the stack. The
same checkpoint in sim, with perfect state and no hardware, grasps and lifts
11.4 cm and reaches none of the 42 goals. Every episode begins 0.301 m from the
first goal in keypoint distance where training sampled at 0.1 and scored at
0.075, and the closest approach is 0.015 against a 0.01 bar. The trajectory is
outside the regime the policy was trained in, and that is true before any camera
or robot is involved.

The calibration is separately wrong and unresolved. A flat board reads 3.79
degrees tilted through it, twice, and the tool's measured corner sits 54.8 mm
from where it was placed. A replacement was solved, installed, measured against
that corner, found worse at 145.1 mm, and reverted.

## Scripts produced

The stack lives under `deployment/fr3_xhand/`, kept out of this zone at the
user's direction because the deliverable is permanent infrastructure.

- `deployment/fr3_xhand/RUNBOOK.md` — the robot session, each step gated by its
  own check.

## Code produced

- `export_contract.py` — dumps orderings, gains, limits and frames from a live
  Isaac Sim env into the one JSON every environment reads.
- `ws/src/fr3_joint_impedance_controller/` — the C++ controller, `tau = kp(q_goal
  - q) - kd*dq`, holding the activation pose until a target arrives.
- `ws/src/fr3_xhand_nodes/` — policy, goal, hand, perception and fake nodes.
- `perception/tracker.py` — the FoundationPose wrapper both paths share.
- `perception/{record_demo,extract_goals,annotate_object,init_scene}.py` — demo
  capture, tracking, canonical frame and grasp box, SAM 2 masking.
- `perception/{replay_parity,check_registration,recompose_goals}.py` — the
  perception gates and the recalibration rebuild.
- `{goals_to_sim,sim_rehearsal,sim_arm_replay,bag_to_targets}.py` — the sim side.
- `{arm_sine_test,hand_sine_test}.py` — the hardware smoke tests.
- `calibration/handeye_capture.py` — eye-to-hand solve, board on the arm,
  `cv2.calibrateHandEye` with inverted robot poses. Written and validated on
  synthetic data, never run on hardware.
- `calibration/results/` — the May solve as `_may_backup`, the live file, and
  three rejected candidates kept for the next attempt.

## Visualizations

- `vis/image/arm-tracking.jpg` — commanded against measured on all seven arm
  joints, uniform lag, no ringing, gravity joints indistinguishable from the rest.
- `vis/image/registration-overlay.jpg` — the registered pose on the live frame,
  box on the tool and axes at the handle arch.
- `vis/image/collision-decomposition.jpg` — the visual mesh, its 22 CoACD hulls,
  and the hulls head on showing the finger gap survived.
- `vis/image/sim-rehearsal-curves.jpg` — distance closed and lift held, our tool
  against the benchmark control.
- `vis/image/checkpoint-comparison.jpg` — episode 33000 against 55000 over eight
  episodes each, the older being the reliable one.

The tabbed session log is an accepted exception to this zone's no-html rule, at
https://claude.ai/code/artifact/66b35bf6-4917-4e1c-af47-ebf2bcad68b0

## User approval

- Approved on. Not yet. The loop runs, the grasp is real, and the goal
  trajectory and the calibration are both open.
- User comments. The corrections that mattered came from the user, not from me.
  Six invented numbers were caught, each only when they asked where it came
  from. Homing was skipped before a rollout and they caught it. The static board
  calibration was their idea and it is what found the tilt. The brush corner
  measurement is what proved my replacement calibration worse than the one it
  replaced, after I had already installed it.
- A calibration was installed on evidence that did not survive their next
  measurement. It is reverted and the rejected file kept. The lesson is in the
  runbook, not just here.

## Reproduction

`deployment/fr3_xhand/RUNBOOK.md` runs the robot session end to end. The offline
chain is `extract_goals.py` then `annotate_object.py` then `goals_to_sim.py`,
all against the canonical asset mesh.
