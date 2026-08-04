# Real-world deploy on FR3 plus XHand, distilled plan

## Approach (one paragraph)

Build the deploy stack under `deployment/fr3_xhand/`, exporting a contract from
a live Isaac Sim env so the orderings, gains, limits and frames that only exist
at runtime become one JSON that every other environment reads. Mirror the sim's
observation and action math in numpy, gate it by replaying a recorded demo
through the deploy path, then wrap it in ROS2 Humble with a custom C++ joint
impedance controller, the only path on the FR3 that exposes both kp and kd.
Perception is FoundationPose in its own conda env, publishing base-frame object
poses on a ROS topic. Prove each layer before the next, contract and parity,
then the arm stack, then the object pipeline, then live perception, then a sim
rehearsal, and only then hardware.

## Key decisions

- ROS2 with franka_ros2 over a hand-rolled bus — the widely adopted vendor path,
  chosen after an explicit options review rather than for convenience.
- Contract JSON as the only cross-environment artifact — three pythons cannot
  share objects, and Lab body ordering invites silent skew.
- Policy on CPU — leaves the 4090 to FoundationPose, and holds 60 Hz.
- FoundationPose and rclpy in one process — the fp env is py310 and resolves
  system rclpy, so the timeboxed bridge risk cost nothing.
- One mesh everywhere, the canonical asset — FoundationPose answers differently
  per mesh file, and two files put goal and live frames 8 mm and 7.25 deg apart.
- Native camera resolution — neither the authors nor the paper downscale.
- Report the sim against real trajectory error, do not threshold it — the line
  that was there first was invented, and Tune to Learn sets none either.
- Safety additions the authors lack — staleness aborts, a track-loss abort, and
  an arm-only launch mode, all of which stop rather than alter behaviour.
- Recoverable device errors are skipped, not fatal — the hand's overcurrent
  warning and a corrupt RS485 exchange each ended a rollout until measurement
  showed the hand carries on through both. The bound for giving up is the
  policy's own staleness tolerance rather than a new number.
- The controller waits for the payload declaration — SetLoad is refused once a
  torque controller has the robot in Move mode, and running them side by side
  is a race whose loss is a rollout with the wrong dynamics.
- Calibration is measured against something the calibration did not produce.
  A solve that fits its own inputs proves nothing, which is how a worse one
  came to be installed.

## Files touched

- `deployment/fr3_xhand/` — the deploy stack, kept out of `.output/` at the
  user's direction because the session's deliverable is permanent infrastructure.
- `.output/002-real-world-deploy/vis/image/` — the five figures.
- `assets/urdf/davian/davian_handle_eraser/` — the reconstructed tool.
- `dextoolbench/` — object registry, metadata, and the interactive viewer fixes.

## References reused

- `.reference/102-paper/appendix/E_humanVideoProcessing.tex` — the 3 Hz
  downsample and the 10 cm lift-off truncation, implemented as written.
- `DexManip/.reference/104-sysid/docs/tune-to-learn/` — the sinusoidal excitation
  protocol and the sim against real trajectory error.
- `DexManip/.reference/102-xhand1_delivery_with_tactile/` — the hand id table,
  the URDF joint limits, and the 83 Hz bus rate.
- `DexManip/sim2real/LeFranX/` — payload, collision thresholds, hand limits.
- `dextoolbench/generate_collision_meshes.py` — the authors' own CoACD path.

## Verification

- Vis artifact. `.output/002-real-world-deploy/vis/image/arm-tracking.jpg`
- Success criterion. Commanded and measured track together on all seven joints
  with no ringing, and the sim replay puts the motor-level gap far inside the
  joint velocity noise the policy trained against.

## Source plan mds distilled

- `001-deploy-plan.md`
- `002-revisions.md`
