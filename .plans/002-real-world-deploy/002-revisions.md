# Revisions to the deploy plan, made during execution

The plan in `001-deploy-plan.md` is the original. These are the changes the work
forced, each with what caused it.

## Numbering became E1 to E9

The plan's phases were renamed to experiment items so the log could carry one
sequence that never renumbers. E1 contract and parity, E2 arm stack, E3 mesh,
E4 goals, E5 live perception, E6 sim rehearsal, E7 robot session, E8 commit and
push, E9 calibration.

## Calibration deferred out of the robot session, E9

The plan put a fresh eye-to-hand solve inside the robot session. The May solve
agreed with two independent checks, the viewer and the tracked object placing
the table surface within 11 mm of the base, so the user deferred it. The
machinery is built and tested, `recompose_goals.py` rebuilds every base-frame
number from saved camera-frame poses without another tracking pass.

## The 2x sim envelope gate was dropped

The plan gated the arm on tracking error staying within about twice the sim
envelope. That threshold was invented, not taken from the authors or any paper.
Four joints landed at 2.1 to 2.4 against it. The user dropped it, and the
comparison now reports the trajectory error from Tune to Learn, position and
velocity together, without a threshold.

## One mesh everywhere

The plan did not say which mesh each stage tracks. Goal extraction ran on the
raw reconstruction and the live node on the canonical asset, and FoundationPose
answers differently per file, so the two routes sat 8 mm and 7.25 degrees apart
on a stationary object. Everything now tracks the canonical asset.

## Full camera resolution, no downscale

Half resolution was introduced to fit alongside another user's 13 GB on the
card. Neither the authors nor the paper downscale anything spatially, so the
path was removed once the card was free.

## SAM 2 for the live mask

The plan said SAM 2. No SAM was installed locally, and a colour threshold stood
in without that being flagged. SAM 2.1 large now runs from a local copy and the
registration was redone on its mask.

## The object is davian_handle_eraser, task wipe_up_down

Named by the user, filed in the eraser category beside the tool whose loop
handle it shares. The bench, the base plate and the whiteboard were measured by
the user and match sim in the robot base frame.

## Homing became a robot session step

The plan never sequenced homing, and `real.launch.py` never called the node
that does it. The policy would have started from wherever the arm stood, while
sim resets every episode to arm 0, 15, 0, -115, 0, 135, -45 degrees with the
hand open, the one starting state it has ever seen.

The node was also unrunnable on hardware. It subscribes `/fr3/joint_states`
and franka_bringup publishes `/joint_states`, a remap the policy node carries
in the launch file and the homing node had nowhere. It waits on the topic
rather than failing, so the symptom is a hang.

Homing now runs by hand before the policy, which is how the authors ran theirs,
under `arm_only` so perception does not hold the camera that scene init needs.
A `no_policy` argument was added alongside, bringing the whole sensing stack up
with nothing commanding motion, as the last check before the loop.

## E9 stopped being deferred

The calibration was deferred because the May solve looked aligned in the viewer
and put the table within 11 mm. Neither of those bounds a lateral or a
rotational error. A flat board on the table reads 3.79 degrees tilted through
that solve, repeated across two placements 219 mm apart, and the object's
measured corner sits 54.8 mm from where the user put it. Both say the
calibration is wrong.

It is still wrong. A replacement was solved, installed, measured against the
user's own corner, found worse at 145.1 mm, and reverted. The evidence
contradicts itself, and the board stack thickness decides which side is right.

## The goals sit outside the trained regime

The closed loop and the sim rehearsal fail the same way, which is the point.
Every rehearsal episode starts 0.301 m from the first goal in keypoint
distance. Training sampled goals at `delta_goal_distance` 0.1 and scored them
at 0.075. The best approach across eight episodes is 0.015 m against a target
tolerance of 0.01. So the policy grasps, lifts 11.4 cm, and reaches none of the
42 goals in sim with perfect state, no perception and no hardware.

That is not a deployment problem and no amount of stack work reaches it.

## Robot address corrected

The config carried 192.168.18.1, stale from the robot that was swapped out. The
arm answers at 172.16.0.2 on the dedicated link.

## Hand state rate corrected

The node polled at 100 Hz. One whole-hand RS485 exchange takes 12 ms, so the
hand caps at 83 Hz, stated in the vendor manual and four API references. The
node now polls at 60 Hz.
