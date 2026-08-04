# FR3 XHand robot session runbook

The order below moves from no motion to full motion, and each step has a check
that has to pass before the next one starts. Stop at the first failed check.

Everything here needs a person at the machine with the e-stop in reach.

## 0. Before power

- FCI is enabled in Desk and the joints are unlocked.
- The tool sits on the table with the camera view clear.
- The 4090 has room, `nvidia-smi` shows the other jobs' usage.
- Nothing else holds the camera, `fuser /dev/video*` returns nothing.

## 1. Build and source

```bash
cd deployment/fr3_xhand/ws
PATH=/usr/bin:$PATH colcon build
source install/setup.bash
```

The conda python shadows the system one, which is why `PATH` is set per call.

## 2. Fake loop regression

```bash
ros2 launch fr3_xhand_bringup fake.launch.py
```

Check the policy node holds 60 Hz, then kill the fake perception node and
check the loop aborts on stale poses.

## 3. Arm alone

```bash
ros2 launch fr3_xhand_bringup real.launch.py robot_ip:=172.16.0.2 arm_only:=true
```

`arm_only` leaves out the hand, the policy, the goals and perception, so this
brings up the franka stack, the payload declaration and the impedance
controller and nothing else.

The controller waits for `arm_bringup` to exit before it spawns. SetLoad is
rejected once an active torque controller has put the robot in Move mode, and
running the two side by side is a race. Losing it means the payload is never
declared and the arm computes its dynamics as though no hand were bolted on,
which is a run worth throwing away. Watch for `payload set` before
`Configured and activated fr3_joint_impedance_controller`, in that order.

The controller captures the measured pose on activation and holds it, so the
arm should not move at all, wherever it happens to be standing. Watch for
reflexes in the franka logs.

Then run the free-air checks with `arm_sine_test.py`, small per-joint sines at
60 Hz, and replay the recorded targets through sim with `bag_to_targets.py` and
`sim_arm_replay.py`.

That reports the trajectory error, sim against real, position and velocity
together, following Tune to Learn. It is a number to read, not a threshold. The
signature that predicts a bad transfer is high frequency oscillation, so watch
the traces for ringing rather than for a smooth lag, which a PD always has.

## 4. Hand alone

```bash
ros2 run fr3_xhand_nodes hand_node --ros-args -p device:=/dev/ttyUSB0
```

Per-finger sines inside the limits. The node asserts the SDK's joint names
against the contract order at startup, so a mismatch fails before motion.

## 5. Home the robot

Every sim episode resets to the contract's default pose, arm at 0, 15, 0, -115,
0, 135, -45 degrees and the hand open at zero, so that is the only starting
state the policy has ever seen. The arm holds wherever it stood when the
controller activated, which is not that pose, so it gets moved there first.

Homing needs the arm and the hand and nothing else. `arm_only` leaves out
perception, which would otherwise hold the camera that scene init is about to
need, so the hand comes up on its own in a second terminal.

```bash
ros2 launch fr3_xhand_bringup real.launch.py robot_ip:=172.16.0.2 arm_only:=true
ros2 run fr3_xhand_nodes hand_node --ros-args -p device:=/dev/ttyUSB0
ros2 run fr3_xhand_nodes home_robot --ros-args -r /fr3/joint_states:=/franka/joint_states
```

Without the remap the node waits forever on a topic nobody publishes. Point it
at `/franka/joint_states`, the broadcaster's own 1 kHz output, rather than
`/joint_states`, which is joint_state_publisher's 30 Hz republish and will emit
URDF defaults for anything it has not heard from.

That broadcaster orders its joints 1, 3, 6, 7, 2, 4, 5. The node maps by name,
which it did not always do, and taking `msg.position` in arrival order put the
measured angles onto the wrong joints and commanded joint 6 to -131 degrees
while it sat at 110. Against kp 400 that is a power limit reflex and a very
loud noise.

It interpolates both arm and hand from measured to default over 10 s at 60 Hz
and exits. Watch the arm the whole way, the interpolation is a straight line in
joint space and takes no account of what is in the way.

Home before scene init, not after, so the frame the object registers against is
the same scene the policy starts from.

## 6. Calibration

No longer deferred. It was deferred because the May solve looked aligned in the
viewer and put the table within 11 mm, but neither of those bounds a lateral or
rotational error, and a flat board on the table later read 3.42 degrees tilted.
See E9 at the end of this file for the procedure and for why the lab's own
script is not the one to run.

Whichever solve produces it, every base-frame number in the demo came from the
old one, so rebuild them
against the new one. Camera-frame poses are calibration independent, which is
why this takes seconds rather than another tracking pass.

```bash
/home/davian/anaconda3/envs/fp/bin/python \
    deployment/fr3_xhand/perception/recompose_goals.py \
    --demo deployment/fr3_xhand/demos/demo_20260803_081042

.venv_isaacsim/bin/python deployment/fr3_xhand/goals_to_sim.py \
    --object_spec deployment/fr3_xhand/objects/davian_handle_eraser.json \
    --category eraser --task wipe_up_down
```

`recompose_goals.py` prints how far the new calibration moved the poses. A
shift of a few millimetres is inside the trained noise, and a large one means
the camera moved and the demo geometry is worth a second look.

## 7. Scene init

```bash
/home/davian/anaconda3/envs/sam3d/bin/python \
    deployment/fr3_xhand/perception/init_scene.py \
    --out deployment/fr3_xhand/init/davian_handle_eraser
```

Left click the tool, right click any background that leaks in, ENTER to
segment with SAM 2, `y` to accept. Leave the tool where it is afterwards.

This one runs on the sam3d env's python, which carries the torch SAM 2 needs.

```bash
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/davian/anaconda3/envs/fp/lib \
    /home/davian/anaconda3/envs/fp/bin/python \
    deployment/fr3_xhand/perception/check_registration.py \
    --init deployment/fr3_xhand/init/davian_handle_eraser \
    --object_spec deployment/fr3_xhand/objects/davian_handle_eraser.json
```

Look at `registration_overlay.jpg`. The box has to wrap the tool and the axes
have to sit at the handle arch. The printed table height should land near zero,
it read +10 mm at E5.

## Every attempt starts here

A rollout moves the tool and leaves the arm wherever it stopped, so both have
to be put back before the next one. This is not optional and it is not a
formality, both were skipped once each and both cost a run.

1. Home the arm, step 5. The policy has only ever started from the contract
   default. An arm left where a previous rollout ended is outside the reset
   distribution the policy trained on, and it will command a step large enough
   to trip a torque reflex. Measured, not argued, four of seven joints were
   outside that distribution after one aborted rollout.
2. Redo scene init, step 7. Registration runs against the stored frame, so it
   is only valid while the object has not moved. A rollout that touches the
   tool voids it. Tracking from a stale registration does not fail loudly, it
   silently reports the object somewhere it is not, once at a height below the
   table it was sitting on.
3. Check the sweep before homing. `home_robot` walks a straight line in joint
   space and knows nothing about the table, the tool, or the whiteboard. From
   some poses that line passes 18 mm above the table, and the check only covers
   frame origins, not the meshes around them. If the margin is small, guide the
   arm somewhere sensible by hand first and home from there.

## 8. Sensing dry run

Everything up, nothing commanding motion. `no_policy` leaves out the policy
node alone, so the arm holds on the impedance controller, the hand holds, and
perception and goals stream against the registration just written.

```bash
ros2 launch fr3_xhand_bringup real.launch.py \
    robot_ip:=172.16.0.2 \
    object_spec:=$PWD/deployment/fr3_xhand/objects/davian_handle_eraser.json \
    init_dir:=$PWD/deployment/fr3_xhand/init/davian_handle_eraser \
    device:=/dev/ttyUSB0 \
    no_policy:=true
```

Check the object pose sits where the tool actually is, the goal pose advances,
and the perception node's capture-to-publish line stays near what E5 measured,
p95 51 ms. This is the last look at the sensing the policy will consume before
it is allowed to command anything.

## 9. Closed loop

```bash
S=$PWD/deployment/fr3_xhand/sessions/$(date +%Y%m%d_%H%M%S)_loop
ros2 launch fr3_xhand_bringup real.launch.py \
    robot_ip:=172.16.0.2 \
    object_spec:=$PWD/deployment/fr3_xhand/objects/davian_handle_eraser.json \
    init_dir:=$PWD/deployment/fr3_xhand/init/davian_handle_eraser \
    device:=/dev/ttyUSB0 \
    frame_dir:=$S/frames
```

Cover the camera once the loop is running and check the policy node aborts on
stale poses.

## What a rollout records

The bag carries the arm state and targets, the hand state and targets, the
object pose and the goal pose, the hand's five fingertip sensors, and the
franka broadcaster's robot state with its measured and external joint torques.
Hand joint torque rides in the JointState `effort` field.

The tactile and the torques cost nothing. One RS485 exchange already carries
all twelve joints and all five sensors, and the broadcaster was already running
with its output going nowhere.

`frame_dir` is separate because it is not free. Depth is aligned to colour, so
both streams are 1280x720 and the pair is 4.61 MB a frame, 138 MB/s at 30 Hz,
or 8.3 GB a minute. Frames go to disk on a writer thread rather than through
DDS. Measured cost inside the tracking loop is 0.17 ms median and 0.23 ms at
p95, and a live run held capture to publish at p95 53 to 57 ms against 51 ms
with recording off, well inside the 250 ms staleness bound, with the queue at
zero throughout. Leave `frame_dir` empty and the rollout keeps poses but not
the frames behind them, which cannot be recovered afterwards.

Watch the disk. Six minutes of recording is 50 GB.

Tear the stack down with one interrupt to the launch and then wait for it. The
perception node drains its writer thread on the way out and holds the frame
files open until it finishes, so a kill that does not wait leaves the space
allocated to files that no longer have a name. `df` will not show it back until
the process is gone.

Frames are fixed-size raw records, `rgb.raw`, `depth.raw` and `stamps_ms.raw`,
with shapes and intrinsics in `meta.json`, so a reader memmaps them:

```python
m = json.load(open(f"{d}/meta.json"))
rgb = np.memmap(f"{d}/rgb.raw", np.uint8).reshape(-1, *m["rgb_shape"])
```

## What the sim rehearsal predicts

E6 rolled this checkpoint on this tool eight times. It grasped every time,
lifted 9.3 to 10.1 cm, and held near the first goal without settling onto it.
Expect the same shape on hardware, a grasp and a hold rather than a completed
trajectory.

## Aborts

- The policy node stops when poses age past 0.25 s.
- The controller holds the current pose when targets age past 0.2 s.
- The perception node stops when the object jumps more than 15 cm in a frame.
- The policy node stops when a target sits more than 10 deg off the measured arm.
- The e-stop covers everything the software misses.

All four have fired on hardware and all four behaved. A dead hand node aged its
state out and the policy stopped 0.257 s later; the controller then held.

## What takes the stack down, and why that is fine

- **Deactivating FCI.** libfranka throws inside its control thread, the
  exception escapes, and the process aborts. Launch tears the rest down because
  `ros2_control_node` is required. Expected, not a fault.
- **Hand guiding.** Pressing the guiding buttons ends the active control loop
  and does the same thing. Also expected. Relaunch afterwards, the hardware
  reinitialises cleanly and needs no recovery service.
- **A reflex.** The arm goes to error state and wants clearing in Desk. A fresh
  bringup then connects normally.

After any of these, relaunch rather than trying to resume. And check the arm
pose before commanding again, it is not where it was.

## E9. Eye-to-hand calibration

Run this when the fingers miss the object by tens of millimetres, which is what
a rotational extrinsics error looks like from the outside.

The board goes on the hand, not the table. A static board tells you where it is
relative to the camera and nothing about where it is relative to the robot, and
that second one is the whole question. Its placement on the hand is never
measured and never appears in the answer, so tape is fine.

```bash
ros2 launch fr3_xhand_bringup real.launch.py robot_ip:=172.16.0.2 arm_only:=true
ros2 control switch_controllers --deactivate fr3_joint_impedance_controller
```

Deactivating matters. Hand guiding ends libfranka's active control loop, so with
a torque controller running the stack aborts, which is what happened the first
time it was tried. With none claiming the command interfaces the arm reads out
happily while you move it.

```bash
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/davian/anaconda3/envs/fp/lib \
    /home/davian/anaconda3/envs/fp/bin/python \
    deployment/fr3_xhand/calibration/handeye_capture.py \
    --out deployment/fr3_xhand/calibration/results
```

Guide the arm, press `s` at each pose, `q` to solve. About 20, and vary the
orientation rather than sliding the board around at a fixed wrist angle, because
two poses whose rotation axes are parallel leave the rotation undetermined. The
overlay refuses a capture whose reprojection error is over 1.5 px.

The board is the lab's own, 5x7, DICT_5X5, squares measured at 47 mm. It is not
the one in byungkunlee's tree, which is DICT_4X4_250 at 33 mm and will not
decode against this one.

The solver reports every OpenCV method beside the scatter of the implied
board-on-hand transform, which is a self-check needing no ground truth, since
that transform is fixed and every pose should agree on it. On synthetic data
PARK, HORAUD and ANDREFF recover an exact answer while TSAI and DANIILIDIS do
not, and the scatter separates them without being told which is which.

Nothing downstream moves until the result is copied over the file the perception
node reads, and `recompose_goals.py` rebuilds the goals against it. The goals
were tracked through the old extrinsics, so leaving them alone would break the
one place the calibration error currently cancels, `keypoints_rel_goal`.
