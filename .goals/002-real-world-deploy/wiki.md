# Real-world deploy on FR3 plus XHand, distilled goal

## Goal

Deploy the sim-trained SimToolReal checkpoint on the lab FR3 v2 plus RobotEra
XHand with one D435i, building the deployment infrastructure rather than
chasing performance.

## Motivation

The user needs a real-world baseline and SimToolReal is the strongest
candidate. The lab robot was already ported into the codebase and a checkpoint
exists, so the missing piece is everything between a trained policy and a
moving arm. The stated success condition is that any future checkpoint deploys
at ease, which makes the infrastructure the deliverable and the first rollout a
by-product. Two principles govern every choice, real must match sim in the
actuator law, the gains and the observation math, and our stack must match the
authors' stack, deviating only where the robot itself differs.

## References cited

- `.reference/102-paper/` — the SimToolReal paper. Appendix E gives the human
  video processing, the 3 Hz downsample and the 10 cm lift-off truncation.
- `DexManip/.reference/104-sysid/docs/tune-to-learn/` — Tune to Learn. Supplies
  the sim against real trajectory error and names high frequency oscillation as
  the transfer failure mode, not a smooth lag.
- `DexManip/.reference/102-xhand1_delivery_with_tactile/` — vendor delivery. The
  joint id table, the URDF joint limits, the 83 Hz whole-hand bus rate.
- `DexManip/sim2real/LeFranX/` — the lab's working arm plus hand stack. Payload,
  collision thresholds, hand joint limits, robot address.
- `DexManip/sim2real/config/arm_ik.yaml` — the real bench geometry.

## Constraints

- Infrastructure over performance, any future checkpoint deploys at ease.
- Match the authors and the vendor, and never invent a constant or a threshold.
- Ideation first, then plan, then deploy. Pull from remote main before starting.
- Adhere to `vault/coding-style.md`, `humanizer`, and `writing-style.md`.
- Commits and pushes follow verification, and pushing is the user's call.
- Never kill a process another person owns, and ask before touching the robot.

## Open questions

- Whether the goal trajectory should be resampled near the 0.1 m spacing
  training sampled at. As extracted it starts each episode 0.301 m from the
  first goal, and the policy reaches none of the 42 even in sim.
- Whether `target_success_tolerance` 0.01 is the right bar to judge a rollout
  against when training scored at 0.075. The rehearsal closes to 0.015.
- What the board stack measures. It was guessed at 20 mm and decides which of
  two contradictory calibrations is right.
- Whether the camera moved since May or the May solve was always wrong. The
  drift reference markers are gone, so this is unrecoverable and does not
  change what to do.
- Whether the arm's 5 to 8 cycle tracking lag is controller dynamics or
  transport delay, which one excitation frequency cannot separate.
- Whether LeFranX's 5 degree floor against mechanical clogging is a real hazard,
  since our limits match the vendor and permit 0.
- Which checkpoint belongs on the robot, episode 33000 being reliable and 55000
  reaching further when it works.

## Final verification artifact (vis)

The closed loop on hardware, the policy driving arm and hand against the tracked
object. Standing evidence is `vis/image/arm-tracking.jpg`, commanded against
measured for all seven arm joints, and `vis/image/registration-overlay.jpg`, the
tracked pose drawn on the live camera frame.

## Source mds distilled

- `001-overview.md`
