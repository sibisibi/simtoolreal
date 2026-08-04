# <session-title>

<!-- This is a structural template. Copy and rename when starting a new session with cp -r 001-template-only NNN-short-name. Replace the placeholders below with the actual prompt. -->

## What I want to do

- Real-world deploy of sim-trained policy

## Why

- I need to setup baseline for my research, where the strongest baseline is SimToolReal
- So, I ported our lab's robot, i.e. Franka FR3 + XHand, into SimToolReal codebase myself
- Then I began training, which is still ongoing, but downloaded checkpoint first
- Goal is not excellent performance
- Goal is to setup infra, i.e. code for deployment, so whenever I am ready in sim, I can deploy at ease
- You will need to first pull or fetch, since remote main has slight update in commit
- You need to think about what is the optimal setup to match sim-real
- First phase should be ideation. Discussion on what we need, i.e. what is, or missing, in SimToolReal deploy side code
- Then we plan code implementation
- Then proceed with sim-trained checkpoint
- I have one D435i mounted.

## References

- `.reference/102-paper/arXiv-2602.16863v2/` — SimToolReal paper
- `runs/a4h1_20260731_085255/0_simtoolreal_sapg/last/model.pth` — Checkpoint to use for deploy

## Constraints

- Web search encouraged whenever investigation needed
- Ask me whenever clarification needed, or some info I must provide you

## Success criterion

- You propose