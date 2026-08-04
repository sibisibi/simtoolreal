# .plans/, agent-generated plans

This zone holds plans the agent produces in response to goals. Each goal session has a matching plan session with identical `NNN-name`.

## What belongs here

- `NNN-short-name/001-overview.md`, the first plan for goal `NNN-short-name`.
- `NNN-short-name/002-*.md`, `003-*.md`, revisions as the plan iterates with the user.
- `NNN-short-name/wiki.md`, the distilled summary, populated only on `distill plan`.

## Naming

- `NNN-short-name` mirrors `.goals/NNN-short-name`. If the goal is `005-tracker-ablation`, the plan is `.plans/005-tracker-ablation/`.
- Plan mds inside a session are numbered `001-`, `002-`, … in revision order.

## Agent rules in this zone

1. **The agent writes plans in plan mode** and saves them here after the user exits plan mode (or as snapshots during iteration).
2. **Plans are full and verbose.** They contain context, file paths, edge cases, and verification steps. `wiki.md` is the readable distillation.
3. **Every plan must end with a vis-artifact verification step.** The user opens the raw vis in `.output/NNN-name/vis/` (image, gif, video) and the deck at `.pages/experiments/NNN-name/slides.html`. This is a hard requirement. See root `CLAUDE.md`.
4. **Bookkeeping.** When a new plan md is added, append a line to `.plans/log.md`. When a new session dir is created, update `.plans/index.md`.
5. **Distillation.** `wiki.md` is written only on `distill plan`. It summarizes every plan md in the session using the skeleton in `001-template-only/wiki.md`.

## Plan-mode artifacts vs persisted plans

Claude Code's plan mode writes a temporary plan file under `~/.claude/plans/`. After the user approves, the agent should copy/adapt that content into `.plans/NNN-name/NNN-*.md` so the plan is preserved with the project.

## Structural reference

`001-template-only/` shows the skeleton.
- `001-template-only/001-template-only.md`, how a plan md is structured.
- `001-template-only/wiki.md`, the distilled-plan skeleton.
