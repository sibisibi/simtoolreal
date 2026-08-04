# .goals/, raw user prompts

This zone holds the user's raw experiment prompts. One session = one numbered directory.

## What belongs here

- `NNN-short-name/001-overview.md`, the initial prompt for session NNN.
- `NNN-short-name/002-*.md`, `003-*.md`, follow-up clarifications and refinements the user adds over time.
- `NNN-short-name/wiki.md`, the distilled summary, populated only on `distill goals`.

## Naming

- `NNN` is zero-padded, starts at `001`, increments by 1.
- `short-name` is kebab-case, ~2-4 words describing the experiment.
- Inside a session, md files are also numbered `001-`, `002-`, … by order of arrival.

## Agent rules in this zone

1. **The agent does not write goal md files**. The user authors them. Agent reads only.
2. **The agent maintains `index.md` and `log.md`** when a new session dir or new md file is added.
3. **The agent writes `wiki.md` only on `distill goals`**. The distillation reads every `*.md` in the session dir and produces one cohesive summary using the skeleton in `001-template-only/wiki.md`.
4. **The agent must not delete or rewrite user-written goal md files**. If clarification is needed, ask the user. Do not overwrite.

## Structural reference

`001-template-only/` shows the expected skeleton.
- `001-template-only/001-template-only.md`, how to start a goal md file.
- `001-template-only/wiki.md`, the distilled-goal skeleton.

## References cited in goals

Goals often cite items from `.reference/`. When the user writes `see .reference/103-pointworld/paper/`, the agent treats that as a hint and reads the referenced paper or code when relevant to planning.
