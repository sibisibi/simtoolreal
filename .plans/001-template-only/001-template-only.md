# Plan: <session-title>

<!-- Structural template for a plan md. Copy and rename to `NNN-session-title.md` inside `.plans/NNN-session-title/`. Iterate by adding 002-, 003-, and so on. -->

## Context

<why this change is being made. Tie back to the goal session. State the gap, the motivation, and the intended outcome.>

## Approach

<the recommended approach in 1-3 paragraphs. Skip alternatives. They belong in conversation, not here.>

## Files to modify / create

- `.output/NNN-name/scripts/<file>.sh` — <what and why>
- `.output/NNN-name/src/<file>.py` — <what and why>
- `.output/NNN-name/vis/<type>/<file>.<ext>` — <the verification artifact>

## Key decisions

- <decision> — <rationale>
- <decision> — <rationale>

## Existing code to reuse

- `.reference/1NN-name/code/<module>` — <how it's used>
- `src/<module>.py` (project-shared) — <how it's used>

## Verification (vis artifact)

The final step of this plan must produce a file under `.output/NNN-name/vis/`. State exactly what artifact and what success looks like when opening it.

- Artifact. `.output/NNN-name/vis/<type>/<filename>`
- Success criterion. <what the user should see when opening it>
