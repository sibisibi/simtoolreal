# .output/, generated artifacts

This zone holds everything the agent produces while executing a session: scripts, code, and visualizations.

## What belongs here

For each session:

```
.output/NNN-short-name/
├── wiki.md             # distilled output summary (on `distill output`)
├── scripts/            # *.sh files
├── src/                # *.py files
└── vis/
    ├── image/          # *.png
    ├── gif/            # *.gif
    └── video/          # *.mp4
```

There is no `vis/html/`. html is not a vis type. The one deck is `slides.html`, built by kai straight to `.pages/experiments/<NNN-name>/slides.html`, never under `.output`.

## Hard rule: output isolation

While executing session `NNN-name`, the agent writes ONLY inside `.output/NNN-name/`. No exceptions.

- `*.sh` → `scripts/`
- `*.py` → `src/`
- `*.png` → `vis/image/`
- `*.gif` → `vis/gif/`
- `*.mp4` → `vis/video/`

There is no `*.html` mapping. The deck `slides.html` is not a session output, kai writes it straight to `.pages/experiments/<NNN-name>/slides.html`.

Anything else, like data dumps or intermediate artifacts the user should not review, goes inside one of these dirs or does not exist. The user assesses success by opening the raw vis here and the deck at `.pages/experiments/<NNN-name>/slides.html`. Every plan must produce at least one vis artifact.

## Naming inside `scripts/`, `src/`, `vis/*`

- Files are named descriptively, e.g. `vis/gif/baseline-vs-ours.gif`.
- Multiple artifacts are fine.
- The deck that ties the vis together is `slides.html` in `.pages/`, not here.

## Agent rules in this zone

1. **Never write outside the assigned session dir** during a session.
2. **Bookkeeping.** When creating a new session dir, append to `.output/log.md` and update `.output/index.md`.
3. **Distillation.** `wiki.md` is written only on `distill output`. It captures what was produced, the paths to vis artifacts, and the date the user approved.
4. **No deletion without user approval**. If a previous session's output looks stale, ask before removing.

## Structural reference

`001-template-only/` contains an example layout.
- `001-template-only/wiki.md`, the distilled-output skeleton
- `001-template-only/scripts/example.sh`
- `001-template-only/src/example.py`
- `001-template-only/vis/image/example.png`
- `001-template-only/vis/gif/example.gif`
- `001-template-only/vis/video/example.mp4`

These example files exist to show the structure. Leave them in place.
