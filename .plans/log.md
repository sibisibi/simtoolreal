# .plans/ — event log

<!-- Append-only. One line per session-affecting action. Format: `YYYY-MM-DD HH:MM  <action>  <path>  <reason>`. Local time. Actions: created, added-md, distilled, archived. -->

2026-05-15 13:42  created  .plans/001-template-only/  structural template
2026-05-15 15:30  created  .plans/002-setup-workflow/  plan for workflow bookkeeping setup
2026-05-15 15:30  added-md  .plans/002-setup-workflow/001-setup-workflow.md  approved plan content persisted from plan mode
2026-05-15 17:04  added-md  .plans/002-setup-workflow/002-merging-workflow.md  approved plan: extract project-agnostic vault (servers + workflow tools)
2026-05-15 17:08  distilled  .plans/002-setup-workflow/wiki.md  distill plan
2026-05-16 12:58  added-md  .plans/002-setup-workflow/003-google-drive.md  approved plan: rewrite google-drive.md (MCP, per-project folders, experiment mapping)
2026-05-16 16:50  distilled  .plans/002-setup-workflow/wiki.md  re-distill plan (incorporate 003-google-drive + post-auth resolution)
2026-05-16 20:11  added-md  .plans/002-setup-workflow/003-google-drive.md  addendum: audit + final-correction sweep (stale text fixed across docs/vis)
2026-05-16 20:11  distilled  .plans/002-setup-workflow/wiki.md  re-distill plan (final: 3 arcs incl. .pages one-tree; supersedes stale 16:50)
2026-05-16 20:47  distilled  .plans/002-setup-workflow/wiki.md  distill plan (explicit cmd; folded in local-only .pages/CLAUDE.md)
2026-05-16 22:03  added-md  .plans/002-setup-workflow/004-writing-style.md  approved plan: vault/writing-style.md taste-driven anti-LLM-ese guide, auto-loaded every session
2026-05-16 22:36  added-md  .plans/002-setup-workflow/005-cleanup-audit.md  approved plan: full workspace distill, audit, restyle, cleanup sweep
2026-05-16 22:36  distilled  .plans/002-setup-workflow/wiki.md  re-distill plan (final: 5 arcs incl. writing-style + cleanup sweep; restyled)
2026-05-17 00:07  added-md  .plans/002-setup-workflow/006-kait-doc-audit.md  approved plan: audit kait chat, rewrite vault/servers/kait/ to 2026-05-14 state + new 14-known-issues
2026-05-17 00:39  distilled  .plans/002-setup-workflow/wiki.md  re-distill plan (add Arc F kait doc audit + port probe)
2026-05-17 00:45  distilled  .plans/002-setup-workflow/wiki.md  tighten to final state (Files touched: 001-006 + kait subtree + rebuilt decks; Verification: Arc F is latest, decks verified live)
2026-05-17 01:53  added-md  .plans/002-setup-workflow/007-notion.md  approved plan: build Notion template hub + 3 empty dbs, rewrite notion.md, decouple readai2notion from DexManip
2026-05-17 02:45  distilled  .plans/002-setup-workflow/wiki.md  re-distill plan (add Arc G: Notion build + readai2notion decoupling)
2026-05-17 18:07  added-md  .plans/002-setup-workflow/008-coding-style.md  approved plan: vault/coding-style.md taste-driven anti-overengineering + fail-loud code guide, auto-loaded every session
2026-05-17 18:25  distilled  .plans/002-setup-workflow/wiki.md  re-distill plan (add Arc H coding-style; house-style row keeps three scopes; publish deferred)
2026-05-17 18:35  distilled  .plans/002-setup-workflow/wiki.md  publish-state tighten (decks rebuilt to Arc H, drop the deferred hedge)
2026-05-17 21:52  added-md  .plans/002-setup-workflow/009-google-slides.md  approved+revised plan: weekly -> native Google Slides via pptx skill + local Drive API OAuth client; spine changed from the MCP at execution
2026-05-18 01:06  added-md  .plans/002-setup-workflow/009-google-slides.md  revised plan: DexManip Roboto house convention + rename weekly file to slides; persisted plan updated with the Refinement section
2026-05-18 01:36  distilled  .plans/002-setup-workflow/wiki.md  add Arc I weekly Google Slides, the local Drive API client spine, the convention, the rename, the promotion
2026-05-18 01:58  distilled  .plans/002-setup-workflow/wiki.md  correction, the "promotion to template" was reverted as an overreach, durable home is the ~/.claude/skills/weekly-slides skill; persisted plan 009 fixed
2026-05-18 02:05  distilled  .plans/002-setup-workflow/wiki.md  distill pass, Arc I key decision now states the skill home, the Claude Code best practice, and the recorded no-write-template boundary
2026-05-18 03:53  added-md  .plans/002-setup-workflow/011-github-pages.md  persisted the approved github-pages plan, Drive-first flow, weekly-first IA, no-html-vis convention, drive-vis skill; user seed 010 left untouched
2026-05-18 04:11  added-md  .plans/002-setup-workflow/011-github-pages.md  rewritten to the corrected close-out, the 002 deck is intentionally not built, gated on explicit distill, the stale-deck reuse and fake 000 were shortcuts around the flow
2026-05-18 04:30  distilled  .plans/002-setup-workflow/wiki.md  Arc J github-pages folded in, Drive-first flow, weekly-first IA, no-html-vis convention, drive-vis skill; 011 rewritten to the finished state, the deck built via the real kai flow and live
2026-05-18 04:45  distilled  .plans/002-setup-workflow/wiki.md  final pass after the 6-agent audit, plans wiki coherent and final through Arc J, no defect remained, session closed
2026-05-18 05:00  renamed  .plans/002-setup-workflow/011-github-pages.md -> 010-github-pages.md  the user's seed was a mis-filed goal moved to .goals, the persisted plan renamed 011 to 010 so goal and plan mirror; H1, wiki source list, and index reconciled
2026-05-19 02:37  distilled  .plans/002-setup-workflow/wiki.md  fold the post-close follow-up into the Notion key decision, hub Page links the Pages site, notion.md aligned
2026-05-19 03:21  distilled  .plans/002-setup-workflow/wiki.md  Arc I bullet spec corrected, L1 14pt Bold / L2 14pt Normal per user convention revision; persisted plan 009 also corrected
2026-05-19 03:30  distilled  .plans/002-setup-workflow/wiki.md + 009  weekly vis aligned to Arc J Drive-first, the live Slides URL is the artifact, no vis/html, diff reported inline; skill SKILL.md aligned too
2026-08-05 04:40  added-dir  .plans/002-real-world-deploy/  backfilled, the session ran two days without this zone being touched; plan-mode output had gone to ~/.claude/plans only
2026-08-05 04:40  added-md  .plans/002-real-world-deploy/001-deploy-plan.md  the approved plan persisted from ~/.claude/plans, unchanged
2026-08-05 04:41  added-md  .plans/002-real-world-deploy/002-revisions.md  what execution forced, E1-E9 numbering, calibration deferred to E9, the invented 2x gate dropped, one mesh everywhere, full resolution, SAM 2, robot address and hand rate corrected
2026-08-05 04:42  distilled  .plans/002-real-world-deploy/wiki.md  approach, key decisions, files, references, verification; index reconciled and the template's 002-setup-workflow line commented out, it names a session dir absent from this repo
2026-08-05 05:26  added-md  .plans/002-real-world-deploy/002-revisions.md  homing added to the robot session, the plan never sequenced it and the launch never called it
2026-08-05 09:28  added-md  .plans/002-real-world-deploy/002-revisions.md  E9 undeferred, the closed loop reached the point where the policy grasps on hardware and the remaining gap is the goal trajectory rather than the stack
