---
name: handoff
description: Summarize the session's work into progress.md as a handoff document. Use when the user asks for a handoff, a progress summary, or to capture session state before stopping work.
---

# Handoff: update progress.md

Write (or update in place — never create a second file) `progress.md` at the
repo root so someone with zero session context can pick the work up.

## Steps

1. Establish what changed: `git log --oneline -10`, `git status --short`, and
   `git diff --stat` against the last handoff commit if one exists.
2. Rewrite `progress.md` with these sections:
   - **Header** — `_Last updated: YYYY-MM-DD_` (absolute date, never "today").
   - **Project state** — one paragraph + bullets on what works overall, not
     just this session.
   - **Just completed** — this session's work: the *why*, the design (name
     the key functions/files with paths like `src/mnc/lyrics.py`), and the
     plumbing (every file touched and its role).
   - **Verification done** — exactly what was run (commands, test counts,
     end-to-end runs) and, critically, **what was NOT verified** with a
     recommended next step.
   - **Possible follow-ups** — concrete, small, prioritized.
   - **Environment notes** — venv location, test command, generated
     artifacts, version pins.
3. Be concrete: file paths, function names, thresholds, commands. No vague
   "improved X". A reader should be able to verify every claim.
4. After writing, offer to commit (or use `/commit-push` if the user asked
   for both).

## Rules

- Never delete "Not yet verified" items without evidence they were verified.
- Keep prior sessions' still-relevant content; fold superseded content into
  "Project state" rather than deleting history the reader needs.
