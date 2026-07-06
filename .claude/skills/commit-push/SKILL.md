---
name: commit-push
description: Commit and push changes following this repo's conventions. Use when the user asks to commit, push, or ship the current changes.
---

# Commit and push

## Pre-flight

1. Run the tests first; do not commit red:
   `.venv/bin/python -m unittest discover tests`
2. `git status --short` — check for stray files (generated audio like
   `tests/*.wav` is gitignored and must stay out; scratch output belongs in
   the scratchpad, not the repo).
3. If anything unexpected is staged or modified, surface it before
   committing instead of sweeping it in with `git add -A`.

## Commit

- One commit per coherent change; don't bundle unrelated work.
- Message style (matches repo history, e.g. `35e4f4e`, `032ddfb`):
  - Imperative subject ≤ 72 chars: "Add user-provided lyrics mapped onto
    the score".
  - Body explains *why* and the design in prose, not a file list — the diff
    already shows the files.
  - End with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Use a heredoc for the message:

```bash
git commit -m "$(cat <<'EOF'
<subject>

<body>

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

## Push

- Work lands directly on `main` (`git push`); remote is
  `github.com/khtam/music_notes_retriever`.
- Report the pushed commit hash and range in the final summary.
