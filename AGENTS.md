# AGENTS.md

Operating instructions for AI agents working in this repository.

## Repo shape
- Study/exploration repo. Primary language: Jupyter Notebook.
  Experiments live in notebooks at the repo root and in `demos/`;
  reusable helpers in `src/`, tests in `tests/`.
- Keep notebook outputs committed so results are reviewable without running them.

## Conventions
- Conventional Commits for every commit message (`feat:`, `fix:`, `chore:`, `docs:`).
- Never push to the default branch. Every change goes through a pull request.
- Keep diffs minimal and reviewable. One concern per PR.
- No secrets in code or notebooks. The `secret-scan` workflow enforces this.

## Verification
- Run lint + tests before opening a PR where applicable.
- Do not merge your own PRs.
