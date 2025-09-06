# Repository Guidelines

## Project Structure & Module Organization
- Source: `main.py` is the current entrypoint. New modules may live alongside it or under `src/` if the project grows.
- Config: `pyproject.toml` tracks metadata and Python requirement (>=3.12).
- Docs: `README.md` for high-level usage and notes.
- Git ignore: `.gitignore` excludes `.venv`, `__pycache__/`, and build artifacts.
- Tests (when added): place under `tests/`, mirroring module paths.

## Build, Test, and Development Commands
- Run locally: `python main.py` (prints a greeting; use as a sanity check).
- Virtual env: `python -m venv .venv && source .venv/bin/activate`.
- Dependencies: none yet. If you add any, declare them under `[project].dependencies` in `pyproject.toml` and install in your venv (e.g., `pip install <pkg>`). Keep `pyproject.toml` in sync.
- Tests: preferred framework is `pytest`. Run with `pytest -q` once tests exist.

## Coding Style & Naming Conventions
- Follow PEP 8: 4-space indentation, max 88–100 cols where reasonable.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, lowercase module names.
- Typing: add type hints for new/modified code; docstrings for public functions.
- Imports: standard library, third-party, then local; keep groups separated.

## Testing Guidelines
- Layout: `tests/test_<module>.py` with function-level tests; mirror package structure.
- Behavior-first: assert outputs and side effects; avoid deep internals.
- Coverage: aim for ~90% on changed code; add tests with new features/bug fixes.
- Quick run: `pytest -q` locally before opening a PR.

## Commit & Pull Request Guidelines
- Commits: short, imperative summaries (e.g., "Add CLI entry", "Fix path bug"). No strict convention in history yet—keep clear and scoped.
- PRs: include purpose, concise description, reproduction/usage steps (`python main.py` or test cmds), linked issues, and sample output when relevant.
- Keep changes focused; update docs and `pyproject.toml` when behavior or deps change.

## Security & Configuration Tips
- Python: target 3.12 (`.python-version`).
- Secrets: never commit tokens or `.env`; use environment variables locally.
- Entrypoint: keep side effects under `if __name__ == "__main__":` to ease testing.

## Agent-Specific Instructions
- Prefer minimal, targeted diffs; avoid unrelated refactors.
- Reflect code changes in tests and docs; do not introduce unconfigured tools.
