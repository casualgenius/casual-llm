# Gotchas & Pitfalls

Things to watch out for in this codebase.

## [2025-12-25 05:04]
Linting tools (ruff, black, uv) are blocked by callback hook in this environment. Manual verification is required.

_Context: When running subtask-5-2 for linting checks, direct invocation of ruff, black, uv, and python -m commands were blocked. User must run linting verification manually outside of the Claude environment._

## [2025-12-25 05:06]
Mypy type checking is blocked by callback hook in this environment. Manual type review must be performed instead.

_Context: When running subtask-5-3 for mypy type checking, both direct mypy invocation and uv run mypy were blocked. Manual type annotation review was performed instead and found no issues._
