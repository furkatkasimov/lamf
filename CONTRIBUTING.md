# Contributing

Thanks for your interest in LAMF. This is a reference implementation
tied to a research paper, so the scope is deliberately narrow.

## What's in scope

- Bug fixes (numerical errors, broken loaders, failing tests)
- Documentation improvements
- New dataset loaders (see `docs/EXTENDING.md`)
- Performance improvements that preserve numerical outputs

## What's out of scope

- Changes to the four paper policies (Static, Triggers, Liquid+Rules,
  Liquid+Bandits). These are fixed by the manuscript.
- Changes to the default hyperparameters that would break reproduction
  of the paper's tables.
- Tuning or ML additions (embeddings, neural ranking, etc.). These
  belong in a downstream fork.

## Workflow

1. Fork the repository and create a branch from `main`.
2. Add or modify tests in `tests/` alongside your change.
3. Run the full test suite: `pytest tests/ -v`. All 30 tests must pass.
4. If you changed a loader or the evaluator, re-run the relevant
   benchmark and confirm numerical output matches the paper to within
   floating-point tolerance.
5. Open a PR describing the change and referencing the paper section
   it touches, if any.

## Coding style

- PEP 8, 4-space indents, ~100-character line limit.
- Type hints on public functions.
- Dataclasses preferred over plain classes for policy objects.
- Docstrings should cite the relevant paper equation or section.
