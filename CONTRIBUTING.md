# Contributing

Contributions should preserve the bounded research claim and the reproducible
repository structure.

Before opening a pull request:

```bash
python scripts/audit-public-release.py
python -m pytest -q
```

Requirements:

- Do not commit secrets, sensitive data, model weights, local artifacts, or
  absolute machine paths.
- Add tests for behavioral changes and negative controls for stronger claims.
- Keep generated evidence traceable to source artifacts and commands.
- Preserve upstream licenses and provenance for source-derived data.
- Do not expand the project claim beyond the evidence demonstrated by tests.
