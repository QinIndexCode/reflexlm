# Public evidence artifacts

This directory exposes only the compact report summaries required to reproduce
the manuscript tables and their boundary checks.

Raw datasets, model packages, runtime outputs, complete experiment directories,
submission archives, and author workflow materials remain excluded from the
public repository. The allowlist in the root `.gitignore` is intentionally
file-specific so that newly generated local artifacts are not published by
accident.

Validate the public boundary before release:

```powershell
python scripts/audit-public-release.py
```
