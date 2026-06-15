# Evidence Artifacts

This directory exposes only the compact report summaries required to reproduce
the manuscript tables and their boundary checks.

Large datasets, model packages, runtime outputs, and complete experiment
directories are distributed through the external dataset archive rather than
tracked directly in Git. The allowlist in the root `.gitignore` is
file-specific so that newly generated local artifacts are not added by accident.

Validate the tracked artifact set:

```powershell
python scripts/audit-public-release.py
```
