# ReflexLM ReflexCore V0 Progress Snapshot

This orphan branch is a clean, history-free progress snapshot for ReflexCore V0 as of 2026-06-21.

Scope:
- Bounded computer-native sensory-motor language core V0.
- Terminal, process, filesystem, and time observations only.
- Typed motor actions and allowlisted RUN_COMMAND safety boundary.
- No GUI, vision, unrestricted shell generation, production autonomy, or full LLM replacement claim.

Key files:
- docs/reflexcore_v0.md
- src/reflexlm/core/
- src/reflexlm/cli/*reflexcore*.py
- configs/reflexcore/
- docs/reflexcore_evidence/
- tests/test_reflexcore_v0.py

Validation:
```powershell
$env:PYTHONPATH='src'
python -m pytest -q tests\test_reflexcore_v0.py
```
