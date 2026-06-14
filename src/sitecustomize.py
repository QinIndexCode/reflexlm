from __future__ import annotations

import os
import platform


def _disable_blocking_platform_wmi_query() -> None:
    """Avoid Windows WMI hangs during heavyweight imports.

    Python 3.12 may ask WMI for OS metadata inside ``platform.system()``. On
    this workstation that call can block indefinitely, which prevents PyTorch
    from importing before any experiment run/progress artifact is created. The
    fallback path in ``platform._win32_ver`` is sufficient for our runtime.
    """

    if os.name != "nt":
        return
    if os.environ.get("REFLEXLM_ENABLE_PLATFORM_WMI") == "1":
        return
    if not hasattr(platform, "_wmi_query"):
        return

    def _raise_wmi_unavailable(*_args: object, **_kwargs: object) -> None:
        raise OSError("platform WMI query disabled by ReflexLM sitecustomize")

    platform._wmi_query = _raise_wmi_unavailable  # type: ignore[attr-defined]


_disable_blocking_platform_wmi_query()
