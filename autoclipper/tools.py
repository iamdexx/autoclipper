"""Which outside tool did what in this run: every call to a model or a voice is counted here,
and `log_report` writes one line per tool at the end of a run. The line is what the channel's
managers read to decide which tool keeps which job (config: nvidia_text_models,
nvidia_vision_models, narration.nvidia_voices)."""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

_calls: dict[str, dict] = {}


def record(tool: str, ok: bool, why: str = "") -> None:
    c = _calls.setdefault(tool, {"ok": 0, "failed": 0, "why": {}})
    if ok:
        c["ok"] += 1
        return
    c["failed"] += 1
    m = re.search(r"\b([45]\d\d)\b", why)
    key = m.group(1) if m else ("timeout" if "timed out" in why.lower() else "other")
    c["why"][key] = c["why"].get(key, 0) + 1


def report() -> dict[str, dict]:
    return {k: dict(v) for k, v in sorted(_calls.items())}


def reset() -> None:
    _calls.clear()


def log_report() -> None:
    for tool, c in report().items():
        why = ", ".join(f"{k} x{n}" for k, n in sorted(c["why"].items()))
        log.info(
            "tool report: %s ok=%d failed=%d%s",
            tool,
            c["ok"],
            c["failed"],
            f" ({why})" if why else "",
        )
