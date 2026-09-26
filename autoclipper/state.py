"""Tiny JSON-backed store: which videos we've processed, which clips were posted where."""

from __future__ import annotations

import json
import time
from pathlib import Path


class State:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict = {"videos": {}, "posts": []}
        if path.exists():
            self.data = json.loads(path.read_text())

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(self.path)

    def seen(self, video_id: str) -> bool:
        return video_id in self.data["videos"]

    def mark_video(self, video_id: str, status: str, **extra) -> None:
        self.data["videos"][video_id] = {"status": status, "at": time.time(), **extra}
        self.save()

    def used_topics(self, kind: str) -> list[str]:
        """Topics already covered for `kind` ('trend' | 'top10'), oldest first."""
        items = self.data.get("topics", {}).get(kind, {})
        return [name for name, _ in sorted(items.items(), key=lambda kv: kv[1].get("at", 0))]

    def mark_topic(self, kind: str, name: str, **extra) -> None:
        self.data.setdefault("topics", {}).setdefault(kind, {})[name] = {
            "at": time.time(),
            **extra,
        }
        self.save()

    def record_post(self, video_id: str, clip: str, platform: str, result: dict) -> None:
        self.data["posts"].append(
            {"video": video_id, "clip": clip, "platform": platform, "at": time.time(), **result}
        )
        self.save()
