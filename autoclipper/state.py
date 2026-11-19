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

    def forget_video(self, video_id: str) -> None:
        """Make a video unseen again, so a later run picks it up."""
        if self.data["videos"].pop(video_id, None) is not None:
            self.save()

    def forget_topic(self, kind: str, name: str) -> None:
        """Make a topic unused again, so a later run can pick it."""
        if self.data.get("topics", {}).get(kind, {}).pop(name, None) is not None:
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

    def used_pillars(self) -> list[str]:
        """Topic-mix pillars of every topic picked so far (trend and top10), oldest first."""
        return list(self.data.get("pillars", []))

    def mark_pillar(self, name: str) -> None:
        if name:
            self.data.setdefault("pillars", []).append(name)
            del self.data["pillars"][:-200]
            self.save()

    def record_post(self, video_id: str, clip: str, platform: str, result: dict, **about) -> None:
        """`about` (creator / hashtags / kind / hook) is what analytics groups performance by.
        A post already recorded (same platform and id) is not counted twice."""
        if result.get("id") and any(
            p["platform"] == platform and p.get("id") == result["id"] for p in self.data["posts"]
        ):
            return
        self.data["posts"].append(
            {
                "video": video_id,
                "clip": clip,
                "platform": platform,
                "at": time.time(),
                **about,
                **result,
            }
        )
        self.save()

    def posted_at(self, platform: str, since_s: float = 24 * 3600) -> list[float]:
        """Timestamps of real posts (not dry-run / failed / skipped) to `platform` in the window."""
        cutoff = time.time() - since_s
        return sorted(
            p["at"]
            for p in self.data["posts"]
            if p["platform"] == platform and p.get("id") and p["at"] >= cutoff
        )

    def posted_kinds(self, n: int) -> list[str]:
        """Format ('clip' / 'top10') of the last `n` videos really posted, oldest first."""
        seen: dict[tuple, str] = {}
        for p in self.data["posts"]:
            if p.get("id"):
                seen[(p["video"], p["clip"])] = p.get("kind", "clip")
        return list(seen.values())[-n:] if n > 0 else []
