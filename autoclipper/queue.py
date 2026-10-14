"""Posting queue: every QA-passed video waits in workdir/queue and the best-scoring one goes out
each time the posting budget (publishing.max_per_day / min_gap_min) opens a slot. Videos that
are never picked age out, so the queue holds the strongest recent candidates, not a backlog."""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import publish
from .config import Config
from .metadata import PostMeta
from .state import State

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 3


@dataclass
class Item:
    id: str  # queue folder name
    video: str  # file name inside the folder
    kind: str  # 'clip' | 'top10'
    source: str  # state key the post is recorded under (video id / 'top10:<slug>')
    score: float
    meta: PostMeta
    hook: str = ""
    creator: str = ""
    hashtags: list[str] = field(default_factory=list)
    added: float = 0.0
    attempts: int = 0

    @property
    def age_h(self) -> float:
        return (time.time() - self.added) / 3600


def _dir(cfg: Config) -> Path:
    return cfg.workdir / "queue"


def _load(d: Path) -> Item | None:
    try:
        j = json.loads((d / "item.json").read_text())
        j["meta"] = PostMeta(**j["meta"])
        it = Item(**j)
        return it if (d / it.video).exists() else None
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _save(cfg: Config, it: Item) -> None:
    (_dir(cfg) / it.id / "item.json").write_text(json.dumps(asdict(it), indent=2))


def enqueue(
    video: Path,
    kind: str,
    source: str,
    score: float,
    meta: PostMeta,
    cfg: Config,
    hook: str = "",
    creator: str = "",
    hashtags: list[str] | None = None,
) -> Item:
    """Copy `video` into the queue (the render dir is per-run scratch; the queue dir is what the
    workflow persists between runs)."""
    qid = f"{int(time.time())}-{video.stem}"[:80]
    d = _dir(cfg) / qid
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(video, d / video.name)
    it = Item(
        id=qid,
        video=video.name,
        kind=kind,
        source=source,
        score=float(score),
        meta=meta,
        hook=hook,
        creator=creator,
        hashtags=list(hashtags or []),
        added=time.time(),
    )
    _save(cfg, it)
    log.info("queued %s (score %.2f): %s", video.name, it.score, meta.title)
    return it


def items(cfg: Config) -> list[Item]:
    """Best first: score, then newest."""
    d = _dir(cfg)
    if not d.exists():
        return []
    found = [it for sub in d.iterdir() if sub.is_dir() and (it := _load(sub))]
    return sorted(found, key=lambda it: (-it.score, -it.added))


def remove(cfg: Config, it: Item) -> None:
    shutil.rmtree(_dir(cfg) / it.id, ignore_errors=True)


def prune(cfg: Config) -> list[Item]:
    """Drop stale (publishing.queue_max_age_h) and surplus (publishing.queue_max_items, weakest
    first) entries; returns what's left, best first."""
    p = cfg.publishing
    keep: list[Item] = []
    for it in items(cfg):
        if it.age_h > p.queue_max_age_h:
            log.info("queue: dropping %s (%.0fh old)", it.video, it.age_h)
            remove(cfg, it)
        elif len(keep) >= p.queue_max_items:
            log.info("queue: dropping %s (score %.2f, queue full)", it.video, it.score)
            remove(cfg, it)
        else:
            keep.append(it)
    return keep


def drain(cfg: Config, state: State, force: bool = False) -> list[Item]:
    """Post queued videos best-first until every enabled platform is over budget. A video is
    removed once nothing is waiting on budget (posted, dry-run, or given up after MAX_ATTEMPTS
    errors); returns the items that went out."""
    posted: list[Item] = []
    for it in prune(cfg):
        res = publish.publish_all(
            _dir(cfg) / it.id / it.video,
            it.hook or it.meta.title,
            it.creator,
            it.hashtags,
            cfg,
            kind=it.kind,
            force=force,
            state=state,
            meta=it.meta,
        )
        for platform, r in res.items():
            state.record_post(
                it.source,
                it.video,
                platform,
                r,
                creator=it.creator,
                hashtags=it.hashtags,
                kind=it.kind,
                hook=it.hook,
            )
        if res and all("skipped" in r for r in res.values()):
            break  # budget: nothing else can go out this run
        if any("skipped" in r for r in res.values()):
            continue  # one platform still waiting; stays queued for it
        if any("error" in r for r in res.values()):
            it.attempts += 1
            if it.attempts < MAX_ATTEMPTS:
                _save(cfg, it)
                continue
            log.warning("queue: giving up on %s after %d errors", it.video, it.attempts)
        else:
            posted.append(it)
        remove(cfg, it)
    left = items(cfg)
    log.info("queue: %d posted, %d waiting", len(posted), len(left))
    return posted
