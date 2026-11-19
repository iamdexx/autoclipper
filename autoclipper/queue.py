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

from . import publish, review, trends
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
    reviewed: bool = False  # passed the editor's pass (review.py)
    review_tries: int = 0  # runs it has waited for a review that could not be run

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


HANDOFF_SCORE = 7.0  # where a finished video handed to a run stands in the queue (a story's score)


def handoff(folder: Path, cfg: Config, state: State | None = None) -> Item | None:
    """Queue a video that was built outside a run: `folder` holds one .mp4 and a post.json with
    its title, description, hashtags, `said` (every word it speaks) and `kind`. It gets no
    shortcut: the output check runs here, and the editor's review and the posting budget apply
    when the queue is drained, as for any video. None when it is refused, queued or posted."""
    from . import qa

    try:
        j = json.loads((folder / "post.json").read_text())
        (video,) = sorted(folder.glob("*.mp4"))
        meta = PostMeta(
            title=str(j["title"]).strip(),
            description=str(j["description"]).strip(),
            hashtags=[str(h) for h in j.get("hashtags", [])],
            **({"category": str(j["category"])} if j.get("category") else {}),
        )
        said, kind = str(j["said"]).strip(), str(j.get("kind") or "top10")
    except Exception as e:
        log.error("handoff %s refused: %s", folder, e)
        return None
    if not meta.title or not said:
        log.error("handoff %s refused: no title or no words", folder)
        return None
    source = f"handoff:{video.stem}"
    posted = state is not None and any(p.get("video") == source for p in state.data["posts"])
    if posted or any(it.source == source for it in items(cfg)):
        log.info("handoff %s is already queued or posted", video.name)
        return None
    if not qa.gate(video, cfg, said, kind=kind):
        log.error("handoff %s refused by the output check", video.name)
        return None
    return enqueue(
        video, kind, source, HANDOFF_SCORE, meta, cfg, hook=meta.title, hashtags=meta.hashtags
    )


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
    """Drop off-focus (trends.focus_keywords, e.g. queued before the channel changed focus), stale
    (publishing.queue_max_age_h) and surplus (publishing.queue_max_items, weakest first) entries;
    returns what's left, best first."""
    p = cfg.publishing
    keep: list[Item] = []
    for it in items(cfg):
        text = " ".join([it.meta.title, it.meta.description, it.hook, *it.meta.hashtags])
        # the channel's own scripted videos chose their subject on purpose (an explainer may be
        # about anything); they only have to stay clear of trends.exclude_keywords
        own = it.kind == "top10" and trends.rights_ok(text, "", cfg)
        if not own and not trends.on_focus(text, cfg):
            log.info("queue: dropping %s (not about %s)", it.video, cfg.trends.focus)
            remove(cfg, it)
        elif it.age_h > p.queue_max_age_h:
            log.info("queue: dropping %s (%.0fh old)", it.video, it.age_h)
            remove(cfg, it)
        elif len(keep) >= p.queue_max_items:
            log.info("queue: dropping %s (score %.2f, queue full)", it.video, it.score)
            remove(cfg, it)
        else:
            keep.append(it)
    return keep


def review_pending(cfg: Config) -> None:
    """The editor's pass (review.py) on queued videos that have not had it, best first: a pass is
    recorded, a fail is deleted, and one whose review could not be run waits for the next run
    (at most review.max_attempts of them). Covers videos queued before the pass existed too."""
    if not cfg.review.enabled:
        return
    down = 0
    for it in items(cfg):
        if it.reviewed:
            continue
        v = review.video(_dir(cfg) / it.id / it.video, it.kind, cfg)
        if v.ok is None:  # one timeout is not an outage: ask once more
            v = review.video(_dir(cfg) / it.id / it.video, it.kind, cfg)
        if v.ok is None:
            down += 1
            it.review_tries += 1
            if it.review_tries >= cfg.review.max_attempts:
                log.info("queue: dropping %s (never reviewed)", it.video)
                remove(cfg, it)
            else:
                _save(cfg, it)
            if down >= 2:
                break  # two videos in a row could not be reviewed: the reviewer is down
            continue
        down = 0
        if v.ok:
            it.reviewed = True
            _save(cfg, it)
        else:
            log.info(
                "queue: dropping %s (review: score %d, %s)",
                it.video,
                v.score,
                ", ".join(v.problems) or v.note,
            )
            remove(cfg, it)


def _next(rest: list[Item], recent: list[str], cfg: Config) -> Item:
    """The best-scoring of `rest` (best first), except that one of the channel's own narrated
    videos goes ahead of clips while they are under publishing.own_share of the last max_per_day
    posts (`recent`: their formats, oldest first)."""
    p = cfg.publishing
    w = max(1, p.max_per_day)
    own = next((it for it in rest if it.kind == "top10"), None)
    last = recent[-(w - 1) :] if w > 1 else []  # the window this post would complete
    if own and p.own_share > 0 and last.count("top10") < p.own_share * w:
        return own
    return rest[0]


def drain(cfg: Config, state: State, force: bool = False) -> list[Item]:
    """Post queued videos best-first until every enabled platform is over budget. A video is
    only posted once it has passed the editor's pass (review.py), and removed once nothing is waiting on budget (posted, dry-run, or given up after MAX_ATTEMPTS
    errors); returns the items that went out."""
    posted: list[Item] = []
    review_pending(cfg)
    rest = [it for it in prune(cfg) if it.reviewed or not cfg.review.enabled]
    while rest:
        # asked again after every upload: one that failed has not used up a slot of its format
        it = _next(rest, state.posted_kinds(cfg.publishing.max_per_day), cfg)
        rest.remove(it)
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
