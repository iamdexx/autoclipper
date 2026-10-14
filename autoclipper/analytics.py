"""What our own uploads actually did, fed back into what we make next.

`refresh` pulls view counts for posted Shorts from the YouTube Data API (videos.list, 1 quota
unit per 50 videos) into state.json once a post is old enough to have been through the feed's
first test batch. `weight` turns that history into a multiplier on a new clip's queue score, so
the kinds of videos that got watched post first and the kinds that died at 20 views wait, and
`summary` gives the topic picker the same history in words.
"""

from __future__ import annotations

import logging
import statistics
import time

from .state import State

log = logging.getLogger(__name__)

SETTLE_H = 24  # a Short's first-day views decide whether the feed keeps pushing it; measured once
MIN_POSTS = 3  # a label needs this many measured posts before it moves anything
MIN_WEIGHT, MAX_WEIGHT = 0.5, 1.5


def _measurable(post: dict, now: float) -> bool:
    return (
        post.get("platform") == "youtube"
        and bool(post.get("id"))
        and post.get("at", now) <= now - SETTLE_H * 3600
        and "views" not in post  # one first-day reading per post, so posts of any age compare
    )


def fetch_stats(ids: list[str]) -> dict[str, dict]:
    """YouTube statistics per video id (views / likes / comments)."""
    from googleapiclient.discovery import build

    from . import publish

    yt = build("youtube", "v3", credentials=publish.youtube_auth(interactive=False))
    out: dict[str, dict] = {}
    for i in range(0, len(ids), 50):
        resp = yt.videos().list(part="statistics", id=",".join(ids[i : i + 50])).execute()
        for item in resp.get("items", []):
            st = item.get("statistics", {})
            out[item["id"]] = {
                "views": int(st.get("viewCount", 0)),
                "likes": int(st.get("likeCount", 0)),
                "comments": int(st.get("commentCount", 0)),
            }
    return out


def refresh(state: State) -> int:
    """Record first-day view counts on every settled YouTube post that has none yet."""
    now = time.time()
    due = [p for p in state.data["posts"] if _measurable(p, now)]
    if not due:
        return 0
    stats = fetch_stats([p["id"] for p in due])
    for p in due:
        if p["id"] in stats:
            p.update(stats[p["id"]])
        p["checked_at"] = now
    state.save()
    log.info("analytics: refreshed %d posts (%d found)", len(due), len(stats))
    return len(stats)


def labels(creator: str = "", hashtags: list[str] | None = None, kind: str = "") -> list[str]:
    out = []
    if kind:
        out.append(f"kind:{kind.lower()}")
    if creator:
        out.append(f"creator:{creator.strip().lower()}")
    out += [f"tag:{h.strip().lower()}" for h in hashtags or [] if h.strip() and h != "#shorts"]
    return out


def measured(state: State) -> list[dict]:
    return [p for p in state.data["posts"] if p.get("platform") == "youtube" and "views" in p]


def report(state: State) -> dict[str, tuple[int, float]]:
    """label -> (posts measured, median views)."""
    by: dict[str, list[int]] = {}
    for p in measured(state):
        for lb in labels(p.get("creator", ""), p.get("hashtags"), p.get("kind", "")):
            by.setdefault(lb, []).append(int(p["views"]))
    return {lb: (len(v), statistics.median(v)) for lb, v in by.items()}


def channel_median(state: State) -> float:
    views = [int(p["views"]) for p in measured(state)]
    return statistics.median(views) if views else 0.0


def weight(
    state: State | None, creator: str = "", hashtags: list[str] | None = None, kind: str = "clip"
) -> float:
    """Queue-score multiplier for a new video: how its creator / topic tags / format have done so
    far against the channel's median, clamped to [0.5, 1.5]; 1.0 until there is enough history."""
    if state is None:
        return 1.0
    base = channel_median(state)
    if not base:
        return 1.0
    rep = report(state)

    def ratios(lbs: list[str]) -> list[float]:
        return [rep[lb][1] / base for lb in lbs if lb in rep and rep[lb][0] >= MIN_POSTS]

    # who / what it is about decides; a new creator or topic stays neutral rather than inheriting
    # the format's record. The format (clip vs top10) only counts for posts that have nothing else.
    ratios = (
        ratios(labels(creator, hashtags)) if (creator or hashtags) else ratios(labels(kind=kind))
    )
    if not ratios:
        return 1.0
    return round(min(MAX_WEIGHT, max(MIN_WEIGHT, statistics.median(ratios))), 2)


def summary(state: State | None, n: int = 6) -> str:
    """Plain-language history for the topic picker: what got watched, what died."""
    if state is None:
        return ""
    base = channel_median(state)
    rows = [
        (lb.split(":", 1)[1], cnt, med)
        for lb, (cnt, med) in report(state).items()
        if cnt >= MIN_POSTS and not lb.startswith("kind:")
    ]
    if not rows or not base:
        return ""
    rows.sort(key=lambda r: -r[2])
    good = [f"{name} ({med:.0f} views/post)" for name, _, med in rows[:n] if med > base]
    bad = [f"{name} ({med:.0f})" for name, _, med in rows[-n:] if med < base * 0.6]
    out = [f"Channel median is {base:.0f} views per Short."]
    if good:
        out.append("Did well: " + ", ".join(good) + ".")
    if bad:
        out.append("Flopped (avoid unless it's huge): " + ", ".join(bad) + ".")
    return " ".join(out)
