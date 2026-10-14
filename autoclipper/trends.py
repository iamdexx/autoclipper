"""Discover what's trending right now and find the best YouTube source video for it.

Sources (all keyless, best-effort - a failing source is skipped):
  google  : Google Trends daily RSS (per geo) - topic, approx traffic, headline
  youtube : yt-dlp search of each configured niche, most viewed uploads of the last 24h

An LLM (same provider order as ranking) then filters the raw list down to topics that fit the
configured niches and are worth a video, and picks the best source video per topic.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

import requests

from . import ingest, rank
from .config import Config

log = logging.getLogger(__name__)

GOOGLE_RSS = "https://trends.google.com/trending/rss?geo={geo}"
_UA = {"User-Agent": "Mozilla/5.0 (autoclipper)"}
_HT = "{https://trends.google.com/trending/rss}"


@dataclass
class Topic:
    name: str
    source: str
    traffic: int = 0
    headline: str = ""
    query: str = ""  # YouTube search query chosen by the LLM
    angle: str = ""  # why it's hot / what the clip should focus on
    hashtags: list[str] = field(default_factory=list)


@dataclass
class Candidate:
    id: str
    title: str
    channel: str
    duration: float
    views: int
    age_h: float | None = None  # hours since upload (yt-dlp's approximate date), None = unknown

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.id}"

    @property
    def age(self) -> str:
        return "age ?" if self.age_h is None else f"{self.age_h:.0f}h old"


def _traffic(s: str) -> int:
    digits = "".join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else 0


def google_trends(geo: str) -> list[Topic]:
    r = requests.get(GOOGLE_RSS.format(geo=geo), headers=_UA, timeout=30)
    r.raise_for_status()
    out = []
    for item in ET.fromstring(r.content).iter("item"):
        name = (item.findtext("title") or "").strip()
        if not name:
            continue
        news = item.find(f"{_HT}news_item")
        out.append(
            Topic(
                name=name,
                source=f"google:{geo}",
                traffic=_traffic(item.findtext(f"{_HT}approx_traffic") or ""),
                headline=(news.findtext(f"{_HT}news_item_title") or "") if news is not None else "",
            )
        )
    return out


# YouTube search filters (the `sp` parameter): type=video, uploaded in the window, most viewed first
_WITHIN = {"hour": "CAISBAgBEAE%3D", "today": "CAISBAgCEAE%3D"}


def yt_search(query: str, n: int = 10, within: str = "") -> list[Candidate]:
    """`within`: 'hour' | 'today' = only uploads from that window, most viewed first; '' = YouTube's
    default relevance ranking."""
    if within:
        q = urllib.parse.quote_plus(query)
        target = f"https://www.youtube.com/results?search_query={q}&sp={_WITHIN[within]}"
    else:
        target = f"ytsearch{n}:{query}"
    out = subprocess.run(
        ingest.ytdlp(
            "--flat-playlist",
            "-J",
            "--playlist-end",
            str(n),
            "--extractor-args",
            "youtubetab:approximate_date",  # flat entries get a timestamp ("3 hours ago")
            target,
        ),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if out.returncode or not out.stdout.strip():
        raise RuntimeError(f"yt-dlp search failed: {out.stderr[-300:]}")
    cands = []
    for e in json.loads(out.stdout).get("entries") or []:
        if not e or not e.get("id"):
            continue
        ts = e.get("timestamp")
        cands.append(
            Candidate(
                id=e["id"],
                title=e.get("title") or "",
                channel=e.get("channel") or e.get("uploader") or "",
                duration=float(e.get("duration") or 0),
                views=int(e.get("view_count") or 0),
                age_h=max(0.0, (time.time() - float(ts)) / 3600) if ts else None,
            )
        )
    return cands


def youtube_hot(niche: str, n: int = 8) -> list[Topic]:
    """Most-viewed uploads of the last 24h for a niche."""
    cands = yt_search(niche, n, within="today")
    return [
        Topic(name=c.title, source="youtube", traffic=c.views, headline=c.channel)
        for c in cands
        if c.views
    ]


def discover(cfg: Config) -> list[Topic]:
    t = cfg.trends
    jobs = [(google_trends, g, f"google trends {g}") for g in t.geos]
    jobs += [(youtube_hot, n, f"youtube search {n!r}") for n in t.niches]

    def _run(job):
        fn, arg, label = job
        try:
            return fn(arg)
        except Exception as e:
            log.warning("%s failed: %s", label, e)
            return []

    topics: list[Topic] = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for res in ex.map(_run, jobs):
            topics += res
    seen: set[str] = set()
    uniq = []
    for tp in topics:
        k = tp.name.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(tp)
    return uniq


PICK_TOPICS = """You run a short-form video channel (TikTok / Reels / Shorts) about: {niches}.
Below are things trending right now. Pick the {n} that (a) fit the channel, (b) people will
search / watch video about today, and (c) have real footage on YouTube to clip from.
Skip anything already covered: {used}.
{history}
Return ONLY a JSON array, each item:
{{"topic": "<trend name exactly as given>", "query": "<best YouTube search to find the key footage>",
  "angle": "<one sentence: why it's blowing up / what the clip should show>",
  "hashtags": ["#tag", "#tag2", "#tag3"]}}

TRENDING:
{trending}
"""

PICK_SOURCE = """Trend: {topic}
Angle: {angle}
Which ONE of these YouTube videos is the best source to clip a viral short from - most relevant,
real footage (not reaction/commentary unless the trend is about that), from a credible channel,
as fresh as possible (a Short about a moment everyone has already seen gets swiped past), high
views? Return ONLY JSON: {{"id": "<video id>", "reason": "<short>"}}

{videos}
"""


def _llm(prompt: str, cfg: Config) -> str:
    provider = rank._pick_provider(cfg)
    if provider == "heuristic":
        raise RuntimeError("no LLM key configured")
    call = {
        "gemini": rank._call_gemini,
        "anthropic": rank._call_anthropic,
        "openai": rank._call_openai,
    }[provider]
    return call(prompt, cfg.llm_model)


def pick_topics(
    topics: list[Topic], cfg: Config, used: list[str], history: str = ""
) -> list[Topic]:
    """`history`: analytics.summary() - what this channel's past Shorts did, so the picker leans
    into what its viewers actually watch."""
    t = cfg.trends
    if not topics:
        return []
    listing = "\n".join(
        f"- {tp.name} [{tp.source}, traffic {tp.traffic}] {tp.headline}".strip() for tp in topics
    )
    try:
        raw = _llm(
            PICK_TOPICS.format(
                niches=", ".join(t.niches) or "anything viral",
                n=t.topics_per_run,
                used=", ".join(used[-50:]) or "nothing",
                history=f"How this channel's past videos did: {history}" if history else "",
                trending=listing,
            ),
            cfg,
        )
        items = rank._parse_json_array(raw)
    except Exception as e:
        log.warning("topic pick failed (%s); using top traffic", e)
        items = [
            {"topic": tp.name, "query": tp.name, "angle": tp.headline, "hashtags": []}
            for tp in sorted(topics, key=lambda x: -x.traffic)
            if tp.name.lower() not in {u.lower() for u in used}
        ][: t.topics_per_run]
    by_name = {tp.name.lower(): tp for tp in topics}
    picked = []
    for it in items:
        name = str(it.get("topic", "")).strip()
        base = by_name.get(name.lower()) or Topic(name=name, source="llm")
        base.query = str(it.get("query") or name)
        base.angle = str(it.get("angle", ""))
        base.hashtags = [str(h) for h in it.get("hashtags", [])][:5]
        if base.name:
            picked.append(base)
    return picked[: t.topics_per_run]


# search windows tried in order; the first one with enough usable videos is the pool
_FRESH_ROUNDS = ("hour", "today", "")
_MIN_POOL = 3


def find_sources(topic: Topic, cfg: Config) -> list[Candidate]:
    """Usable source videos for a trend, freshest window first: uploaded in the last hour, then
    today, then anything. A clip of a moment the feed has already seen for a day is dead on
    arrival, so an older video is only considered when nothing fresh fits (and never past
    trends.max_source_age_h when its age is known)."""
    t = cfg.trends
    q = topic.query or topic.name
    pool: list[Candidate] = []
    seen: set[str] = set()
    for within in _FRESH_ROUNDS:
        try:
            found = yt_search(q, 12, within=within)
        except Exception as e:
            log.warning("source search (%s) failed for %r: %s", within or "any", q, e)
            found = []
        for c in found:
            if c.id in seen or not (t.min_duration_s <= c.duration <= t.max_duration_s):
                continue
            if t.max_source_age_h and c.age_h is not None and c.age_h > t.max_source_age_h:
                continue
            seen.add(c.id)
            pool.append(c)
        if len(pool) >= _MIN_POOL:
            break
    return pool


def pick_source(topic: Topic, cfg: Config) -> Candidate | None:
    cands = find_sources(topic, cfg)
    if not cands:
        return None
    # fresh first, most viewed within the same window
    cands.sort(key=lambda c: (c.age_h if c.age_h is not None else 1e9, -c.views))
    listing = "\n".join(
        f"id={c.id} | {c.title} | {c.channel} | {int(c.duration)}s | {c.views:,} views | {c.age}"
        for c in cands
    )
    try:
        raw = _llm(PICK_SOURCE.format(topic=topic.name, angle=topic.angle, videos=listing), cfg)
        chosen = json.loads(raw.strip().strip("`"))["id"]
        for c in cands:
            if c.id == chosen:
                return c
    except Exception as e:
        log.warning("source pick failed (%s); using the freshest", e)
    return cands[0]


def dump(topics: list[Topic]) -> list[dict]:
    return [asdict(t) for t in topics]
