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

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.id}"


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


# YouTube search filter: type=video, uploaded today, sorted by view count
_HOT_TODAY = "CAISBAgCEAE%3D"


def yt_search(query: str, n: int = 10, hot_today: bool = False) -> list[Candidate]:
    if hot_today:
        q = urllib.parse.quote_plus(query)
        target = f"https://www.youtube.com/results?search_query={q}&sp={_HOT_TODAY}"
    else:
        target = f"ytsearch{n}:{query}"
    out = subprocess.run(
        ingest.ytdlp("--flat-playlist", "-J", "--playlist-end", str(n), target),
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
        cands.append(
            Candidate(
                id=e["id"],
                title=e.get("title") or "",
                channel=e.get("channel") or e.get("uploader") or "",
                duration=float(e.get("duration") or 0),
                views=int(e.get("view_count") or 0),
            )
        )
    return cands


def youtube_hot(niche: str, n: int = 8) -> list[Topic]:
    """Most-viewed uploads of the last 24h for a niche."""
    cands = yt_search(niche, n, hot_today=True)
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
recent, high views? Return ONLY JSON: {{"id": "<video id>", "reason": "<short>"}}

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


def pick_topics(topics: list[Topic], cfg: Config, used: list[str]) -> list[Topic]:
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


def pick_source(topic: Topic, cfg: Config) -> Candidate | None:
    t = cfg.trends
    cands = [
        c
        for c in yt_search(topic.query or topic.name, 12)
        if t.min_duration_s <= c.duration <= t.max_duration_s
    ]
    if not cands:
        return None
    cands.sort(key=lambda c: -c.views)
    listing = "\n".join(
        f"id={c.id} | {c.title} | {c.channel} | {int(c.duration)}s | {c.views:,} views"
        for c in cands
    )
    try:
        raw = _llm(PICK_SOURCE.format(topic=topic.name, angle=topic.angle, videos=listing), cfg)
        chosen = json.loads(raw.strip().strip("`"))["id"]
        for c in cands:
            if c.id == chosen:
                return c
    except Exception as e:
        log.warning("source pick failed (%s); using most viewed", e)
    return cands[0]


def dump(topics: list[Topic]) -> list[dict]:
    return [asdict(t) for t in topics]
