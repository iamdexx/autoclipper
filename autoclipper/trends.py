"""Discover what's trending right now and find the best YouTube source video for it.

Sources (all keyless, best-effort - a failing source is skipped):
  google  : Google Trends daily RSS (per geo) - topic, approx traffic, headline
  youtube : yt-dlp search of each configured niche, most viewed uploads of the last 24h

With `trends.pillars` set the channel has a fixed topic mix instead: every pillar's own searches
are run, anything off `trends.focus_keywords` is dropped, and each run's topics come from the
pillars used least recently (`next_pillars`), one topic per pillar.

An LLM (same provider order as ranking) then filters the raw list down to topics that fit the
configured niches and are worth a video, and picks the best source video per topic.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import time
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

import requests

from . import ingest, rank
from .config import Config, Pillar

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
    pillar: str = ""  # topic-mix pillar it was found under ("" = fits any)


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


def _names(text: str, keywords: list[str]) -> bool:
    kws = [k.strip() for k in keywords if k.strip()]
    if not kws:
        return False
    pat = "|".join(r"\s*".join(re.escape(w) for w in k.split()) for k in kws)
    return re.search(rf"\b(?:{pat})\b", text or "", re.I) is not None


def rights_ok(title: str, channel: str, cfg: Config) -> bool:
    """False for footage the channel never uses: an upload by a trends.blocked_channels channel
    or one of its regional channels ("Rockstar Games" also blocks "Rockstar Games Polska"), or a
    title naming one of trends.exclude_keywords."""
    t = cfg.trends
    name = channel.strip().lower()
    for b in (b.strip().lower() for b in t.blocked_channels):
        if b and name.startswith(b) and not name[len(b) : len(b) + 1].isalnum():
            return False
    return not _names(title, t.exclude_keywords)


def on_focus(text: str, cfg: Config, pillar: str | None = None) -> bool:
    """True when `text` fits the channel: it names none of trends.exclude_keywords and one of
    trends.focus_keywords (whole words; anything fits with none set). A pillar with its own
    `keywords` is matched on those instead; `pillar=None` accepts any pillar's keywords and
    `pillar=""` only the channel's own."""
    t = cfg.trends
    if _names(text, t.exclude_keywords):
        return False
    if not [k for k in t.focus_keywords if k.strip()]:
        return True
    own = [p.keywords for p in t.pillars if p.keywords and pillar in (None, p.name)]
    if pillar is not None:
        return _names(text, own[0] if own else t.focus_keywords)
    return _names(text, t.focus_keywords) or any(_names(text, k) for k in own)


def pillar_hot(query: str, n: int = 8) -> list[Topic]:
    """Today's most-viewed uploads for a pillar search; a narrow pillar with nothing new today
    falls back to YouTube's relevance ranking so it still gets its turn."""
    return youtube_hot(query, n) or [
        Topic(name=c.title, source="youtube", traffic=c.views, headline=c.channel)
        for c in yt_search(query, n)
        if c.views
    ]


def next_pillars(cfg: Config, used: list[str], n: int) -> list[Pillar]:
    """The `n` pillars whose turn it is: never-used first (config order), then least recently used,
    skipping any that already fills its max_share of a `pillar_window` of topics (at least one
    topic per window, so a capped pillar still gets a turn)."""
    t = cfg.trends
    w = t.pillar_window
    recent = list(used[-w:]) if w > 0 else []
    out: list[Pillar] = []

    def fits(p: Pillar) -> bool:
        if p.max_share >= 1 or w <= 0:
            return True
        return [*recent, p.name][-w:].count(p.name) <= max(1, int(p.max_share * w))

    for _ in range(min(n, len(t.pillars))):
        last = {name: i for i, name in enumerate(recent)}
        open_ = [p for p in t.pillars if p not in out and fits(p)]
        if not open_:
            break
        pick = min(open_, key=lambda p: last.get(p.name, -1))
        out.append(pick)
        recent.append(pick.name)
    return out


def discover(cfg: Config) -> list[Topic]:
    t = cfg.trends
    jobs = [(google_trends, g, f"google trends {g}", "") for g in t.geos]
    if t.pillars:
        jobs += [
            (pillar_hot, q, f"youtube search {q!r}", p.name) for p in t.pillars for q in p.queries
        ]
    else:
        jobs += [(youtube_hot, n, f"youtube search {n!r}", "") for n in t.niches]

    def _run(job):
        fn, arg, label, pillar = job
        try:
            found = fn(arg)
        except Exception as e:
            log.warning("%s failed: %s", label, e)
            return []
        for tp in found:
            tp.pillar = pillar
        return [tp for tp in found if on_focus(f"{tp.name} {tp.headline}", cfg, pillar or None)]

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
{history}{mix}
Return ONLY a JSON array, each item:
{{"topic": "<trend name exactly as given>", "query": "<best YouTube search to find the key footage>",
  "angle": "<one sentence: why it's blowing up / what the clip should show>",
  "hashtags": ["#tag", "#tag2", "#tag3"]}}

TRENDING:
{trending}
"""

PICK_SOURCE = """Trend: {topic}
Angle: {angle}
Which ONE of these YouTube videos is the best source to clip a short from? Prefer, in this order:
a real person on camera who made or reported the thing themselves (a creator, a journalist, a
developer interview) speaking English; most relevant to the trend; as fresh as possible; high
views. Avoid faceless channels that read articles over footage with a synthetic voice,
compilations, re-uploads, "N things" listicles and videos not in English - return an empty id if
every option is one of those.
Return ONLY JSON: {{"id": "<video id or empty>", "reason": "<short>"}}

{videos}
"""


def _llm(prompt: str, cfg: Config) -> str:
    provider = rank._pick_provider(cfg)
    if provider == "heuristic":
        raise RuntimeError("no LLM key configured")
    call = {
        "nvidia": rank._call_nvidia,
        "gemini": rank._call_gemini,
        "anthropic": rank._call_anthropic,
        "openai": rank._call_openai,
    }[provider]
    return call(prompt, cfg.llm_model)


PILLAR_MIX = """
The channel is about {focus} and rotates through a fixed topic mix. This time pick exactly ONE
topic for EACH of these pillars, from the items tagged with that pillar (or tagged "any"):
{pillars}
Never present rumours or leaks as confirmed, and never promise an income or a result."""


def pick_topics(
    topics: list[Topic],
    cfg: Config,
    used: list[str],
    history: str = "",
    pillars_used: list[str] | None = None,
) -> list[Topic]:
    """`history`: analytics.summary() - what this channel's past Shorts did, so the picker leans
    into what its viewers actually watch. `pillars_used`: State.used_pillars(), which decides
    whose turn it is when trends.pillars is set."""
    t = cfg.trends
    if not topics:
        return []
    used_l = {u.lower() for u in used}
    slots = next_pillars(cfg, pillars_used or [], t.topics_per_run) if t.pillars else []
    if t.pillars:
        names = {p.name for p in slots}
        topics = [tp for tp in topics if tp.pillar in names or not tp.pillar]
        if not slots or not topics:
            return []
    listing = "\n".join(
        f"- {tp.name} [{tp.pillar or 'any'}, {tp.source}, traffic {tp.traffic}] "
        f"{tp.headline}".strip()
        if slots
        else f"- {tp.name} [{tp.source}, traffic {tp.traffic}] {tp.headline}".strip()
        for tp in topics
    )
    mix = (
        PILLAR_MIX.format(
            focus=t.focus or "the niches above",
            pillars="\n".join(f"- {p.name}: {p.angle}".rstrip(": ") for p in slots),
        )
        if slots
        else ""
    )
    prompt = PICK_TOPICS.format(
        niches=t.focus or ", ".join(t.niches) or "anything viral",
        n=len(slots) or t.topics_per_run,
        used=", ".join(used[-50:]) or "nothing",
        history=f"How this channel's past videos did: {history}" if history else "",
        mix=mix,
        trending=listing,
    )
    try:
        for last in (False, True):  # a model now and then answers with no JSON: ask again
            try:
                items = rank._parse_json_array(_llm(prompt, cfg))
                break
            except ValueError as e:
                if last:
                    raise
                log.warning("topic pick answer had no JSON (%s); asking again", e)
    except Exception as e:
        log.warning("topic pick failed (%s); using top traffic", e)
        items = [
            {"topic": tp.name, "query": tp.name, "angle": tp.headline, "hashtags": []}
            for tp in sorted(topics, key=lambda x: -x.traffic)
            if tp.name.lower() not in used_l
        ]
        # with a topic mix, _one_per_pillar fills each slot from that pillar's own results
        items = [] if slots else items[: t.topics_per_run]
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
    if slots:
        return _one_per_pillar(picked, topics, slots, used_l, cfg)
    return picked[: t.topics_per_run]


def _one_per_pillar(
    picked: list[Topic], topics: list[Topic], slots: list[Pillar], used: set[str], cfg: Config
) -> list[Topic]:
    """One topic per pillar whose turn it is: the picker's choice when it is a real, unused item
    of that pillar, else that pillar's most-viewed unused item, else the most-viewed untagged
    one (a Google trend) that is about that pillar's subject."""
    by_traffic = sorted(topics, key=lambda x: -x.traffic)
    out: list[Topic] = []
    taken: set[str] = set()
    for p in slots:
        own = [tp for tp in by_traffic if tp.pillar == p.name]
        for tp in [*picked, *own, *by_traffic]:
            k = tp.name.lower()
            if tp.source == "llm" or k in used or k in taken or tp.pillar not in (p.name, ""):
                continue
            if not tp.pillar and not on_focus(f"{tp.name} {tp.headline}", cfg, p.name):
                continue
            tp.pillar = p.name
            tp.query = tp.query or tp.name
            tp.angle = tp.angle or p.angle or tp.headline
            taken.add(k)
            out.append(tp)
            break
        else:
            log.info("pillar %r: nothing usable this run", p.name)
    return out


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
            if not rights_ok(c.title, c.channel, cfg):
                continue
            if not on_focus(c.title, cfg, topic.pillar or None):
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
        if not chosen:
            log.info("no source worth clipping for %r", topic.name)
            return None
        for c in cands:
            if c.id == chosen:
                return c
    except Exception as e:
        log.warning("source pick failed (%s); using the freshest", e)
    return cands[0]


def dump(topics: list[Topic]) -> list[dict]:
    return [asdict(t) for t in topics]
