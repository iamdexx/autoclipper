"""'Top 10 ...' countdown videos: an LLM writes the list + narration, each entry is voiced by the
narrator over a few seconds of the most relevant YouTube footage, everything is stitched into
one vertical video with a big rank badge, entry title and word-highlight captions."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import narrate, render, trends, verify
from .config import Config

log = logging.getLogger(__name__)


@dataclass
class Entry:
    rank: int
    name: str
    line: str  # what the narrator says
    query: str  # YouTube search for footage
    video_id: str = ""
    verdict: str = ""  # fact-check result (verify.py); '' = not checked


@dataclass
class Script:
    topic: str
    title: str  # on-screen / upload title, e.g. "Top 10 Craziest NFL Plays This Week"
    intro: str  # narrator's opening line
    hashtags: list[str] = field(default_factory=list)
    entries: list[Entry] = field(default_factory=list)


SCRIPT_PROMPT = """Write a viral "Top {n}" countdown short (TikTok / Reels / Shorts) about: {topic}
Context: {angle}
Rules: countdown from {n} to 1, best last. Every entry needs real footage findable on YouTube.
Only REAL, well-documented events/facts - nothing invented; if unsure about a detail leave it out.
{avoid}
Narration is read aloud by an energetic narrator: plain spoken words only, no emojis, hashtags,
stage directions or rank numbers inside the sentences (the video shows the number).
Return ONLY JSON:
{{"title": "<Top {n} ... - punchy, max 8 words>",
  "intro": "<1 sentence hook the narrator opens with, max 14 words>",
  "hashtags": ["#a", "#b", "#c", "#d"],
  "entries": [{{"rank": {n}, "name": "<entry, max 5 words>",
                "line": "<1-2 sentences, 12-24 words, specific and punchy>",
                "query": "<YouTube search that finds footage of exactly this>"}}, ...]}}
"""


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "top10"


def write_script(topic: str, angle: str, cfg: Config, avoid: list[str] | None = None) -> Script:
    note = ""
    if avoid:
        note = "Do NOT use these (they failed a fact-check): " + "; ".join(avoid[:30])
    raw = trends._llm(
        SCRIPT_PROMPT.format(n=cfg.top10.items, topic=topic, angle=angle, avoid=note), cfg
    )
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError("no JSON in LLM output")
    j = json.loads(m.group(0))
    entries = [
        Entry(int(e["rank"]), str(e["name"])[:60], str(e["line"])[:300], str(e["query"]))
        for e in j["entries"]
    ]
    entries.sort(key=lambda e: -e.rank)
    return Script(
        topic=topic,
        title=str(j["title"])[:80],
        intro=str(j.get("intro", ""))[:200],
        hashtags=[str(h) for h in j.get("hashtags", [])][:6],
        entries=entries[: cfg.top10.items],
    )


def fact_check(script: Script, cfg: Config) -> list[verify.Check]:
    """Check every entry's narration (and the intro, if any) against real-world evidence;
    sets Entry.verdict. Returns one Check per entry, then the intro's Check last."""
    claims = [
        (f"{e.name}: {e.line}", e.query or f"{script.topic} {e.name}") for e in script.entries
    ]
    if script.intro:
        claims.append((f"{script.title}: {script.intro}", script.topic))
    checks = verify.check_claims(claims, cfg)
    for e, c in zip(script.entries, checks, strict=False):
        e.verdict = c.verdict
        log.info(
            "fact-check #%d %s -> %s (%.2f) %s", e.rank, e.name, c.verdict, c.confidence, c.note
        )
    if script.intro:
        c = checks[-1]
        log.info("fact-check intro -> %s (%.2f) %s", c.verdict, c.confidence, c.note)
        if not verify.passes(c, cfg):
            log.warning("intro rejected, dropping it: %r / %r", script.title, script.intro)
            script.intro = ""
            script.title = f"Top {len(script.entries)} {script.topic}".strip()
    return checks


def verified_script(topic: str, angle: str, cfg: Config, vdir: Path) -> Script:
    """Write a script, drop entries that fail the fact-check, refill from a fresh draft (up to
    verify.max_rounds), re-rank 10..1. Raises if fewer than verify.min_entries survive."""
    v = cfg.verify
    script = first = write_script(topic, angle, cfg)
    if not v.enabled:
        return script
    vdir.mkdir(parents=True, exist_ok=True)
    report: list[dict] = []
    keep: list[Entry] = []
    rejected: list[str] = []
    seen: set[str] = set()
    for rnd in range(max(1, v.max_rounds)):
        checks = fact_check(script, cfg)
        report += verify.dump(checks)
        fresh: list[Entry] = []
        for e, c in zip(script.entries, checks, strict=False):
            k = e.name.lower()
            if k in seen:
                continue
            seen.add(k)
            if verify.passes(c, cfg):
                fresh.append(e)
            else:
                rejected.append(e.name)
        keep = fresh + keep  # refills take the low ranks; the original top picks stay on top
        (vdir / "factcheck.json").write_text(json.dumps(report, indent=2))
        if len(keep) >= cfg.top10.items or rnd == v.max_rounds - 1:
            break
        log.info("round %d: %d/%d entries verified; rewriting", rnd + 1, len(keep), cfg.top10.items)
        script = write_script(topic, angle, cfg, avoid=rejected + [e.name for e in keep])
    if len(keep) < v.min_entries:
        raise RuntimeError(
            f"fact-check: only {len(keep)} verified entries for {topic!r} (need {v.min_entries})"
        )
    keep = keep[-cfg.top10.items :]
    for i, e in enumerate(keep):
        e.rank = len(keep) - i
    first.entries = keep
    return first


_STOP = set(
    "the a an of in on at to and or vs v for with his her their its this that from by".split()
)


def _keywords(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2 and w not in _STOP}


def footage_matches(entry: Entry, title: str) -> bool:
    """The search result must actually be about the entry: the video title has to share the
    proper-noun-ish keywords of the entry name (or most of the query)."""
    t = _keywords(title)
    name = _keywords(entry.name)
    if name and len(name & t) >= max(1, len(name) // 2):
        return True
    q = _keywords(entry.query)
    return bool(q) and len(q & t) >= max(2, (len(q) * 2) // 3)


def fetch_footage(entry: Entry, secs: float, dest: Path, cfg: Config) -> Path | None:
    """Download only a `secs`-long window from the best-matching video (yt-dlp sections)."""
    key = hashlib.sha1(f"{entry.name}|{entry.query}".encode()).hexdigest()[:8]
    out = dest / f"{entry.rank:02d}-{_slug(entry.name)[:30]}-{key}.mp4"
    if out.exists():
        return out
    try:
        cands = [
            c
            for c in trends.yt_search(entry.query, 8)
            if 20 <= c.duration <= cfg.top10.source_max_duration_s
        ]
    except Exception as e:
        log.warning("footage search failed for %r: %s", entry.query, e)
        return None
    cands = [c for c in cands if footage_matches(entry, c.title)]
    cands.sort(key=lambda c: -c.views)
    for c in cands[:3]:
        # skip the first ~20% (intros) and pull a window from there
        start = max(0.0, min(c.duration * 0.2, c.duration - secs - 1))
        cmd = [
            "yt-dlp",
            "--no-playlist",
            "-f",
            "b[height<=720][ext=mp4]/bv*[height<=720][ext=mp4]+ba[ext=m4a]/b",
            "--download-sections",
            f"*{start:.0f}-{start + secs:.0f}",
            "--force-keyframes-at-cuts",
            "--merge-output-format",
            "mp4",
            "-o",
            str(out),
            c.url,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode == 0 and out.exists():
            entry.video_id = c.id
            return out
        log.warning("download failed for %s: %s", c.id, r.stderr[-200:])
    return None


def build(script: Script, cfg: Config, vdir: Path) -> Path:
    t = cfg.top10
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "script.json").write_text(json.dumps(asdict(script), indent=2))
    parts: list[Path] = []

    # intro + entries share one loop; the intro rides on the first entry's footage
    segments: list[tuple[str, str, str, Entry]] = []
    if script.intro:
        segments.append((script.intro, script.title, "", script.entries[0]))
    for e in script.entries:
        segments.append((e.line, e.name, f"#{e.rank}", e))

    for i, (text, title, badge, entry) in enumerate(segments):
        seg = vdir / "segments" / f"{i:02d}.mp4"
        voiced = narrate.synthesize_timed(text, cfg, seg.with_suffix(".voice.m4a"))
        if voiced:
            voice, words = voiced
            dur = max(t.item_min_s, narrate.duration(voice) + 0.6)  # never cut speech
        else:
            voice, words = None, narrate.even_words(text, t.item_min_s)
            dur = t.item_min_s
        footage = fetch_footage(entry, max(dur, t.item_max_s) + 3, vdir / "footage", cfg)
        if footage is None:
            log.warning("no footage for #%d %s; skipping", entry.rank, entry.name)
            continue
        # intro uses a different offset of the same footage so the visuals differ from entry 1
        start = 0.0 if badge else 1.0
        dur = min(dur, max(1.0, narrate.duration(footage) - start))
        render.render_narrated(footage, start, dur, voice, words, title, badge, cfg, seg)
        parts.append(seg)

    if len(parts) < 3:
        raise RuntimeError(f"only {len(parts)} usable segments for {script.title!r}")
    out = vdir / f"{_slug(script.title)}.mp4"
    return render.concat(parts, out)


def pick_topic(cfg: Config, used: list[str]) -> tuple[str, str] | None:
    """(topic, angle) for the next Top-10: from config.top10.topics, else from what's trending."""
    fixed = [x for x in cfg.top10.topics if x.lower() not in {u.lower() for u in used}]
    if fixed:
        return fixed[0], ""
    if cfg.top10.topics:
        return None
    picked = trends.pick_topics(trends.discover(cfg), cfg, used)
    if not picked:
        return None
    tp = picked[0]
    return tp.name, tp.angle or tp.headline


def make(topic: str, angle: str, cfg: Config, vdir: Path) -> tuple[Path, Script]:
    script = verified_script(topic, angle, cfg, vdir)
    log.info("top10 %r: %d entries", script.title, len(script.entries))
    return build(script, cfg, vdir), script
