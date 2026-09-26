"""'Top 10 ...' countdown videos: an LLM writes the list + narration, each entry is voiced by the
narrator over a few seconds of the most relevant YouTube footage, everything is stitched into
one vertical video with a big rank badge, entry title and word-highlight captions."""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import narrate, render, trends
from .config import Config

log = logging.getLogger(__name__)


@dataclass
class Entry:
    rank: int
    name: str
    line: str  # what the narrator says
    query: str  # YouTube search for footage
    video_id: str = ""


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


def write_script(topic: str, angle: str, cfg: Config) -> Script:
    raw = trends._llm(SCRIPT_PROMPT.format(n=cfg.top10.items, topic=topic, angle=angle), cfg)
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


def fetch_footage(entry: Entry, secs: float, dest: Path, cfg: Config) -> Path | None:
    """Download only a `secs`-long window from the best-matching video (yt-dlp sections)."""
    out = dest / f"{entry.rank:02d}.mp4"
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
    script = write_script(topic, angle, cfg)
    log.info("top10 %r: %d entries", script.title, len(script.entries))
    return build(script, cfg, vdir), script
