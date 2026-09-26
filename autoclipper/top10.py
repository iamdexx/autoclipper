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

from . import images, ingest, narrate, render, sfx, trends, verify
from .config import Config
from .ingest import Word

log = logging.getLogger(__name__)


@dataclass
class Entry:
    rank: int
    name: str
    line: str  # what the narrator says
    query: str  # YouTube search for footage
    video_id: str = ""
    verdict: str = ""  # fact-check result (verify.py); '' = not checked
    label: str = ""  # ranking style: 1-3 word tag shown in the on-screen list


@dataclass
class Script:
    topic: str
    title: str  # on-screen / upload title, e.g. "Top 10 Craziest NFL Plays This Week"
    intro: str  # narrator's opening line
    hashtags: list[str] = field(default_factory=list)
    entries: list[Entry] = field(default_factory=list)
    tease: str = ""  # ranking style: small line under the title, e.g. "the last one is insane"


SCRIPT_PROMPT = """Write a viral "Top {n}" countdown short (TikTok / Reels / Shorts) about: {topic}
Context: {angle}
Rules: countdown from {n} to 1, best last. Every entry needs real footage findable on YouTube.
Only REAL, well-documented events/facts - nothing invented; if unsure about a detail leave it out.
{avoid}
Narration is read aloud by a hyped, fast-talking narrator: plain spoken words only, no emojis,
hashtags, stage directions or rank numbers inside the sentences (the video shows the number).
Style: the intro is a cold open that promises a payoff ("...and number one broke the internet");
every line leads with the most shocking concrete detail (a number, a name, a consequence), stays
under 20 words, and every 2nd or 3rd entry ends with a short tease toward what's next.
Return ONLY JSON:
{{"title": "<Top {n} ... - punchy, max 8 words>",
  "intro": "<1 sentence cold-open hook, max 14 words>",
  "hashtags": ["#a", "#b", "#c", "#d"],
  "entries": [{{"rank": {n}, "name": "<entry, max 5 words>",
                "line": "<1-2 sentences, 10-20 words, specific and punchy>",
                "query": "<YouTube search that finds footage of exactly this>"}}, ...]}}
"""


LISTICLE_PROMPT = """Write a viral {n}-item listicle short (TikTok / Reels / Shorts) about: {topic}
Context: {angle}
Format: "{n} <things> that <hook>" - e.g. "{n} moments nobody expected this week". Each item is
one still photo on screen while a narrator says ONE short sentence about it, so every item must
be something a real, findable photo can show (a person, a place, a play, a product, a scene).
Only REAL, well-documented events/facts - nothing invented; if unsure about a detail leave it out.
{avoid}
Narration is read aloud by a calm, confident narrator with no pauses between items: plain spoken
words only, no emojis, hashtags, stage directions or numbering inside the sentences.
Style: the intro is the title said as a hook (max 10 words), each line leads with the surprising
concrete detail, stays under 14 words, and the last item is the biggest payoff.
Return ONLY JSON:
{{"title": "<{n} ... - punchy, max 8 words>",
  "intro": "<the hook sentence, max 10 words>",
  "hashtags": ["#a", "#b", "#c", "#d"],
  "entries": [{{"rank": {n}, "name": "<item, max 5 words>",
                "line": "<1 sentence, 8-14 words, specific>",
                "query": "<image search: the exact subject, 2-5 words, proper nouns first>"}}, ...]}}
"""


RANKING_PROMPT = """Write a viral "Ranking ..." short (TikTok / Reels / Shorts) about: {topic}
Context: {angle}
Format: {n} real clips shown back to back, ranked {n} (shown first) to 1 (best, shown last). A
title banner stays on screen ("Ranking The Most Insane <X> Moments") with a small tease under it,
and a 1-{n} list on the left fills in with a 1-3 word label as each clip plays. A hyped,
fast-talking narrator voices a cold-open hook and one line per clip over the footage, so every
entry must be a specific moment findable as a YouTube video.
Only REAL, well-documented moments - nothing invented; if unsure about a detail leave it out.
{avoid}
Narration is plain spoken words only: no emojis, hashtags, stage directions or rank numbers
inside the sentences (the video shows the number). Every line leads with the most shocking
concrete detail and the last one or two tease toward number one.
Return ONLY JSON:
{{"title": "<Ranking ... - starts with 'Ranking', max 6 words>",
  "tease": "<max 7 words, e.g. 'the last one broke the internet'>",
  "intro": "<1 sentence cold-open hook, max 12 words>",
  "hashtags": ["#a", "#b", "#c", "#d"],
  "entries": [{{"rank": {n}, "name": "<moment, max 5 words>",
                "label": "<1-3 word list tag, punchy, e.g. 'Skilled', 'Hell nah', 'Smooth'>",
                "line": "<1 sentence, 10-16 words, specific and punchy>",
                "query": "<YouTube search that finds footage of exactly this>"}}, ...]}}
"""

_PROMPTS = {"countdown": SCRIPT_PROMPT, "listicle": LISTICLE_PROMPT, "ranking": RANKING_PROMPT}


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "top10"


def write_script(topic: str, angle: str, cfg: Config, avoid: list[str] | None = None) -> Script:
    note = ""
    if avoid:
        note = "Do NOT use these (they failed a fact-check): " + "; ".join(avoid[:30])
    prompt = _PROMPTS.get(cfg.top10.style, SCRIPT_PROMPT)
    raw = trends._llm(prompt.format(n=cfg.top10.items, topic=topic, angle=angle, avoid=note), cfg)
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError("no JSON in LLM output")
    j = json.loads(m.group(0))
    entries = [
        Entry(
            int(e["rank"]),
            str(e["name"])[:60],
            str(e["line"])[:300],
            str(e["query"]),
            label=str(e.get("label", ""))[:24],
        )
        for e in j["entries"]
    ]
    entries.sort(key=lambda e: -e.rank)
    return Script(
        topic=topic,
        title=str(j["title"])[:80],
        intro=str(j.get("intro", ""))[:200],
        hashtags=[str(h) for h in j.get("hashtags", [])][:6],
        entries=entries[: cfg.top10.items],
        tease=str(j.get("tease", ""))[:60],
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
            if cfg.top10.style != "ranking":  # ranking's title is the banner, not the hook
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


def has_video(path: Path) -> bool:
    """A partial download (e.g. a 403 mid-stream) can leave an audio-only or empty file."""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
             "stream=codec_type", "-of", "csv=p=0", str(path)],
            text=True,
            timeout=60,
        )  # fmt: skip
    except Exception:
        return False
    return "video" in out


def fetch_footage(entry: Entry, secs: float, dest: Path, cfg: Config) -> Path | None:
    """Download only a `secs`-long window from the best-matching video (yt-dlp sections)."""
    key = hashlib.sha1(f"{entry.name}|{entry.query}".encode()).hexdigest()[:8]
    out = dest / f"{entry.rank:02d}-{_slug(entry.name)[:30]}-{key}.mp4"
    if out.exists():
        if has_video(out):
            return out
        out.unlink()
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
        if r.returncode == 0 and out.exists() and has_video(out):
            entry.video_id = c.id
            return out
        log.warning("download failed for %s: %s", c.id, r.stderr[-200:])
        for leftover in dest.glob(out.stem + "*"):
            leftover.unlink()
    return None


def _renumber(script: Script, cfg: Config, kept: int) -> None:
    """After dropping entries: ranks N..1 with no holes, and the title never promises more
    items than the video shows."""
    n = len(script.entries)
    for i, e in enumerate(script.entries):
        e.rank = n - i
    if n != cfg.top10.items:
        script.title = re.sub(rf"\b{cfg.top10.items}\b", str(n), script.title, count=1)
        if script.intro:
            script.intro = re.sub(rf"\b{cfg.top10.items}\b", str(n), script.intro, count=1)


def _finish(parts: list[Path], title: str, cfg: Config, vdir: Path) -> Path:
    """Stitch the segments and lay one continuous music bed under the whole thing."""
    out = vdir / f"{_slug(title)}.mp4"
    render.concat(parts, out, fade_out=render.OUTRO_FADE_S)
    bed = None
    if sfx.music_enabled(cfg):
        bed = sfx.music_bed(narrate.duration(out), cfg, vdir / "music.m4a")
    if bed:
        raw = out.with_name(out.stem + ".nomusic.mp4")
        out.rename(raw)
        sfx.mix_under(raw, bed, cfg.top10.music_db, out)
        raw.unlink()
    return out


def build_listicle(script: Script, cfg: Config, vdir: Path) -> Path:
    """Listicle style: one photo per item with a slow zoom, one word on screen at a time, the
    narrator reads straight through with no gaps, no badges/SFX/clip audio - just voice + bed."""
    t = cfg.top10
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "script.json").write_text(json.dumps(asdict(script), indent=2))

    pics: dict[int, Path] = {}
    for e in script.entries:
        p = images.fetch(e.query, e.name, vdir / "images")
        if p is None:
            log.warning("no image for #%d %s; dropping it", e.rank, e.name)
        else:
            pics[id(e)] = p
    script.entries = [e for e in script.entries if id(e) in pics]
    if len(script.entries) < max(3, cfg.verify.min_entries if cfg.verify.enabled else 3):
        raise RuntimeError(f"only {len(script.entries)} entries with images for {script.title!r}")
    _renumber(script, cfg, len(script.entries))
    (vdir / "script.json").write_text(json.dumps(asdict(script), indent=2))

    segments: list[tuple[str, Entry]] = []
    if script.intro:
        segments.append((script.intro, script.entries[0]))
    segments += [(e.line, e) for e in script.entries]
    parts: list[Path] = []
    for i, (text, entry) in enumerate(segments):
        seg = vdir / "segments" / f"{i:02d}.mp4"
        voiced = narrate.synthesize_timed(text, cfg, seg.with_suffix(".voice.m4a"))
        if voiced:
            voice, words = voiced
            dur = narrate.duration(voice) + t.listicle_gap_s
        else:
            voice, words = None, narrate.even_words(text, t.item_min_s)
            dur = t.item_min_s
        if i == len(segments) - 1:
            dur += t.listicle_outro_s  # the last photo holds a beat instead of cutting off
        # the intro shares the first item's photo: zoom out on the intro, in on the item
        zoom_in = not (script.intro and i == 0)
        render.render_still(pics[id(entry)], dur, voice, words, cfg, seg, zoom_in=zoom_in)
        parts.append(seg)
    return _finish(parts, script.title, cfg, vdir)


def _clip_words(src: Path, cuts: list[tuple[float, float]], cfg: Config) -> list[Word]:
    """What is said in the footage windows, re-timed to the segment's clock."""
    try:
        segs = ingest.transcribe(src, cfg, src.with_suffix(".words.json"))
    except Exception as e:
        log.warning("could not transcribe %s: %s", src.name, e)
        return []
    out: list[Word] = []
    at = 0.0
    for s, ln in cuts:
        out += [Word(w.start + at, w.end + at, w.text) for w in render.words_in(segs, s, s + ln)]
        at += ln
    return out


def build(script: Script, cfg: Config, vdir: Path) -> Path:
    if cfg.top10.style == "listicle":
        return build_listicle(script, cfg, vdir)
    ranking = cfg.top10.style == "ranking"
    t = cfg.top10
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "script.json").write_text(json.dumps(asdict(script), indent=2))
    parts: list[Path] = []

    # footage first: entries with nothing relevant on YouTube are dropped and the countdown is
    # renumbered, so the badges always run N..1 with no holes
    footage: dict[int, Path] = {}
    for e in script.entries:
        f = fetch_footage(e, t.item_max_s + 3, vdir / "footage", cfg)
        if f is None:
            log.warning("no footage for #%d %s; dropping it", e.rank, e.name)
        else:
            footage[id(e)] = f
    script.entries = [e for e in script.entries if id(e) in footage]
    if len(script.entries) < max(3, cfg.verify.min_entries if cfg.verify.enabled else 3):
        raise RuntimeError(f"only {len(script.entries)} entries with footage for {script.title!r}")
    _renumber(script, cfg, len(script.entries))
    (vdir / "script.json").write_text(json.dumps(asdict(script), indent=2))

    # intro + entries share one loop; the intro rides on the first entry's footage
    segments: list[tuple[str, str, str, Entry]] = []
    if script.intro and (not ranking or t.narrate):
        segments.append((script.intro, script.title, "", script.entries[0]))
    for e in script.entries:
        segments.append((e.line, e.name, f"#{e.rank}", e))
    labels = [(e.rank, e.label or e.name) for e in script.entries]

    whoosh = sfx.whoosh(cfg.workdir / "assets" / "reveal.m4a") if t.sfx else None
    for i, (text, title, badge, entry) in enumerate(segments):
        seg = vdir / "segments" / f"{i:02d}.mp4"
        voiced = None
        if not ranking or t.narrate:
            voiced = narrate.synthesize_timed(text, cfg, seg.with_suffix(".voice.m4a"))
        unduck_at = None
        if voiced:
            voice, words = voiced
            dur = max(t.item_min_s, narrate.duration(voice) + 0.4)  # never cut speech
        else:
            voice, words = None, [] if ranking else narrate.even_words(text, t.item_min_s)
            dur = t.item_min_s
        if i == len(segments) - 1 and t.outro_s > 0:
            # #1 is the payoff: its footage plays on after the last line with its own audio
            unduck_at = dur - 0.2
            dur += t.outro_s
        src = footage[id(entry)]
        avail = narrate.duration(src)
        # the intro rides on entry N's footage: play its cuts in reverse order so the two
        # segments don't open on the same frame
        cuts = render.cut_plan(avail, min(dur, max(1.0, avail)), t.cuts_per_entry)
        if not badge:
            cuts = cuts[::-1]
        events = None
        duck = t.source_db
        if ranking:
            # banner + running list instead of a badge (the intro shows the list still empty),
            # without a narrator the clip's own speech is what gets captioned and its audio
            # is left alone
            revealed = entry.rank if badge else entry.rank + 1
            title, badge = "", ""
            events = render.ranking_events(script.title, script.tease, labels, revealed, dur)
            if not voice:
                words, duck = _clip_words(src, cuts, cfg), 0.0
        render.render_narrated(
            src,
            cuts,
            voice,
            words,
            title,
            badge,
            cfg,
            seg,
            sfx=whoosh,
            duck_db=duck,
            events=events,
            unduck_at=unduck_at,
        )
        parts.append(seg)
    return _finish(parts, script.title, cfg, vdir)


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


def narration_text(script: Script) -> str:
    """Everything the narrator is meant to say, for the QA speech check."""
    return " ".join([script.intro, *(e.line for e in script.entries)]).strip()


def make(topic: str, angle: str, cfg: Config, vdir: Path) -> tuple[Path, Script]:
    script = verified_script(topic, angle, cfg, vdir)
    log.info("top10 %r: %d entries", script.title, len(script.entries))
    return build(script, cfg, vdir), script
