"""'Top 10 ...' countdown videos: an LLM writes the list + narration, each entry is voiced by the
narrator over a few seconds of the most relevant YouTube footage, everything is stitched into
one vertical video with a big rank badge, entry title and word-highlight captions."""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import images, ingest, narrate, render, review, sfx, slop, trends, verify
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
    footage: str = ""  # explainer about a game, film or show: a YouTube search for footage of it


SCRIPT_PROMPT = """Write a viral "Top {n}" countdown short (TikTok / Reels / Shorts) about: {topic}
Context: {angle}
Rules: countdown from {n} to 1, best last. Every entry needs real footage findable on YouTube.
Only REAL, well-documented events/facts - nothing invented; if unsure about a detail leave it out.
{sources}{avoid}
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
{sources}{avoid}
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
{sources}{avoid}
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

EXPLAINER_PROMPT = """Write the script for a short narrated explainer (TikTok / Reels / Shorts,
45-60 seconds) about: {topic}
Context: {angle}
It is a story, not a list. The title is a question or claim people want answered ("How Humans
Survived The Ice Age", "Why GTA 6 Took 13 Years"). The intro states the puzzle in one sentence.
Then {n} beats, each one or two sentences, each adding ONE specific fact that moves the story
forward (who, what number, what happened next), in the order a good teacher would tell it. The
last beat answers the title. A viewer who stops after any beat should already have learned
something.
{rules}
Use ONLY facts that appear in the SOURCES below; if the sources do not support {n} beats, return
fewer. No predictions, no rumours stated as fact.
{sources}{avoid}
Every beat is shown over ONE real photograph from Wikimedia Commons, so "query" must name a
real-world subject a photo exists of (a place, a person, an object, a building, an animal), 2-4
words, proper nouns first - never a game screenshot, a logo or an abstract idea.
Use one word for one thing from the title to the last beat (if the title says "railway", never
say "railroad"), and the spelling of the place the story is set in. Open the intro on the single
most surprising fact. Each beat's photo must show what that beat talks about.
Narration is plain spoken words only: no emojis, hashtags, stage directions or numbering.
Write every number in digits ("475 million", "1.65 billion", "13 years"), never spelled out: the
line is shown on screen a few words at a time, and "seventy-five million" alone reads as wrong.
Return ONLY JSON:
{{"title": "<the question or claim, max 9 words, Title Case>",
  "intro": "<the puzzle in one sentence, max 18 words>",
  "footage": "<only when the story is about ONE video game, film or show: its name plus
              'gameplay no commentary' or 'official trailer'. Otherwise an empty string>",
  "hashtags": ["#a", "#b", "#c", "#d"],
  "entries": [{{"rank": {n}, "name": "<beat label, max 4 words>",
                "line": "<1-2 sentences, 14-26 words, one concrete fact>",
                "query": "<Wikimedia Commons photo search>"}}, ...]}}
"rank" counts down: the first beat told is {n}, the last is 1.
"""

IDEA_PROMPT = """Propose ONE subject for a 60-second narrated explainer video in the style of "How
humans survived the Ice Age" or "Why the Titanic's lookouts had no binoculars": a concrete
question from history, science, engineering, nature, games or film whose real answer surprises
people, and that the English Wikipedia covers well. Not any of: {used}.
Return ONLY JSON: {{"topic": "<the question, max 10 words>", "wiki": "<the Wikipedia article that answers it>"}}
"""

_PROMPTS = {
    "countdown": SCRIPT_PROMPT,
    "listicle": LISTICLE_PROMPT,
    "ranking": RANKING_PROMPT,
    "explainer": EXPLAINER_PROMPT,
}

SOURCES_NOTE = """SOURCES - real headlines and video titles about this topic, fetched just now. Your own
knowledge ends before recent events, so every entry MUST be a moment that appears in (or is
directly implied by) these sources, and names/numbers must match them. Skip anything the sources don't
mention rather than guessing. If the sources cover fewer than {n} distinct moments, return fewer.
{items}
"""


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "top10"


SUBJECT_PROMPT = """A short video will answer: "{topic}"
Which ONE English Wikipedia article tells that story? Answer with the exact title of the article
and nothing else."""

ARTICLE_RETRY_S = 20
_articles: dict[str, tuple[verify.Evidence, ...]] = {}


def _article(topic: str, cfg: Config | None) -> tuple[verify.Evidence, ...]:
    """A search by the question's own words finds whatever article shares them ("Why did the US
    plan to nuke the Moon?" found the Democratic Party), so the model names the article first."""
    queries = [topic]
    if cfg is not None:
        try:
            lines = trends._llm(SUBJECT_PROMPT.format(topic=topic), cfg).strip().splitlines()
            name = lines[-1].strip(" \"'*.") if lines else ""
            if 0 < len(name) <= 100:
                queries.insert(0, name)
        except Exception as e:
            log.warning("could not name the article for %r: %s", topic, e)
    for q in queries:
        # capped once, here: the writer and the fact-check judge are shown the same paragraphs
        found = tuple(verify.wiki_extract(q, chars=24000)[:60])
        if found:
            log.info("article for %r: %s", topic, found[0].title.split(":")[0])
            return found
    return ()


def article_for(topic: str, cfg: Config | None = None) -> list[verify.Evidence]:
    """The Wikipedia article an explainer is written from and checked against ([] when the
    lookup fails): a search by each line's wording rarely finds the paragraph that backs it."""
    for last in (False, True):  # Wikipedia now and then answers 429: once more after a pause
        if topic in _articles:
            break
        try:
            _articles[topic] = _article(topic, cfg)
        except Exception as e:
            log.warning("no article for %r: %s", topic, e)
            if last:
                return []
            time.sleep(ARTICLE_RETRY_S)
    return list(_articles[topic])


def sources_for(
    topic: str, angle: str, n: int, titles: bool = True, cfg: Config | None = None
) -> str:
    """Evidence block for the script prompt: the same keyless retrieval the fact-check judges
    against (news, wiki, and YouTube titles unless `titles` is off), so the writer picks moments
    that will later verify."""
    if titles:
        ev = verify.gather(topic)
        if angle and angle.lower() not in topic.lower():
            ev += verify.gather(f"{topic} {angle}"[:120])
    else:
        ev = verify.gather(topic, titles=False)
        if angle and angle.lower() not in topic.lower():
            ev += verify.gather(angle[:120], titles=False)
    if not titles:  # the article first and in full: every beat has to come out of it
        ev = [*article_for(topic, cfg), *ev]
    seen: set[str] = set()
    items = []
    for e in ev:
        k = e.title.lower()
        if k in seen:
            continue
        seen.add(k)
        meta = " | ".join(x for x in (e.source, e.date) if x)
        cap = (700 if titles else 1200) if e.kind == "article" else 160
        items.append(f"- [{e.kind}] {e.title[:cap]}" + (f" ({meta})" if meta else ""))
    if not items:
        return ""
    return SOURCES_NOTE.format(n=n, items="\n".join(items[: 40 if titles else 70]))


def write_script(topic: str, angle: str, cfg: Config, avoid: list[str] | None = None) -> Script:
    note = ""
    if avoid:
        note = "Do NOT use these (they failed a fact-check): " + "; ".join(avoid[:30])
    prompt = _PROMPTS.get(cfg.top10.style, SCRIPT_PROMPT)
    story = cfg.top10.style == "explainer"
    sources = ""
    if cfg.verify.enabled or story:
        sources = sources_for(topic, angle, cfg.top10.items, titles=not story, cfg=cfg)
    if story and not sources:
        raise RuntimeError(f"no sources found to write {topic!r} from")
    filled = prompt.format(
        n=cfg.top10.items, topic=topic, angle=angle, avoid=note, sources=sources, rules=slop.RULES
    )
    for last in (False, True):  # a model now and then answers with no or broken JSON: ask again
        raw = trends._llm(filled, cfg)
        try:
            j = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
            break
        except (AttributeError, ValueError) as e:
            if last:
                raise ValueError("no JSON in LLM output") from e
            log.warning("script answer was not valid JSON (%s); asking again", e)
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
        footage=str(j.get("footage") or "")[:80] if story else "",
    )


def fact_check(script: Script, cfg: Config) -> list[verify.Check]:
    """Check every entry's narration (and the intro, if any) against real-world evidence;
    sets Entry.verdict. Returns one Check per entry, then the intro's Check last."""
    story = cfg.top10.style == "explainer"
    # an explainer's "query" is a photo search; its facts are looked up by what the line says
    claims = [
        (f"{e.name}: {e.line}", f"{script.topic} {e.name}" if story else e.query or e.name)
        for e in script.entries
    ]
    if script.intro:
        claims.append((f"{script.title}: {script.intro}", script.topic))
    checks = (
        verify.check_claims(claims, cfg, titles=False, shared=article_for(script.topic, cfg))
        if story
        else verify.check_claims(claims, cfg)
    )
    for e, c in zip(script.entries, checks, strict=False):
        found = slop.filler(e.line)
        if found:  # hype instead of information: never spoken, whatever the evidence says
            c.verdict, c.note = "unsupported", f"filler: {', '.join(found)}"
        e.verdict = c.verdict
        log.info(
            "fact-check #%d %s -> %s (%.2f) %s", e.rank, e.name, c.verdict, c.confidence, c.note
        )
    if script.intro:
        c = checks[-1]
        log.info("fact-check intro -> %s (%.2f) %s", c.verdict, c.confidence, c.note)
        if not verify.passes(c, cfg) or slop.filler(script.intro):
            log.warning("intro rejected, dropping it: %r / %r", script.title, script.intro)
            script.intro = ""
            if cfg.top10.style not in ("ranking", "explainer"):  # their title is not the hook
                script.title = f"Top {len(script.entries)} {script.topic}".strip()
    return checks


def verified_script(topic: str, angle: str, cfg: Config, vdir: Path) -> Script:
    """Write a script, drop entries that fail the fact-check, refill from a fresh draft (up to
    verify.max_rounds), re-rank 10..1. Raises if fewer than verify.min_entries survive."""
    v = cfg.verify
    strict = v.strict or v.strict_top10
    story = cfg.top10.style == "explainer"
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
            if story and any(_same_fact(e.line, o.line) for o in fresh + keep):
                log.info("beat %r says what an earlier beat says; dropping it", e.name)
            elif verify.passes(c, cfg, strict=strict):
                fresh.append(e)
            else:
                rejected.append(e.name)
        keep = fresh + keep  # refills take the low ranks; the original top picks stay on top
        (vdir / "factcheck.json").write_text(json.dumps(report, indent=2))
        if len(keep) >= cfg.top10.items or rnd == v.max_rounds - 1:
            break
        if story and len(keep) >= v.min_entries:
            break  # a story is told in order: beats from a second draft would land out of place
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


def _same_fact(a: str, b: str) -> bool:
    """Two narration lines that tell the viewer the same thing: most of the shorter one's words
    are in the other, and neither carries a figure the other contradicts ("Trailer 1 drew 93
    million views" and "Trailer 2 drew 475 million views" are two facts)."""
    m, n = (set(re.findall(r"\d[\d.]*\d|\d", s.replace(",", ""))) for s in (a, b))
    if m and n and not (m <= n or n <= m):
        return False
    x, y = _keywords(a), _keywords(b)
    return bool(x and y) and len(x & y) >= 0.5 * min(len(x), len(y))


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


def fetch_footage(
    entry: Entry,
    secs: float,
    dest: Path,
    cfg: Config,
    max_s: float = 0,
    skip: frozenset[str] = frozenset(),
) -> Path | None:
    """Download only a `secs`-long window from the best-matching video (yt-dlp sections).
    `max_s`: the longest video to take it from (default: top10.source_max_duration_s).
    `skip`: ids of videos not to take it from."""
    key = hashlib.sha1(f"{entry.name}|{entry.query}".encode()).hexdigest()[:8]
    out = dest / f"{entry.rank:02d}-{_slug(entry.name)[:30]}-{key}.mp4"
    source = out.with_suffix(".id")  # which video a kept clip came from
    if out.exists():
        if has_video(out):
            entry.video_id = entry.video_id or (source.read_text() if source.exists() else "")
            return out
        out.unlink()
    try:
        cands = [
            c
            for c in trends.yt_search(entry.query, 8)
            if 20 <= c.duration <= (max_s or cfg.top10.source_max_duration_s)
        ]
    except Exception as e:
        log.warning("footage search failed for %r: %s", entry.query, e)
        return None
    cands = [
        c
        for c in cands
        if c.id not in skip
        and footage_matches(entry, c.title)
        and trends.rights_ok(c.title, c.channel, cfg)
    ]
    cands.sort(key=lambda c: -c.views)
    for c in cands[:3]:
        # skip the first ~20% (intros) and pull a window from there
        start = max(0.0, min(c.duration * 0.2, c.duration - secs - 1))
        cmd = ingest.ytdlp(
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
        )
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode == 0 and out.exists() and has_video(out):
            entry.video_id = c.id
            source.write_text(c.id)
            return out
        log.warning("download failed for %s: %s", c.id, r.stderr[-200:])
        for leftover in dest.glob(out.stem + "*"):
            leftover.unlink()
    return None


def footage_stills(script: Script, n: int, cfg: Config, dest: Path) -> list[Path]:
    """Up to `n` clean frames of the game, film or show a story is about, for the beats no
    photograph fits: a photo library has nothing for a game. Each frame is checked first."""
    if n <= 0 or not script.footage:
        return []
    dest.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    tried: set[str] = set()
    for k in range(3):  # one upload may carry its channel's watermark throughout: try others
        if len(out) >= n:
            break
        # clean footage of a game is mostly hours-long walkthroughs; only a window is downloaded
        src = Entry(k, "", "", script.footage)
        clip = fetch_footage(src, 120, dest, cfg, max_s=12 * 3600, skip=frozenset(tried))
        if clip is None:
            log.warning("no more footage found for %r", script.footage)
            break
        tried.add(src.video_id)
        dur = narrate.duration(clip)
        looks = (n - len(out)) * 3
        for i in range(looks):
            if len(out) == n:
                break
            f = dest / f"still-{k}-{i:02d}.jpg"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{dur * (i + 0.5) / looks:.3f}"]
                + ["-i", str(clip), "-frames:v", "1", "-q:v", "2"]
                # a game's own subtitles and button prompts sit at the bottom of the frame
                + ["-vf", "crop=iw:ih*0.78:0:0", str(f)],
                check=False,
            )
            if f.exists() and review.backdrop(f, script.footage, cfg):
                out.append(f)
    return out


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


def _pictures(script: Script, cfg: Config, vdir: Path, story: bool) -> dict[int, Path]:
    """The picture each entry is shown over, keyed by id(entry); an entry with none is absent."""
    pics: dict[int, Path] = {}
    if story and script.footage and cfg.review.enabled:
        # a story about a game is shown over the game: the photo library answers a line about
        # its revenue with a dollar bill, and the finished video failed for wrong pictures
        stills = footage_stills(script, len(script.entries), cfg, vdir / "footage")
        pics.update({id(e): s for e, s in zip(script.entries, stills, strict=False)})
    for e in script.entries:
        if id(e) in pics:
            continue
        fits = None
        if story and cfg.review.enabled:  # each photo has to show what its line says
            fits = functools.partial(review.photo, line=e.line, name=e.name, cfg=cfg)
        p = images.fetch(
            e.query,
            e.name,
            vdir / "images",
            thumbs=not story,
            accept=fits,
            beat=e.line if fits else "",
        )
        if p is not None:
            pics[id(e)] = p
    return pics


def build_listicle(script: Script, cfg: Config, vdir: Path) -> Path:
    """Listicle style: one photo per item with a slow zoom, one word on screen at a time, the
    narrator reads straight through with no gaps, no badges/SFX/clip audio - just voice + bed.
    Explainer style is the same build with the story's title over the opening, photos from
    Wikimedia Commons only (never another channel's thumbnail) and the push direction
    alternating beat to beat."""
    t = cfg.top10
    story = t.style == "explainer"
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "script.json").write_text(json.dumps(asdict(script), indent=2))

    pics = _pictures(script, cfg, vdir, story)
    for e in script.entries:
        if id(e) not in pics:
            log.warning("no image for #%d %s; dropping it", e.rank, e.name)
    script.entries = [e for e in script.entries if id(e) in pics]
    if len(script.entries) < max(3, cfg.verify.min_entries if cfg.verify.enabled else 3):
        raise RuntimeError(f"only {len(script.entries)} entries with images for {script.title!r}")
    if not story:
        _renumber(script, cfg, len(script.entries))
    (vdir / "script.json").write_text(json.dumps(asdict(script), indent=2))

    segments: list[tuple[str, Entry]] = []
    if script.intro or story:  # a story always opens on its title, said aloud if nothing else
        segments.append((script.intro or script.title, script.entries[0]))
    segments += [(e.line, e) for e in script.entries]
    outro = cfg.publishing.outro.strip() if story else ""
    if outro:  # the subscribe line rides on the last photo
        segments.append((outro, script.entries[-1]))
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
        zoom_in = i % 2 == 1 if story else not (script.intro and i == 0)
        title = script.title if story and i == 0 else ""
        render.render_still(
            pics[id(entry)], dur, voice, words, cfg, seg, zoom_in=zoom_in, title=title, fill=story
        )
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
    if cfg.top10.style in ("listicle", "explainer"):
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


def open_idea(cfg: Config, used: list[str]) -> tuple[str, str] | None:
    """(question, Wikipedia article) for an explainer on any subject, or None when the writer is
    unavailable."""
    try:
        raw = trends._llm(IDEA_PROMPT.format(used="; ".join(used[-40:]) or "(none yet)"), cfg)
        j = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        topic, wiki = str(j["topic"]).strip()[:100], str(j.get("wiki", "")).strip()[:100]
    except Exception as e:
        log.warning("no open explainer idea (%s)", e)
        return None
    return (topic, wiki) if topic and trends.rights_ok(topic, "", cfg) else None


def pick_topic(cfg: Config, used: list[str], state=None) -> tuple[str, str] | None:
    """(topic, angle) for the next Top-10: from config.top10.topics, else from what's trending.
    With `state`, the pick takes its turn in the topic mix (trends.pillars) and is recorded."""
    fixed = [x for x in cfg.top10.topics if x.lower() not in {u.lower() for u in used}]
    if fixed:
        return fixed[0], ""
    if cfg.top10.topics:
        return None
    t = cfg.top10
    if t.style == "explainer" and t.open_topic_every and (len(used) + 1) % t.open_topic_every == 0:
        idea = open_idea(cfg, used)
        if idea:
            return idea
    picked = trends.pick_topics(
        trends.discover(cfg), cfg, used, pillars_used=state.used_pillars() if state else None
    )
    if not picked:
        return None
    tp = picked[0]
    if state:
        state.mark_pillar(tp.pillar)
    return tp.name, tp.angle or tp.headline


def narration_text(script: Script, cfg: Config | None = None) -> str:
    """Everything the narrator is meant to say, for the QA speech check."""
    intro = script.intro
    if not intro and cfg is not None and cfg.top10.style == "explainer":
        intro = script.title
    outro = ""
    if cfg is not None and cfg.top10.style == "explainer":
        outro = cfg.publishing.outro.strip()
    return " ".join([intro, *(e.line for e in script.entries), outro]).strip()


def make(topic: str, angle: str, cfg: Config, vdir: Path) -> tuple[Path, Script]:
    script = verified_script(topic, angle, cfg, vdir)
    log.info("top10 %r: %d entries", script.title, len(script.entries))
    return build(script, cfg, vdir), script
