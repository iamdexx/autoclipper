"""Upload metadata (title / description / hashtags) for a finished video.

The on-screen hook is written to stop a thumb, not to be found by search: "Career Threatening
Injury In The Ring" says nothing about who, where or when. The LLM rewrites it into a searchable
title that names the real people / teams / event from the source video and the clip's own
transcript, writes a 1-2 sentence description that credits the source, and picks specific
hashtags (people, event, niche) instead of the generic #clips #fyp spam. Anything the clip doesn't
show stays out - the inputs are the same fact-checked hook and transcript the video was cut from.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from . import top10, trends
from .config import Config

log = logging.getLogger(__name__)

TITLE_MAX = 90  # YouTube caps titles at 100; leave room for " #shorts" and never mid-word cut
DESC_MAX = 600
BRAND_TAGS = ["#shorts"]  # always present, always last

PROMPT = """You write the upload metadata for a short-form video (YouTube Shorts / TikTok / Reels).
{kind_note}
Source video: "{source_title}" by {channel}
On-screen hook: {hook}
Narrator says: {narration}
What is said in the clip (transcript):
{transcript}
Hashtags already suggested for this topic: {hashtags}

Only describe what the hook, source title and transcript support - nothing invented, no outcomes
the clip doesn't show.
Return ONLY JSON:
{{"title": "<max 70 chars. Names the real people / teams / event / show so it's searchable, then the
   payoff or curiosity (e.g. 'Mahomes to Kelce: The 4th-Down Throw That Beat the Bills'). No emojis,
   no hashtags, no ALL CAPS, no clickbait the clip doesn't deliver.>",
 "description": "<1-2 sentences: who, what, where/when (the event / stream), written for someone
   deciding whether to watch. {credit}>",
 "hashtags": ["<5-8 tags, most specific first: the people, the event, the show/team, then the niche
   (e.g. #wrestling), then one or two broad ones that genuinely fit (#sports). No spaces, no
   generic filler like #fyp #viral #clips #trending unless it truly fits.>"]}}
"""

KIND_NOTES = {
    "clip": "It's a clip cut from the source video below.",
    "top10": "It's an original countdown / ranking video the channel made; the transcript is its "
    "narration and the source title is its topic.",
}
CREDIT = {
    "clip": "End with a credit line: 'Clip from {channel}.'",
    "top10": "No credit line.",
}


@dataclass
class PostMeta:
    title: str
    description: str
    hashtags: list[str] = field(default_factory=list)


def _tag(h: str) -> str:
    h = "#" + re.sub(r"[^0-9A-Za-z_]", "", str(h))
    return h if len(h) > 1 else ""


def clean_tags(tags: list[str], limit: int = 8) -> list[str]:
    """#-prefixed, alphanumeric, de-duplicated (case-insensitive), brand tags last."""
    out: list[str] = []
    seen: set[str] = set()
    for h in [_tag(t) for t in tags] + BRAND_TAGS:
        if h and h.lower() not in seen:
            seen.add(h.lower())
            out.append(h)
    picked = [h for h in out if h.lower() not in {b.lower() for b in BRAND_TAGS}][: limit - 1]
    return picked + BRAND_TAGS


def clip_title(s: str) -> str:
    """Whole words, under the cap, no trailing punctuation."""
    s = re.sub(r"\s+", " ", str(s)).strip().strip("#").strip()
    if len(s) > TITLE_MAX:
        s = s[:TITLE_MAX].rsplit(" ", 1)[0]
    return s.rstrip(" -:,;|")


def fallback(hook: str, source_title: str, channel: str, hashtags: list[str]) -> PostMeta:
    """No LLM: hook + source name is still more searchable than the hook alone."""
    title = clip_title(f"{hook} | {source_title}" if source_title else hook)
    desc = f"{hook}\n\n{source_title}".strip() + (f"\nClip from {channel}." if channel else "")
    return PostMeta(title, desc, clean_tags(hashtags))


def describe(
    hook: str,
    narration: str,
    transcript: str,
    source_title: str,
    channel: str,
    hashtags: list[str],
    cfg: Config,
    kind: str = "clip",
) -> PostMeta:
    """Title / description / hashtags for a finished video. Falls back to a template when no LLM
    is configured or the call fails - a post never blocks on this."""
    fb = fallback(hook, source_title, channel, hashtags)
    try:
        raw = trends._llm(
            PROMPT.format(
                kind_note=KIND_NOTES.get(kind, KIND_NOTES["clip"]),
                credit=CREDIT.get(kind, CREDIT["clip"]).format(
                    channel=channel or "the original creator"
                ),
                source_title=source_title or hook,
                channel=channel or "the original creator",
                hook=hook,
                narration=narration or "(none)",
                transcript=transcript[:3000] or "(none)",
                hashtags=" ".join(hashtags) or "(none)",
            ),
            cfg,
        )
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            raise ValueError("no JSON in LLM output")
        j = json.loads(m.group(0))
        title = clip_title(j.get("title", ""))
        desc = re.sub(r"\s+\n", "\n", str(j.get("description", "")).strip())[:DESC_MAX]
        tags = clean_tags([str(h) for h in j.get("hashtags", [])] + hashtags)
    except Exception as e:  # metadata is best-effort; the template is always acceptable
        log.warning("metadata failed (%s); using template", e)
        return fb
    if len(title) < 12 or not desc:
        log.warning("metadata too thin (%r); using template", title)
        return fb
    log.info("metadata: %r %s", title, " ".join(tags))
    return PostMeta(title, desc, tags)


def moment_transcript(segments, start: float, end: float) -> str:
    return " ".join(s.text for s in segments if s.end > start and s.start < end).strip()


def top10_transcript(script) -> str:
    """The countdown's own narration plus the entry list, so the title can name the entries."""
    listing = "\n".join(f"#{e.rank} {e.name}" for e in script.entries)
    return f"{top10.narration_text(script)}\n\nEntries:\n{listing}"
