"""Editor's pass: a vision model looks at frames of a video and reads what is said in it, then
answers the checklist a viewer applies in two seconds. The technical gate (qa.py) proves a file
plays; this one decides whether it is worth anyone's time. Nothing is posted on a review that
could not be run - the video waits in the queue for the next run."""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import ingest, narrate, rank
from .config import Config

log = logging.getLogger(__name__)

PASS_SCORE = 7


@dataclass
class Verdict:
    ok: bool | None  # None = the review could not be run
    score: int = 0
    problems: list[str] = field(default_factory=list)
    note: str = ""


SOURCE_PROMPT = """These are {n} frames sampled evenly across a YouTube video titled "{title}" by
"{channel}", followed by the start of its transcript. A channel wants to cut short clips from it
and will add its own captions. Answer strictly from what you see and read.

Return ONLY JSON:
{{"presenter": "<'on_camera' if a real person is filmed talking in a good share of the frames; 'voice_over' if it is footage or gameplay with no person filmed; 'slideshow' if it is mostly screenshots of articles, text cards or still images>",
  "synthetic_voice": <true if the transcript reads like text-to-speech narration of an article rather than a person talking>,
  "burned_captions": <true if the video already has captions or large text burned into most frames>,
  "overlays": <true if subscribe/like buttons, 'follow' animations, donation or chat pop-ups, or another brand's banner appear. The creator's own webcam inset of themselves and the game's own HUD are not overlays>,
  "watermark": <true if a channel's or brand's logo, name or @handle sits in the same corner or edge of the picture in several frames (a watermark). The game's own HUD and logos inside the footage itself are not watermarks>,
  "screenshot_frames": <how many of the {n} frames are filled or covered by a screenshot of a social-media post, an article or a text card, as a number>,
  "english": <true if the speech is English a general audience can follow>,
  "note": "<one sentence>"}}

TRANSCRIPT START:
{said}
"""

VIDEO_PROMPT = """These are {n} frames sampled evenly, in order, from a finished vertical short our
channel is about to publish, followed by everything said in it. Our own elements are: one
caption line in the lower middle, a small title at the top for the first seconds, and {mark}.
Our own elements are never foreign_overlays, but still check our captions and title for
garbled_captions and inconsistent_wording. Judge it the way a viewer who likes the subject would, strictly
from what you see and read. Be hard to please: most videos should not pass.

{format}

Return ONLY JSON:
{{"problems": [<zero or more of these exact strings>
     "not_english"        - the speech is not English, or cannot be followed,
     "garbled_captions"   - captions show broken characters, mixed scripts or nonsense words,
     "foreign_overlays"   - subscribe/like buttons, another channel's captions, banners or logos,
     "empty_frame"        - the picture is a small strip inside black bars, or mostly blank,
     "slideshow_of_text"  - the picture is mostly screenshots of articles or text cards,
     "wrong_pictures"     - what is shown does not match what is being said,
     "inconsistent_wording" - the on-screen title and the narration name the same thing with
                            different words (title says "railway", narrator says "railroad"),
                            or a caption shows a different word from the one spoken,
     "says_nothing"       - a viewer learns no specific fact or argued opinion,
     "hype_filler"        - our host's lines are hype instead of information,
     "repeats_itself"     - the same sentence or point is said twice,
     "cut_off"            - it starts or ends in the middle of a thought],
  "score": <1-10: would this viewer watch to the end and trust the channel more afterwards>,
  "note": "<one sentence: the single biggest reason for the score>"}}

SAID IN THE VIDEO:
{said}
"""

_FORMATS = {
    "clip": "FORMAT: a clip of someone else's video. Our host says one line before it and may "
    "add one line after it; everything in between is the original speaker.",
    "top10": "FORMAT: our own scripted story, narrated start to finish by our host over photos "
    "or footage. It must open with a question or claim worth staying for, give a new specific "
    "fact in every beat, and end on a payoff.",
}


def frames(video: Path, n: int, width: int = 360) -> list[bytes]:
    """`n` JPEG frames spread evenly over the video."""
    dur = narrate.duration(video)
    out: list[bytes] = []
    with tempfile.TemporaryDirectory() as td:
        for i in range(n):
            f = Path(td) / f"{i}.jpg"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{dur * (i + 0.5) / n:.3f}"]
                + ["-i", str(video), "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "5"]
                + [str(f)],
                check=False,
            )
            if f.exists():
                out.append(f.read_bytes())
    return out


def _ask(prompt: str, pics: list[bytes], cfg: Config) -> dict:
    provider = rank._pick_provider(cfg)
    if provider == "nvidia" and rank._nvidia_key():
        vision = "nvidia"
    elif rank._gemini_key():
        vision = "gemini"
    elif rank._nvidia_key():
        vision = "nvidia"
    else:
        raise RuntimeError("no vision model configured (needs an NVIDIA or Gemini key)")
    if not pics:
        raise RuntimeError("no frames could be read")

    def ask() -> str:
        if vision == "nvidia":
            # llm_model names the text model; the review needs one that takes images
            return rank._call_nvidia(prompt, None, images=pics)
        # llm_model names a model of the configured provider, which need not be Gemini
        model = cfg.llm_model if provider == "gemini" else None
        return rank._call_gemini(prompt, model, images=pics)

    for last in (False, False, True):
        raw = ask()
        try:
            return json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        except (AttributeError, ValueError) as e:
            if last:
                raise
            log.warning("review answer was not valid JSON (%s); asking again", e)


PHOTO_PROMPT = """This picture will fill the screen of a short video while the narrator says:
"{line}"
The picture was searched for as: {name}.
Does the picture show what the narrator is talking about, so a viewer would not be confused?
A picture of something else (a different object, place, person, game or product), a logo, a
diagram, a screenshot of text or an article, or a picture carrying a watermark is a "no".
Return JSON only: {{"shows": "<what the picture shows, a few words>", "fits": true|false}}"""


def photo(path: Path, line: str, name: str, cfg: Config) -> bool:
    """Does this photo fit the line it is shown over? True when the check could not run, so an
    outage does not stop every explainer (the finished video is still reviewed)."""
    try:
        j = _ask(PHOTO_PROMPT.format(line=line[:400], name=name), [path.read_bytes()], cfg)
    except Exception as e:
        log.warning("photo check could not run (%s)", e)
        return True
    fits = j.get("fits") is not False
    log.info("photo for %r %s: %s", name, "fits" if fits else "does NOT fit", j.get("shows"))
    return fits


BACKDROP_PROMPT = """This frame will fill the screen of a short video, behind a narrator telling a
story about: {subject}
Is it a clean picture of that subject (for a game: the game itself being played or one of its
scenes)? A frame with a person in a webcam box, a channel's logo, name or @handle, captions or
other added text, subtitles or button prompts, a menu, a loading or title screen, a screenshot
of a post or an article, or a frame of something else is a "no".
Return JSON only: {{"shows": "<what the frame shows, a few words>", "fits": true|false}}"""


def backdrop(path: Path, subject: str, cfg: Config) -> bool:
    """Is this video frame clean footage of the subject? False when the check could not run:
    another channel's footage is never shown unchecked."""
    try:
        j = _ask(BACKDROP_PROMPT.format(subject=subject[:200]), [path.read_bytes()], cfg)
    except Exception as e:
        log.warning("footage check could not run (%s)", e)
        return False
    fits = j.get("fits") is True
    log.info("footage frame %s: %s", "fits" if fits else "does NOT fit", j.get("shows"))
    return fits


# a source where this share of sampled frames is somebody's post or an article: clips cut from it
# carry that account's name and logo, and fail the finished-video review
MAX_SCREENSHOT_SHARE = 0.2


def source(video: Path, title: str, channel: str, said: str, cfg: Config) -> Verdict:
    """Is this a video worth cutting from: a real person talking in English, with a clean
    picture? ok=None when the check could not be run."""
    try:
        pics = frames(video, max(cfg.review.frames, 12), width=640)  # a corner mark is small
        j = _ask(
            SOURCE_PROMPT.format(n=len(pics), title=title, channel=channel, said=said[:3000]),
            pics,
            cfg,
        )
    except Exception as e:
        log.warning("source review could not run (%s)", e)
        return Verdict(None, note=str(e)[:200])
    problems = []
    try:
        shots = float(j.get("screenshot_frames") or 0)
    except (TypeError, ValueError):
        shots = 0
    if j.get("presenter") == "slideshow" or shots >= MAX_SCREENSHOT_SHARE * len(pics):
        problems.append("slideshow_of_text")
    if j.get("synthetic_voice") is True and j.get("presenter") != "on_camera":
        problems.append("synthetic_voice")
    if j.get("burned_captions") is True:
        problems.append("burned_captions")
    if j.get("overlays") is True or j.get("watermark") is True:
        problems.append("foreign_overlays")
    if j.get("english") is False:
        problems.append("not_english")
    return Verdict(not problems, problems=problems, note=str(j.get("note", ""))[:300])


def _own_mark(cfg: Config) -> str:
    """How the reviewer is told about our handle, so it is not taken for another channel's."""
    if not cfg.look.watermark:
        return "no watermark"
    return f'our channel handle "{cfg.look.watermark}" as a small watermark in a top corner'


def video(path: Path, kind: str, cfg: Config) -> Verdict:
    """The finished video against the viewer's checklist; the report lands in <video>.review.json.
    ok=None when the review could not be run."""
    try:
        cache = path.with_suffix(".review-transcript.json")
        said = " ".join(s.text for s in ingest.transcribe(path, cfg, cache))
        pics = frames(path, cfg.review.frames)
        j = _ask(
            VIDEO_PROMPT.format(
                n=len(pics),
                format=_FORMATS.get(kind, _FORMATS["clip"]),
                said=said[:6000],
                mark=_own_mark(cfg),
            ),
            pics,
            cfg,
        )
        problems = [str(p) for p in j.get("problems") or []]
        score = int(float(j.get("score", 0) or 0))
        v = Verdict(
            not problems and score >= PASS_SCORE, score, problems, str(j.get("note", ""))[:300]
        )
    except Exception as e:
        log.warning("review could not run for %s (%s)", path.name, e)
        return Verdict(None, note=str(e)[:200])
    path.with_suffix(".review.json").write_text(json.dumps(v.__dict__, indent=2))
    log.info(
        "review %s %s: score %d%s - %s",
        "PASS" if v.ok else "FAIL",
        path.name,
        v.score,
        f" ({', '.join(v.problems)})" if v.problems else "",
        v.note,
    )
    return v
