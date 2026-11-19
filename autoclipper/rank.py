"""Pick the moments in a transcript most likely to perform as short-form clips.

Provider order for `llm_provider: auto`: NVIDIA -> Gemini -> Anthropic -> OpenAI -> heuristic,
based on which API key is present in the environment.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass

from . import slop, tools
from .config import Config
from .ingest import Segment

log = logging.getLogger(__name__)


@dataclass
class Moment:
    start: float
    end: float
    hook: str  # short title / on-screen hook line
    reason: str
    score: float
    narration: str = ""  # 1-2 spoken sentences read by the narrator over the intro
    take: str = ""  # the narrator's commentary after the clip: what it means, from the clip only


# ---------- candidate windows ----------


def _windows(segments: list[Segment], cfg: Config) -> list[tuple[float, float, str]]:
    """Slide over segments producing candidate windows close to target length, snapped to
    sentence boundaries so clips don't start/end mid-word."""
    out = []
    n = len(segments)
    i = 0
    while i < n:
        start = segments[i].start
        j = i
        text = []
        while j < n and segments[j].end - start < cfg.clips.max_len_s:
            text.append(segments[j].text)
            if segments[j].end - start >= cfg.clips.target_len_s:
                break
            j += 1
        j = min(j, n - 1)
        end = segments[j].end
        if end - start >= cfg.clips.min_len_s:
            out.append((start, end, " ".join(text)))
        # advance ~half a window for overlap
        step_until = start + cfg.clips.target_len_s / 2
        while i < n and segments[i].start < step_until:
            i += 1
        if i == j and j < n - 1:
            i += 1
        elif i <= j and j == n - 1:
            break
    return out


# ---------- heuristic ----------

_HOT = re.compile(
    r"\b(oh my god|what the|no way|bro|insane|crazy|wait|actually|never|always|worst|best|"
    r"secret|truth|money|\$\d|million|hate|love|literally|shut up|are you kidding|"
    r"let me tell you|here's the thing|nobody|everyone|stop|dude|holy|wtf|lmao|haha)\b",
    re.I,
)


def heuristic_score(text: str, dur: float) -> float:
    words = text.split()
    if not words:
        return 0.0
    wps = len(words) / max(dur, 1)  # speech density
    hot = len(_HOT.findall(text))
    excl = text.count("!") + text.count("?")
    caps = sum(1 for w in words if len(w) > 2 and w.isupper())
    score = 0.4 * min(wps / 3.0, 1.5) + 0.35 * hot + 0.15 * excl + 0.1 * caps
    # penalise filler-heavy windows
    filler = len(re.findall(r"\b(um|uh|like|you know)\b", text, re.I))
    score -= 0.05 * filler
    return round(score, 3)


def _hook_from_text(text: str) -> str:
    sent = re.split(r"(?<=[.!?])\s+", text.strip())
    for s in sent:
        if 15 <= len(s) <= 70:
            return s.rstrip(".")[:60]
    return (text[:60] + "...") if len(text) > 60 else text


def rank_heuristic(cands, cfg: Config) -> list[Moment]:
    scored = [
        Moment(
            s,
            e,
            _hook_from_text(t),
            "heuristic: energy/keywords",
            heuristic_score(t, e - s),
        )
        for s, e, t in cands
    ]
    return _dedupe(sorted(scored, key=lambda m: -m.score), cfg.clips.count)


def _dedupe(moments: list[Moment], n: int) -> list[Moment]:
    picked: list[Moment] = []
    for m in moments:
        if all(m.end <= p.start or m.start >= p.end for p in picked):
            picked.append(m)
        if len(picked) >= n:
            break
    return picked


# ---------- LLM ----------

PROMPT = """You are the editor of a short-form video channel (TikTok / Reels / Shorts) that
viewers trust because it never wastes their time. Below is a timestamped transcript of a video
by {creator}. Pick up to {n} moments worth posting. A moment is worth posting only if a stranger
learns one specific, surprising thing from it or hears a strong opinion argued with a reason:
a new rule, a number, a quote, a comparison, a reveal. Skip intros, sponsor reads, "like and
subscribe", recaps of things everyone knows, and anything that needs the rest of the video to
make sense. If fewer than {n} moments clear that bar, return fewer; an empty array is fine.

Each clip must be between {min_len} and {max_len} seconds. It must start on the first word of a
sentence, with the point arriving in the first 3 seconds, and end on the last word of a
sentence that completes the thought - never mid-explanation, never on a question that the next
line answers.

{rules}

Return ONLY a JSON array, no prose, each item:
{{"start": <seconds float>, "end": <seconds float>,
  "hook": "<on-screen title, 4-8 words: the specific claim made in the clip, e.g. 'GTA 6 Locks Supercars Behind The Story'. Not a tease.>",
  "narration": "<ONE sentence, max 12 words, said by our host before the clip: the clip's actual claim plus who says it, e.g. 'Rockstar says some parked cars need a key cloner.' Must not repeat the clip's first sentence. Plain spoken words only.>",
  "take": "<our host's line after the clip: ONE or two sentences, 14-26 words, giving context the clip itself leaves out, taken from elsewhere in THIS transcript (a caveat, a number, what is still unconfirmed, how it compares to something said earlier or later). It must not restate the clip. If the rest of the transcript adds nothing, return an empty string.>",
  "reason": "<one sentence: what the viewer learns>", "score": <0-10>}}

TRANSCRIPT:
{transcript}
"""


class Unavailable(RuntimeError):
    """The model that picks moments could not be reached; nothing was picked."""


def _transcript_text(segments: list[Segment], max_chars: int = 180_000) -> str:
    lines = [f"[{s.start:.1f}-{s.end:.1f}] {s.text}" for s in segments]
    text = "\n".join(lines)
    if len(text) > max_chars:
        # keep even coverage by sampling every k-th line
        k = len(text) // max_chars + 1
        text = "\n".join(lines[::k])
    return text


def _gemini_key() -> str | None:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "Geminikey"):
        if os.getenv(name):
            return os.getenv(name)
    return None


def _nvidia_key() -> str | None:
    return os.getenv("NVIDIA_API_KEY") or None


def _pick_provider(cfg: Config) -> str:
    p = cfg.llm_provider
    if p != "auto":
        return p
    if _nvidia_key():
        return "nvidia"
    if _gemini_key():
        return "gemini"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "heuristic"


# tried in order when the request is throttled / the model is overloaded
GEMINI_MODELS = [
    "gemini-flash-latest",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
]


def _call_gemini(prompt: str, model: str | None, images: list[bytes] | None = None) -> str:
    """`images`: JPEGs sent along with the prompt, in order."""
    import requests

    key = _gemini_key()
    models = [model] if model else GEMINI_MODELS
    pics = [
        {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(b).decode()}}
        for b in images or []
    ]
    body = {
        "contents": [{"parts": [*pics, {"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, "responseMimeType": "application/json"},
    }
    last: Exception | None = None
    for attempt in range(4):
        for m in models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
            r = requests.post(url, headers={"x-goog-api-key": key}, json=body, timeout=180)
            if r.status_code in (429, 500, 503):
                last = RuntimeError(f"gemini {m} {r.status_code}: {r.text[:200]}")
                log.warning("%s - trying next model", last)
                tools.record(f"gemini/{m}", False, str(r.status_code))
                continue
            tools.record(f"gemini/{m}", r.status_code == 200, str(r.status_code))
            r.raise_for_status()
            parts = r.json()["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        time.sleep(5 * (attempt + 1))
    raise last or RuntimeError("gemini unavailable")


# build.nvidia.com: free, but it refuses bursts, so calls are spaced out. Only the VL models
# take images.
NVIDIA_MODELS = ["nvidia/nemotron-3-ultra-550b-a55b", "nvidia/nemotron-3.5-lightning-30b-a3b"]
NVIDIA_VISION_MODELS = [
    "nvidia/nemotron-3.5-super-vl-120b-a12b",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
]
NVIDIA_MIN_GAP_S = 2.0
_nvidia_last = 0.0


def configure(cfg: Config) -> None:
    """Take the NVIDIA model order from the config, when it names one."""
    global NVIDIA_MODELS, NVIDIA_VISION_MODELS
    NVIDIA_MODELS = list(cfg.nvidia_text_models) or NVIDIA_MODELS
    NVIDIA_VISION_MODELS = list(cfg.nvidia_vision_models) or NVIDIA_VISION_MODELS


def _nvidia_post(model: str, content, timeout: float = 240):
    import requests

    global _nvidia_last
    wait = _nvidia_last + NVIDIA_MIN_GAP_S - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    try:
        return requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {_nvidia_key()}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.4,
                "max_tokens": 8000,
            },
            timeout=timeout,
        )
    finally:
        _nvidia_last = time.monotonic()


def _call_nvidia(prompt: str, model: str | None, images: list[bytes] | None = None) -> str:
    """`images`: JPEGs sent along with the prompt, in order. Falls back to Gemini (when a key
    is set) if NVIDIA stays unavailable."""
    import requests

    models = [model] if model else (NVIDIA_VISION_MODELS if images else NVIDIA_MODELS)
    content: str | list = prompt
    if images:
        content = [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b).decode()},
            }
            for b in images
        ] + [{"type": "text", "text": prompt}]
    last: Exception | None = None
    r = None
    for attempt in range(3):
        for m in models:
            try:
                r = _nvidia_post(m, content)
            except requests.RequestException as e:
                last = RuntimeError(f"nvidia {m}: {e}")
                log.warning("%s - trying next model", last)
                tools.record(m, False, str(e))
                continue
            if r.status_code != 200:
                last = RuntimeError(f"nvidia {m} {r.status_code}: {r.text[:200]}")
                log.warning("%s - trying next model", last)
                tools.record(m, False, str(r.status_code))
                continue
            text = r.json()["choices"][0]["message"].get("content") or ""
            tools.record(m, bool(text.strip()), "empty answer")
            if text.strip():
                return text
            last = RuntimeError(f"nvidia {m}: empty answer")
        if r is not None and r.status_code in (401, 403, 404):
            break  # the key or the model is refused: waiting will not help
        time.sleep(15 * (attempt + 1))
    if _gemini_key():
        log.warning("nvidia unavailable (%s); using gemini", last)
        return _call_gemini(prompt, None, images) if images else _call_gemini(prompt, None)
    raise last or RuntimeError("nvidia unavailable")


def _call_anthropic(prompt: str, model: str | None) -> str:
    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model or "claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def _call_openai(prompt: str, model: str | None) -> str:
    from openai import OpenAI

    client = OpenAI()
    r = client.chat.completions.create(
        model=model or "gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return r.choices[0].message.content or ""


def _parse_json_array(text: str) -> list[dict]:
    text = text.strip()
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise ValueError("no JSON array in LLM output")
    return json.loads(m.group(0))


_SENTENCE_END = re.compile(r"[.!?][\"')\]]*$")


def _snap(t: float, segments: list[Segment], to_end: bool) -> float:
    """Snap a timestamp to a nearby segment boundary so clips start/end on sentences. Whisper
    segments break mid-sentence too, so a boundary where a sentence really ends (punctuation)
    within 4s wins over a nearer one that does not."""
    best = t
    bd = 1e9
    for i, s in enumerate(segments):
        b = s.end if to_end else s.start
        d = abs(b - t)
        closes = s if to_end else (segments[i - 1] if i else None)
        if closes is None or _SENTENCE_END.search(closes.text.strip()):
            d -= 4  # a real sentence boundary beats a closer mid-sentence one
        if d < bd:
            bd, best = d, b
    return best if bd <= 4 else t


def rank_llm(segments: list[Segment], cfg: Config, creator: str, provider: str) -> list[Moment]:
    prompt = PROMPT.format(
        creator=creator,
        n=cfg.clips.count,
        rules=slop.RULES,
        min_len=int(cfg.clips.min_len_s),
        max_len=int(cfg.clips.max_len_s),
        transcript=_transcript_text(segments),
    )
    call = {
        "nvidia": _call_nvidia,
        "gemini": _call_gemini,
        "anthropic": _call_anthropic,
        "openai": _call_openai,
    }[provider]
    raw = call(prompt, cfg.llm_model)
    items = _parse_json_array(raw)
    moments = []
    for it in items:
        start = _snap(float(it["start"]), segments, to_end=False)
        end = _snap(float(it["end"]), segments, to_end=True)
        dur = end - start
        if dur < cfg.clips.min_len_s * 0.7 or dur > cfg.clips.max_len_s * 1.3:
            continue
        end = min(end, start + cfg.clips.max_len_s)
        narration = str(it.get("narration", "")).strip()[:240]
        take = str(it.get("take", "")).strip()[:320] if cfg.narration.take else ""
        for what, line in (("narration", narration), ("take", take)):
            if slop.filler(line):
                log.info("%s dropped (filler: %s): %r", what, ", ".join(slop.filler(line)), line)
        moments.append(
            Moment(
                start,
                end,
                str(it.get("hook", ""))[:80],
                str(it.get("reason", "")),
                float(it.get("score", 5)),
                narration=narration if slop.clean(narration) else "",
                take=take if slop.clean(take) else "",
            )
        )
    return _dedupe(sorted(moments, key=lambda m: -m.score), cfg.clips.count)


def rank(segments: list[Segment], cfg: Config, creator: str) -> list[Moment]:
    cands = _windows(segments, cfg)
    if not cands:
        return []
    provider = _pick_provider(cfg)
    if provider != "heuristic":
        try:
            picked = rank_llm(segments, cfg, creator, provider)
            if picked:
                log.info("ranked %d moments with %s", len(picked), provider)
                return picked
            log.info("%s found no moment worth posting", provider)
            return []
        except Exception as e:
            if not cfg.clips.heuristic_fallback:
                raise Unavailable(f"{provider} ranking failed: {e}") from e
            log.warning("%s ranking failed (%s); falling back to heuristic", provider, e)
    return rank_heuristic(cands, cfg)
