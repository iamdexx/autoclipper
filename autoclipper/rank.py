"""Pick the moments in a transcript most likely to perform as short-form clips.

Provider order for `llm_provider: auto`: Gemini -> Anthropic -> OpenAI -> heuristic,
based on which API key is present in the environment.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass

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
            narration=f"{_hook_from_text(t)}.",
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

PROMPT = """You are a professional short-form clipper for TikTok / Reels / Shorts.
Below is a timestamped transcript of a livestream by {creator}. Pick the {n} moments with the
highest viral potential: the action or the line that matters starts within the first 2 seconds
(no run-up, no context), emotional spikes, hot takes, drama, laughter, money/numbers,
controversy, or a clear story with payoff. Each clip must be between {min_len} and {max_len}
seconds, must start right where it gets good (a natural sentence beginning) and end on a payoff.
Return ONLY a JSON array, no prose, each item:
{{"start": <seconds float>, "end": <seconds float>, "hook": "<punchy 3-6 word on-screen title>",
  "narration": "<ONE short punchy sentence (max 9 words) an excited narrator says before the clip so viewers stay, e.g. 'Wait for what he says about his boss.' Plain spoken words only: no stage directions, hashtags or emojis.>",
  "reason": "<one sentence>", "score": <0-10>}}

TRANSCRIPT:
{transcript}
"""


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


def _pick_provider(cfg: Config) -> str:
    p = cfg.llm_provider
    if p != "auto":
        return p
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


def _call_gemini(prompt: str, model: str | None) -> str:
    import requests

    key = _gemini_key()
    models = [model] if model else GEMINI_MODELS
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
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
                continue
            r.raise_for_status()
            parts = r.json()["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        time.sleep(5 * (attempt + 1))
    raise last or RuntimeError("gemini unavailable")


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


def _snap(t: float, segments: list[Segment], to_end: bool) -> float:
    """Snap a timestamp to the nearest segment boundary so clips start/end on sentences."""
    best = t
    bd = 1e9
    for s in segments:
        b = s.end if to_end else s.start
        d = abs(b - t)
        if d < bd:
            bd, best = d, b
    return best if bd <= 4 else t


def rank_llm(segments: list[Segment], cfg: Config, creator: str, provider: str) -> list[Moment]:
    prompt = PROMPT.format(
        creator=creator,
        n=cfg.clips.count,
        min_len=int(cfg.clips.min_len_s),
        max_len=int(cfg.clips.max_len_s),
        transcript=_transcript_text(segments),
    )
    call = {"gemini": _call_gemini, "anthropic": _call_anthropic, "openai": _call_openai}[provider]
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
        moments.append(
            Moment(
                start,
                end,
                str(it.get("hook", ""))[:80],
                str(it.get("reason", "")),
                float(it.get("score", 5)),
                narration=str(it.get("narration", ""))[:240],
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
            log.warning("%s returned no usable moments; falling back to heuristic", provider)
        except Exception as e:
            log.warning("%s ranking failed (%s); falling back to heuristic", provider, e)
    return rank_heuristic(cands, cfg)
