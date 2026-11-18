"""Spoken hook intro for each clip.

The ranker's `hook` becomes a 1-2 sentence narrator line, voiced by a natural TTS voice and
mixed over the first seconds of the clip with the stream audio ducked underneath.

Providers (config `narration.provider`; `auto` tries them in this order and moves on to the
next one when a call fails, e.g. the endpoint being down or a monthly quota running out):
  nvidia     : NVIDIA_API_KEY      (Magpie TTS on build.nvidia.com, free; a different voice from
               narration.nvidia_voices for each video, the same one throughout a video)
  edge       : no key at all - the Microsoft neural voices (Andrew, Brian, Guy, ...) through the
               endpoint Edge's read-aloud uses (unofficial, hence the fallbacks below)
  azure      : AZURE_SPEECH_KEY + AZURE_SPEECH_REGION (same voices, official; 500k chars/month free)
  google     : GOOGLE_TTS_API_KEY  (Cloud Text-to-Speech Chirp 3 HD; 1M chars/month free)
  elevenlabs : ELEVENLABS_API_KEY  (most natural; free tier 10k chars/month)
  gemini     : GEMINI_API_KEY      (Gemini TTS, prebuilt voices like Puck / Kore / Charon)
  openai     : OPENAI_API_KEY      (tts-1-hd, voices alloy/echo/nova/...)

Word timings for the captions come straight from ElevenLabs and Edge; for every other provider
the narration is run through Whisper and aligned to the script, so the highlight follows the
voice the same way whichever provider spoke.
"""

from __future__ import annotations

import base64
import difflib
import logging
import os
import re
import subprocess
import time
import zlib
from pathlib import Path

import requests

from .config import Config
from .ingest import Word

log = logging.getLogger(__name__)

PROVIDERS = ("nvidia", "edge", "azure", "google", "elevenlabs", "gemini", "openai")

# lower stability + higher style = more expressive, "hyped" delivery (still Liam)
_VOICE_SETTINGS = {
    "stability": 0.3,
    "similarity_boost": 0.8,
    "style": 0.6,
    "use_speaker_boost": True,
}

GEMINI_TTS_MODEL = "gemini-3.8-flash-tts"


def _gemini_key() -> str | None:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "Geminikey"):
        if os.getenv(name):
            return os.getenv(name)
    return None


def _elevenlabs_key() -> str | None:
    for name in ("ELEVENLABS_API_KEY", "elevenlabskey"):
        if os.getenv(name):
            return os.getenv(name)
    return None


def _google_key() -> str | None:
    for name in ("GOOGLE_TTS_API_KEY", "GOOGLE_CLOUD_API_KEY"):
        if os.getenv(name):
            return os.getenv(name)
    return None


def _azure_key() -> str | None:
    return os.getenv("AZURE_SPEECH_KEY") or None


def _azure_region() -> str:
    return os.getenv("AZURE_SPEECH_REGION", "eastus")


_KEYS = {
    "edge": lambda: True,
    "azure": _azure_key,
    "google": _google_key,
    "elevenlabs": _elevenlabs_key,
    "gemini": _gemini_key,
    "nvidia": lambda: os.getenv("NVIDIA_API_KEY"),
    "openai": lambda: os.getenv("OPENAI_API_KEY"),
}


def providers(cfg: Config) -> list[str]:
    """Providers to try, in order: the configured one, or every one with a key for `auto`."""
    p = cfg.narration.provider
    if p == "off":
        return []
    if p != "auto":
        return [p]
    return [name for name in PROVIDERS if _KEYS[name]()]


def pick_provider(cfg: Config) -> str | None:
    found = providers(cfg)
    return found[0] if found else None


def tts_azure(text: str, voice: str, rate: float, style: str, out: Path) -> Path:
    """Azure Speech neural voice (`voice` is the short name, e.g. en-US-GuyNeural).
    `style` (excited, cheerful, ...) is applied when the voice supports it; ignored otherwise."""
    body = _esc_xml(text)
    if rate != 1.0:
        body = f"<prosody rate='{rate:.2f}'>{body}</prosody>"
    if style:
        body = f"<mstts:express-as style='{style}'>{body}</mstts:express-as>"
    ssml = (
        "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' "
        "xmlns:mstts='https://www.w3.org/2001/mstts' xml:lang='en-US'>"
        f"<voice name='{voice}'>{body}</voice></speak>"
    )
    r = requests.post(
        f"https://{_azure_region()}.tts.speech.microsoft.com/cognitiveservices/v1",
        headers={
            "Ocp-Apim-Subscription-Key": _azure_key(),
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "audio-24khz-96kbitrate-mono-mp3",
            "User-Agent": "autoclipper",
        },
        data=ssml.encode(),
        timeout=120,
    )
    r.raise_for_status()
    mp3 = out.with_suffix(".mp3")
    mp3.write_bytes(r.content)
    return _mp3_to_aac(mp3, out)


def _edge_stream(text: str, voice: str, rate: float, out: Path) -> tuple[Path, list[Word]]:
    import asyncio

    import edge_tts

    async def run() -> list[Word]:
        heard: list[Word] = []
        com = edge_tts.Communicate(
            text, voice, rate=f"{(rate - 1) * 100:+.0f}%", boundary="WordBoundary"
        )
        with mp3.open("wb") as f:
            async for chunk in com.stream():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":  # offsets in 100ns ticks
                    s = chunk["offset"] / 1e7
                    heard.append(Word(s, s + chunk["duration"] / 1e7, chunk["text"]))
        return heard

    mp3 = out.with_suffix(".mp3")
    heard = asyncio.run(run())
    if not mp3.stat().st_size:
        raise RuntimeError("edge tts returned no audio")
    _mp3_to_aac(mp3, out)
    return out, heard


def tts_edge(text: str, voice: str, rate: float, out: Path) -> Path:
    """Microsoft neural voice via the Edge read-aloud endpoint (no account or key)."""
    return _edge_stream(text, voice, rate, out)[0]


def tts_edge_timed(text: str, voice: str, rate: float, out: Path) -> tuple[Path, list[Word]]:
    """Like tts_edge, plus word timings from the stream's word boundaries."""
    voice_path, heard = _edge_stream(text, voice, rate, out)
    return voice_path, align_words(text, heard, duration(voice_path))


def _esc_xml(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tts_google(text: str, voice: str, rate: float, out: Path) -> Path:
    """Cloud Text-to-Speech, Chirp 3 HD voice (`voice` is the short name, e.g. Charon)."""
    r = requests.post(
        "https://texttospeech.googleapis.com/v1/text:synthesize",
        headers={"x-goog-api-key": _google_key()},
        json={
            "input": {"text": text},
            "voice": {"languageCode": "en-US", "name": f"en-US-Chirp3-HD-{voice}"},
            "audioConfig": {"audioEncoding": "MP3", "speakingRate": rate},
        },
        timeout=120,
    )
    r.raise_for_status()
    mp3 = out.with_suffix(".mp3")
    mp3.write_bytes(base64.b64decode(r.json()["audioContent"]))
    return _mp3_to_aac(mp3, out)


def tts_gemini(text: str, voice: str, out: Path) -> Path:
    # Gemini TTS reads any style/stage direction aloud, so only the hook itself is sent;
    # delivery comes from the voice choice and how the LLM phrases the line.
    prompt = text
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_TTS_MODEL}:generateContent",
        headers={"x-goog-api-key": _gemini_key()},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
            },
        },
        timeout=120,
    )
    r.raise_for_status()
    part = r.json()["candidates"][0]["content"]["parts"][0]["inlineData"]
    raw = out.with_suffix(".raw")
    raw.write_bytes(base64.b64decode(part["data"]))
    mime = part["mimeType"]
    if mime.startswith("audio/L16") or mime.startswith("audio/pcm"):
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            "24000",
            "-ac",
            "1",
            "-i",
            str(raw),
        ]
    else:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw)]
    subprocess.check_call([*cmd, "-af", TRIM_TAIL, "-c:a", "aac", "-b:a", "128k", str(out)])
    raw.unlink()
    return out


def tts_elevenlabs(text: str, voice: str, out: Path) -> Path:
    r = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice}",
        headers={"xi-api-key": _elevenlabs_key(), "Accept": "audio/mpeg"},
        json={
            "text": text,
            "model_id": os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2"),
            "voice_settings": _VOICE_SETTINGS,
        },
        timeout=120,
    )
    r.raise_for_status()
    mp3 = out.with_suffix(".mp3")
    mp3.write_bytes(r.content)
    return _mp3_to_aac(mp3, out)


# TTS engines pad the end of the speech with up to ~1s of silence; keep 0.15s of it so the
# narration ends when the voice does and the gap to whatever follows stays a beat, not dead air
TRIM_TAIL = (
    "areverse,silenceremove=start_periods=1:start_silence=0.15:start_threshold=-45dB,areverse"
)


def _mp3_to_aac(mp3: Path, out: Path) -> Path:
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(mp3),
            "-af",
            TRIM_TAIL,
            "-c:a",
            "aac",
            str(out),
        ]
    )
    mp3.unlink()
    return out


def tts_elevenlabs_timed(text: str, voice: str, out: Path) -> tuple[Path, list[Word]]:
    """Like tts_elevenlabs but also returns per-word timings (for captions on narrated videos)."""
    r = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice}/with-timestamps",
        headers={"xi-api-key": _elevenlabs_key()},
        json={
            "text": text,
            "model_id": os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2"),
            "voice_settings": _VOICE_SETTINGS,
        },
        timeout=120,
    )
    r.raise_for_status()
    j = r.json()
    mp3 = out.with_suffix(".mp3")
    mp3.write_bytes(base64.b64decode(j["audio_base64"]))
    _mp3_to_aac(mp3, out)
    al = j["alignment"]
    words: list[Word] = []
    cur = ""
    ws = prev_end = 0.0
    for ch, s, e in zip(
        al["characters"],
        al["character_start_times_seconds"],
        al["character_end_times_seconds"],
        strict=False,
    ):
        if ch.isspace():
            if cur:
                words.append(Word(ws, prev_end, cur))
                cur = ""
            continue
        if not cur:
            ws = s
        cur += ch
        prev_end = e
    if cur:
        words.append(Word(ws, prev_end, cur))
    return out, words


def even_words(text: str, dur: float) -> list[Word]:
    """Fallback word timing when the TTS provider gives none: spread words evenly."""
    toks = text.split()
    if not toks:
        return []
    step = dur / len(toks)
    return [Word(i * step, (i + 1) * step, t) for i, t in enumerate(toks)]


def _norm(tok: str) -> str:
    return re.sub(r"[^a-z0-9]", "", tok.lower())


def align_words(text: str, heard: list[Word], dur: float) -> list[Word]:
    """Give every script word a timing from what Whisper heard in the narration: matched words
    take the heard timing, the rest are spread evenly between their timed neighbours."""
    toks = text.split()
    if not toks:
        return []
    if not heard:
        return even_words(text, dur)
    sm = difflib.SequenceMatcher(None, [_norm(t) for t in toks], [_norm(w.text) for w in heard])
    timed: list[tuple[float, float] | None] = [None] * len(toks)
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            timed[a + k] = (heard[b + k].start, heard[b + k].end)
    if not any(timed):
        return even_words(text, dur)
    words: list[Word] = []
    i = 0
    while i < len(toks):
        if timed[i]:
            words.append(Word(timed[i][0], timed[i][1], toks[i]))
            i += 1
            continue
        j = i
        while j < len(toks) and not timed[j]:
            j += 1
        lo = words[-1].end if words else 0.0
        hi = timed[j][0] if j < len(toks) else max(dur, lo + 0.3 * (j - i))
        step = max(hi - lo, 0.05 * (j - i)) / (j - i)
        for k in range(i, j):
            words.append(Word(lo + (k - i) * step, lo + (k - i + 1) * step, toks[k]))
        i = j
    return words


def _heard(voice: Path, cfg: Config) -> list[Word]:
    from . import ingest

    try:
        segs = ingest.transcribe(voice, cfg, voice.with_suffix(".words.json"))
    except Exception as e:
        log.warning("could not align narration with whisper (%s); spreading words evenly", e)
        return []
    return [w for s in segs for w in s.words]


def synthesize_timed(text: str, cfg: Config, out: Path) -> tuple[Path, list[Word]] | None:
    """Narration plus word timings: exact from ElevenLabs, Whisper-aligned for other providers."""
    if not text.strip():
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    for provider in providers(cfg):
        try:
            n = cfg.narration
            if provider == "elevenlabs":
                return tts_elevenlabs_timed(text, n.elevenlabs_voice, out)
            if provider == "edge":
                return tts_edge_timed(text, n.edge_voice, n.edge_rate, out)
            voice = _tts(provider, text, cfg, out)
        except Exception as e:
            log.warning("narration with %s failed: %s", provider, e)
            continue
        out.with_suffix(".words.json").unlink(missing_ok=True)
        return voice, align_words(text, _heard(voice, cfg), duration(voice))
    return None


NVIDIA_TTS_URL = "https://877104f7-e885-42b9-8de8-f6e4c6303969.invocation.api.nvcf.nvidia.com"


def nvidia_voice(cfg: Config, out: Path) -> str:
    """The voice for the video `out` belongs to: every line of one video gets the same voice,
    different videos get different ones."""
    voices = cfg.narration.nvidia_voices or ["Aria"]
    video = out.parent.parent.name if out.parent.name == "segments" else out.name.split(".")[0]
    name = voices[zlib.crc32(video.encode()) % len(voices)]
    return name if "." in name else f"Magpie-Multilingual.EN-US.{name}"


def tts_nvidia(text: str, voice: str, out: Path) -> Path:
    r = None
    for attempt in range(3):
        r = requests.post(
            f"{NVIDIA_TTS_URL}/v1/audio/synthesize",
            headers={"Authorization": f"Bearer {os.getenv('NVIDIA_API_KEY')}"},
            files={
                "text": (None, text),
                "language": (None, "en-US"),
                "voice": (None, voice),
                "encoding": (None, "LINEAR_PCM"),
                "sample_rate_hz": (None, "44100"),
            },
            timeout=120,
        )
        if r.status_code not in (429, 500, 502, 503, 504):
            break
        time.sleep(5 * (attempt + 1))
    r.raise_for_status()
    if r.content[:4] != b"RIFF":
        raise RuntimeError(f"nvidia tts returned no audio: {r.content[:120]!r}")
    wav = out.with_suffix(".wav")
    wav.write_bytes(r.content)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-c:a", "aac", "-b:a", "160k"]
        + [str(out)],
        check=True,
    )
    wav.unlink(missing_ok=True)
    return out


def tts_openai(text: str, voice: str, out: Path) -> Path:
    from openai import OpenAI

    resp = OpenAI().audio.speech.create(
        model="tts-1-hd", voice=voice, input=text, response_format="aac"
    )
    out.write_bytes(resp.content)
    return out


def _tts(provider: str, text: str, cfg: Config, out: Path) -> Path:
    n = cfg.narration
    if provider == "azure":
        return tts_azure(text, n.azure_voice, n.azure_rate, n.azure_style, out)
    if provider == "google":
        return tts_google(text, n.google_voice, n.google_rate, out)
    if provider == "elevenlabs":
        return tts_elevenlabs(text, n.elevenlabs_voice, out)
    if provider == "gemini":
        return tts_gemini(text, n.gemini_voice, out)
    if provider == "nvidia":
        return tts_nvidia(text, nvidia_voice(cfg, out), out)
    if provider == "openai":
        return tts_openai(text, n.openai_voice, out)
    if provider == "edge":
        return tts_edge(text, n.edge_voice, n.edge_rate, out)
    raise ValueError(f"unknown narration provider {provider!r}")


def synthesize(text: str, cfg: Config, out: Path) -> Path | None:
    if not text.strip():
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    for provider in providers(cfg):
        try:
            return _tts(provider, text, cfg, out)
        except Exception as e:
            log.warning("narration with %s failed: %s", provider, e)
    return None


def duration(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]
    )
    return float(out.decode().strip())
