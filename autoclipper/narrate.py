"""Spoken hook intro for each clip.

The ranker's `hook` becomes a 1-2 sentence narrator line, voiced by a natural TTS voice and
mixed over the first seconds of the clip with the stream audio ducked underneath.

Providers (config `narration.provider`, `auto` picks the first available):
  elevenlabs : ELEVENLABS_API_KEY  (most natural; free tier 10k chars/month)
  gemini     : GEMINI_API_KEY      (Gemini TTS, prebuilt voices like Puck / Kore / Charon)
  openai     : OPENAI_API_KEY      (tts-1-hd, voices alloy/echo/nova/...)
"""

from __future__ import annotations

import base64
import logging
import os
import subprocess
from pathlib import Path

import requests

from .config import Config

log = logging.getLogger(__name__)

GEMINI_TTS_MODEL = "gemini-3.8-flash-tts"


def _gemini_key() -> str | None:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "Geminikey"):
        if os.getenv(name):
            return os.getenv(name)
    return None


def pick_provider(cfg: Config) -> str | None:
    p = cfg.narration.provider
    if p == "off":
        return None
    if p != "auto":
        return p
    if os.getenv("ELEVENLABS_API_KEY"):
        return "elevenlabs"
    if _gemini_key():
        return "gemini"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return None


def tts_gemini(text: str, voice: str, style: str, out: Path) -> Path:
    prompt = f"{style}: {text}" if style else text
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
    subprocess.check_call([*cmd, "-c:a", "aac", "-b:a", "128k", str(out)])
    raw.unlink()
    return out


def tts_elevenlabs(text: str, voice: str, out: Path) -> Path:
    r = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice}",
        headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"], "Accept": "audio/mpeg"},
        json={
            "text": text,
            "model_id": os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2"),
            "voice_settings": {"stability": 0.45, "similarity_boost": 0.8, "style": 0.35},
        },
        timeout=120,
    )
    r.raise_for_status()
    mp3 = out.with_suffix(".mp3")
    mp3.write_bytes(r.content)
    subprocess.check_call(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3), "-c:a", "aac", str(out)]
    )
    mp3.unlink()
    return out


def tts_openai(text: str, voice: str, out: Path) -> Path:
    from openai import OpenAI

    resp = OpenAI().audio.speech.create(
        model="tts-1-hd", voice=voice, input=text, response_format="aac"
    )
    out.write_bytes(resp.content)
    return out


def synthesize(text: str, cfg: Config, out: Path) -> Path | None:
    provider = pick_provider(cfg)
    if not provider or not text.strip():
        return None
    n = cfg.narration
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        if provider == "elevenlabs":
            return tts_elevenlabs(text, n.elevenlabs_voice, out)
        if provider == "gemini":
            return tts_gemini(text, n.gemini_voice, n.style, out)
        if provider == "openai":
            return tts_openai(text, n.openai_voice, out)
    except Exception as e:
        log.warning("narration with %s failed: %s", provider, e)
    return None


def duration(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]
    )
    return float(out.decode().strip())
