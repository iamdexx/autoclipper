"""Sound design for Top-10 videos: a rank-reveal whoosh/hit and a continuous music bed.

The whoosh is synthesised with ffmpeg. Music comes from the ElevenLabs Music API by default
(one continuous instrumental generated per video, cached), or your own tracks in
`top10.music_dir`; the ffmpeg synth pulse is an explicit opt-in (`top10.music: synth`).
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import subprocess
from pathlib import Path

import requests

from . import narrate
from .config import Config

log = logging.getLogger(__name__)

_AUDIO_EXT = {".mp3", ".m4a", ".wav", ".aac", ".ogg", ".flac"}


def _ffmpeg(*args: str) -> None:
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", *args])


def whoosh(out: Path) -> Path:
    """0.4s: a soft, band-limited air sweep with a faint low thump - the sound of a rank
    appearing. Deliberately muted so it reads as a transition, not a hit."""
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    graph = (
        "anoisesrc=color=brown:amplitude=0.5:d=0.4,highpass=f=300,lowpass=f=2200,"
        "afade=t=in:d=0.18:curve=qsin,afade=t=out:st=0.2:d=0.2:curve=qsin[sw];"
        "sine=f=60:d=0.3,afade=t=in:d=0.03,afade=t=out:st=0.05:d=0.25,volume=0.5[hit];"
        "[hit]adelay=200|200[hitd];"
        "[sw][hitd]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0,"
        "aformat=channel_layouts=stereo,volume=2.2,alimiter=limit=0.6"
    )
    _ffmpeg("-filter_complex", graph, "-ar", "48000", "-c:a", "aac", "-b:a", "128k", str(out))
    return out


def _synth_bed(dur: float, out: Path) -> Path:
    """Minimal 128bpm pulse: soft kick + sub + hi-hat ticks. Deliberately sparse so it sits
    under speech without fighting it."""
    bar = 60 / 128  # one beat
    graph = (
        # kick: pitched sine burst every beat
        f"sine=f=52:d={dur:.3f},"
        f"volume='0.9*exp(-mod(t,{bar:.5f})*18)':eval=frame[kick];"
        # sub bass drone, gently pumping against the kick
        f"sine=f=41.2:d={dur:.3f},"
        f"volume='0.28*(0.6+0.4*(1-exp(-mod(t,{bar:.5f})*10)))':eval=frame[sub];"
        # hats: filtered noise ticks on the off-beats
        f"anoisesrc=color=white:amplitude=0.25:d={dur:.3f},highpass=f=7000,"
        f"volume='0.35*exp(-mod(t+{bar / 2:.5f},{bar:.5f})*40)':eval=frame[hat];"
        "[kick][sub][hat]amix=inputs=3:duration=first:dropout_transition=0:normalize=0,"
        "aformat=channel_layouts=stereo,volume=4,alimiter=limit=0.8,"
        f"afade=t=in:d=0.4,afade=t=out:st={max(0.0, dur - 1.8):.3f}:d=1.8"
    )
    _ffmpeg("-filter_complex", graph, "-ar", "48000", "-c:a", "aac", "-b:a", "160k", str(out))
    return out


def generate_music(prompt: str, secs: float, cfg: Config, cache: Path) -> Path | None:
    """One instrumental track from the ElevenLabs Music API (same key as the narrator; needs the
    `music_generation` permission). Cached per prompt+length so repeat videos reuse it."""
    key = narrate._elevenlabs_key()
    if not key:
        return None
    ms = int(math.ceil(secs / 15) * 15_000)  # round up to 15s buckets for the cache
    out = cache / f"{hashlib.sha1(f'{prompt}|{ms}'.encode()).hexdigest()[:12]}.mp3"
    if out.exists():
        return out
    cache.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.post(
            "https://api.elevenlabs.io/v1/music",
            headers={"xi-api-key": key},
            json={"prompt": prompt, "music_length_ms": ms},
            timeout=300,
        )
        r.raise_for_status()
        out.write_bytes(r.content)
        return out
    except requests.HTTPError as e:
        log.warning("elevenlabs music generation failed: %s %s", e, e.response.text[:200])
        return None
    except requests.RequestException as e:
        log.warning("elevenlabs music generation failed: %s", e)
        return None


def music_enabled(cfg: Config) -> bool:
    return str(cfg.top10.music).lower() not in ("off", "false", "none", "")


def music_bed(dur: float, cfg: Config, out: Path, prompt: str = "") -> Path | None:
    """A bed exactly `dur` long, faded out at the end. Source, in order: a generated track
    (`top10.music: elevenlabs`), a random file from `top10.music_dir`, the built-in synth pulse
    (`top10.music: synth`). None if music is off or no source is available."""
    t = cfg.top10
    mode = str(t.music).lower()
    if not music_enabled(cfg):
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    track: Path | None = None
    if mode in ("elevenlabs", "auto", "true"):
        track = generate_music(
            prompt or t.music_prompt, dur + 2, cfg, cfg.workdir / "assets" / "music"
        )
    if track is None and t.music_dir:
        d = Path(t.music_dir)
        tracks = sorted(p for p in d.glob("*") if p.suffix.lower() in _AUDIO_EXT)
        if tracks:
            track = random.choice(tracks)
        else:
            log.warning("music_dir %s has no audio files", d)
    if track is None:
        if mode == "synth":
            return _synth_bed(dur, out)
        log.warning("no music source available; rendering without a bed")
        return None
    _ffmpeg(
        *("-stream_loop", "-1", "-i", str(track), "-t", f"{dur:.3f}"),
        *(
            "-af",
            f"afade=t=in:d=0.5,afade=t=out:st={max(0.0, dur - 2.0):.3f}:d=2,"
            "aformat=channel_layouts=stereo,aresample=48000",
        ),  # fmt: skip
        *("-c:a", "aac", "-b:a", "160k", str(out)),
    )
    return out


def mix_under(video: Path, bed: Path, db: float, out: Path) -> Path:
    """Lay `bed` under the video's existing audio at `db` (relative), video stream copied."""
    gain = 10 ** (db / 20)
    _ffmpeg(
        *("-i", str(video), "-i", str(bed)),
        *(
            "-filter_complex",
            f"[1:a]volume={gain:.4f}[m];[0:a][m]amix=inputs=2:duration=first:"
            "dropout_transition=0:normalize=0[a]",
        ),  # fmt: skip
        *("-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k"),
        *("-movflags", "+faststart", str(out)),
    )
    return out
