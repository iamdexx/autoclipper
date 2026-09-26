"""Sound design for Top-10 videos: a rank-reveal whoosh/hit and a continuous music bed.

Everything is synthesised with ffmpeg so the repo ships no audio files; drop your own tracks in
`top10.music_dir` to use real music instead of the built-in pulse.
"""

from __future__ import annotations

import logging
import random
import subprocess
from pathlib import Path

from .config import Config

log = logging.getLogger(__name__)

_AUDIO_EXT = {".mp3", ".m4a", ".wav", ".aac", ".ogg", ".flac"}


def _ffmpeg(*args: str) -> None:
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", *args])


def whoosh(out: Path) -> Path:
    """0.5s: a filtered-noise sweep into a short sub 'hit' - the sound of a rank appearing."""
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    graph = (
        "anoisesrc=color=pink:amplitude=0.45:d=0.45,highpass=f=900,"
        "afade=t=in:d=0.15,afade=t=out:st=0.25:d=0.2[sw];"
        "sine=f=55:d=0.35,afade=t=out:d=0.35,volume=1.6[hit];"
        "[hit]adelay=280|280[hitd];"
        "[sw][hitd]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0,"
        "aformat=channel_layouts=stereo,volume=6,alimiter=limit=0.85"
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


def music_bed(dur: float, cfg: Config, out: Path) -> Path | None:
    """A bed exactly `dur` long: a random track from top10.music_dir (looped/cut, faded out) or
    the synth pulse. None if music is disabled."""
    t = cfg.top10
    if not t.music:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    tracks = []
    if t.music_dir:
        d = Path(t.music_dir)
        tracks = sorted(p for p in d.glob("*") if p.suffix.lower() in _AUDIO_EXT)
        if not tracks:
            log.warning("music_dir %s has no audio files; using the synth bed", d)
    if not tracks:
        return _synth_bed(dur, out)
    track = random.choice(tracks)
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
