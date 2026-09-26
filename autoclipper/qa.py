"""Output quality gate: every finished video is inspected before it may be published.

Checks the file itself (streams, 9:16, duration, fps), the audio (level, clipping, dead air,
narration audible), the picture (no black / frozen stretches) and - by re-transcribing the
finished video with whisper - that what the narrator actually says matches the script.
A failing video is written up in qa.json next to it and never published."""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import ingest
from .config import Config

log = logging.getLogger(__name__)

W, H = 1080, 1920


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


@dataclass
class Report:
    video: str
    ok: bool = True
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.checks.append(Check(name, ok, detail))
        self.ok = self.ok and ok
        (log.info if ok else log.warning)("qa %-11s %s  %s", name, "PASS" if ok else "FAIL", detail)


def _ffprobe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out)


def _filter_log(path: Path, af: str = "", vf: str = "") -> str:
    """Run detection filters over the whole file and return ffmpeg's stderr."""
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path)]
    if af:
        cmd += ["-af", af]
    else:
        cmd += ["-an"]
    if vf:
        cmd += ["-vf", vf]
    else:
        cmd += ["-vn"]
    cmd += ["-f", "null", "-"]
    return subprocess.run(cmd, capture_output=True, text=True).stderr


def _floats(pattern: str, text: str) -> list[float]:
    return [float(m) for m in re.findall(pattern, text)]


_STOP = {"the", "a", "an", "and", "of", "to", "in", "on", "is", "it", "this", "that", "with"}


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in _STOP and len(w) > 2}


def script_match(expected: str, heard: str) -> float:
    """Share of the script's content words that whisper heard in the finished video."""
    want = _tokens(expected)
    if not want:
        return 1.0
    return len(want & _tokens(heard)) / len(want)


def check(video: Path, cfg: Config, expect_speech: str = "", kind: str = "clip") -> Report:
    q = cfg.qa
    rep = Report(video=video.name)
    try:
        info = _ffprobe(video)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        rep.add("container", False, f"ffprobe failed: {e}")
        return rep
    streams = info.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    dur = float(info.get("format", {}).get("duration", 0) or 0)
    rep.add("streams", bool(v and a), f"video={'yes' if v else 'no'} audio={'yes' if a else 'no'}")
    if not v or not a:
        return rep
    w, h = int(v.get("width", 0)), int(v.get("height", 0))
    rep.add("aspect", (w, h) == (W, H), f"{w}x{h}")
    lo, hi = (q.top10_min_s, q.top10_max_s) if kind == "top10" else (q.clip_min_s, q.clip_max_s)
    rep.add("duration", lo <= dur <= hi, f"{dur:.1f}s (allowed {lo:.0f}-{hi:.0f}s)")
    adur = float(a.get("duration", dur) or dur)
    rep.add("av_sync", abs(adur - dur) <= 0.5, f"audio {adur:.2f}s vs video {dur:.2f}s")

    alog = _filter_log(
        video,
        af=f"volumedetect,silencedetect=n={q.silence_db}dB:d={q.max_silence_s}",
    )
    mean = _floats(r"mean_volume: (-?[\d.]+) dB", alog)
    peak = _floats(r"max_volume: (-?[\d.]+) dB", alog)
    if mean and peak:
        rep.add(
            "loudness",
            q.min_mean_db <= mean[0] <= q.max_mean_db,
            f"mean {mean[0]:.1f} dB (allowed {q.min_mean_db:.0f}..{q.max_mean_db:.0f})",
        )
        rep.add("clipping", peak[0] <= q.max_peak_db, f"peak {peak[0]:.1f} dB")
    else:
        rep.add("loudness", False, "volumedetect produced no reading")
    starts = _floats(r"silence_start: (-?[\d.]+)", alog)
    gaps = _floats(r"silence_duration: ([\d.]+)", alog)
    if len(starts) > len(gaps):  # silence still running at EOF gets no duration line
        gaps.append(dur - max(0.0, starts[-1]))
    # a trailing fade-out is fine; anything inside the video is dead air
    inside = [g for g, s in zip(gaps, starts, strict=True) if s + g < dur - 0.3 or g > 2.0]
    rep.add(
        "dead_air",
        not inside,
        f"longest gap {max(inside):.1f}s" if inside else f"none over {q.max_silence_s:.1f}s",
    )

    vlog = _filter_log(
        video,
        vf=f"blackdetect=d={q.max_black_s}:pix_th=0.10,freezedetect=n=-60dB:d={q.max_freeze_s}",
    )
    black = _floats(r"black_duration:([\d.]+)", vlog)
    frozen = len(re.findall(r"freeze_start", vlog))
    rep.add("black", not black, f"{len(black)} black stretch(es)" if black else "none")
    rep.add("frozen", frozen == 0, f"{frozen} frozen stretch(es)" if frozen else "none")

    if expect_speech.strip() and q.check_speech:
        try:
            cache = video.with_suffix(".qa-transcript.json")
            if cache.exists() and cache.stat().st_mtime < video.stat().st_mtime:
                cache.unlink()  # transcript of a previous render under the same name
            segs = ingest.transcribe(video, cfg, cache)
            heard = " ".join(s.text for s in segs)
            score = script_match(expect_speech, heard)
            rep.add(
                "speech",
                score >= q.min_speech_match,
                f"{score:.0%} of the script's words heard (need {q.min_speech_match:.0%})",
            )
        except Exception as e:  # whisper missing / model download failed: don't pass silently
            rep.add("speech", not q.speech_required, f"could not transcribe output: {e}")
    return rep


def gate(video: Path, cfg: Config, expect_speech: str = "", kind: str = "clip") -> bool:
    """Run the checks, write qa.json next to the video, return whether it may be published."""
    if not cfg.qa.enabled:
        return True
    rep = check(video, cfg, expect_speech, kind)
    video.with_suffix(".qa.json").write_text(json.dumps(asdict(rep), indent=2))
    if not rep.ok:
        failed = ", ".join(c.name for c in rep.checks if not c.ok)
        log.error("qa FAILED for %s (%s) - not publishing", video.name, failed)
    return rep.ok
