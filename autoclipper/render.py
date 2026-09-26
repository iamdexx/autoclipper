"""Cut a moment out of the source, reframe to 1080x1920 and burn word-highlight captions."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from . import narrate
from .config import Config
from .ingest import Segment, Word
from .rank import Moment

log = logging.getLogger(__name__)

W, H = 1080, 1920


def _ts(t: float) -> str:
    t = max(t, 0)
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _esc(s: str) -> str:
    return s.replace("{", "(").replace("}", ")").replace("\\", "/")


def words_in(segments: list[Segment], start: float, end: float) -> list[Word]:
    return [
        Word(w.start - start, w.end - start, w.text)
        for s in segments
        for w in s.words
        if w.start >= start - 0.05 and w.end <= end + 0.05
    ]


def _wrap(text: str, max_chars: int) -> list[str]:
    """Greedy word wrap; never splits a word, never emits an empty line."""
    out: list[str] = []
    cur = ""
    for w in text.split():
        if cur and len(cur) + 1 + len(w) > max_chars:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        out.append(cur)
    return out or [text]


def _group_words(words: list[Word], max_words: int, max_chars: int) -> list[list[Word]]:
    """Chunk words so each caption line fits: at most max_words and max_chars per line.
    A sentence-ending word closes the line so lines don't straddle sentences."""
    groups: list[list[Word]] = []
    cur: list[Word] = []
    cur_len = 0
    for w in words:
        add = len(w.text) + (1 if cur else 0)
        if cur and (len(cur) >= max_words or cur_len + add > max_chars):
            groups.append(cur)
            cur, cur_len = [], 0
            add = len(w.text)
        cur.append(w)
        cur_len += add
        if w.text[-1:] in ".!?" and len(cur) > 1:
            groups.append(cur)
            cur, cur_len = [], 0
    if cur:
        groups.append(cur)
    return groups


def build_ass(words: list[Word], hook: str, dur: float, cfg: Config, hook_secs: float = 4.0) -> str:
    c = cfg.captions
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{c.font},{c.size},{c.base_color},{c.base_color},&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{c.outline},2,2,60,60,{c.margin_v},1
Style: Hook,{c.font},{int(c.size * 0.85)},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{c.outline},2,8,60,60,300,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    if hook:
        hook_text = "\\N".join(_esc(ln) for ln in _wrap(hook.upper(), c.hook_chars_per_line))
        lines.append(f"Dialogue: 0,{_ts(0)},{_ts(min(dur, hook_secs))},Hook,,0,0,0,,{hook_text}")
    for group in _group_words(words, c.words_per_line, c.chars_per_line):
        g_start = group[0].start
        g_end = max(group[-1].end, g_start + 0.3)
        # one dialogue line per word so the current word is highlighted
        for k, w in enumerate(group):
            ws = w.start if k else g_start
            we = group[k + 1].start if k + 1 < len(group) else g_end
            if we <= ws:
                continue
            parts = []
            for j, other in enumerate(group):
                t = _esc(other.text)
                parts.append(f"{{\\c{c.highlight_color}}}{t}{{\\c{c.base_color}}}" if j == k else t)
            lines.append(f"Dialogue: 0,{_ts(ws)},{_ts(we)},Cap,,0,0,0,,{' '.join(parts)}")
    return header + "\n".join(lines) + "\n"


def render_clip(
    source: Path, moment: Moment, segments: list[Segment], cfg: Config, out: Path
) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    dur = moment.end - moment.start

    voice = narrate.synthesize(moment.narration, cfg, out.with_suffix(".voice.m4a"))
    voice_len = 0.0
    tempo = 1.0
    if voice:
        voice_len = narrate.duration(voice)
        if voice_len > cfg.narration.max_len_s:
            tempo = min(1.3, voice_len / cfg.narration.max_len_s)
            voice_len /= tempo

    ass_path = out.with_suffix(".ass")
    ass_path.write_text(
        build_ass(
            words_in(segments, moment.start, moment.end),
            moment.hook,
            dur,
            cfg,
            hook_secs=max(4.0, voice_len + 0.5),
        )
    )

    # Vertical reframe: blurred, zoomed copy of the frame as background, sharp centre crop on top.
    vf = (
        f"[0:v]split=2[bg][fg];"
        f"[bg]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=20:5[bgb];"
        f"[fg]scale={W}:-2:force_original_aspect_ratio=decrease[fgs];"
        f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,"
        f"ass={ass_path.as_posix()}[v]"
    )
    inputs = ["-ss", f"{moment.start:.3f}", "-t", f"{dur:.3f}", "-i", str(source)]
    audio_map = ["-map", "0:a?"]
    if voice:
        duck = 10 ** (cfg.narration.duck_db / 20)
        lead = 0.25
        vf += (
            f";[0:a]volume={duck:.3f}:enable='lt(t,{voice_len + lead + 0.3:.2f})'[duck];"
            f"[1:a]atempo={tempo:.3f},adelay={int(lead * 1000)}|{int(lead * 1000)},"
            f"aresample=48000,apad[nar];"
            f"[duck][nar]amix=inputs=2:duration=first:dropout_transition=0,volume=2[a]"
        )
        inputs += ["-i", str(voice)]
        audio_map = ["-map", "[a]"]
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        *inputs,
        "-filter_complex",
        vf,
        "-map",
        "[v]",
        *audio_map,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "30",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
        str(out),
    ]
    log.info("rendering %s (%.1fs)", out.name, dur)
    subprocess.check_call(cmd)
    return out
