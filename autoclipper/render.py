"""Cut a moment out of the source, reframe to 1080x1920 and burn word-highlight captions."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

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


def build_ass(words: list[Word], hook: str, dur: float, cfg: Config) -> str:
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
        lines.append(f"Dialogue: 0,{_ts(0)},{_ts(min(dur, 4.0))},Hook,,0,0,0,,{_esc(hook.upper())}")
    n = c.words_per_line
    for i in range(0, len(words), n):
        group = words[i : i + n]
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
    ass_path = out.with_suffix(".ass")
    ass_path.write_text(
        build_ass(words_in(segments, moment.start, moment.end), moment.hook, dur, cfg)
    )

    # Vertical reframe: blurred, zoomed copy of the frame as background, sharp centre crop on top.
    vf = (
        f"[0:v]split=2[bg][fg];"
        f"[bg]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=20:5[bgb];"
        f"[fg]scale={W}:-2:force_original_aspect_ratio=decrease[fgs];"
        f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,"
        f"ass={ass_path.as_posix()}[v]"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{moment.start:.3f}",
        "-t",
        f"{dur:.3f}",
        "-i",
        str(source),
        "-filter_complex",
        vf,
        "-map",
        "[v]",
        "-map",
        "0:a?",
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
