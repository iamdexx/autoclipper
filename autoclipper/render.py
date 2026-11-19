"""Cut a moment out of the source, reframe to 1080x1920 and burn word-highlight captions."""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

from . import narrate, reframe
from .config import Config
from .ingest import Segment, Word
from .rank import Moment

log = logging.getLogger(__name__)

W, H = 1080, 1920
FPS = 30


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


PAUSE_GAP = 0.5  # silence this long between two words starts a new caption line
HOOK_MIN_S = (
    3.0  # the on-screen hook stays at least this long, into the clip if the intro is shorter
)
# pop-in: scale 40% -> 115% -> 100% over the first third of a second
POP = "{\\fscx40\\fscy40\\t(0,220,\\fscx115\\fscy115)\\t(220,340,\\fscx100\\fscy100)}"
MIN_TAIL = 0.3  # a creator clip never ends closer than this to its last word
UNDUCK_S = 0.6  # how long the footage audio takes to come back up after the narrator
TAKE_LEAD_S = 0.4  # the footage ducks this long before the narrator's take starts
OUTRO_FADE_S = 0.5  # every video eases out over its last half second


def _group_words(words: list[Word], max_words: int, max_chars: int) -> list[list[Word]]:
    """Chunk words so each caption line fits: at most max_words and max_chars per line.
    A sentence-ending word or a pause closes the line so lines don't straddle sentences,
    silences, or a change of speaker."""
    groups: list[list[Word]] = []
    cur: list[Word] = []
    cur_len = 0
    for w in words:
        add = len(w.text) + (1 if cur else 0)
        if cur and (
            len(cur) >= max_words or cur_len + add > max_chars or w.start - cur[-1].end >= PAUSE_GAP
        ):
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


def build_ass(
    words: list[Word],
    hook: str,
    dur: float,
    cfg: Config,
    hook_secs: float = 4.0,
    badge: str = "",
    one_word: bool | None = None,
    events: list[str] | None = None,
) -> str:
    """`one_word`: every word is its own big caption, no highlight colouring (defaults to
    captions.style == "word" so every format looks the same).
    `events`: extra pre-built Dialogue lines (e.g. the ranking overlay)."""
    c = cfg.captions
    if one_word is None:
        one_word = c.style == "word"
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{c.font},{c.size},{c.base_color},{c.base_color},&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{c.outline},2,2,60,60,{c.margin_v},1
Style: Hook,{c.font},{int(c.size * 1.05)},{c.highlight_color},{c.highlight_color},&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{c.outline + 1},3,8,60,60,300,1
Style: Badge,{c.font},{int(c.size * 2.6)},{c.highlight_color},{c.highlight_color},&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{c.outline + 2},3,8,60,60,110,1
Style: Word,{c.font},{int(c.size * 1.45)},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{c.outline + 1},3,2,60,60,660,1
Style: Title,{c.font},{int(c.size * 1.35)},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{c.outline + 1},3,8,60,60,150,1
Style: Rank,{c.font},{int(c.size * 0.95)},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{c.outline},2,7,70,60,560,1
Style: Brand,{c.font},{int(c.size * 0.5)},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,1,9,60,50,170,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = list(events or [])
    if cfg.look.watermark:
        mark = "{\\alpha&H40&}" + _esc(cfg.look.watermark)
        lines.append(f"Dialogue: 0,{_ts(0)},{_ts(dur)},Brand,,0,0,0,,{mark}")
    if badge:
        lines.append(f"Dialogue: 0,{_ts(0)},{_ts(dur)},Badge,,0,0,0,,{POP}{_esc(badge)}")
    if hook:
        hook_text = "\\N".join(_esc(ln) for ln in _wrap(hook.upper(), c.hook_chars_per_line))
        end = _ts(min(dur, hook_secs))
        lines.append(f"Dialogue: 0,{_ts(0)},{end},Hook,,0,0,0,,{POP}{hook_text}")
    if one_word:
        for k, w in enumerate(words):
            # hold until the next word, but never through a long pause
            nxt = words[k + 1].start if k + 1 < len(words) else w.end + 0.6
            we = min(nxt, max(w.start + 0.3, w.end + 0.6))  # never overlaps the next word
            if we > w.start:
                txt = _esc(w.text.strip(".,!?;:").upper())
                lines.append(f"Dialogue: 0,{_ts(w.start)},{_ts(we)},Word,,0,0,0,,{txt}")
        return header + "\n".join(lines) + "\n"
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


def _has_audio(path: Path) -> bool:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
         "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
    )  # fmt: skip
    return bool(out.stdout.strip())


def _vf(
    ass_path: Path,
    src: str = "[0:v]",
    punch_at: list[float] | None = None,
    mode: str = "crop",
    focus: float | None = None,
) -> str:
    """Vertical reframe. `mode` crop: the picture is scaled to fill the 9:16 frame and cropped
    around `focus` (0..1 horizontal centre, default the middle); blur: the full picture sits in
    the middle over a blurred, zoomed copy of itself.
    `punch_at`: times (s) where the picture snaps in ~10% and settles back - a cut accent."""
    punch = ""
    if punch_at:
        terms = "+".join(
            f"exp(-(on-{round(t * FPS)})/5)*gte(on,{round(t * FPS)})" for t in punch_at
        )
        punch = (
            f"fps={FPS},zoompan=z='1+0.10*({terms})':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d=1:s={W}x{H}:fps={FPS},"
        )
    if mode == "crop":
        cx = 0.5 if focus is None else min(1.0, max(0.0, focus))
        # the crop window centred on the face, clamped to the scaled picture's edges
        x = f"max(0\\,min(iw-{W}\\,iw*{cx:.4f}-{W // 2}))"
        return (
            f"{src}scale={W}:{H}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H}:{x}:(ih-{H})/2,{punch}ass={ass_path.as_posix()}[v]"
        )
    return (
        f"{src}split=2[bg][fg];"
        f"[bg]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=20:5[bgb];"
        f"[fg]scale={W}:-2:force_original_aspect_ratio=decrease[fgs];"
        f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,{punch}"
        f"ass={ass_path.as_posix()}[v]"
    )


def _frame(cfg: Config, source: Path, start: float, dur: float) -> dict:
    """_vf keyword arguments for this clip: the configured mode and, when cropping, where the
    speaker is in the footage. A crop with no speaker found becomes "blur"."""
    mode = cfg.look.reframe
    focus = reframe.focus_x(source, start, dur) if mode == "crop" else None
    if mode == "crop" and focus is None:
        # nobody found to crop to (a speaker in profile, gameplay): the middle of the frame may
        # cut them off, so the whole picture is shown over a blurred copy instead
        mode = "blur"
    return {"mode": mode, "focus": focus}


def cut_plan(avail: float, dur: float, cuts: int) -> list[tuple[float, float]]:
    """Split `dur` seconds of screen time into `cuts` (start, length) windows spread across
    `avail` seconds of footage, so an entry shows two different moments instead of one."""
    cuts = max(1, cuts)
    if avail <= 0:
        return [(0.0, dur)]
    if dur >= avail or cuts == 1:
        return [(0.0, min(dur, avail))]
    piece = dur / cuts
    step = (avail - piece) / (cuts - 1)
    return [(round(i * step, 3), round(piece, 3)) for i in range(cuts)]


# every output lands at the same loudness (-16 LUFS, the short-form norm) whatever the source
NORM = "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000"

_ENC = [
    *("-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30"),
    *("-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2"),
    *("-movflags", "+faststart"),
]


def render_narrated(
    source: Path,
    cuts: list[tuple[float, float]],
    voice: Path | None,
    words: list[Word],
    title: str,
    badge: str,
    cfg: Config,
    out: Path,
    sfx: Path | None = None,
    duck_db: float | None = None,
    events: list[str] | None = None,
    one_word: bool | None = None,
    unduck_at: float | None = None,
) -> Path:
    """A segment fully voiced by the narrator (Top-10 entries): source audio stays ducked the
    whole time, captions come from the narration's word timings, badge = big rank number.
    `cuts` = (start, length) windows of `source` shown back to back, each with a zoom punch;
    `sfx` plays at the start (the rank reveal). `unduck_at` = when the narrator is done and the
    footage's own audio comes back up (the payoff clip playing out)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    dur = sum(ln for _, ln in cuts)
    ass_path = out.with_suffix(".ass")
    ass_path.write_text(
        build_ass(
            words, title, dur, cfg, hook_secs=dur, badge=badge, one_word=one_word, events=events
        )
    )
    src_audio = _has_audio(source)
    punch_at, t = [], 0.0
    for _, ln in cuts:
        punch_at.append(t)
        t += ln
    vf = ""
    for i, (s, ln) in enumerate(cuts):
        vf += f"[0:v]trim=start={s:.3f}:duration={ln:.3f},setpts=PTS-STARTPTS[c{i}];"
        if src_audio:
            vf += (
                f"[0:a]atrim=start={s:.3f}:duration={ln:.3f},asetpts=PTS-STARTPTS,"
                f"afade=t=in:d=0.06,afade=t=out:st={max(0.0, ln - 0.08):.3f}:d=0.08[ca{i}];"
            )
    vf += "".join(f"[c{i}]" for i in range(len(cuts))) + f"concat=n={len(cuts)}:v=1:a=0[src];"
    if src_audio:
        vf += "".join(f"[ca{i}]" for i in range(len(cuts)))
        vf += f"concat=n={len(cuts)}:v=0:a=1[srca];"
    vf += _vf(ass_path, "[src]", punch_at, **_frame(cfg, source, cuts[0][0], cuts[0][1]))
    inputs = ["-i", str(source)]
    duck = 10 ** ((cfg.narration.duck_db if duck_db is None else duck_db) / 20)
    pad = f"apad=whole_dur={dur:.3f},atrim=0:{dur:.3f}"
    tracks = []
    if src_audio:
        vol = f"volume={duck:.3f}"
        if unduck_at is not None and unduck_at < dur:
            up = f"{duck:.3f}+(1-{duck:.3f})*min(1,(t-{unduck_at:.3f})/{UNDUCK_S})"
            vol = f"volume='if(lt(t,{unduck_at:.3f}),{duck:.3f},{up})':eval=frame"
        vf += f";[srca]{vol},{pad}[duck]"
        tracks.append("[duck]")
    if voice:
        vf += f";[{len(inputs) // 2}:a]aresample=48000,{pad}[nar]"
        inputs += ["-i", str(voice)]
        tracks.append("[nar]")
    if sfx:
        vf += f";[{len(inputs) // 2}:a]aresample=48000,volume=0.35,{pad}[fx]"
        inputs += ["-i", str(sfx)]
        tracks.append("[fx]")
    # short fades at both ends so segment joins never click
    edge = f"afade=t=in:d=0.04,afade=t=out:st={max(0.0, dur - 0.06):.3f}:d=0.06"
    if len(tracks) > 1:
        vf += (
            f";{''.join(tracks)}amix=inputs={len(tracks)}:duration=shortest"
            f":dropout_transition=0:normalize=0,{NORM},{edge}[a]"
        )
    elif tracks:
        vf += f";{tracks[0]}{NORM},{edge}[a]"
    else:
        vf += f";anullsrc=r=48000:cl=stereo,atrim=0:{dur:.3f}[a]"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", vf]
    cmd += ["-map", "[v]", "-map", "[a]", *_ENC, str(out)]
    subprocess.check_call(cmd)
    return out


# ASS colours are &HAABBGGRR: blue, green, yellow, pink, white
PALETTE = ["&H00FF9A1F", "&H0050E03C", "&H0000D9FF", "&H00B36CFF", "&H00FFFFFF"]


def ranking_events(
    title: str, tease: str, labels: list[tuple[int, str]], revealed: int, dur: float
) -> list[str]:
    """The 'Ranking ...' overlay: a colour-per-word title banner with a small tease line under
    it, and a 1..N list down the left edge where every rank <= `revealed` shows its label (the
    list fills in from the bottom as the video plays). `labels` = [(rank, label)], any order."""
    words = [f"{{\\c{PALETTE[i % 4]}}}{_esc(w)}" for i, w in enumerate(title.split())]
    banner = " ".join(words)
    if tease:
        banner += f"\\N{{\\c&H00FFFFFF\\fs44}}({_esc(tease.upper())})"
    out = [f"Dialogue: 0,{_ts(0)},{_ts(dur)},Title,,0,0,0,,{banner}"]
    by_rank = dict(labels)
    rows = []
    for r in sorted(by_rank):
        num = f"{{\\c{PALETTE[(r - 1) % 4]}\\fs78}}{r}{{\\c&H00FFFFFF\\fs40}}"
        if r >= revealed:
            pop = "" if r != revealed else "{\\fscx60\\fscy60\\t(0,200,\\fscx100\\fscy100)}"
            rows.append(f"{num} {pop}{_esc(by_rank[r][:22])}")
        else:
            rows.append(num)
    gap = "\\N{\\fs22} \\N"  # breathing room between rows
    out.append(f"Dialogue: 0,{_ts(0)},{_ts(dur)},Rank,,0,0,0,," + gap.join(rows))
    return out


def render_still(
    image: Path,
    dur: float,
    voice: Path | None,
    words: list[Word],
    cfg: Config,
    out: Path,
    zoom_in: bool = True,
    title: str = "",
    fill: bool = False,
) -> Path:
    """One listicle item: a still photo with a slow Ken Burns zoom (in or out) over a blurred
    fill, captions in the global style, narration as the only audio. `fill`: the photo covers
    the whole frame instead and the camera travels across it (left to right when `zoom_in`,
    else right to left) while pushing in - a wide photo becomes a moving shot."""
    out.parent.mkdir(parents=True, exist_ok=True)
    ass_path = out.with_suffix(".ass")
    ass_path.write_text(build_ass(words, title, dur, cfg, badge=""))
    frames = max(1, round(dur * FPS))
    if fill:
        # cover the frame with 15% to spare, then slide a growing crop over at most 60% of the
        # spare width so a panorama does not race past
        p = f"n/{frames}"
        travel = f"min(iw-ow,{0.6 * W:.0f})"
        x = f"(iw-ow)/2+{travel}*({p}-0.5)" if zoom_in else f"(iw-ow)/2-{travel}*({p}-0.5)"
        vf = (
            f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
            f"scale=iw*1.15:ih*1.15:flags=lanczos,fps={FPS},"
            f"crop=w='{W}*(1.12-0.12*{p})':h='{H}*(1.12-0.12*{p})':x='{x}':y='(ih-oh)/2',"
            f"scale={W}:{H}:flags=lanczos,setsar=1,ass={ass_path.as_posix()}[v]"
        )
        inputs = ["-loop", "1", "-framerate", str(FPS), "-t", f"{dur:.3f}", "-i", str(image)]
    else:
        z = f"1+0.12*on/{frames}" if zoom_in else f"1.12-0.12*on/{frames}"
        # the composite is built at 2x and zoompan downsamples it, keeping the slow zoom smooth
        vf = (
            f"[0:v]split=2[b0][f0];"
            f"[b0]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=24:6[bg];"
            f"[f0]scale={W}:-2:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,scale={2 * W}:{2 * H},"
            f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={W}x{H}"
            f":fps={FPS},ass={ass_path.as_posix()}[v]"
        )
        inputs = ["-i", str(image)]
    if voice:
        inputs += ["-i", str(voice)]
        vf += f";[1:a]{NORM},apad=whole_dur={dur:.3f},atrim=0:{dur:.3f}[a]"
    else:
        vf += f";anullsrc=r=48000:cl=stereo,atrim=0:{dur:.3f}[a]"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", vf]
    cmd += ["-map", "[v]", "-map", "[a]", "-t", f"{dur:.3f}", *_ENC, str(out)]
    subprocess.check_call(cmd)
    return out


# a game's own subtitles and button prompts sit in the bottom of its frame
SHOT_CROP = "crop=iw:ih*0.78:0:0"
SHOT_ASPECT = 0.8  # width over height of the part of the footage that is shown (9:16 is 0.5625)


def letterbox(clip: Path) -> str:
    """A crop that takes the black bars off a letterboxed clip (a trailer in a cinema ratio),
    "" when it has none. Bars are cut off rather than shown: the picture between them is fine."""
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(clip), "-vf", "fps=1/4,cropdetect=24:2:0"]
        + ["-frames:v", "40", "-an", "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    found = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", r.stderr)
    size = re.search(r"Stream #0.*Video:.* (\d{3,5})x(\d{3,5})", r.stderr)
    if not found or not size:
        return ""
    w, h = int(size.group(1)), int(size.group(2))
    # the tallest picture any sampled frame had: a dark scene reads as a bigger bar than there is
    ch, y = max((int(f[1]), int(f[3])) for f in found)
    if ch >= h * 0.95 or ch < h * 0.6:
        return ""
    return f"crop={w}:{ch}:0:{y}"


def shot_view(bars: str = "") -> str:
    """The filters that leave the part of a footage frame the viewer sees sharp: without its
    black bars and the game's subtitle strip, the middle SHOT_ASPECT of what is left."""
    return f"{bars + ',' if bars else ''}{SHOT_CROP},crop='min(iw,ih*{SHOT_ASPECT})':ih"


def render_shot(
    clip: Path,
    start: float,
    dur: float,
    voice: Path | None,
    words: list[Word],
    cfg: Config,
    out: Path,
    title: str = "",
    footage_s: float | None = None,
    bars: str = "",
) -> Path:
    """One beat of a story over moving footage: `dur` seconds of `clip` from `start`, its middle
    filling the frame, its own sound off, captions in the global style, narration as the only
    audio. At most `footage_s` seconds of the footage are shown; footage that runs out holds
    its last frame. `bars`: letterbox() of the clip."""
    out.parent.mkdir(parents=True, exist_ok=True)
    ass_path = out.with_suffix(".ass")
    ass_path.write_text(build_ass(words, title, dur, cfg, badge=""))
    vf = (
        # the middle 4:5 of the picture across the full width, over a blurred copy of itself:
        # filling the tall screen with the picture alone cut a third-person character off
        f"[0:v]{bars + ',' if bars else ''}{SHOT_CROP},fps={FPS},split=2[b0][f0];"
        f"[b0]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=24:6[bg];"
        f"[f0]crop='min(iw,ih*{SHOT_ASPECT})':ih,scale={W}:-2:flags=lanczos[fg];"
        f"[bg][fg]overlay=0:(H-h)/2,setsar=1,tpad=stop_mode=clone:stop_duration={dur:.3f},"
        f"ass={ass_path.as_posix()}[v]"
    )
    shown = dur if footage_s is None else max(0.1, min(dur, footage_s))
    inputs = ["-ss", f"{max(0.0, start):.3f}", "-t", f"{shown:.3f}", "-i", str(clip)]
    if voice:
        inputs += ["-i", str(voice)]
        vf += f";[1:a]{NORM},apad=whole_dur={dur:.3f},atrim=0:{dur:.3f}[a]"
    else:
        vf += f";anullsrc=r=48000:cl=stereo,atrim=0:{dur:.3f}[a]"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", vf]
    cmd += ["-map", "[v]", "-map", "[a]", "-t", f"{dur:.3f}", *_ENC, str(out)]
    subprocess.check_call(cmd)
    return out


def concat(parts: list[Path], out: Path, fade_out: float = 0.0) -> Path:
    """Join segments rendered with the same _ENC settings; `fade_out` eases the audio out over
    the last N seconds so the video ends on a beat instead of a cut."""
    lst = out.with_suffix(".txt")
    lst.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in parts))
    audio = ["-c:a", "copy"]
    if fade_out > 0:
        total = sum(narrate.duration(p) for p in parts)
        st = max(0.0, total - fade_out)
        audio = ["-af", f"afade=t=out:st={st:.3f}:d={fade_out:.3f}", "-c:a", "aac", "-b:a", "160k"]
    subprocess.check_call(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst)]
        + ["-c:v", "copy", *audio, "-movflags", "+faststart", str(out)]
    )
    lst.unlink()
    return out


def clip_tail(moment: Moment, segments: list[Segment], source_len: float, cfg: Config) -> float:
    """How long the footage rolls on after the moment's last line: up to clips.tail_s, but never
    into the next sentence (so the fade-out doesn't catch half a word) or past the source."""
    tail = min(cfg.clips.tail_s, max(0.0, source_len - moment.end))
    # by words as well as segments: a moment may end inside a segment
    starts = [s.start for s in segments] + [w.start for s in segments for w in s.words]
    nxt = [t for t in starts if t >= moment.end - 0.01]
    if nxt:
        tail = min(tail, max(MIN_TAIL, min(nxt) - moment.end))
    return round(max(0.0, tail), 3)


def _voice(
    text: str, cfg: Config, out: Path, max_s: float
) -> tuple[Path | None, list[Word], float]:
    """(narration file, its words, tempo) with timings already at the tempo that fits `max_s`."""
    timed = narrate.synthesize_timed(text, cfg, out)
    if not timed:
        return None, [], 1.0
    voice, words = timed
    tempo = min(1.3, max(1.0, narrate.duration(voice) / max_s))
    return voice, [Word(w.start / tempo, w.end / tempo, w.text) for w in words], tempo


def render_clip(
    source: Path, moment: Moment, segments: list[Segment], cfg: Config, out: Path
) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    n = cfg.narration
    source_len = narrate.duration(source)
    tail = clip_tail(moment, segments, source_len, cfg)
    body = moment.end - moment.start
    has_audio = _has_audio(source)

    # the narrator's hook is its own intro: the clip's first frame is held and its audio
    # silent while he speaks, captions follow his words; then the clip plays with its own
    # words captioned - one voice at a time, like every other format
    voice, voice_words, tempo = None, [], 1.0
    lead = 0.25
    if has_audio:
        voice, voice_words, tempo = _voice(
            moment.narration, cfg, out.with_suffix(".voice.m4a"), n.max_len_s
        )
    voice_len = narrate.duration(voice) / tempo if voice else 0.0
    if voice and voice_len > (body + tail) / 2:
        log.warning("narration %.1fs too long for %.1fs clip, skipping", voice_len, body + tail)
        voice, voice_words, voice_len, tempo = None, [], 0.0, 1.0
    intro = voice_len + lead + PAUSE_GAP + 0.1 if voice else 0.0

    # the channel's take comes after the last line: the footage rolls on (its last frame held
    # when the source ends first) with its audio ducked while the narrator speaks
    take, take_words, take_tempo = None, [], 1.0
    if has_audio and n.take:
        take, take_words, take_tempo = _voice(
            moment.take, cfg, out.with_suffix(".take.m4a"), n.take_max_s
        )
    end_hold = 0.0
    if take:
        outro = TAKE_LEAD_S + narrate.duration(take) / take_tempo + OUTRO_FADE_S
        tail = min(outro, max(0.0, source_len - moment.end))
        end_hold = outro - tail
    dur = body + tail
    total = intro + dur + end_hold
    take_at = intro + body + TAKE_LEAD_S
    # the ending is a beat, not a cut: the picture rolls on while the audio eases out
    fade_s = OUTRO_FADE_S if take else tail
    fade = f"afade=t=out:st={max(0.0, total - fade_s):.3f}:d={max(fade_s, 0.05):.3f}"
    fade = f",{fade}" if fade_s else ""

    words = [Word(w.start + lead, w.end + lead, w.text) for w in voice_words]
    words += [
        Word(w.start + intro, w.end + intro, w.text)
        for w in words_in(segments, moment.start, moment.end)
    ]
    words += [Word(w.start + take_at, w.end + take_at, w.text) for w in take_words]
    ass_path = out.with_suffix(".ass")
    ass_path.write_text(build_ass(words, moment.hook, total, cfg, hook_secs=max(intro, HOOK_MIN_S)))

    # the intro shows the footage leading into the moment (muted) when there is enough of it,
    # otherwise the first frame is held
    pre = min(intro, moment.start)
    hold = intro - pre
    fr = _frame(cfg, source, moment.start, dur)
    vf = _vf(ass_path, **fr)
    pads = [f"start_duration={hold:.3f}"] if hold > 0.01 else []
    if end_hold > 0.01:
        pads.append(f"stop_mode=clone:stop_duration={end_hold:.3f}")
    if pads:
        vf = f"[0:v]tpad={':'.join(pads)}[hv];" + _vf(ass_path, src="[hv]", **fr)
    inputs = ["-ss", f"{moment.start - pre:.3f}", "-t", f"{dur + pre:.3f}", "-i", str(source)]
    audio_map = ["-map", "[a]"] if has_audio else []
    if has_audio and not (voice or take):
        vf += f";[0:a]{NORM}{fade}[a]"
    elif has_audio:
        ms = int(hold * 1000)
        src_a = [f"volume=0:enable='lt(t,{pre:.3f})'", f"adelay={ms}|{ms}"] if voice else []
        if take:
            duck = 10 ** (n.duck_db / 20)
            t0 = intro + body
            down = f"max({duck:.3f},1-(1-{duck:.3f})*(t-{t0:.3f})/{TAKE_LEAD_S})"
            src_a += [
                f"volume='if(lt(t,{t0:.3f}),1,{down})':eval=frame",
                f"apad=whole_dur={total:.3f}",
            ]
        vf += f";[0:a]{','.join([*src_a, 'aresample=48000'])}[held]"
        mix = ["[held]"]
        for file, speed, at in ((voice, tempo, lead), (take, take_tempo, take_at)):
            if not file:
                continue
            inputs += ["-i", str(file)]
            i = len(mix)
            vf += (
                f";[{i}:a]atempo={speed:.3f},adelay={int(at * 1000)}|{int(at * 1000)},"
                f"aresample=48000,apad[nar{i}]"
            )
            mix.append(f"[nar{i}]")
        vf += (
            f";{''.join(mix)}amix=inputs={len(mix)}:duration=first:dropout_transition=0,"
            f"{NORM}{fade}[a]"
        )
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
        *_ENC,
        str(out),
    ]
    log.info("rendering %s (%.1fs)", out.name, total)
    subprocess.check_call(cmd)
    return out
