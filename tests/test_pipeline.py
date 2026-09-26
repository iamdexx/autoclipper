import json
import subprocess
from pathlib import Path

from autoclipper import rank, render
from autoclipper.config import Config
from autoclipper.ingest import Segment, Word
from autoclipper.rank import Moment

ROOT = Path(__file__).resolve().parents[1]


def _cfg(tmp_path):
    cfg = Config.load(ROOT / "config.yaml")
    cfg.workdir = tmp_path
    cfg.llm_provider = "heuristic"
    cfg.clips.min_len_s = 4
    cfg.clips.target_len_s = 6
    cfg.clips.max_len_s = 10
    return cfg


def _segments(n=20, dur=2.0):
    segs = []
    for i in range(n):
        start = i * dur
        text = (
            "oh my god that is insane bro!" if i % 5 == 0 else "so um yeah we kept going like this."
        )
        words = []
        toks = text.split()
        for k, t in enumerate(toks):
            ws = start + k * dur / len(toks)
            words.append(Word(ws, ws + dur / len(toks), t))
        segs.append(Segment(start, start + dur, text, words))
    return segs


def test_heuristic_prefers_hype(tmp_path):
    cfg = _cfg(tmp_path)
    moments = rank.rank(_segments(), cfg, "tester")
    assert 1 <= len(moments) <= cfg.clips.count
    for m in moments:
        assert cfg.clips.min_len_s <= m.end - m.start <= cfg.clips.max_len_s
    assert "insane" in moments[0].hook.lower() or moments[0].score > 0


def test_llm_output_parsing_and_snapping(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    segs = _segments()
    fake = json.dumps(
        [
            {"start": 0.3, "end": 6.1, "hook": "He lost it", "reason": "r", "score": 9},
            {"start": 100, "end": 200, "hook": "too long", "reason": "r", "score": 8},
        ]
    )
    monkeypatch.setattr(rank, "_call_gemini", lambda prompt, model: fake)
    moments = rank.rank_llm(segs, cfg, "tester", "gemini")
    assert len(moments) == 1
    assert moments[0].start == 0.0 and moments[0].end == 6.0


def test_render_produces_vertical_mp4(tmp_path):
    cfg = _cfg(tmp_path)
    src = tmp_path / "src.mp4"
    subprocess.check_call(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=440",
            "-t", "8", "-pix_fmt", "yuv420p", "-c:a", "aac", str(src),
        ]
    )  # fmt: skip
    segs = _segments(4)
    m = Moment(0.0, 6.0, "TEST HOOK", "r", 1.0)
    out = render.render_clip(src, m, segs, cfg, tmp_path / "clip.mp4")
    probe = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0", str(out),
        ]
    ).decode()  # fmt: skip
    assert probe.strip() == "1080,1920"
    assert (tmp_path / "clip.ass").read_text().count("Dialogue:") > 5
