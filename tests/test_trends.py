import json
import subprocess
from pathlib import Path

from autoclipper import narrate, render, top10, trends
from autoclipper.config import Config
from autoclipper.state import State
from autoclipper.trends import Candidate, Topic

ROOT = Path(__file__).resolve().parents[1]

RSS = b"""<?xml version="1.0"?>
<rss xmlns:ht="https://trends.google.com/trending/rss" version="2.0"><channel>
<item><title>phillies</title><ht:approx_traffic>5,000+</ht:approx_traffic>
  <ht:news_item><ht:news_item_title>Bill Giles dies</ht:news_item_title></ht:news_item></item>
<item><title>chip kelly</title><ht:approx_traffic>2000+</ht:approx_traffic></item>
</channel></rss>"""


def _cfg(tmp_path):
    cfg = Config.load(ROOT / "config.yaml")
    cfg.workdir = tmp_path
    cfg.llm_provider = "heuristic"
    cfg.narration.provider = "off"
    cfg.trends.topics_per_run = 1
    return cfg


def test_config_sections_load():
    cfg = Config.load(ROOT / "config.yaml")
    assert cfg.trends.geos == ["US"]
    assert cfg.top10.items == 10
    assert cfg.narration.elevenlabs_voice == "TX3LPaxmHKxFdv7VOQHJ"  # Liam


def test_google_trends_parsing(monkeypatch):
    class R:
        content = RSS

        def raise_for_status(self):
            pass

    monkeypatch.setattr(trends.requests, "get", lambda *a, **k: R())
    got = trends.google_trends("US")
    assert [t.name for t in got] == ["phillies", "chip kelly"]
    assert got[0].traffic == 5000 and got[0].headline == "Bill Giles dies"
    assert got[1].headline == ""


def test_yt_search_parses_flat_playlist(monkeypatch):
    payload = {
        "entries": [
            {"id": "abc", "title": "T", "channel": "C", "duration": 61, "view_count": 12},
            None,
            {"id": None},
        ]
    }

    def fake_run(cmd, **kw):
        assert "--flat-playlist" in cmd
        return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")

    monkeypatch.setattr(trends.subprocess, "run", fake_run)
    got = trends.yt_search("x", 3)
    assert len(got) == 1 and got[0].url.endswith("abc") and got[0].views == 12


def test_discover_dedupes_and_survives_failures(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.trends.niches = ["gaming"]
    monkeypatch.setattr(
        trends, "google_trends", lambda geo: [Topic("A", "google"), Topic("a", "google")]
    )

    def boom(niche, n=8):
        raise RuntimeError("blocked")

    monkeypatch.setattr(trends, "youtube_hot", boom)
    got = trends.discover(cfg)
    assert [t.name for t in got] == ["A"]


def test_pick_topics_falls_back_to_traffic_and_skips_used(tmp_path):
    cfg = _cfg(tmp_path)  # heuristic provider -> _llm raises -> fallback
    topics = [Topic("low", "g", 10), Topic("hot", "g", 900), Topic("used", "g", 5000)]
    picked = trends.pick_topics(topics, cfg, used=["USED"])
    assert [t.name for t in picked] == ["hot"]
    assert picked[0].query == "hot"


def test_pick_topics_uses_llm_choice(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    fake = json.dumps(
        [{"topic": "Hot", "query": "hot highlights", "angle": "why", "hashtags": ["#a", "#b"]}]
    )
    monkeypatch.setattr(trends, "_llm", lambda prompt, cfg_: fake)
    picked = trends.pick_topics([Topic("hot", "g", 1)], cfg, used=[])
    assert picked[0].source == "g" and picked[0].query == "hot highlights"
    assert picked[0].hashtags == ["#a", "#b"]


def test_pick_source_filters_duration_and_falls_back(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.trends.min_duration_s = 100
    cands = [
        Candidate("short", "s", "c", 30, 10**9),
        Candidate("ok", "o", "c", 600, 100),
        Candidate("best", "b", "c", 700, 5000),
    ]
    monkeypatch.setattr(trends, "yt_search", lambda q, n=10, **kw: cands)
    assert trends.pick_source(Topic("x", "g"), cfg).id == "best"
    monkeypatch.setattr(trends, "_llm", lambda p, c: '{"id": "ok"}')
    assert trends.pick_source(Topic("x", "g"), cfg).id == "ok"


def test_state_topics(tmp_path):
    st = State(tmp_path / "state.json")
    assert st.used_topics("trend") == []
    st.mark_topic("trend", "b")
    st.mark_topic("trend", "a", video="v1")
    st.mark_topic("top10", "z")
    assert State(tmp_path / "state.json").used_topics("trend") == ["b", "a"]
    assert st.used_topics("top10") == ["z"]


def test_elevenlabs_alignment_to_words(tmp_path, monkeypatch):
    chars = list("hi there")
    j = {
        "audio_base64": "",
        "alignment": {
            "characters": chars,
            "character_start_times_seconds": [i * 0.1 for i in range(len(chars))],
            "character_end_times_seconds": [(i + 1) * 0.1 for i in range(len(chars))],
        },
    }

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return j

    monkeypatch.setattr(narrate.requests, "post", lambda *a, **k: R())
    monkeypatch.setattr(narrate, "_elevenlabs_key", lambda: "k")
    monkeypatch.setattr(narrate, "_mp3_to_aac", lambda mp3, out: out)
    _, words = narrate.tts_elevenlabs_timed("hi there", "v", tmp_path / "v.m4a")
    assert [w.text for w in words] == ["hi", "there"]
    assert words[0].start == 0.0 and abs(words[0].end - 0.2) < 1e-9
    assert abs(words[1].start - 0.3) < 1e-9 and abs(words[1].end - 0.8) < 1e-9


def test_even_words():
    ws = narrate.even_words("a b c d", 2.0)
    assert len(ws) == 4 and ws[-1].end == 2.0 and ws[1].start == 0.5


def test_top10_script_parsing(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.top10.items = 3
    j = {
        "title": "Top 3 things",
        "intro": "Hook.",
        "hashtags": ["#x"],
        "entries": [
            {"rank": 1, "name": "one", "line": "l1", "query": "q1"},
            {"rank": 3, "name": "three", "line": "l3", "query": "q3"},
            {"rank": 2, "name": "two", "line": "l2", "query": "q2"},
        ],
    }
    monkeypatch.setattr(trends, "_llm", lambda p, c: "```json\n" + json.dumps(j) + "\n```")
    s = top10.write_script("things", "", cfg)
    assert s.title == "Top 3 things"
    assert [e.rank for e in s.entries] == [3, 2, 1]  # countdown order


def _src(tmp_path, secs=8):
    src = tmp_path / "src.mp4"
    subprocess.check_call(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=440",
            "-t", str(secs), "-pix_fmt", "yuv420p", "-c:a", "aac", str(src),
        ]
    )  # fmt: skip
    return src


def _probe(path):
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height",
            "-of", "json", str(path),
        ]
    )  # fmt: skip
    return json.loads(out)["streams"]


def test_render_narrated_and_concat(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    src = _src(tmp_path)
    voice = tmp_path / "voice.m4a"
    subprocess.check_call(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=1000",
         "-t", "3", "-c:a", "aac", str(voice)]
    )  # fmt: skip
    words = narrate.even_words("this is the number ten entry", 3.0)
    parts = []
    from autoclipper import sfx

    fx = sfx.whoosh(tmp_path / "whoosh.m4a")
    assert abs(narrate.duration(fx) - 0.6) < 0.2
    for i, badge in enumerate(["#10", "#9"]):
        cuts = [(0.5, 2.0), (3.0, 2.0)]  # two cuts, a zoom punch on each
        out = render.render_narrated(
            src, cuts, voice, words, f"ENTRY {i}", badge, cfg, tmp_path / f"{i}.mp4", sfx=fx
        )
        streams = _probe(out)
        assert [s["codec_type"] for s in streams] == ["video", "audio"]
        assert (streams[0]["width"], streams[0]["height"]) == (1080, 1920)
        assert abs(narrate.duration(out) - 4.0) < 0.3
        ass = (tmp_path / f"{i}.ass").read_text()
        assert badge in ass and f"ENTRY {i}" in ass
        assert "\\fscx40" in ass  # badge pops in
        parts.append(out)
    # source without audio track still renders
    mute = tmp_path / "mute.mp4"
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-an", str(mute)])
    parts.append(
        render.render_narrated(mute, [(0.0, 2.0)], None, [], "MUTE", "#8", cfg, tmp_path / "m.mp4")
    )
    final = render.concat(parts, tmp_path / "final.mp4")
    assert abs(narrate.duration(final) - 10.0) < 0.5
    assert [s["codec_type"] for s in _probe(final)] == ["video", "audio"]
    # one continuous music bed under the whole thing, video stream untouched
    cfg.top10.music = "synth"
    bed = sfx.music_bed(narrate.duration(final), cfg, tmp_path / "bed.m4a")
    assert bed and abs(narrate.duration(bed) - narrate.duration(final)) < 0.2
    mixed = sfx.mix_under(final, bed, -14.0, tmp_path / "mixed.mp4")
    assert abs(narrate.duration(mixed) - narrate.duration(final)) < 0.3
    assert [s["codec_type"] for s in _probe(mixed)] == ["video", "audio"]
    cfg.top10.music = "off"
    assert sfx.music_bed(5.0, cfg, tmp_path / "none.m4a") is None
    # generated music: falls back to music_dir, then to nothing (never to the synth)
    cfg.top10.music = "elevenlabs"
    monkeypatch.setattr(sfx, "generate_music", lambda *a: None)
    assert sfx.music_bed(5.0, cfg, tmp_path / "none2.m4a") is None
    cfg.top10.music_dir = str(tmp_path / "tracks")
    (tmp_path / "tracks").mkdir()
    (tmp_path / "tracks" / "song.m4a").write_bytes(voice.read_bytes())
    own = sfx.music_bed(5.0, cfg, tmp_path / "own.m4a")
    assert own and abs(narrate.duration(own) - 5.0) < 0.2


def test_cut_plan():
    # two 2s windows spread over 13s of footage: one at the start, one at the end
    assert render.cut_plan(13.0, 4.0, 2) == [(0.0, 2.0), (11.0, 2.0)]
    assert render.cut_plan(13.0, 6.0, 3) == [(0.0, 2.0), (5.5, 2.0), (11.0, 2.0)]
    # not enough footage to spread: one window, never past the end
    assert render.cut_plan(3.0, 4.0, 2) == [(0.0, 3.0)]
    assert render.cut_plan(13.0, 4.0, 1) == [(0.0, 4.0)]
    assert render.cut_plan(0.0, 4.0, 2) == [(0.0, 4.0)]
    assert "zoompan" in render._vf(Path("x.ass"), "[src]", [0.0, 2.0])
    assert "zoompan" not in render._vf(Path("x.ass"))
