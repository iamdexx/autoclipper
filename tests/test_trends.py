import json
import subprocess
from pathlib import Path

from autoclipper import images, narrate, render, top10, trends
from autoclipper.config import Config
from autoclipper.ingest import Word
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
    cfg.look.watermark = ""  # caption tests count Dialogue lines
    cfg.trends.topics_per_run = 1
    cfg.trends.focus_keywords, cfg.trends.pillars = [], []  # generic channel; _mix() sets a focus
    return cfg


def test_config_sections_load():
    cfg = Config.load(ROOT / "config.yaml")
    assert cfg.trends.geos == ["US"]
    assert cfg.top10.items == 6 and cfg.top10.style == "ranking"
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
    import time

    payload = {
        "entries": [
            {
                "id": "abc",
                "title": "T",
                "channel": "C",
                "duration": 61,
                "view_count": 12,
                "timestamp": time.time() - 2 * 3600,
            },
            {"id": "old", "title": "T", "channel": "C", "duration": 61, "view_count": 1},
            None,
            {"id": None},
        ]
    }
    targets = []

    def fake_run(cmd, **kw):
        assert "--flat-playlist" in cmd and "youtubetab:approximate_date" in cmd
        targets.append(cmd[-1])
        return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")

    monkeypatch.setattr(trends.subprocess, "run", fake_run)
    got = trends.yt_search("x", 3)
    assert len(got) == 2 and got[0].url.endswith("abc") and got[0].views == 12
    assert 1.9 < got[0].age_h < 2.1 and got[1].age_h is None
    trends.yt_search("x", 3, within="hour")
    assert targets[0].startswith("ytsearch3:") and "sp=CAISBAgBEAE" in targets[1]


def test_find_sources_prefers_the_freshest_window_and_drops_stale_videos(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.trends.min_duration_s = 100
    cfg.trends.max_source_age_h = 24
    by_window = {
        "hour": [Candidate("h1", "t", "c", 600, 10, age_h=0.5)],
        "today": [
            Candidate("h1", "t", "c", 600, 10, age_h=0.5),  # already seen
            Candidate("d1", "t", "c", 600, 5000, age_h=6),
            Candidate("short", "t", "c", 30, 10**9, age_h=1),
        ],
        "": [
            Candidate("stale", "t", "c", 600, 10**6, age_h=90),
            Candidate("unknown", "t", "c", 600, 100),
        ],
    }
    asked = []

    def search(q, n=10, within=""):
        asked.append(within)
        return by_window[within]

    monkeypatch.setattr(trends, "yt_search", search)
    pool = trends.find_sources(Topic("x", "g"), cfg)
    assert asked == ["hour", "today", ""]  # kept widening until it had a pool
    assert [c.id for c in pool] == ["h1", "d1", "unknown"]
    # the fallback pick is the freshest, not the most viewed
    assert trends.pick_source(Topic("x", "g"), cfg).id == "h1"
    # a fresh window with enough videos ends the search
    by_window["hour"] = [Candidate(f"h{i}", "t", "c", 600, i, age_h=0.2) for i in range(3)]
    asked.clear()
    trends.find_sources(Topic("x", "g"), cfg)
    assert asked == ["hour"]


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


def _mix(cfg):
    from autoclipper.config import Pillar

    cfg.trends.focus_keywords = ["gta 6", "gta"]
    cfg.trends.pillars = [
        Pillar("news", ["gta 6 news"]),
        Pillar("map", ["gta 6 map"]),
        Pillar("money", ["clipping gta 6"], max_share=0.15),
    ]
    return cfg


def test_on_focus_matches_whole_keywords_only(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.trends.focus_keywords = []
    assert trends.on_focus("NFL week 3", cfg)  # no focus configured: everything passes
    _mix(cfg)
    assert trends.on_focus("GTA6 map leaked?", cfg) and trends.on_focus("New GTA VI trailer", cfg)
    assert not trends.on_focus("Regatta highlights", cfg) and not trends.on_focus("", cfg)


def test_config_pillars_cover_a_range_and_cap_the_clipping_one():
    t = Config.load(ROOT / "config.yaml").trends
    assert len(t.pillars) >= 6 and all(p.queries for p in t.pillars)
    capped = [p for p in t.pillars if p.max_share < 1]
    assert [p.name for p in capped] == ["clipping gta 6"]


def test_next_pillars_rotates_and_respects_max_share(tmp_path):
    cfg = _mix(_cfg(tmp_path))
    names = lambda used, n: [p.name for p in trends.next_pillars(cfg, used, n)]  # noqa: E731
    assert names([], 2) == ["news", "map"]
    # money is capped at 15% of recent topics, so it waits while the others rotate
    assert names(["news", "map"], 2) == ["news", "map"]
    assert names(["news", "map"] * 3, 2) == ["money", "news"]
    assert names(["news", "map", "money"], 3) == ["news", "map"]
    # a full window already at the cap stays shut even when the topic falling out is another's
    full = ["news", "map"] * 5 + ["money", "news", "map"] * 3 + ["news"]
    assert "money" not in names(full, 3)
    used: list[str] = []
    for _ in range(40):
        used += names(used, 1)
    assert 3 <= used.count("money") <= 7 and used.count("news") > used.count("money")
    assert "money" not in {a for a, b in zip(used, used[1:], strict=False) if a == b}
    assert all(used[i : i + 20].count("money") <= 3 for i in range(len(used) - 19))


def test_discover_tags_pillars_and_drops_off_focus(tmp_path, monkeypatch):
    cfg = _mix(_cfg(tmp_path))
    monkeypatch.setattr(trends, "google_trends", lambda geo: [Topic("phillies", "google")])
    hits = {
        "gta 6 news": [Topic("GTA 6 delayed?", "youtube", 9), Topic("Fortnite live", "youtube", 99)]
    }
    monkeypatch.setattr(trends, "pillar_hot", lambda q, n=8: hits.get(q, []))
    got = trends.discover(cfg)
    assert [(t.name, t.pillar) for t in got] == [("GTA 6 delayed?", "news")]


def test_pick_topics_takes_one_per_pillar_in_turn(tmp_path, monkeypatch):
    cfg = _mix(_cfg(tmp_path))
    cfg.trends.topics_per_run = 2
    topics = [
        Topic("GTA 6 news a", "y", 5, pillar="news"),
        Topic("GTA 6 news b", "y", 50, pillar="news"),
        Topic("GTA 6 map a", "y", 7, pillar="map"),
        Topic("GTA 6 clipping", "y", 900, pillar="money"),
    ]
    trend = Topic("GTA 6 trend", "google", 9000)  # untagged: only fills a pillar with nothing
    picked = trends.pick_topics([trend, *topics], cfg, used=[], pillars_used=[])  # no LLM
    assert [(t.name, t.pillar) for t in picked] == [
        ("GTA 6 news b", "news"),
        ("GTA 6 map a", "map"),
    ]
    assert trend.pillar == ""
    # the LLM's pick wins inside its pillar; an invented topic or a pillar not up is ignored
    fake = json.dumps(
        [
            {"topic": "GTA 6 news a", "query": "q1", "angle": "a"},
            {"topic": "made up", "query": "q2"},
            {"topic": "GTA 6 clipping", "query": "q3"},
        ]
    )
    seen = {}
    monkeypatch.setattr(trends, "_llm", lambda prompt, c: seen.setdefault("p", prompt) and fake)
    picked = trends.pick_topics(topics, cfg, used=[], pillars_used=[])
    assert [(t.name, t.query) for t in picked] == [
        ("GTA 6 news a", "q1"),
        ("GTA 6 map a", "GTA 6 map a"),
    ]
    assert "GTA 6 clipping" not in seen["p"] and "- news" in seen["p"]
    picked = trends.pick_topics(topics, cfg, used=[], pillars_used=["news", "map"] * 3)
    assert picked[0].pillar == "money"


def test_find_sources_drops_off_focus_videos(tmp_path, monkeypatch):
    cfg = _mix(_cfg(tmp_path))
    cands = [
        Candidate("a", "Fortnite stream", "c", 600, 9),
        Candidate("b", "GTA 6 map", "c", 600, 1),
    ]
    monkeypatch.setattr(trends, "yt_search", lambda q, n=10, **kw: cands)
    assert [c.id for c in trends.find_sources(Topic("x", "g"), cfg)] == ["b"]


def test_state_pillars(tmp_path):
    st = State(tmp_path / "state.json")
    st.mark_pillar("news")
    st.mark_pillar("")
    st.mark_pillar("map")
    assert State(tmp_path / "state.json").used_pillars() == ["news", "map"]


def test_pick_source_filters_duration_and_falls_back(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.trends.min_duration_s = 100
    cands = [
        Candidate("short", "s", "c", 30, 10**9),
        Candidate("ok", "o", "c", 600, 100),
        Candidate("best", "b", "c", 700, 5000),
    ]
    monkeypatch.setattr(trends, "yt_search", lambda q, n=10, **kw: cands)
    assert trends.pick_source(Topic("x", "g"), cfg).id == "best"  # no ages known: most viewed
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


def _mean_db(path: Path, start: float, end: float) -> float:
    err = subprocess.run(
        ["ffmpeg", "-loglevel", "info", "-ss", f"{start}", "-to", f"{end}", "-i", str(path),
         "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True,
    ).stderr  # fmt: skip
    return float(err.split("mean_volume:")[1].split("dB")[0])


def test_payoff_footage_comes_back_up_after_the_narrator_and_the_video_fades_out(tmp_path):
    cfg = _cfg(tmp_path)
    src = _src(tmp_path)
    voice = tmp_path / "voice.m4a"
    subprocess.check_call(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=1000",
         "-t", "2", "-c:a", "aac", str(voice)]
    )  # fmt: skip
    words = narrate.even_words("number one", 2.0)
    # 2s of narration, then 3s of the footage playing out with its audio back up
    out = render.render_narrated(
        src,
        [(0.5, 5.0)],
        voice,
        words,
        "",
        "#1",
        cfg,
        tmp_path / "1.mp4",
        duck_db=-40,
        unduck_at=2.0,
    )
    ducked = _mean_db(out, 2.0, 2.4)  # narrator's over, footage still coming up
    up = _mean_db(out, 3.5, 5.0)
    assert up - ducked > 6, (ducked, up)
    # without unduck_at the footage stays buried the whole way
    flat = render.render_narrated(
        src, [(0.5, 5.0)], voice, words, "", "#1", cfg, tmp_path / "flat.mp4", duck_db=-40
    )
    assert _mean_db(flat, 3.5, 5.0) < up - 10
    final = render.concat([out, out], tmp_path / "final.mp4", fade_out=1.0)
    assert abs(narrate.duration(final) - 10.0) < 0.3
    assert _mean_db(final, 9.6, 10.0) < _mean_db(final, 8.5, 9.0) - 6


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


def test_ranking_script_parsing_and_prompt(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.top10.items, cfg.top10.style = 2, "ranking"
    j = {
        "title": "Ranking Insane Moments",
        "tease": "last one is wild",
        "intro": "Hook.",
        "hashtags": ["#x"],
        "entries": [
            {"rank": 2, "name": "two", "label": "Smooth", "line": "l2", "query": "q2"},
            {"rank": 1, "name": "one", "label": "Hell nah", "line": "l1", "query": "q1"},
        ],
    }
    seen = {}
    monkeypatch.setattr(trends, "_llm", lambda p, c: seen.setdefault("prompt", p) and json.dumps(j))
    s = top10.write_script("things", "", cfg)
    assert "Ranking" in seen["prompt"] and "label" in seen["prompt"]
    assert s.tease == "last one is wild" and s.intro == "Hook."
    assert [e.label for e in s.entries] == ["Smooth", "Hell nah"]


def test_ranking_events_fill_from_the_bottom():
    labels = [(1, "Best"), (2, "Ouch"), (3, "Smooth")]
    ev = render.ranking_events("Ranking The Best Moments", "wait for it", labels, 2, 5.0)
    banner, rows = ev
    assert "Ranking" in banner and "WAIT FOR IT" in banner
    assert "Smooth" in rows and "Ouch" in rows and "Best" not in rows
    assert rows.count("\\N") >= 2


def test_one_word_captions(tmp_path):
    cfg = _cfg(tmp_path)
    words = [Word(0.0, 0.4, "Bro"), Word(0.5, 0.9, "took"), Word(1.0, 1.4, "the bait!")]
    ass = render.build_ass(words, "", 2.0, cfg, one_word=True)
    lines = [ln for ln in ass.splitlines() if ln.startswith("Dialogue")]
    assert len(lines) == 3 and all(",Word," in ln for ln in lines)
    assert lines[-1].endswith("THE BAIT")


def test_one_word_caption_clears_during_a_long_pause(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.captions.style = "word"
    words = [Word(0.0, 0.4, "Wait"), Word(3.0, 3.4, "what")]
    lines = [ln for ln in render.build_ass(words, "", 4.0, cfg).splitlines() if ln.startswith("D")]
    assert lines[0].split(",")[2] == "0:00:01.00"  # 0.4 + 0.6 hold, not 3.0


def test_one_word_captions_never_overlap(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.captions.style = "word"
    words = [Word(0.0, 0.1, "you"), Word(0.15, 0.3, "hear"), Word(0.35, 0.6, "how")]
    lines = [ln for ln in render.build_ass(words, "", 2.0, cfg).splitlines() if ln.startswith("D")]
    ends = [ln.split(",")[2] for ln in lines]
    starts = [ln.split(",")[1] for ln in lines]
    assert ends[:-1] == starts[1:]  # each word leaves the moment the next arrives, no stacking


def test_caption_style_is_one_setting_for_every_format(tmp_path):
    cfg = _cfg(tmp_path)
    words = [Word(0.0, 0.4, "Bro"), Word(0.5, 0.9, "took"), Word(1.0, 1.4, "the bait!")]

    def dialogue(ass: str) -> str:
        return "\n".join(ln for ln in ass.splitlines() if ln.startswith("Dialogue"))

    default = dialogue(render.build_ass(words, "", 2.0, cfg))  # what every format gets
    assert cfg.captions.style == "highlight"
    assert cfg.captions.highlight_color in default and ",Word," not in default
    cfg.captions.style = "word"
    alt = dialogue(render.build_ass(words, "", 2.0, cfg))
    assert cfg.captions.highlight_color not in alt and ",Cap," not in alt


def test_image_matching():
    assert images.matches("Justin Tucker field goal", "File:Justin Tucker kicks a field goal.jpg")
    assert not images.matches("Justin Tucker field goal", "File:Sunset over the bay.jpg")
