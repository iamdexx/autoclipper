import json
import os
import subprocess
import time
from pathlib import Path

from autoclipper import narrate, qa, rank, render
from autoclipper.config import Config
from autoclipper.ingest import Segment, Word
from autoclipper.rank import Moment
from autoclipper.state import State

ROOT = Path(__file__).resolve().parents[1]


def _cfg(tmp_path):
    cfg = Config.load(ROOT / "config.yaml")
    cfg.workdir = tmp_path
    cfg.llm_provider = "heuristic"
    cfg.clips.min_len_s = 4
    cfg.clips.target_len_s = 6
    cfg.clips.max_len_s = 10
    cfg.narration.provider = "off"
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


def _src(tmp_path):
    src = tmp_path / "src.mp4"
    subprocess.check_call(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=440",
            "-t", "8", "-pix_fmt", "yuv420p", "-c:a", "aac", str(src),
        ]
    )  # fmt: skip
    return src


def test_render_produces_vertical_mp4(tmp_path):
    cfg = _cfg(tmp_path)
    src = _src(tmp_path)
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


def test_reframe_crops_to_the_face_or_the_middle_and_brands_every_video(tmp_path):
    from autoclipper import reframe

    cfg = _cfg(tmp_path)
    cfg.look.watermark = "@Clipverse-HQ"
    ass = tmp_path / "x.ass"
    # crop mode fills the frame; the window follows the face and is clamped to the picture
    vf = render._vf(ass, mode="crop", focus=0.8)
    assert "force_original_aspect_ratio=increase" in vf and "boxblur" not in vf
    assert "iw*0.8000-540" in vf and "max(0\\,min(iw-1080" in vf
    assert "iw*0.5000" in render._vf(ass, mode="crop", focus=None)
    assert "iw*1.0000" in render._vf(ass, mode="crop", focus=1.7)
    assert "boxblur" in render._vf(ass, mode="blur")
    # no face in a test pattern -> centre crop, and the handle sits on every frame
    src = _src(tmp_path)
    assert reframe.focus_x(src, 0.0, 6.0) is None
    text = render.build_ass([], "HOOK", 6.0, cfg)
    assert "Style: Brand," in text and ",Brand,,0,0,0,,{\\alpha&H40&}@Clipverse-HQ" in text
    cfg.look.watermark = ""
    assert ",Brand," not in render.build_ass([], "HOOK", 6.0, cfg)
    cfg.look.reframe = "blur"
    out = render.render_clip(
        src, Moment(0.0, 6.0, "H", "r", 1.0), _segments(4), cfg, tmp_path / "b.mp4"
    )
    assert out.exists()


def test_hook_card_pops_in_and_outlasts_a_short_intro(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.look.watermark = ""
    ass = render.build_ass([], "BIG MOMENT", 10.0, cfg, hook_secs=max(1.2, render.HOOK_MIN_S))
    hook = [ln for ln in ass.splitlines() if ",Hook," in ln]
    assert len(hook) == 1 and "\\fscx40" in hook[0] and "BIG MOMENT" in hook[0]
    assert hook[0].split(",")[2] == render._ts(render.HOOK_MIN_S)
    # the hook style is the caption highlight colour so the first frame reads as one look
    assert (
        f"Style: Hook,{cfg.captions.font},{int(cfg.captions.size * 1.05)},{cfg.captions.highlight_color}"
        in ass
    )


def test_heuristic_narration_is_one_line(tmp_path):
    cfg = _cfg(tmp_path)
    m = rank.rank_heuristic([(0.0, 6.0, "No way, he actually did it. Insane, bro!")], cfg)[0]
    assert m.narration == f"{m.hook}." and len(m.narration.split()) <= 9
    assert "max 9 words" in rank.PROMPT and "3-6 word" in rank.PROMPT


def test_render_mixes_narration(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.narration.provider = "gemini"
    cfg.narration.max_len_s = 1.0
    src = _src(tmp_path)

    def fake_tts(text, voice, out):
        # 2s of a 1 kHz tone stands in for the narrator; longer than max_len so tempo kicks in
        subprocess.check_call(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=1000",
             "-t", "2", "-c:a", "aac", str(out)]
        )  # fmt: skip
        return out

    monkeypatch.setattr(narrate, "tts_gemini", fake_tts)
    monkeypatch.setattr(narrate, "_heard", lambda voice, cfg_: [])  # a tone has no words to hear
    m = Moment(0.0, 6.0, "TEST HOOK", "r", 1.0, narration="wait for it")
    out = render.render_clip(src, m, _segments(4), cfg, tmp_path / "clip.mp4")
    streams = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(out)]
    ).decode()  # fmt: skip
    assert streams.split() == ["video", "audio"]
    # 2s tone at 1.3x = 1.54s of hook + 0.85s of lead/gap ahead of the 6s clip, then the clip
    # rolls on for the 0.3s minimum tail (the next sentence starts right at 6.0)
    assert abs(narrate.duration(out) - (6.0 + 1.54 + 0.85 + 0.3)) < 0.3
    assert (tmp_path / "clip.voice.m4a").exists()
    ass = (tmp_path / "clip.ass").read_text()
    caps = [ln for ln in ass.splitlines() if ",Cap," in ln]
    assert "WAIT" in caps[0].upper() and caps[3].split(",")[1] >= "0:00:02.0"  # clip words after


def test_clip_tail_rolls_on_but_never_into_the_next_sentence(tmp_path):
    cfg = _cfg(tmp_path)
    segs = _segments(4)  # sentences at 0-2, 2-4, 4-6, 6-8
    # a moment that ends between sentences gets the full tail
    m = Moment(0.0, 6.0, "h", "r", 1.0)
    assert render.clip_tail(Moment(0.0, 4.6, "h", "r", 1.0), segs, 30.0, cfg) == 1.0
    # ...but stops short of the next sentence (with a floor so the ending is still a beat)
    assert render.clip_tail(Moment(0.0, 5.5, "h", "r", 1.0), segs, 30.0, cfg) == 0.5
    assert render.clip_tail(m, segs, 30.0, cfg) == 0.3
    # and never runs past the end of the source
    assert render.clip_tail(Moment(0.0, 8.0, "h", "r", 1.0), segs, 8.4, cfg) == 0.4
    assert render.clip_tail(Moment(0.0, 8.0, "h", "r", 1.0), segs, 8.0, cfg) == 0.0
    out = render.render_clip(_src(tmp_path), m, segs, cfg, tmp_path / "clip.mp4")
    assert abs(narrate.duration(out) - 6.3) < 0.2


def test_narration_provider_selection(monkeypatch):
    cfg = Config.load(ROOT / "config.yaml")
    for k in (
        "ELEVENLABS_API_KEY",
        "elevenlabskey",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "Geminikey",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)
    assert narrate.providers(cfg) == ["edge"]  # no key needed
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert narrate.providers(cfg) == ["edge", "gemini"]
    monkeypatch.setenv("ELEVENLABS_API_KEY", "x")
    assert narrate.providers(cfg) == ["edge", "elevenlabs", "gemini"]
    monkeypatch.setenv("GOOGLE_TTS_API_KEY", "x")
    monkeypatch.setenv("AZURE_SPEECH_KEY", "x")
    assert narrate.providers(cfg) == ["edge", "azure", "google", "elevenlabs", "gemini"]
    cfg.narration.provider = "google"
    assert narrate.providers(cfg) == ["google"]
    cfg.narration.provider = "off"
    assert narrate.pick_provider(cfg) is None


def test_narration_falls_through_to_the_next_provider(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    for k in ("GOOGLE_TTS_API_KEY", "GOOGLE_CLOUD_API_KEY", "AZURE_SPEECH_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AZURE_SPEECH_KEY", "x")
    monkeypatch.setenv("GOOGLE_TTS_API_KEY", "x")
    cfg.narration.provider = "auto"
    calls = []

    def edge(*a):
        calls.append("edge")
        raise RuntimeError("endpoint changed")

    def azure(*a):
        calls.append("azure")
        raise RuntimeError("quota exceeded")

    def google(text, voice, rate, out):
        calls.append("google")
        out.write_bytes(b"x")
        return out

    monkeypatch.setattr(narrate, "tts_edge", edge)
    monkeypatch.setattr(narrate, "tts_azure", azure)
    monkeypatch.setattr(narrate, "tts_google", google)
    assert narrate.synthesize("hi there", cfg, tmp_path / "v.m4a") == tmp_path / "v.m4a"
    assert calls == ["edge", "azure", "google"]


def test_timed_narration_tries_each_provider_once(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    for k in ("GOOGLE_TTS_API_KEY", "GOOGLE_CLOUD_API_KEY", "AZURE_SPEECH_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "x")
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    cfg.narration.provider = "auto"
    calls = []

    def edge_timed(*a):
        calls.append("edge")
        raise RuntimeError("endpoint changed")

    def eleven_timed(*a):
        calls.append("elevenlabs")
        raise RuntimeError("quota_exceeded")

    def gemini(text, voice, out):
        calls.append("gemini")
        out.write_bytes(b"x")
        return out

    monkeypatch.setattr(narrate, "tts_edge_timed", edge_timed)
    monkeypatch.setattr(narrate, "tts_edge", lambda *a: calls.append("edge-plain"))
    monkeypatch.setattr(narrate, "tts_elevenlabs_timed", eleven_timed)
    monkeypatch.setattr(narrate, "tts_elevenlabs", lambda *a: calls.append("elevenlabs-plain"))
    monkeypatch.setattr(narrate, "tts_gemini", gemini)
    monkeypatch.setattr(narrate, "_heard", lambda voice, cfg: [])
    monkeypatch.setattr(narrate, "duration", lambda p: 1.0)
    got = narrate.synthesize_timed("hi there", cfg, tmp_path / "v.m4a")
    assert got is not None and got[0] == tmp_path / "v.m4a"
    assert [w.text for w in got[1]] == ["hi", "there"]  # evenly spread when whisper hears nothing
    assert calls == ["edge", "elevenlabs", "gemini"]


def test_edge_timed_uses_the_streams_word_boundaries(tmp_path, monkeypatch):
    heard = [Word(0.1, 0.4, "Hi"), Word(0.5, 0.9, "there")]

    def stream(text, voice, rate, out):
        out.write_bytes(b"x")
        return out, heard

    monkeypatch.setattr(narrate, "_edge_stream", stream)
    monkeypatch.setattr(narrate, "duration", lambda p: 1.0)
    cfg = _cfg(tmp_path)
    cfg.narration.provider = "edge"
    voice, words = narrate.synthesize_timed("Hi there!", cfg, tmp_path / "v.m4a")
    assert voice == tmp_path / "v.m4a"
    assert [(w.start, w.end, w.text) for w in words] == [(0.1, 0.4, "Hi"), (0.5, 0.9, "there!")]


def test_align_words_follows_what_whisper_heard():
    script = "Imagine getting fired from the company you built."
    heard = [Word(0.1, 0.5, "Imagine"), Word(0.5, 0.9, "getting"), Word(0.9, 1.2, "fired"),
             Word(1.6, 1.8, "you"), Word(1.8, 2.3, "built.")]  # fmt: skip
    words = narrate.align_words(script, heard, 2.5)
    assert [w.text for w in words] == script.split()
    assert (words[0].start, words[2].end) == (0.1, 1.2)  # matched words keep heard timing
    assert words[3].start == 1.2 and words[5].end <= 1.6  # the unheard run fills the gap
    assert words[-1].end == 2.3
    assert all(b.start >= a.start for a, b in zip(words, words[1:], strict=False))
    # nothing heard: even spread
    assert len(narrate.align_words(script, [], 2.5)) == 8


def test_captions_never_merge_narrator_and_clip_words(tmp_path):
    cfg = _cfg(tmp_path)
    words = [Word(0.3, 0.7, "Watch.")] + [
        Word(1.4 + i * 0.3, 1.6 + i * 0.3, t) for i, t in enumerate("so I got fired".split())
    ]
    ass = render.build_ass(words, "", 4.0, cfg, hook_secs=1.3)
    caps = [ln.split(",,")[-1] for ln in ass.splitlines() if ",Cap," in ln]
    assert "Watch." in caps[0] and "so" not in caps[0]


def test_top10_build_drops_entries_without_footage(tmp_path, monkeypatch):
    import pytest

    from autoclipper import top10

    cfg = _cfg(tmp_path)
    cfg.verify.min_entries = 3
    cfg.top10.items, cfg.top10.style = 4, "countdown"
    cfg.top10.music = "off"  # concat is faked below, so there is no file to mix under
    script = top10.Script(
        topic="plays",
        title="Top 4 plays of the week",
        intro="",
        entries=[top10.Entry(4 - i, f"play {i}", f"line {i}", f"q{i}") for i in range(4)],
    )
    src = _src(tmp_path)

    def fake_fetch(entry, secs, dest, cfg):
        return None if entry.name == "play 1" else src

    monkeypatch.setattr(top10, "fetch_footage", fake_fetch)
    rendered = []

    def fake_render(src, cuts, voice, words, title, badge, cfg, out, **kw):
        rendered.append(badge)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"x")

    monkeypatch.setattr(top10.render, "render_narrated", fake_render)
    monkeypatch.setattr(top10.render, "concat", lambda parts, out, **kw: out)

    top10.build(script, cfg, tmp_path / "t")
    assert [e.name for e in script.entries] == ["play 0", "play 2", "play 3"]
    assert [e.rank for e in script.entries] == [3, 2, 1]
    assert script.title == "Top 3 plays of the week"
    assert rendered == ["#3", "#2", "#1"]
    saved = json.loads((tmp_path / "t" / "script.json").read_text())
    assert len(saved["entries"]) == 3

    monkeypatch.setattr(top10, "fetch_footage", lambda *a: None)
    with pytest.raises(RuntimeError):
        top10.build(script, cfg, tmp_path / "t2")


def test_live_streams_are_not_processed(tmp_path, monkeypatch):
    from autoclipper import ingest, pipeline
    from autoclipper.state import State

    cfg = _cfg(tmp_path)
    meta = ingest.VideoMeta("live1", "u", "live now", 0, "c", live_status="is_live")
    monkeypatch.setattr(ingest, "probe", lambda url, name: meta)
    monkeypatch.setattr(ingest, "download", lambda *a: pytest_fail("downloaded a live stream"))
    st = State(tmp_path / "state.json")
    assert pipeline.process_video("u", cfg.creators[0], cfg, st) == []
    assert not st.seen("live1")  # retried once the VOD exists


def pytest_fail(msg):
    raise AssertionError(msg)


def test_resolve_channel_id_uses_the_playlist_field(monkeypatch):
    import pytest

    from autoclipper import ingest
    from autoclipper.config import Creator

    seen = {}

    def fake(cmd, text):
        seen["cmd"] = cmd
        return "UCmDTrq0LNgPodDOFZiSbsww\n" if "@xQcOW/videos" in cmd[-1] else ""

    monkeypatch.setattr(ingest.subprocess, "check_output", fake)
    c = Creator("xQc", handle="xQcOW")
    assert ingest.resolve_channel_id(c) == "UCmDTrq0LNgPodDOFZiSbsww" == c.channel_id
    assert (
        "playlist:channel_id" in seen["cmd"]
        and "https://www.youtube.com/@xQcOW/videos" in seen["cmd"]
    )
    with pytest.raises(ValueError, match="could not resolve @nobody"):
        ingest.resolve_channel_id(Creator("n", handle="@nobody"))


def test_new_videos_dates_undated_uploads_and_applies_the_age_window(tmp_path, monkeypatch):
    import time

    from autoclipper import ingest
    from autoclipper.config import Creator

    cfg = _cfg(tmp_path)
    cfg.max_video_age_h = 72
    now = time.time()
    cfg.creators = [Creator("A", channel_id="UCa"), Creator("B", channel_id="UCb"), Creator("C")]

    def uploads(creator, limit=ingest.FEED_DEPTH):
        if creator.name == "A":  # Data API: dated
            return [("a-new", now - 3600), ("a-old", now - 100 * 3600), ("a-seen", now)]
        if creator.name == "B":  # yt-dlp flat listing: undated
            return [("b-new", None), ("b-old", None), ("b-gone", None), ("b-seen", None)]
        raise ValueError("needs channel_id or handle")

    dated = []

    def times(ids):
        dated.append(list(ids))
        return {"b-new": now - 7200, "b-old": now - 200 * 3600}  # b-gone failed to extract

    monkeypatch.setattr(ingest, "latest_uploads", uploads)
    monkeypatch.setattr(ingest, "upload_times", times)
    got = ingest.new_videos(cfg, lambda v: v.endswith("-seen"))
    assert [(c.name, v) for c, v in got] == [("A", "a-new"), ("B", "b-new")]
    assert dated == [["b-new", "b-old", "b-gone"]]  # one batched call, seen ids never re-probed


def test_latest_uploads_falls_back_to_ytdlp_without_a_token(monkeypatch):
    from autoclipper import ingest
    from autoclipper.config import Creator

    monkeypatch.setattr(ingest, "_uploads_via_api", lambda cid, limit: pytest_fail("no token"))
    cmds = []

    def fake(cmd, text):
        cmds.append(cmd)
        return "v1\nv2\n\n"

    monkeypatch.setattr(ingest.subprocess, "check_output", fake)
    assert ingest.latest_uploads(Creator("x", channel_id="UCx"), limit=5) == [
        ("v1", None),
        ("v2", None),
    ]
    assert "--flat-playlist" in cmds[0] and "1:5" in cmds[0]
    assert cmds[0][-1] == "https://www.youtube.com/channel/UCx/videos"


def test_publish_top10_templates(tmp_path):
    from autoclipper import publish

    cfg = _cfg(tmp_path)
    cfg.publishing.dry_run = True
    cfg.publishing.youtube = True
    res = publish.publish_all(tmp_path / "v.mp4", "Top 10 plays", "", ["#nfl"], cfg, kind="top10")
    assert res["youtube"]["title"] == "Top 10 plays #shorts"
    assert "clip from" not in res["youtube"]["caption"]


def test_publish_only_and_force(tmp_path, monkeypatch):
    from autoclipper import publish

    cfg = _cfg(tmp_path)
    cfg.publishing.dry_run = True
    cfg.publishing.youtube = cfg.publishing.tiktok = True
    monkeypatch.setattr(publish, "publish_tiktok", lambda v, c: {"id": "x", "status": "ok"})
    res = publish.publish_all(tmp_path / "v.mp4", "h", "c", [], cfg, only="tiktok", force=True)
    assert res == {"tiktok": {"id": "x", "status": "ok"}}


def test_publish_budget_caps_and_spaces_real_posts(tmp_path, monkeypatch):
    from autoclipper import publish
    from autoclipper.state import State

    cfg = _cfg(tmp_path)
    cfg.publishing.dry_run = False
    cfg.publishing.youtube = True
    cfg.publishing.max_per_day, cfg.publishing.min_gap_min = 2, 60
    cfg.publishing.schedule_ahead_h = 0  # this test: no scheduling, slots open or they don't
    posted = []
    monkeypatch.setattr(
        publish,
        "publish_youtube",
        lambda v, t, d, tags, publish_at=None, **kw: posted.append(t) or {"id": f"y{len(posted)}"},
    )
    st = State(tmp_path / "state.json")

    def go(hook):
        res = publish.publish_all(tmp_path / "v.mp4", hook, "c", [], cfg, state=st)
        st.record_post("vid", "v.mp4", "youtube", res["youtube"])
        return res["youtube"]

    assert go("a")["id"] == "y1"
    assert "next slot" in go("b")["skipped"]  # too soon after the first
    st.data["posts"][0]["at"] -= 61 * 60  # ...an hour later it's fine
    assert go("c")["id"] == "y2"
    for p in st.data["posts"]:
        p["at"] -= 61 * 60
    assert "2/2 posts" in go("d")["skipped"]  # daily cap; skipped/dry-run rows don't count
    for p in st.data["posts"]:
        p["at"] -= 24 * 3600
    assert go("e")["id"] == "y3"
    assert posted == ["a | c #shorts", "c | c #shorts", "e | c #shorts"]
    # force (manual `autoclipper publish`) ignores the budget
    res = publish.publish_all(tmp_path / "v.mp4", "f", "c", [], cfg, state=st, force=True)
    assert res["youtube"]["id"] == "y4"


def test_publish_schedules_youtube_into_future_slots(tmp_path, monkeypatch):
    """One run fills the coming slots: the first video goes out now, the next is uploaded as a
    scheduled post one gap later, and the budget counts the scheduled one like a real post."""
    from autoclipper import publish
    from autoclipper.state import State

    cfg = _cfg(tmp_path)
    cfg.publishing.dry_run = False
    cfg.publishing.youtube = True
    cfg.publishing.max_per_day, cfg.publishing.min_gap_min = 3, 60
    cfg.publishing.schedule_ahead_h = 1
    calls = []

    def fake_upload(v, t, d, tags, publish_at=None, **kw):
        calls.append(publish_at)
        out = {"id": t}
        if publish_at and publish_at > time.time() + publish.SCHEDULE_MIN_AHEAD_S:
            out.update(at=publish_at, scheduled=publish._rfc3339(publish_at))
        return out

    monkeypatch.setattr(publish, "publish_youtube", fake_upload)
    st = State(tmp_path / "state.json")

    def go(hook):
        res = publish.publish_all(tmp_path / "v.mp4", hook, "c", [], cfg, state=st)
        st.record_post("vid", "v.mp4", "youtube", res["youtube"])
        return res["youtube"]

    now = time.time()
    assert go("a")["id"] == "a | c #shorts" and abs(calls[0] - now) < 5
    b = go("b")
    assert b["scheduled"] and 3590 < b["at"] - now < 3610  # one gap after the first
    assert "schedule_ahead_h" in go("c")["skipped"]  # 2h out is past the horizon
    cfg.publishing.schedule_ahead_h = 3
    assert "2/3" not in go("c").get("skipped", "")
    assert 7190 < st.data["posts"][-1]["at"] - now < 7210
    assert "3/3 posts" in go("d")["skipped"]  # daily cap counts the scheduled ones
    # tiktok has no scheduler: it waits for the slot instead
    cfg.publishing.tiktok = True
    monkeypatch.setattr(publish, "publish_tiktok", lambda v, c: {"id": "t1"})
    st.record_post("vid", "v.mp4", "tiktok", {"id": "t0"})
    res = publish.publish_all(tmp_path / "v.mp4", "e", "c", [], cfg, state=st)
    assert "no scheduling on tiktok" in res["tiktok"]["skipped"]


def test_publish_youtube_body_schedules_private_until_publish_at(tmp_path, monkeypatch):
    from autoclipper import publish

    bodies = []

    class Req:
        def next_chunk(self):
            return None, {"id": "vid1"}

    class Videos:
        def insert(self, part, body, media_body):
            bodies.append(body)
            return Req()

    class YT:
        def videos(self):
            return Videos()

    monkeypatch.setattr(publish, "youtube_auth", lambda interactive=True, manual=False: None)
    monkeypatch.setattr("googleapiclient.discovery.build", lambda *a, **k: YT())
    monkeypatch.setattr("googleapiclient.http.MediaFileUpload", lambda *a, **k: object())
    (tmp_path / "v.mp4").write_bytes(b"x")
    later = time.time() + 3600
    res = publish.publish_youtube(
        tmp_path / "v.mp4", "t", "d", ["#shorts"], publish_at=later, category="sports"
    )
    assert bodies[0]["status"]["privacyStatus"] == "private"
    assert bodies[0]["snippet"]["categoryId"] == "17"
    assert bodies[0]["status"]["publishAt"] == publish._rfc3339(later)
    assert res["at"] == later and res["scheduled"] == publish._rfc3339(later)
    res = publish.publish_youtube(tmp_path / "v.mp4", "t", "d", ["#shorts"], publish_at=None)
    assert (
        bodies[1]["status"]["privacyStatus"] == "public" and "publishAt" not in bodies[1]["status"]
    )
    assert bodies[1]["snippet"]["categoryId"] == "24"
    assert "at" not in res


def test_config_goes_live_on_youtube_only():
    cfg = Config.load(ROOT / "config.yaml")
    p = cfg.publishing
    assert not p.dry_run and p.youtube and not p.tiktok and not p.instagram
    assert p.max_per_day == 6 and p.min_gap_min == 150


def test_tiktok_privacy_downgrades_to_allowed(monkeypatch):
    from autoclipper import publish

    monkeypatch.setenv("TIKTOK_PRIVACY", "PUBLIC_TO_EVERYONE")
    assert publish.tiktok_privacy({"privacy_level_options": ["SELF_ONLY"]}) == "SELF_ONLY"
    assert (
        publish.tiktok_privacy({"privacy_level_options": ["SELF_ONLY", "PUBLIC_TO_EVERYONE"]})
        == "PUBLIC_TO_EVERYONE"
    )


def test_tiktok_token_refreshes(monkeypatch):
    from autoclipper import publish

    for k in ("TIKTOK_CLIENT_KEY", "TIKTOK_CLIENT_SECRET", "TIKTOK_REFRESH_TOKEN"):
        monkeypatch.setenv(k, "x")
    seen = {}

    class R:
        def json(self):
            return {"access_token": "fresh"}

    def post(url, data, timeout):
        seen.update(data)
        return R()

    monkeypatch.setattr(publish.requests, "post", post)
    assert publish.tiktok_token() == "fresh"
    assert seen["grant_type"] == "refresh_token"


def test_host_public_copies_without_bucket(tmp_path, monkeypatch):
    from autoclipper import publish

    monkeypatch.setenv("PUBLIC_BASE_URL", "https://cdn.example.com/")
    monkeypatch.delenv("PUBLIC_S3_BUCKET", raising=False)
    v = tmp_path / "a.mp4"
    v.write_bytes(b"x")
    url = publish.host_public(v, tmp_path / "public")
    assert url == "https://cdn.example.com/a.mp4"
    assert (tmp_path / "public" / "a.mp4").exists()


def _synthetic(tmp_path, name, dur, audio="anoisesrc=color=pink:amplitude=0.3", video="testsrc2="):
    out = tmp_path / name
    subprocess.check_call(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"{video}size=1080x1920:rate=30:d={dur}",
            "-f", "lavfi", "-i", f"{audio}:d={dur}",
            "-shortest", "-c:v", "libx264", "-preset", "ultrafast",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000", "-ac", "2", str(out),
        ]
    )  # fmt: skip
    return out


def test_qa_gate_passes_a_sane_video_and_writes_report(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.qa.top10_min_s = 3
    monkeypatch.setattr(
        qa.ingest,
        "transcribe",
        lambda v, c, cache: [Segment(0, 1, "the miami miracle lateral", [])],
    )
    vid = _synthetic(tmp_path, "ok.mp4", 5)
    assert qa.gate(vid, cfg, "Miami Miracle lateral", kind="top10")
    rep = json.loads((tmp_path / "ok.qa.json").read_text())
    assert rep["ok"] and {c["name"] for c in rep["checks"]} >= {"aspect", "loudness", "speech"}


def test_qa_drops_transcript_cached_from_an_older_render(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.qa.top10_min_s = 3
    seen: list[bool] = []

    def fake(v, c, cache):
        seen.append(cache.exists())
        return [Segment(0, 1, "miami miracle", [])]

    monkeypatch.setattr(qa.ingest, "transcribe", fake)
    vid = _synthetic(tmp_path, "re.mp4", 5)
    stale = vid.with_suffix(".qa-transcript.json")
    stale.write_text("[]")
    os.utime(stale, (0, 0))  # older than the video -> must be ignored
    qa.check(vid, cfg, "Miami Miracle", kind="top10")
    assert seen == [False]


def test_qa_gate_rejects_silence_black_and_wrong_narration(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.qa.top10_min_s = 3
    monkeypatch.setattr(
        qa.ingest, "transcribe", lambda v, c, cache: [Segment(0, 1, "completely different", [])]
    )
    vid = _synthetic(tmp_path, "bad.mp4", 5, audio="anullsrc=r=48000", video="color=c=black:")
    rep = qa.check(vid, cfg, "Miami Miracle lateral touchdown", kind="top10")
    failed = {c.name for c in rep.checks if not c.ok}
    assert not rep.ok and {"loudness", "dead_air", "black", "speech"} <= failed
    assert qa.check(_synthetic(tmp_path, "short.mp4", 1), cfg, "", kind="clip").ok is False


def test_qa_speech_match_ignores_stopwords():
    assert qa.script_match("the Miami Miracle, and it is on", "miami miracle") == 1.0
    assert qa.script_match("the Dolphins pulled off the Miami Miracle", "cats") == 0.0


def test_narration_drops_the_engines_trailing_silence(tmp_path):
    mp3 = tmp_path / "v.mp3"
    subprocess.check_call(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
         "sine=frequency=440:duration=1.0", "-af", "apad=pad_dur=1.0", str(mp3)]
    )  # fmt: skip
    assert 1.9 < narrate.duration(mp3) < 2.2
    out = narrate._mp3_to_aac(mp3, tmp_path / "v.m4a")
    assert 1.05 < narrate.duration(out) < 1.35  # 1s of tone + the 0.15s kept
    assert not mp3.exists()


def test_ytdlp_gets_cookies_and_proxy_from_the_environment(tmp_path, monkeypatch):
    from autoclipper import ingest

    for k in ("YT_COOKIES", "YT_COOKIES_FILE", "YT_PROXY"):
        monkeypatch.delenv(k, raising=False)
    assert ingest.ytdlp("-J", "u") == ["yt-dlp", "-J", "u"]

    monkeypatch.setenv(
        "YT_COOKIES", "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tA\tb"
    )
    monkeypatch.setenv("YT_PROXY", "http://u:p@host:8080")
    cmd = ingest.ytdlp("-J", "u")
    assert cmd[:1] + cmd[3:] == ["yt-dlp", "--proxy", "http://u:p@host:8080", "-J", "u"]
    assert cmd[1] == "--cookies" and Path(cmd[2]).read_text().startswith("# Netscape")
    assert oct(Path(cmd[2]).stat().st_mode)[-3:] == "600"
    assert ingest.ytdlp("x")[2] == cmd[2]  # written once, reused by every call
    Path(cmd[2]).unlink()

    f = tmp_path / "c.txt"
    f.write_text("x")
    monkeypatch.setenv("YT_COOKIES_FILE", str(f))
    assert ingest.ytdlp()[:3] == ["yt-dlp", "--cookies", str(f)]


def test_youtube_auth_manual_swaps_code_for_token_from_a_pasted_url(tmp_path, monkeypatch):
    import json

    from autoclipper import publish

    class FakeFlow:
        redirect_uri = None

        def authorization_url(self, **kw):
            return "https://accounts.google.com/o/oauth2/auth?x=1", "state"

        def fetch_token(self, authorization_response):
            assert authorization_response == "http://localhost:8765/?code=abc&scope=s"
            self.credentials = type(
                "C", (), {"to_json": lambda self: json.dumps({"refresh_token": "r"})}
            )()

    fake = FakeFlow()
    monkeypatch.setattr(
        "google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file",
        lambda client, scopes: fake,
    )
    monkeypatch.setenv("YT_TOKEN", str(tmp_path / "tok.json"))
    monkeypatch.delenv("YT_TOKEN_JSON", raising=False)
    monkeypatch.setattr("builtins.input", lambda _: " http://localhost:8765/?code=abc&scope=s \n")
    publish.youtube_auth(interactive=True, manual=True)
    assert fake.redirect_uri == "http://localhost:8765/"
    assert json.loads((tmp_path / "tok.json").read_text()) == {"refresh_token": "r"}


def test_metadata_describe_writes_searchable_title_and_specific_tags(tmp_path, monkeypatch):
    import json

    from autoclipper import metadata

    cfg = _cfg(tmp_path)
    prompts = []

    def llm(prompt, cfg):
        prompts.append(prompt)
        return json.dumps(
            {
                "title": "Will Ospreay vs Jon Moxley: The Injury That Silenced AEW All Out 2026 ",
                "description": "Ospreay goes down in the title match.  \nClip from All Elite Wrestling.",
                "hashtags": ["Will Ospreay", "#JonMoxley", "#aew", "#AEW", "#shorts", "#Wrestling"],
            }
        )

    monkeypatch.setattr(metadata.trends, "_llm", llm)
    pm = metadata.describe(
        "Career Threatening Injury In The Ring",
        "Wait for the arena to go quiet.",
        "Ospreay is down, he's not moving.",
        "Will Ospreay vs Jon Moxley - AEW All Out 2026",
        "All Elite Wrestling",
        ["#AEW", "#AllOut"],
        cfg,
    )
    assert pm.title == "Will Ospreay vs Jon Moxley: The Injury That Silenced AEW All Out 2026"
    assert pm.description.endswith("Clip from All Elite Wrestling.")
    # #-prefixed, de-duplicated case-insensitively, topic tags appended, #shorts once and last
    assert pm.hashtags == ["#WillOspreay", "#JonMoxley", "#aew", "#Wrestling", "#AllOut", "#shorts"]
    p = prompts[0]
    assert "Ospreay is down" in p and "All Elite Wrestling" in p and "Clip from" in p


def test_metadata_falls_back_to_template_when_llm_is_off_or_thin(tmp_path, monkeypatch):
    from autoclipper import metadata

    cfg = _cfg(tmp_path)  # heuristic provider -> _llm raises
    pm = metadata.describe("Big hook", "", "text", "Source Title", "Chan", ["#a"], cfg)
    assert pm.title == "Big hook | Source Title"
    assert pm.description == "Big hook\n\nSource Title\nClip from Chan."
    assert pm.hashtags == ["#a", "#shorts"]
    monkeypatch.setattr(metadata.trends, "_llm", lambda p, c: '{"title": "x", "description": ""}')
    assert metadata.describe("Big hook", "", "t", "Source Title", "Chan", [], cfg).title == (
        "Big hook | Source Title"
    )
    # top10: no credit line asked for, entries listed for the title to name them
    seen = {}
    monkeypatch.setattr(
        metadata.trends,
        "_llm",
        lambda p, c: (
            seen.setdefault("p", p)
            and '{"title": "Top 5 Taylor Swift Moments Ranked", "description": "d", "hashtags": []}'
        ),
    )
    pm = metadata.describe(
        "Ranking Taylor",
        "intro",
        "#2 Grammys\n#1 Proposal",
        "taylor swift",
        "",
        ["#ts"],
        cfg,
        kind="top10",
    )
    assert "No credit line" in seen["p"] and "#1 Proposal" in seen["p"]
    assert pm.hashtags == ["#ts", "#shorts"]
    assert (
        metadata.clip_title("a " * 60).endswith("a") and len(metadata.clip_title("a " * 60)) <= 90
    )


def test_publish_all_uses_written_metadata_over_templates(tmp_path):
    from autoclipper import publish
    from autoclipper.metadata import PostMeta

    cfg = _cfg(tmp_path)
    cfg.publishing.dry_run = True
    cfg.publishing.youtube = True
    pm = PostMeta("Real Title", "What happens.\nClip from X.", ["#a", "#shorts"])
    res = publish.publish_all(tmp_path / "v.mp4", "hook", "c", ["#zzz"], cfg, meta=pm)
    assert res["youtube"]["title"] == "Real Title"
    assert res["youtube"]["caption"] == (
        "What happens.\nClip from X.\n\n" + cfg.publishing.cta + "\n\n#a #shorts"
    )
    cfg.publishing.cta = ""
    res = publish.publish_all(tmp_path / "v.mp4", "hook", "c", ["#zzz"], cfg, meta=pm)
    assert res["youtube"]["caption"] == "What happens.\nClip from X.\n\n#a #shorts"


def test_metadata_category_from_llm_or_keywords(tmp_path, monkeypatch):
    import json

    from autoclipper import metadata

    cfg = _cfg(tmp_path)
    good = {
        "title": "Bills vs Rams: Josh Allen Overtime Winner",
        "description": "x",
        "hashtags": [],
    }
    monkeypatch.setattr(
        metadata.trends, "_llm", lambda p, c: json.dumps({**good, "category": "Sports"})
    )
    assert metadata.describe("h", "", "", "s", "c", [], cfg).category == "sports"
    # unknown / missing category -> keywords in the title, transcript and tags decide
    monkeypatch.setattr(
        metadata.trends, "_llm", lambda p, c: json.dumps({**good, "category": "vlog"})
    )
    assert (
        metadata.describe("h", "", "touchdown in overtime", "s", "c", ["#nfl"], cfg).category
        == "sports"
    )
    assert metadata.guess_category("xQc rages at a GTA RP cop on stream") == "gaming"
    assert metadata.guess_category("a quiet day") == "entertainment"
    # only whole words count: Emma / matcha / coda are not mma / match / cod
    assert (
        metadata.guess_category("Emma tries matcha, then a coda in the warehouse")
        == "entertainment"
    )
    assert metadata.guess_category("UFC mma knockout in the title match") == "sports"
    assert (
        metadata.fallback("Kai Cenat loses it", "Kai Cenat stream", "Kai", []).category == "gaming"
    )
    assert metadata.youtube_category_id("sports") == "17"
    assert metadata.youtube_category_id("whatever") == "24"


def test_cta_lands_even_when_template_keeps_hashtags_inline(tmp_path, monkeypatch):
    from autoclipper import publish

    cfg = _queue_cfg(tmp_path)
    cfg.publishing.dry_run = False
    cfg.publishing.cta = "Subscribe for more."
    cfg.publishing.caption_template = "{hashtags} {hook}"
    seen = {}
    monkeypatch.setattr(
        publish, "publish_youtube", lambda v, t, d, tags, **kw: seen.update(d=d) or {"id": "y"}
    )
    publish.publish_all(
        tmp_path / "x.mp4", "hook", "creator", ["#a"], cfg, state=State(tmp_path / "s.json")
    )
    assert seen["d"].endswith("\n\nSubscribe for more.") and seen["d"].startswith("#")


def _queue_cfg(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.publishing.dry_run = False
    cfg.publishing.youtube = True
    cfg.publishing.max_per_day, cfg.publishing.min_gap_min = 2, 60
    return cfg


def _enqueue(cfg, name, score, **kw):
    from autoclipper import queue
    from autoclipper.metadata import PostMeta

    v = cfg.workdir / f"{name}.mp4"
    v.write_bytes(b"x")
    return queue.enqueue(
        v, "clip", f"src-{name}", score, PostMeta(name, "d", ["#shorts"]), cfg, **kw
    )


def test_queue_posts_best_first_and_keeps_the_rest_for_later_runs(tmp_path, monkeypatch):
    from autoclipper import publish, queue
    from autoclipper.state import State

    cfg = _queue_cfg(tmp_path)
    cfg.publishing.schedule_ahead_h = 0
    posted = []
    monkeypatch.setattr(
        publish,
        "publish_youtube",
        lambda v, t, d, tags, publish_at=None, **kw: posted.append(t) or {"id": t},
    )
    st = State(tmp_path / "state.json")
    for name, score in [("meh", 3.0), ("best", 9.0), ("good", 7.0)]:
        _enqueue(cfg, name, score)
    (tmp_path / "best.mp4").unlink()  # the render dir is scratch; the queue has its own copy

    # run 1: one slot open -> the strongest goes out, the other two wait
    assert [it.meta.title for it in queue.drain(cfg, st)] == ["best"]
    assert posted == ["best"]
    assert [it.meta.title for it in queue.items(cfg)] == ["good", "meh"]
    assert st.data["posts"][0]["video"] == "src-best" and st.data["posts"][0]["id"] == "best"

    # run 2, right after: the gap blocks everything, nothing is lost and nothing re-posted
    assert queue.drain(cfg, st) == []
    assert posted == ["best"] and len(queue.items(cfg)) == 2

    # run 3, an hour later: next best goes out, then the daily cap holds the last one
    for p in st.data["posts"]:
        if p.get("id"):
            p["at"] -= 61 * 60
    assert [it.meta.title for it in queue.drain(cfg, st)] == ["good"]
    assert posted == ["best", "good"]
    assert [it.meta.title for it in queue.items(cfg)] == ["meh"]
    for p in st.data["posts"]:
        if p.get("id"):
            p["at"] -= 61 * 60
    assert queue.drain(cfg, st) == [] and "posts" in publish.over_budget("youtube", cfg, st)


def test_queue_drain_fills_the_coming_slots_with_scheduled_posts(tmp_path, monkeypatch):
    """With scheduling on, one run empties the queue into now + the next gaps (best first) and
    stops at the daily cap; the leftovers wait for the next run."""
    from autoclipper import publish, queue
    from autoclipper.state import State

    cfg = _queue_cfg(tmp_path)
    cfg.publishing.max_per_day, cfg.publishing.schedule_ahead_h = 3, 12
    uploads = []

    def fake_upload(v, t, d, tags, publish_at=None, **kw):
        uploads.append((t, publish_at))
        out = {"id": t}
        if publish_at > time.time() + publish.SCHEDULE_MIN_AHEAD_S:
            out.update(at=publish_at, scheduled=publish._rfc3339(publish_at))
        return out

    monkeypatch.setattr(publish, "publish_youtube", fake_upload)
    st = State(tmp_path / "state.json")
    for name, score in [("meh", 3.0), ("best", 9.0), ("good", 7.0), ("ok", 5.0)]:
        _enqueue(cfg, name, score)
    now = time.time()
    assert [it.meta.title for it in queue.drain(cfg, st)] == ["best", "good", "ok"]
    assert [t for t, _ in uploads] == ["best", "good", "ok"]
    gaps = [round((at - now) / 60) for _, at in uploads]
    assert gaps == [0, 60, 120]
    assert [it.meta.title for it in queue.items(cfg)] == ["meh"]
    live = [p for p in st.data["posts"] if p.get("id")]
    assert [bool(p.get("scheduled")) for p in live] == [False, True, True]


def test_queue_prunes_stale_and_surplus_weakest_first(tmp_path):
    from autoclipper import queue

    cfg = _queue_cfg(tmp_path)
    cfg.publishing.queue_max_items = 2
    old = _enqueue(cfg, "old", 9.5)
    old.added -= (cfg.publishing.queue_max_age_h + 1) * 3600
    queue._save(cfg, old)
    for name, score in [("a", 5.0), ("b", 8.0), ("c", 6.0)]:
        _enqueue(cfg, name, score)
    assert [it.meta.title for it in queue.prune(cfg)] == ["b", "c"]
    assert [it.meta.title for it in queue.items(cfg)] == ["b", "c"]


def test_queue_survives_a_missing_file_and_gives_up_after_repeated_errors(tmp_path, monkeypatch):
    from autoclipper import publish, queue
    from autoclipper.state import State

    cfg = _queue_cfg(tmp_path)
    cfg.publishing.min_gap_min = 0
    st = State(tmp_path / "state.json")
    it = _enqueue(cfg, "broken", 8.0)
    (cfg.workdir / "queue" / it.id / it.video).unlink()  # corrupt entry is ignored, not fatal
    assert queue.items(cfg) == []

    _enqueue(cfg, "flaky", 8.0)

    def boom(v, t, d, tags):
        raise RuntimeError("quota")

    monkeypatch.setattr(publish, "publish_youtube", boom)
    for n in (1, 2):
        assert queue.drain(cfg, st) == []
        assert queue.items(cfg)[0].attempts == n
    assert queue.drain(cfg, st) == [] and queue.items(cfg) == []
    assert all("error" in p for p in st.data["posts"])


def test_pipeline_enqueues_instead_of_posting_and_run_once_drains(tmp_path, monkeypatch):
    from autoclipper import pipeline, queue

    cfg = _queue_cfg(tmp_path)
    monkeypatch.setattr(pipeline, "run_creators", lambda c, s: 0)
    cfg.trends.enabled = cfg.top10.enabled = False
    drained = []
    monkeypatch.setattr(queue, "drain", lambda c, s, force=False: drained.append(True) or [])
    pipeline.run_once(cfg, None)
    assert drained == [True]
    assert "publish.publish_all" not in (ROOT / "autoclipper" / "pipeline.py").read_text()


def _history(tmp_path, rows):
    st = State(tmp_path / "state.json")
    for i, (creator, tags, views) in enumerate(rows):
        st.record_post(
            f"v{i}",
            "c.mp4",
            "youtube",
            {"id": f"y{i}", "views": views},
            creator=creator,
            hashtags=tags,
            kind="clip",
        )
    return st


def test_analytics_weights_by_what_got_watched(tmp_path):
    from autoclipper import analytics

    st = _history(
        tmp_path,
        [("penguinz0", ["#commentary", "#shorts"], v) for v in (300, 120, 90)]
        + [("Kai Cenat", ["#kaicenat"], v) for v in (3, 8, 12)]
        + [("NFL", ["#nfl"], v) for v in (150, 110, 100)],
    )
    assert analytics.channel_median(st) == 100
    rep = analytics.report(st)
    assert rep["creator:penguinz0"] == (3, 120) and rep["tag:#kaicenat"] == (3, 8)
    assert "tag:#shorts" not in rep
    assert analytics.weight(st, "penguinz0", ["#commentary"]) == 1.2
    assert analytics.weight(st, "Kai Cenat", ["#kaicenat"]) == 0.5  # clamped
    assert analytics.weight(st, "someone new", ["#new"]) == 1.0  # no history: untouched
    assert analytics.weight(None, "x") == 1.0
    one = _history(tmp_path / "one", [("a", ["#a"], 500)])
    assert analytics.weight(one, "a", ["#a"]) == 1.0  # fewer than MIN_POSTS
    s = analytics.summary(st)
    assert (
        "median is 100" in s and "penguinz0 (120" in s and "Flopped" in s and "kai cenat (8)" in s
    )


def test_analytics_refresh_pulls_views_for_settled_posts_only(tmp_path, monkeypatch):
    from autoclipper import analytics

    st = State(tmp_path / "state.json")
    now = time.time()
    st.record_post("v1", "c", "youtube", {"id": "old", "at": now - 48 * 3600})
    st.record_post("v2", "c", "youtube", {"id": "fresh", "at": now - 3600})
    st.record_post(
        "v3",
        "c",
        "youtube",
        {"id": "done", "at": now - 48 * 3600, "views": 5, "checked_at": now - 60},
    )
    st.record_post("v4", "c", "tiktok", {"id": "tt", "at": now - 48 * 3600})
    asked = []
    monkeypatch.setattr(
        analytics,
        "fetch_stats",
        lambda ids: asked.append(ids) or {"old": {"views": 77, "likes": 1, "comments": 0}},
    )
    assert analytics.refresh(st) == 1
    assert asked == [["old"]]
    saved = json.loads((tmp_path / "state.json").read_text())["posts"]
    assert saved[0]["views"] == 77 and "views" not in saved[1] and saved[2]["views"] == 5
    assert analytics.refresh(st) == 0  # first-day reading already taken; never overwritten
    assert "fresh" not in asked[0]


def test_analytics_format_only_counts_without_creator_history(tmp_path):
    from autoclipper import analytics

    st = _history(tmp_path, [("a", ["#a"], 200)] * 3)
    st.record_post(
        "v9", "c", "youtube", {"id": "t", "views": 10}, creator="", hashtags=[], kind="top10"
    )
    st.record_post(
        "v8", "c", "youtube", {"id": "u", "views": 10}, creator="", hashtags=[], kind="top10"
    )
    st.record_post(
        "v7", "c", "youtube", {"id": "w", "views": 10}, creator="", hashtags=[], kind="top10"
    )
    assert analytics.weight(st, "newcomer", ["#new"], "clip") == 1.0  # not inherited from kind:clip
    assert analytics.weight(st, "", [], "top10") == 0.5  # format is all we know: it counts
    assert analytics.weight(st, "a", ["#a"], "top10") == 1.5  # creator history beats format


def test_queue_records_what_the_post_was_about(tmp_path, monkeypatch):
    from autoclipper import publish, queue

    cfg = _queue_cfg(tmp_path)
    cfg.publishing.dry_run = False
    monkeypatch.setattr(publish, "publish_youtube", lambda *a, **k: {"id": "y1"})
    _enqueue(cfg, "a", 5.0, creator="NFL", hashtags=["#nfl"])
    st = State(tmp_path / "state.json")
    queue.drain(cfg, st)
    p = st.data["posts"][0]
    assert p["creator"] == "NFL" and p["hashtags"] == ["#nfl"] and p["kind"] == "clip"


def test_workflow_persists_queue_with_state():
    wf = (ROOT / ".github" / "workflows" / "clip.yml").read_text()
    assert wf.count("data/queue") == 2 and wf.count("data/state.json") == 2
