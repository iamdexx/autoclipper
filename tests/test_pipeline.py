import json
import os
import subprocess
from pathlib import Path

from autoclipper import narrate, qa, rank, render
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
    # 2s tone at 1.3x = 1.54s of hook + 0.85s of lead/tail ahead of the 6s clip
    assert abs(narrate.duration(out) - (6.0 + 1.54 + 0.85)) < 0.3
    assert (tmp_path / "clip.voice.m4a").exists()
    ass = (tmp_path / "clip.ass").read_text()
    caps = [ln for ln in ass.splitlines() if ",Cap," in ln]
    assert "WAIT" in caps[0].upper() and caps[3].split(",")[1] >= "0:00:02.0"  # clip words after


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
    monkeypatch.setattr(top10.render, "concat", lambda parts, out: out)

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
