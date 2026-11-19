import hashlib
import json
from pathlib import Path

import pytest

from autoclipper import cli, pipeline, top10, trends, verify
from autoclipper.config import Config
from autoclipper.ingest import Segment
from autoclipper.rank import Moment

ROOT = Path(__file__).resolve().parents[1]

NEWS_RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel>
<item><title>Bowers hurdles defender - ESPN</title><link>http://e/1</link>
  <pubDate>Sun, 20 Sep 2026 13:24:48 GMT</pubDate><source url="http://e">ESPN</source></item>
<item><title>Raiders notes</title><link>http://e/2</link></item>
</channel></rss>"""


def _cfg(tmp_path):
    cfg = Config.load(ROOT / "config.yaml")
    cfg.workdir = tmp_path
    cfg.llm_provider = "heuristic"
    cfg.narration.provider = "off"
    cfg.review.enabled = False
    cfg.top10.style, cfg.top10.items, cfg.verify.min_entries = "ranking", 6, 4
    return cfg


def _no_net(monkeypatch, ev=None):
    monkeypatch.setattr(verify, "gather", lambda q: ev or [])


def _script(n=10):
    return top10.Script(
        topic="nfl",
        title="Top 10",
        intro="hi",
        entries=[top10.Entry(r, f"play {r}", f"line {r}", f"q {r}") for r in range(n, 0, -1)],
    )


def test_verify_config_loads():
    cfg = Config.load(ROOT / "config.yaml")
    assert cfg.verify.enabled and cfg.verify.min_entries == 5 and not cfg.verify.strict
    assert cfg.verify.strict_top10


def test_script_is_written_from_fetched_sources(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.top10.items, cfg.top10.style = 6, "ranking"
    ev = [
        verify.Evidence("news", "Bauman 62-yard pick-six seals it - ESPN", source="ESPN"),
        verify.Evidence("youtube", "Oregon vs USC Highlights"),
        verify.Evidence("youtube", "oregon vs usc highlights"),  # dup (case) dropped
    ]
    monkeypatch.setattr(verify, "gather", lambda q: list(ev) if "Oregon" in q else [])
    prompts = []

    def fake_llm(prompt, c):
        prompts.append(prompt)
        return json.dumps({"title": "Ranking", "intro": "", "hashtags": [], "entries": []})

    monkeypatch.setattr(trends, "_llm", fake_llm)
    top10.write_script("Oregon USC", "insane moments", cfg)
    p = prompts[0]
    assert "SOURCES" in p and "MUST be a moment that appears in" in p
    assert "- [news] Bauman 62-yard pick-six seals it - ESPN (ESPN)" in p
    assert p.count("Oregon vs USC Highlights") == 1 and "return fewer" in p
    # nothing found (offline / obscure topic): the prompt simply has no sources block
    monkeypatch.setattr(verify, "gather", lambda q: [])
    top10.write_script("nothing", "", cfg)
    assert "SOURCES" not in prompts[1]


def test_top10_drops_unverifiable_entries_by_default(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.top10.items, cfg.verify.max_rounds, cfg.verify.min_entries = 6, 1, 2
    _no_net(monkeypatch)
    monkeypatch.setattr(top10, "write_script", lambda *a, **k: _script(6))

    def fake_check(claims, c):
        out = []
        for claim, _ in claims:
            invented = claim.startswith(("play 6:", "play 5:", "play 3:"))
            out.append(verify.Check(claim, "unverifiable" if invented else "supported", 0.9))
        return out

    monkeypatch.setattr(verify, "check_claims", fake_check)
    s = top10.verified_script("Oregon USC", "", cfg, tmp_path)
    assert [e.name for e in s.entries] == ["play 4", "play 2", "play 1"]
    assert [e.rank for e in s.entries] == [3, 2, 1]
    # the old permissive behaviour is still one switch away
    cfg.verify.strict_top10 = False
    assert len(top10.verified_script("Oregon USC", "", cfg, tmp_path).entries) == 6
    # ...and creator-clip hooks are unaffected by strict_top10 (they follow `strict`)
    assert verify.passes(verify.Check("h", "unverifiable", 0.0), cfg)


def test_google_news_parses_rss(monkeypatch):
    class R:
        content = NEWS_RSS

        def raise_for_status(self):
            pass

    monkeypatch.setattr(verify.requests, "get", lambda *a, **k: R())
    ev = verify.google_news("bowers")
    assert [e.title for e in ev] == ["Bowers hurdles defender - ESPN", "Raiders notes"]
    assert ev[0].source == "ESPN" and ev[0].date.startswith("Sun, 20 Sep 2026")


def _wiki_page(title, redirects=()):
    para = f"{title} is a video game whose history is long enough to be a real paragraph here."
    return {
        "title": title,
        "redirects": [{"title": r} for r in redirects],
        "extract": para,
    }


def test_wiki_extract_skips_article_that_only_cites_the_query(monkeypatch):
    pages = {}

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"query": {"pages": pages}}

    monkeypatch.setattr(verify.requests, "get", lambda *a, **k: R())
    # a YouTube title as the topic: the only full-text hit cites "GTA" in a reference
    pages["1"] = _wiki_page("Space Invaders", ["Space Invaders (arcade game)"])
    assert verify.wiki_extract("GTA 6 Review Roundup - Luke Reacts") == []
    # an article the query names through a redirect is used
    pages["1"] = _wiki_page("Grand Theft Auto VI", ["GTA 6", "GTA VI"])
    ev = verify.wiki_extract("GTA 6 Review Roundup - Luke Reacts")
    assert ev and ev[0].title.startswith("Grand Theft Auto VI:")
    pages["1"] = _wiki_page("Space Invaders")
    assert verify.wiki_extract("Why did Space Invaders speed up?")
    # a subject made only of short or common words is matched by its whole name
    pages["1"] = _wiki_page("The Who")
    assert verify.wiki_extract("The Who")
    assert not verify.names_query(_wiki_page("The Who"), "Who")


def test_gather_survives_failing_sources(monkeypatch):
    def boom(q):
        raise RuntimeError("down")

    monkeypatch.setattr(verify, "google_news", boom)
    monkeypatch.setattr(verify, "wikipedia", lambda q, n=3: [verify.Evidence("wiki", "W")])
    monkeypatch.setattr(verify, "youtube", boom)
    assert [e.title for e in verify.gather("x")] == ["W"]


def test_check_claims_parses_verdicts(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _no_net(monkeypatch, [verify.Evidence("news", "some headline")])
    seen = {}

    def fake(prompt, c):
        seen["p"] = prompt
        return json.dumps(
            [
                {"i": 0, "verdict": "supported", "confidence": 0.9, "note": "ok"},
                {"i": 1, "verdict": "UNSUPPORTED", "confidence": 0.8, "note": "nope"},
                {"i": 2, "verdict": "garbage"},
            ]
        )

    monkeypatch.setattr(trends, "_llm", fake)
    checks = verify.check_claims([("a", "qa"), ("b", "qb"), ("c", "qc")], cfg)
    assert [c.verdict for c in checks] == ["supported", "unsupported", "unverifiable"]
    assert "some headline" in seen["p"] and "CLAIM 2: c" in seen["p"]
    assert checks[0].evidence[0].title == "some headline"


def test_judge_failure_is_unverifiable_not_pass(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _no_net(monkeypatch)

    def boom(p, c):
        raise RuntimeError("quota")

    monkeypatch.setattr(trends, "_llm", boom)
    (c,) = verify.check_claims([("a", "q")], cfg)
    assert c.verdict == "unverifiable" and "quota" in c.note
    assert verify.passes(c, cfg)  # permissive default
    cfg.verify.strict = True
    assert not verify.passes(c, cfg)


def test_passes_thresholds(tmp_path):
    cfg = _cfg(tmp_path)
    assert not verify.passes(verify.Check("x", "unsupported", 0.9), cfg)
    assert verify.passes(verify.Check("x", "supported", 0.9), cfg)
    assert not verify.passes(verify.Check("x", "supported", 0.2), cfg)


def test_verified_script_drops_refills_and_reranks(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.top10.items, cfg.top10.style, cfg.verify.min_entries = 10, "countdown", 7
    _no_net(monkeypatch)
    drafts = iter([_script(), None])
    prompts = []

    def fake_write(topic, angle, c, avoid=None):
        prompts.append(avoid or [])
        s = next(drafts)
        if s is None:  # refill draft: two new verified entries + a repeat
            return top10.Script(
                "nfl",
                "T2",
                "",
                entries=[
                    top10.Entry(10, "new a", "l", "q"),
                    top10.Entry(9, "new b", "l", "q"),
                    top10.Entry(8, "play 4", "l", "q"),
                ],
            )
        return s

    def fake_check(claims, c):
        out = []
        for claim, _ in claims:
            bad = claim.startswith("play 3:") or claim.startswith("play 8:")
            out.append(verify.Check(claim, "unsupported" if bad else "supported", 0.9))
        return out

    monkeypatch.setattr(top10, "write_script", fake_write)
    monkeypatch.setattr(verify, "check_claims", fake_check)
    s = top10.verified_script("nfl", "", cfg, tmp_path)
    names = [e.name for e in s.entries]
    assert "play 3" not in names and "play 8" not in names
    assert names[:2] == ["new a", "new b"]  # refills take the bottom ranks
    assert names[-1] == "play 1" and [e.rank for e in s.entries] == list(range(10, 0, -1))
    assert s.title == "Top 10"
    assert "play 3" in prompts[1] and "play 8" in prompts[1]
    report = json.loads((tmp_path / "factcheck.json").read_text())
    assert len(report) == 14 and all(r["verdict"] in verify.VERDICTS for r in report)


def test_verified_script_skips_video_when_too_few_survive(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.verify.max_rounds = 1
    monkeypatch.setattr(top10, "write_script", lambda *a, **k: _script())
    monkeypatch.setattr(
        verify,
        "check_claims",
        lambda claims, c: [verify.Check(cl, "unsupported", 0.9) for cl, _ in claims],
    )
    with pytest.raises(RuntimeError, match="only 0 verified"):
        top10.verified_script("nfl", "", cfg, tmp_path)


def test_verify_disabled_passes_through(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.verify.enabled = False
    monkeypatch.setattr(top10, "write_script", lambda *a, **k: _script())
    monkeypatch.setattr(verify, "check_claims", lambda *a: pytest.fail("should not check"))
    assert len(top10.verified_script("nfl", "", cfg, tmp_path).entries) == 10


def test_footage_must_match_entry():
    e = top10.Entry(1, "Brock Bowers hurdle", "l", "Brock Bowers hurdle Raiders 2026")
    assert top10.footage_matches(e, "Brock Bowers LEAPS over defender | Raiders highlights")
    assert not top10.footage_matches(e, "NFL Top 100 Players of 2026")
    assert not top10.footage_matches(e, "Best Hurdles in Football History")


def test_check_hook_rewrites_unfaithful(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(
        trends,
        "_llm",
        lambda p, c: (
            '{"verdict": "unsupported", "confidence": 0.9, "note": "no boss", '
            '"fixed": "He says the company started in a garage."}'
        ),
    )
    segs = [Segment(0, 5, "we started in a garage", []), Segment(5, 10, "then apple grew", [])]
    m = Moment(0, 8, "hook", "r", 1.0, narration="Wait for what he says about his boss!")
    pipeline.check_hooks([m], segs, cfg, tmp_path / "fc.json")
    assert m.narration == "He says the company started in a garage."
    rows = json.loads((tmp_path / "fc.json").read_text())
    assert rows[0]["verdict"] == "unsupported" and rows[0]["fixed"]


def test_check_hook_failure_keeps_narration(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)

    def boom(p, c):
        raise RuntimeError("503")

    monkeypatch.setattr(trends, "_llm", boom)
    chk, fixed = verify.check_hook("hook", "transcript", cfg)
    assert chk.verdict == "unverifiable" and fixed == ""


def test_factcheck_cli_exit_code(tmp_path, monkeypatch):
    d = tmp_path / "t"
    d.mkdir()
    (d / "script.json").write_text(
        json.dumps({**vars(_script(3)), "entries": [vars(e) for e in _script(3).entries]})
    )
    verdicts = iter(["supported", "unsupported", "supported", "supported"])
    monkeypatch.setattr(
        verify,
        "check_claims",
        lambda claims, c, **k: [verify.Check(cl, next(verdicts), 0.9, "n") for cl, _ in claims],
    )
    args = ["-c", str(ROOT / "config.yaml"), "factcheck", str(d / "script.json")]
    monkeypatch.chdir(tmp_path)
    assert cli.main(args) == 1
    assert len(json.loads((d / "factcheck.json").read_text())) == 4
    verdicts = iter(["supported"] * 4)
    assert cli.main(args) == 0


def test_intro_is_checked_and_dropped(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.top10.style = "countdown"
    seen = {}

    def fake(claims, c):
        seen["claims"] = claims
        return [
            verify.Check(cl, "unsupported" if cl.startswith("Top 10:") else "supported", 0.9)
            for cl, _ in claims
        ]

    monkeypatch.setattr(verify, "check_claims", fake)
    s = _script(3)
    checks = top10.fact_check(s, cfg)
    assert len(checks) == 4 and seen["claims"][-1] == ("Top 10: hi", "nfl")
    assert s.intro == "" and all(e.verdict == "supported" for e in s.entries)
    assert s.title == "Top 3 nfl"


def test_footage_cache_is_per_entry(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(trends, "yt_search", lambda q, n=8: [])
    old = top10.Entry(1, "old play", "l", "q")
    assert top10.fetch_footage(old, 5, tmp_path, cfg) is None
    cached = tmp_path / "01-old-play-cafe1234.mp4"
    key = hashlib.sha1(b"old play|q").hexdigest()[:8]
    cached = cached.with_name(f"01-old-play-{key}.mp4")
    cached.write_bytes(b"x")
    assert top10.fetch_footage(old, 5, tmp_path, cfg) is None  # not a playable video: purged
    assert not cached.exists()
    cached.write_bytes(b"x")
    monkeypatch.setattr(top10, "has_video", lambda p: True)
    assert top10.fetch_footage(old, 5, tmp_path, cfg) == cached
    cached.with_suffix(".id").write_text("vid9")  # a kept clip still names its source
    again = top10.Entry(1, "old play", "l", "q")
    assert top10.fetch_footage(again, 5, tmp_path, cfg) == cached and again.video_id == "vid9"
    assert top10.fetch_footage(top10.Entry(1, "new play", "l", "q"), 5, tmp_path, cfg) is None
    assert top10.fetch_footage(top10.Entry(1, "old play", "l", "other q"), 5, tmp_path, cfg) is None


def test_rejected_hook_replaces_title_too(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(
        trends, "_llm", lambda p, c: '{"verdict": "unsupported", "confidence": 0.9, "note": "n"}'
    )
    segs = [Segment(0, 5, "We started Apple in my parents garage back in 1976.", [])]
    m = Moment(0, 5, "HE SOLD APPLE FOR $1", "r", 1.0, narration="He sold Apple for a dollar!")
    pipeline.check_hooks([m], segs, cfg, tmp_path / "fc.json")
    assert m.narration == "" and m.hook == "We started Apple in my parents garage back in 1976"


def test_take_is_only_spoken_when_supported(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    segs = [Segment(0, 5, "Rockstar confirmed the map has six regions.", [])]
    verdict = {"v": "supported"}

    def llm(prompt, c):
        if "TAKE:" not in prompt:
            return '{"verdict": "supported", "confidence": 0.9, "note": "ok"}'
        if verdict["v"] == "boom":
            raise RuntimeError("503")
        return json.dumps(
            {"verdict": verdict["v"], "adds": verdict.get("adds", True), "confidence": 0.9}
        )

    monkeypatch.setattr(trends, "_llm", llm)
    take = "Six regions means the biggest map Rockstar has made."

    def run(name):
        m = Moment(0, 5, "SIX REGIONS", "r", 1.0, narration="Six regions confirmed.", take=take)
        pipeline.check_hooks([m], segs, cfg, tmp_path / name)
        return m

    assert run("a.json").take == take
    rows = json.loads((tmp_path / "a.json").read_text())
    assert [r.get("kind") for r in rows] == [None, "take"]
    verdict["v"] = "unsupported"
    m = run("b.json")
    assert m.take == "" and m.narration == "Six regions confirmed."  # the hook is untouched
    verdict["v"] = "boom"  # a check that could not run never lets commentary through
    assert run("c.json").take == ""
    cfg.verify.check_hooks = False  # the take is checked on its own switch
    assert run("d.json").take == "" and run("d.json").hook == "SIX REGIONS"
    verdict.update(v="supported", adds=False)  # faithful, but only restates the clip
    assert run("e.json").take == ""
    verdict.update(v="supported", adds=True)
    assert run("e.json").take == take
    assert [r.get("kind") for r in json.loads((tmp_path / "e.json").read_text())] == ["take"]


def test_top10_footage_skips_blocked_channels_and_excluded_titles(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.trends.blocked_channels = ["Rockstar Games"]
    cfg.trends.exclude_keywords = ["highlights"]
    cands = [
        trends.Candidate("a", "GTA 6 map reveal", "rockstar games", 60, 9),
        trends.Candidate("b", "GTA 6 map reveal highlights", "Fan", 60, 8),
    ]
    monkeypatch.setattr(trends, "yt_search", lambda q, n, **k: cands)
    monkeypatch.setattr(top10, "footage_matches", lambda e, t: True)

    def no_download(*a, **k):
        raise AssertionError("prohibited footage was downloaded")

    monkeypatch.setattr(top10.subprocess, "run", no_download)
    assert top10.fetch_footage(top10.Entry(1, "map reveal", "l", "q"), 5, tmp_path, cfg) is None


def test_unverifiable_hook_dropped_in_strict_mode(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)

    def boom(p, c):
        raise RuntimeError("503")

    monkeypatch.setattr(trends, "_llm", boom)
    segs = [Segment(0, 5, "hello there everyone welcome back", [])]
    m = Moment(0, 5, "hook", "r", 1.0, narration="narration")
    pipeline.check_hooks([m], segs, cfg, tmp_path / "a.json")
    assert m.narration == "narration"  # permissive default keeps it
    cfg.verify.strict = True
    pipeline.check_hooks([m], segs, cfg, tmp_path / "b.json")
    assert m.narration == ""
    m = Moment(0, 5, "hook", "r", 1.0, narration="narration")
    pipeline.check_hooks([m], [], cfg, tmp_path / "c.json")  # no transcript = unverifiable
    assert m.narration == ""


def test_explainer_is_checked_without_video_titles_and_drops_filler(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.top10.style = "explainer"
    seen = {}

    def gather(q, titles=True):
        seen[q] = titles
        return [
            verify.Evidence("article", "Ice age: people sewed fitted clothing with bone needles")
        ]

    monkeypatch.setattr(verify, "gather", gather)
    monkeypatch.setattr(
        trends,
        "_llm",
        lambda p, c: json.dumps(
            [{"i": i, "verdict": "supported", "confidence": 0.9, "note": "n"} for i in range(3)]
        ),
    )
    script = top10.Script(
        "How humans survived the Ice Age",
        "How Humans Survived The Ice Age",
        "Half of Europe was under ice, and people stayed.",
        entries=[
            top10.Entry(
                2, "Bone needles", "They sewed fitted clothing with bone needles.", "bone needle"
            ),
            top10.Entry(1, "Fire", "This insane trick changes everything.", "hearth"),
        ],
    )
    checks = top10.fact_check(script, cfg)
    assert set(seen.values()) == {False}  # a video title is not evidence for a story
    assert "bone needle" not in seen  # the photo search is not what gets fact-checked
    assert [c.verdict for c in checks[:2]] == ["supported", "unsupported"]
    assert "filler" in checks[1].note and script.intro
