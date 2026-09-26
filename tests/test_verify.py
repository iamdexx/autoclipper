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
    assert cfg.verify.enabled and cfg.verify.min_entries == 7 and not cfg.verify.strict


def test_google_news_parses_rss(monkeypatch):
    class R:
        content = NEWS_RSS

        def raise_for_status(self):
            pass

    monkeypatch.setattr(verify.requests, "get", lambda *a, **k: R())
    ev = verify.google_news("bowers")
    assert [e.title for e in ev] == ["Bowers hurdles defender - ESPN", "Raiders notes"]
    assert ev[0].source == "ESPN" and ev[0].date.startswith("Sun, 20 Sep 2026")


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
    assert len(report) == 13 and all(r["verdict"] in verify.VERDICTS for r in report)


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
            '"fixed": "Wait for what he says about his garage."}'
        ),
    )
    segs = [Segment(0, 5, "we started in a garage", []), Segment(5, 10, "then apple grew", [])]
    m = Moment(0, 8, "hook", "r", 1.0, narration="Wait for what he says about his boss!")
    pipeline.check_hooks([m], segs, cfg, tmp_path / "fc.json")
    assert m.narration == "Wait for what he says about his garage."
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
    verdicts = iter(["supported", "unsupported", "supported"])
    monkeypatch.setattr(
        verify,
        "check_claims",
        lambda claims, c: [verify.Check(cl, next(verdicts), 0.9, "n") for cl, _ in claims],
    )
    args = ["-c", str(ROOT / "config.yaml"), "factcheck", str(d / "script.json")]
    monkeypatch.chdir(tmp_path)
    assert cli.main(args) == 1
    assert len(json.loads((d / "factcheck.json").read_text())) == 3
    verdicts = iter(["supported"] * 3)
    assert cli.main(args) == 0
