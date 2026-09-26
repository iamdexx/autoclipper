from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from . import publish, top10, trends, verify
from .config import Config, Creator
from .pipeline import make_top10, process_video, run_once, run_trends
from .state import State


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="autoclipper")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("clip", help="clip one YouTube URL (what the video does by hand)")
    p.add_argument("url")
    p.add_argument("--creator", default=None, help="creator name used in captions")
    p.add_argument(
        "--publish", action="store_true", help="also publish (respects publishing.dry_run)"
    )

    t = sub.add_parser("trends", help="show what's trending now (and what the LLM would pick)")
    t.add_argument("--clip", action="store_true", help="also clip the picked topics")
    t.add_argument("--publish", action="store_true")

    tt = sub.add_parser("top10", help="make one Top-10 countdown video")
    tt.add_argument(
        "topic", nargs="?", help="e.g. 'craziest NFL plays this week'; default: trending"
    )
    tt.add_argument("--angle", default="", help="extra context for the script")
    tt.add_argument("--publish", action="store_true")

    fc = sub.add_parser("factcheck", help="fact-check a Top-10 script.json or a free-text claim")
    fc.add_argument("target", help="path to data/top10/<slug>/script.json, or a claim in quotes")
    fc.add_argument("--query", default="", help="search query for a free-text claim")

    sub.add_parser("run", help="one pass: creators + trends + top10 (whatever is enabled)")
    sub.add_parser("watch", help="loop `run` forever every poll_interval_s")

    a = sub.add_parser("auth", help="interactive OAuth setup")
    a.add_argument("platform", choices=["youtube"])

    sub.add_parser("status", help="show processed videos and posts")

    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpcore", "httpx", "urllib3", "google", "googleapiclient"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = Config.load(args.config)
    state = State(cfg.workdir / "state.json")

    if args.cmd == "clip":
        creator = next((c for c in cfg.creators if c.name == args.creator), None) or Creator(
            name=args.creator or "creator", min_duration_s=0
        )
        clips = process_video(args.url, creator, cfg, state, do_publish=args.publish)
        for c in clips:
            print(c)
        return 0
    if args.cmd == "trends":
        found = trends.discover(cfg)
        for tp in found[:40]:
            print(f"{tp.traffic:>8}  {tp.source:<10} {tp.name}  {tp.headline[:50]}")
        picked = trends.pick_topics(found, cfg, state.used_topics("trend"))
        print("\npicked:")
        for tp in picked:
            print(f"  {tp.name}  q={tp.query!r}  {tp.angle}")
        if args.clip:
            run_trends(cfg, state, do_publish=args.publish)
        return 0
    if args.cmd == "top10":
        if args.topic:
            topic, angle = args.topic, args.angle
        else:
            pick = top10.pick_topic(cfg, state.used_topics("top10"))
            if not pick:
                print("no topic available", file=sys.stderr)
                return 1
            topic, angle = pick
        out = make_top10(topic, angle, cfg, state, do_publish=args.publish)
        if not out:
            return 1
        print(out)
        return 0
    if args.cmd == "factcheck":
        p = Path(args.target)
        if p.is_file():
            j = json.loads(p.read_text())
            script = top10.Script(
                topic=j["topic"],
                title=j["title"],
                intro=j.get("intro", ""),
                entries=[
                    top10.Entry(e["rank"], e["name"], e["line"], e["query"]) for e in j["entries"]
                ],
            )
            checks = top10.fact_check(script, cfg)
            labels = [f"#{e.rank} {e.name}" for e in script.entries]
            if len(checks) > len(labels):
                labels.append("intro")
            (p.parent / "factcheck.json").write_text(json.dumps(verify.dump(checks), indent=2))
        else:
            checks = verify.check_claims([(args.target, args.query or args.target)], cfg)
            labels = ["claim"]
        bad = 0
        for label, c in zip(labels, checks, strict=True):
            ok = verify.passes(c, cfg)
            bad += not ok
            print(f"{'PASS' if ok else 'FAIL'} {c.verdict:<13} {c.confidence:.2f}  {label}")
            print(f"     {c.note}")
            for ev in c.evidence[:3]:
                print(f"     - [{ev.kind}] {ev.title[:90]}  {ev.url}")
        print(f"\n{len(checks) - bad}/{len(checks)} passed")
        return 1 if bad else 0
    if args.cmd == "run":
        run_once(cfg, state)
        return 0
    if args.cmd == "watch":
        while True:
            try:
                run_once(cfg, state)
            except Exception:  # keep the daemon alive
                logging.exception("run failed")
            time.sleep(cfg.poll_interval_s)
    if args.cmd == "auth":
        publish.youtube_auth(interactive=True)
        print("YouTube authorised.")
        return 0
    if args.cmd == "status":
        for vid, v in state.data["videos"].items():
            print(f"{vid}  {v['status']:<18} {v.get('title', '')[:60]}")
        for p in state.data["posts"]:
            print(f"  -> {p['platform']:<10} {p['clip']}  {p.get('url') or p.get('id') or p}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
