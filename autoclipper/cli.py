from __future__ import annotations

import argparse
import logging
import sys
import time

from . import publish
from .config import Config, Creator
from .pipeline import process_video, run_once
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

    sub.add_parser("run", help="one pass: check feeds, clip + publish anything new")
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
