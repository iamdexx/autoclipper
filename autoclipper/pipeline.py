from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from pathlib import Path

from . import ingest, publish, rank, render
from .config import Config, Creator
from .state import State

log = logging.getLogger(__name__)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "clip"


def process_video(
    url: str, creator: Creator, cfg: Config, state: State, do_publish: bool = True
) -> list[Path]:
    meta = ingest.probe(url, creator.name)
    vdir = cfg.workdir / "videos" / meta.id
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "meta.json").write_text(json.dumps(asdict(meta), indent=2))

    if meta.duration and not (creator.min_duration_s <= meta.duration <= creator.max_duration_s):
        log.info(
            "skipping %s (%s): duration %.0fs outside window", meta.id, meta.title, meta.duration
        )
        state.mark_video(meta.id, "skipped_duration", title=meta.title)
        return []

    state.mark_video(meta.id, "downloading", title=meta.title, creator=creator.name)
    src = ingest.download(meta.url, vdir)
    segments = ingest.transcribe(src, cfg, vdir / "transcript.json")
    moments = rank.rank(segments, cfg, creator.name)
    (vdir / "moments.json").write_text(json.dumps([asdict(m) for m in moments], indent=2))
    log.info("%s: %d moments picked", meta.id, len(moments))

    clips: list[Path] = []
    for i, m in enumerate(moments, 1):
        out = vdir / "clips" / f"{i:02d}-{_slug(m.hook)}.mp4"
        try:
            render.render_clip(src, m, segments, cfg, out)
        except Exception as e:
            log.error("render failed for moment %d: %s", i, e)
            continue
        clips.append(out)
        if do_publish:
            res = publish.publish_all(out, m.hook, creator.name, creator.hashtags, cfg)
            for platform, r in res.items():
                state.record_post(meta.id, out.name, platform, r)

    state.mark_video(
        meta.id, "done", title=meta.title, creator=creator.name, clips=[c.name for c in clips]
    )
    return clips


def run_once(cfg: Config, state: State) -> int:
    todo = ingest.new_videos(cfg, state.seen)
    if not todo:
        log.info("no new videos")
        return 0
    n = 0
    for creator, vid in todo[: cfg.max_videos_per_run]:
        url = f"https://www.youtube.com/watch?v={vid}"
        try:
            process_video(url, creator, cfg, state)
            n += 1
        except Exception as e:
            log.exception("failed on %s: %s", vid, e)
            state.mark_video(vid, "error", error=str(e), creator=creator.name)
    return n
