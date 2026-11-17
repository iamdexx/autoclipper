from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from pathlib import Path

from . import analytics, ingest, metadata, qa, queue, rank, render, top10, trends, verify
from .config import Config, Creator
from .state import State

log = logging.getLogger(__name__)

OFF_FOCUS = "skipped_off_focus"  # video status: title names none of trends.focus_keywords

# Queue priority for an original Top-N video, on the 0-10 scale rank.rank() scores clip moments.
TOP10_SCORE = 7.0


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "clip"


def process_video(
    url: str,
    creator: Creator,
    cfg: Config,
    state: State,
    do_publish: bool = True,
    pillar: str | None = "",
) -> list[Path]:
    """`pillar`: the topic-mix pillar the video was found for (None = any); a creator's own
    upload ("") has to be about the channel's main focus."""
    meta = ingest.probe(url, creator.name)
    vdir = cfg.workdir / "videos" / meta.id
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "meta.json").write_text(json.dumps(asdict(meta), indent=2))

    if meta.live_status in ("is_live", "is_upcoming", "post_live"):
        log.info(
            "skipping %s (%s): %s, retry once the VOD is ready",
            meta.id,
            meta.title,
            meta.live_status,
        )
        return []
    if not trends.on_focus(meta.title, cfg, pillar):
        log.info("skipping %s (%s): not about %s", meta.id, meta.title, cfg.trends.focus)
        state.mark_video(meta.id, OFF_FOCUS, title=meta.title, creator=creator.name)
        return []
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
    if cfg.verify.enabled:
        check_hooks(moments, segments, cfg, vdir / "factcheck.json")
    (vdir / "moments.json").write_text(json.dumps([asdict(m) for m in moments], indent=2))
    log.info("%s: %d moments picked", meta.id, len(moments))

    clips: list[Path] = []
    rejected: list[Path] = []
    w = analytics.weight(state, creator.name, creator.hashtags, "clip")
    if w != 1.0:
        log.info("%s: queue scores x%.2f from how %s has performed", meta.id, w, creator.name)
    for i, m in enumerate(moments, 1):
        out = vdir / "clips" / f"{i:02d}-{_slug(m.hook)}.mp4"
        try:
            render.render_clip(src, m, segments, cfg, out)
        except Exception as e:
            log.error("render failed for moment %d: %s", i, e)
            continue
        if not qa.gate(out, cfg, f"{m.narration} {m.take}".strip(), kind="clip"):
            rejected.append(out)
            continue
        clips.append(out)
        if do_publish:
            pm = metadata.describe(
                m.hook,
                f"{m.narration} {m.take}".strip(),
                metadata.moment_transcript(segments, m.start, m.end),
                meta.title,
                meta.uploader or creator.name,
                creator.hashtags,
                cfg,
            )
            (vdir / "clips" / f"{out.stem}.post.json").write_text(json.dumps(asdict(pm), indent=2))
            queue.enqueue(
                out,
                "clip",
                meta.id,
                m.score * w,
                pm,
                cfg,
                hook=m.hook,
                creator=creator.name,
                hashtags=creator.hashtags,
            )

    state.mark_video(
        meta.id,
        "done",
        title=meta.title,
        creator=creator.name,
        clips=[c.name for c in clips],
        qa_failed=[c.name for c in rejected],
    )
    return clips


def check_hooks(moments, segments, cfg: Config, report: Path) -> None:
    """The on-screen hook and the narrator's line must be faithful to the clip. Anything that
    fails verify.passes() (unsupported; unverifiable too in strict mode) is replaced: narration by
    the checker's faithful rewrite (or dropped - the clip keeps the creator's own audio), the
    on-screen title by a line taken straight from the transcript. The take is checked even with
    verify.check_hooks off, and is dropped unless it is positively supported."""
    rows = []
    for m in moments:
        text = " ".join(s.text for s in segments if s.end > m.start and s.start < m.end)
        claim = m.hook if not m.narration else f"{m.hook}. {m.narration}"
        chk = None
        if cfg.verify.check_hooks:
            chk, fixed = verify.check_hook(claim, text, cfg)
            rows.append({**verify.dump([chk])[0], "fixed": fixed})
        if chk and not verify.passes(chk, cfg):
            safe_hook = rank._hook_from_text(text) if text else ""
            log.warning(
                "hook rejected (%s %s): %r / %r -> %r / %r",
                chk.verdict,
                chk.note,
                m.hook,
                m.narration,
                safe_hook,
                fixed,
            )
            m.hook = safe_hook or m.hook
            m.narration = fixed
        if m.take:
            tchk = verify.check_take(m.take, text, cfg)
            rows.append({**verify.dump([tchk])[0], "kind": "take"})
            if not verify.passes(tchk, cfg, strict=True):
                log.warning("take rejected (%s %s): %r", tchk.verdict, tchk.note, m.take)
                m.take = ""
    report.write_text(json.dumps(rows, indent=2))


def run_creators(cfg: Config, state: State) -> int:
    todo = ingest.new_videos(cfg, state.seen)
    if not todo:
        log.info("no new videos")
        return 0
    n = 0
    for creator, vid in todo:
        if n >= cfg.max_videos_per_run:
            break
        url = f"https://www.youtube.com/watch?v={vid}"
        try:
            process_video(url, creator, cfg, state)
            if state.data["videos"].get(vid, {}).get("status") != OFF_FOCUS:
                n += 1  # an off-focus upload costs one probe, not one of the run's slots
        except Exception as e:
            log.exception("failed on %s: %s", vid, e)
            state.mark_video(vid, "error", error=str(e), creator=creator.name)
    return n


def run_trends(cfg: Config, state: State, do_publish: bool = True) -> int:
    """Clip the best source video for each trending topic the LLM picks."""
    found = trends.discover(cfg)
    log.info("%d trending topics found", len(found))
    picked = trends.pick_topics(
        found,
        cfg,
        state.used_topics("trend"),
        history=analytics.summary(state),
        pillars_used=state.used_pillars(),
    )
    n = 0
    for tp in picked:
        state.mark_topic("trend", tp.name, status="picked", query=tp.query)
        state.mark_pillar(tp.pillar)
        try:
            src = trends.pick_source(tp, cfg)
        except Exception as e:
            log.warning("source search failed for %r: %s", tp.name, e)
            src = None
        if not src:
            state.mark_topic("trend", tp.name, status="no_source")
            continue
        if state.seen(src.id):
            state.mark_topic("trend", tp.name, status="already_clipped", video=src.id)
            continue
        creator = Creator(
            name=src.channel or tp.name,
            min_duration_s=cfg.trends.min_duration_s,
            max_duration_s=cfg.trends.max_duration_s,
            hashtags=tp.hashtags,
        )
        log.info("trend %r -> %s (%s)", tp.name, src.title, src.url)
        try:
            clips = process_video(
                src.url, creator, cfg, state, do_publish=do_publish, pillar=tp.pillar or None
            )
            state.mark_topic("trend", tp.name, status="done", video=src.id, clips=len(clips))
            n += 1
        except Exception as e:
            log.exception("trend %r failed: %s", tp.name, e)
            state.mark_topic("trend", tp.name, status="error", error=str(e))
    return n


def make_top10(
    topic: str, angle: str, cfg: Config, state: State, do_publish: bool = True
) -> Path | None:
    key = _slug(topic)
    vdir = cfg.workdir / "top10" / key
    state.mark_topic("top10", topic, status="building")
    try:
        out, script = top10.make(topic, angle, cfg, vdir)
    except Exception as e:
        log.exception("top10 %r failed: %s", topic, e)
        state.mark_topic("top10", topic, status="error", error=str(e))
        return None
    if not qa.gate(out, cfg, top10.narration_text(script), kind="top10"):
        state.mark_topic("top10", topic, status="qa_failed", video=out.name, title=script.title)
        return None
    state.mark_topic("top10", topic, status="done", video=out.name, title=script.title)
    if do_publish:
        pm = metadata.describe(
            script.title,
            script.intro,
            metadata.top10_transcript(script),
            topic,
            "",
            script.hashtags,
            cfg,
            kind="top10",
        )
        (vdir / "post.json").write_text(json.dumps(asdict(pm), indent=2))
        queue.enqueue(
            out,
            "top10",
            f"top10:{key}",
            TOP10_SCORE,
            pm,
            cfg,
            hook=script.title,
            hashtags=script.hashtags,
        )
    return out


def run_top10(cfg: Config, state: State) -> int:
    n = 0
    for _ in range(cfg.top10.per_run):
        pick = top10.pick_topic(cfg, state.used_topics("top10"), state)
        if not pick:
            log.info("no top10 topic available")
            break
        if make_top10(pick[0], pick[1], cfg, state):
            n += 1
    return n


def run_once(cfg: Config, state: State) -> int:
    """Make this run's videos, then post the best of everything waiting (this run's and earlier
    runs') as far as the budget allows."""
    if state is not None:
        try:
            analytics.refresh(state)
        except Exception as e:  # stats are a nice-to-have; never block a run on them
            log.warning("analytics refresh failed: %s", e)
    n = run_creators(cfg, state) if cfg.creators else 0
    if cfg.trends.enabled:
        n += run_trends(cfg, state)
    if cfg.top10.enabled:
        n += run_top10(cfg, state)
    queue.drain(cfg, state)
    return n
