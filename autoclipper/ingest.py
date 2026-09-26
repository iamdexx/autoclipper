"""Find new long-form videos from watched creators, download them, transcribe with word timings."""

from __future__ import annotations

import calendar
import json
import logging
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import feedparser

from .config import Config, Creator

log = logging.getLogger(__name__)

YT_FEED = "https://www.youtube.com/feeds/videos.xml?channel_id={cid}"


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word]


@dataclass
class VideoMeta:
    id: str
    url: str
    title: str
    duration: float
    creator: str
    uploader: str = ""
    live_status: str = ""  # is_live | is_upcoming | was_live | not_live | post_live


def resolve_channel_id(creator: Creator) -> str:
    if creator.channel_id:
        return creator.channel_id
    if not creator.handle:
        raise ValueError(f"creator {creator.name} needs channel_id or handle")
    handle = creator.handle if creator.handle.startswith("@") else "@" + creator.handle
    out = subprocess.check_output(
        [
            "yt-dlp",
            "--playlist-items",
            "0",
            "--print",
            "channel_id",
            f"https://www.youtube.com/{handle}/videos",
        ],
        text=True,
    ).strip()
    cid = out.splitlines()[0]
    creator.channel_id = cid
    return cid


def latest_video_ids(creator: Creator, max_age_h: float | None = None) -> list[str]:
    """Newest first. With max_age_h, uploads published earlier are left out so the first run on a
    channel doesn't work through its whole back catalogue."""
    cid = resolve_channel_id(creator)
    feed = feedparser.parse(YT_FEED.format(cid=cid))
    out = []
    for e in feed.entries:
        pub = e.get("published_parsed")
        if max_age_h is not None and pub is not None:
            if time.time() - calendar.timegm(pub) > max_age_h * 3600:
                continue
        out.append(e.yt_videoid)
    return out


def probe(url: str, creator_name: str) -> VideoMeta:
    out = subprocess.check_output(["yt-dlp", "-J", "--no-playlist", url], text=True)
    j = json.loads(out)
    return VideoMeta(
        id=j["id"],
        url=j.get("webpage_url", url),
        title=j.get("title", ""),
        duration=float(j.get("duration") or 0),
        creator=creator_name,
        uploader=j.get("uploader", ""),
        live_status=j.get("live_status") or ("is_live" if j.get("is_live") else ""),
    )


def download(url: str, dest: Path, max_height: int = 1080) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "source.mp4"
    if out.exists():
        return out
    subprocess.check_call(
        [
            "yt-dlp",
            "--no-playlist",
            "-f",
            f"bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]/b[height<={max_height}][ext=mp4]/b",
            "--merge-output-format",
            "mp4",
            "-o",
            str(out),
            url,
        ]
    )
    return out


def transcribe(video: Path, cfg: Config, cache: Path) -> list[Segment]:
    if cache.exists():
        raw = json.loads(cache.read_text())
        return [
            Segment(s["start"], s["end"], s["text"], [Word(**w) for w in s["words"]]) for s in raw
        ]

    from faster_whisper import WhisperModel

    log.info("transcribing %s with whisper-%s", video.name, cfg.whisper_model)
    model = WhisperModel(cfg.whisper_model, device="cpu", compute_type=cfg.whisper_compute)
    segs, _ = model.transcribe(str(video), word_timestamps=True, vad_filter=True)
    result: list[Segment] = []
    for s in segs:
        words = [Word(w.start, w.end, w.word.strip()) for w in (s.words or []) if w.word.strip()]
        result.append(Segment(s.start, s.end, s.text.strip(), words))
    cache.write_text(json.dumps([asdict(s) for s in result]))
    return result


def new_videos(cfg: Config, seen) -> list[tuple[Creator, str]]:
    found: list[tuple[Creator, str]] = []
    for c in cfg.creators:
        try:
            ids = latest_video_ids(c, cfg.max_video_age_h)
        except Exception as e:  # network / bad handle: keep going with other creators
            log.warning("feed failed for %s: %s", c.name, e)
            continue
        for vid in ids:
            if not seen(vid):
                found.append((c, vid))
    return found
