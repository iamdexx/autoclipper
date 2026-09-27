"""Find new long-form videos from watched creators, download them, transcribe with word timings."""

from __future__ import annotations

import atexit
import calendar
import functools
import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import feedparser

from .config import Config, Creator

log = logging.getLogger(__name__)

YT_FEED = "https://www.youtube.com/feeds/videos.xml?channel_id={cid}"


_cookie_tmp: Path | None = None


def _cookies_file() -> str | None:
    """YouTube answers datacenter IPs (GitHub runners) with 'sign in to confirm you're not a bot';
    a logged-in browser's cookies get past it. YT_COOKIES_FILE = path to a Netscape cookies.txt,
    or YT_COOKIES = its contents (the form a repo secret takes), written once to a private temp
    file that goes away with the process."""
    global _cookie_tmp
    path = os.environ.get("YT_COOKIES_FILE")
    if path and Path(path).is_file():
        return path
    blob = os.environ.get("YT_COOKIES", "").strip()
    if not blob:
        return None
    if _cookie_tmp is None or not _cookie_tmp.exists():
        fd, name = tempfile.mkstemp(suffix=".txt", prefix="yt-cookies-")
        with os.fdopen(fd, "w") as f:  # mkstemp => 0600, owner only
            f.write(blob + "\n")
        _cookie_tmp = Path(name)
        atexit.register(lambda p=_cookie_tmp: p.unlink(missing_ok=True))
    return str(_cookie_tmp)


def ytdlp(*args: str) -> list[str]:
    """`yt-dlp` plus the env-driven access options every call needs (cookies, proxy)."""
    cmd = ["yt-dlp"]
    if cookies := _cookies_file():
        cmd += ["--cookies", cookies]
    if proxy := os.environ.get("YT_PROXY"):
        cmd += ["--proxy", proxy]
    return cmd + list(args)


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
        ytdlp(
            "--playlist-items",
            "0",
            "--print",
            "channel_id",
            f"https://www.youtube.com/{handle}/videos",
        ),
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
    out = subprocess.check_output(ytdlp("-J", "--no-playlist", url), text=True)
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
        ytdlp(
            "--no-playlist",
            "-f",
            f"bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]/b[height<={max_height}][ext=mp4]/b",
            "--merge-output-format",
            "mp4",
            "-o",
            str(out),
            url,
        )
    )
    return out


@functools.lru_cache(maxsize=1)
def _whisper(model: str, compute: str):
    from faster_whisper import WhisperModel

    return WhisperModel(model, device="cpu", compute_type=compute)


def transcribe(video: Path, cfg: Config, cache: Path) -> list[Segment]:
    if cache.exists():
        raw = json.loads(cache.read_text())
        return [
            Segment(s["start"], s["end"], s["text"], [Word(**w) for w in s["words"]]) for s in raw
        ]

    log.info("transcribing %s with whisper-%s", video.name, cfg.whisper_model)
    segs, _ = _whisper(cfg.whisper_model, cfg.whisper_compute).transcribe(
        str(video), word_timestamps=True, vad_filter=True
    )
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
