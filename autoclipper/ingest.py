"""Find new long-form videos from watched creators, download them, transcribe with word timings."""

from __future__ import annotations

import atexit
import functools
import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .config import Config, Creator

log = logging.getLogger(__name__)

FEED_DEPTH = 8  # newest uploads looked at per creator per run


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
            "playlist:channel_id",  # channel-level field; a bare `channel_id` is per video
            f"https://www.youtube.com/{handle}/videos",
        ),
        text=True,
    ).strip()
    if not out.startswith("UC"):
        raise ValueError(f"could not resolve {handle} to a channel id: {out[:200]!r}")
    cid = out.splitlines()[0]
    creator.channel_id = cid
    return cid


def latest_uploads(creator: Creator, limit: int = FEED_DEPTH) -> list[tuple[str, float | None]]:
    """Newest first: (video id, publish time as epoch). YouTube's RSS feed used to serve this but
    answers 404 now, so: the Data API's uploads playlist when the channel's OAuth token is around
    (1 quota unit, exact dates), else yt-dlp's flat listing of the uploads tab, which has no dates
    (None - see upload_times)."""
    cid = resolve_channel_id(creator)
    try:
        return _uploads_via_api(cid, limit)
    except Exception as e:  # no token / quota / API down: yt-dlp still works
        log.info("%s: Data API unavailable (%s); listing uploads with yt-dlp", creator.name, e)
    return [(v, None) for v in latest_video_ids(cid, limit)]


def _uploads_via_api(cid: str, limit: int) -> list[tuple[str, float | None]]:
    from googleapiclient.discovery import build

    from .publish import youtube_auth

    yt = build("youtube", "v3", credentials=youtube_auth(interactive=False), cache_discovery=False)
    res = (
        yt.playlistItems()
        .list(part="contentDetails", playlistId="UU" + cid[2:], maxResults=min(limit, 50))
        .execute()
    )
    out = []
    for it in res.get("items", []):
        cd = it["contentDetails"]
        ts = None
        if at := cd.get("videoPublishedAt"):
            ts = datetime.fromisoformat(at.replace("Z", "+00:00")).timestamp()
        out.append((cd["videoId"], ts))
    return out


def latest_video_ids(cid: str, limit: int = FEED_DEPTH) -> list[str]:
    """Newest first, off the channel's uploads tab (same cookies/proxy as every other YouTube call)."""
    out = subprocess.check_output(
        ytdlp(
            "--flat-playlist",
            "--playlist-items",
            f"1:{limit}",
            "--print",
            "id",
            f"https://www.youtube.com/channel/{cid}/videos",
        ),
        text=True,
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def upload_times(ids: list[str]) -> dict[str, float]:
    """Publish time (epoch) per video id, one yt-dlp call for the lot (a full extraction each, so
    only for ids the flat listing left undated). Videos that fail to extract are simply absent."""
    if not ids:
        return {}
    res = subprocess.run(
        ytdlp(
            "--ignore-errors",
            "--no-playlist",
            "--print",
            "%(id)s %(timestamp)s",
            *(f"https://www.youtube.com/watch?v={i}" for i in ids),
        ),
        text=True,
        capture_output=True,
    )
    out: dict[str, float] = {}
    for line in res.stdout.splitlines():
        vid, _, ts = line.strip().partition(" ")
        if vid in ids and ts.replace(".", "", 1).isdigit():
            out[vid] = float(ts)
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
    """Unseen uploads no older than max_video_age_h, newest first per creator - so the first run on
    a channel doesn't work through its back catalogue."""
    found: list[tuple[Creator, str]] = []
    cutoff = time.time() - cfg.max_video_age_h * 3600
    for c in cfg.creators:
        try:
            times = {v: ts for v, ts in latest_uploads(c) if not seen(v)}
            if undated := [v for v, ts in times.items() if ts is None]:
                times.update(upload_times(undated))
        except Exception as e:  # network / bad handle: keep going with other creators
            log.warning("feed failed for %s: %s", c.name, e)
            continue
        fresh = [v for v, ts in times.items() if (ts or 0) >= cutoff]
        log.info(
            "%s: %d unseen uploads, %d within %gh",
            c.name,
            len(times),
            len(fresh),
            cfg.max_video_age_h,
        )
        found.extend((c, v) for v in fresh)
    return found
