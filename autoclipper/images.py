"""Still images for listicle videos, keyless: Wikimedia Commons search first (real photos of
people/places/things), then the thumbnail of the best-matching YouTube video (great for
sports/news/drama where Commons has nothing). Every pick must share keywords with the entry."""

from __future__ import annotations

import hashlib
import logging
import re
import subprocess
from pathlib import Path

import requests

from . import trends

log = logging.getLogger(__name__)

_UA = {"User-Agent": "autoclipper/0.1 (https://github.com/iamdexx/autoclipper)"}
_COMMONS = "https://commons.wikimedia.org/w/api.php"
_STOP = set(
    "the a an of in on at to and or vs v for with his her their its this that from by photo "
    "image picture".split()
)


def _keywords(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2 and w not in _STOP}


def matches(query: str, title: str) -> bool:
    q, t = _keywords(query), _keywords(title)
    return bool(q) and len(q & t) >= max(1, (len(q) + 1) // 2)


def commons(query: str, n: int = 8) -> list[tuple[str, str]]:
    """[(title, url)] of landscape-ish bitmap files matching the query."""
    r = requests.get(
        _COMMONS,
        params={
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": f"{query} filetype:bitmap",
            "gsrnamespace": 6,
            "gsrlimit": n,
            "prop": "imageinfo",
            "iiprop": "url|size|mime",
            "iiurlwidth": 1600,
        },
        headers=_UA,
        timeout=30,
    )
    r.raise_for_status()
    out = []
    for p in (r.json().get("query") or {}).get("pages", {}).values():
        for ii in p.get("imageinfo") or []:
            if ii.get("mime") not in ("image/jpeg", "image/png"):
                continue
            if int(ii.get("width") or 0) < 600:
                continue
            out.append((p.get("title", ""), ii.get("thumburl") or ii["url"]))
    return out


def youtube_thumbs(query: str, n: int = 8) -> list[tuple[str, str]]:
    return [
        (c.title, f"https://i.ytimg.com/vi/{c.id}/maxresdefault.jpg")
        for c in trends.yt_search(query, n)
    ]


def _download(url: str, out: Path) -> bool:
    try:
        r = requests.get(url, headers=_UA, timeout=60)
        if r.status_code == 404 and "maxresdefault" in url:
            r = requests.get(url.replace("maxresdefault", "hqdefault"), headers=_UA, timeout=60)
        r.raise_for_status()
        out.write_bytes(r.content)
    except requests.RequestException as e:
        log.warning("image download failed %s: %s", url, e)
        return False
    return is_image(out)


def is_image(path: Path) -> bool:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0",
             str(path)],
            text=True,
            timeout=30,
        )  # fmt: skip
    except Exception:
        return False
    try:
        w, h = (int(x) for x in out.strip().split(",")[:2])
    except ValueError:
        return False
    return w >= 320 and h >= 240


def fetch(query: str, name: str, dest: Path) -> Path | None:
    """One relevant image for `query` (entry `name` also counts as a match), cached on disk."""
    dest.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{name}|{query}".encode()).hexdigest()[:8]
    out = dest / f"{key}.jpg"
    if out.exists() and is_image(out):
        return out
    for source in (commons, youtube_thumbs):
        try:
            cands = source(query)
        except Exception as e:
            log.warning("%s search failed for %r: %s", source.__name__, query, e)
            continue
        for title, url in cands:
            if not (matches(query, title) or matches(name, title)):
                continue
            if _download(url, out):
                log.info("image for %r: %s", name, title)
                return out
    if out.exists():
        out.unlink()
    return None
