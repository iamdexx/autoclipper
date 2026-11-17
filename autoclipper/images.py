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


_NOT_A_PHOTO = re.compile(
    r"\b(map|plan|drawing|diagram|chart|logo|icon|flag|seal|coat of arms|blueprint|sketch|"
    r"engraving|scan|page|pdf|svg|locator|schematic|poster|cover|screenshot)\b",
    re.I,
)


def photo(title: str) -> bool:
    """False for file names that say the image is a document or a graphic, not a photograph."""
    return not _NOT_A_PHOTO.search(title)


def wikipedia_lead(query: str, n: int = 3) -> list[tuple[str, str]]:
    """[(article title, url)] - the freely licensed lead photo of the Wikipedia articles that
    best match the query: picked by editors to show the subject, unlike a file-name search."""
    r = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": n,
            "prop": "pageimages",
            "piprop": "thumbnail|name",
            "pithumbsize": 1600,
            "pilicense": "free",
        },
        headers=_UA,
        timeout=30,
    )
    r.raise_for_status()
    pages = sorted(
        (r.json().get("query") or {}).get("pages", {}).values(), key=lambda p: p.get("index", 99)
    )
    return [
        (p.get("title", ""), p["thumbnail"]["source"])
        for p in pages
        if p.get("thumbnail")
        and int(p["thumbnail"].get("width") or 0) >= 600
        and photo(p.get("pageimage", ""))
    ]


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


def fetch(query: str, name: str, dest: Path, thumbs: bool = True) -> Path | None:
    """One relevant image for `query` (entry `name` also counts as a match), cached on disk.
    `thumbs=False`: real photos only, never a YouTube thumbnail."""
    dest.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{name}|{query}".encode()).hexdigest()[:8]
    out = dest / f"{key}.jpg"
    if out.exists() and is_image(out):
        return out
    for source in (commons, youtube_thumbs) if thumbs else (wikipedia_lead, commons):
        try:
            cands = source(query)
        except Exception as e:
            log.warning("%s search failed for %r: %s", source.__name__, query, e)
            continue
        for title, url in cands:
            if not (matches(query, title) or matches(name, title)):
                continue
            if not thumbs and not photo(title):
                continue
            if _download(url, out):
                log.info("image for %r: %s", name, title)
                return out
    if out.exists():
        out.unlink()
    return None
