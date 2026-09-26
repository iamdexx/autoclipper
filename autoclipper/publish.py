"""Upload a rendered clip to YouTube Shorts, TikTok and Instagram Reels.

Credentials (all via environment variables):
  YouTube   : YT_CLIENT_SECRETS (path to OAuth client json) + YT_TOKEN (path where the refresh
              token is cached; created by `autoclipper auth youtube`).
  TikTok    : TIKTOK_ACCESS_TOKEN  (Content Posting API, scope video.publish).
  Instagram : IG_ACCESS_TOKEN + IG_USER_ID (Graph API, instagram_content_publish) and
              PUBLIC_BASE_URL - Instagram fetches the video from a URL, so clips are served from
              the `public/` dir (see server.py) or you set PUBLIC_BASE_URL to a bucket/CDN.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path

import requests

from .config import Config

log = logging.getLogger(__name__)

YT_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


class PublishError(RuntimeError):
    pass


# ---------------- YouTube ----------------


def youtube_auth(interactive: bool = True):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    token_path = Path(os.getenv("YT_TOKEN", "secrets/yt_token.json"))
    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), YT_SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    elif not creds or not creds.valid:
        if not interactive:
            raise PublishError("YouTube not authorised: run `autoclipper auth youtube`")
        secrets = os.getenv("YT_CLIENT_SECRETS", "secrets/yt_client_secret.json")
        flow = InstalledAppFlow.from_client_secrets_file(secrets, YT_SCOPES)
        creds = flow.run_console() if os.getenv("HEADLESS") else flow.run_local_server(port=0)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json())
    return creds


def publish_youtube(video: Path, title: str, description: str, tags: list[str]) -> dict:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    yt = build("youtube", "v3", credentials=youtube_auth(interactive=False))
    body = {
        "snippet": {
            "title": title[:100],
            "description": description[:5000],
            "tags": [t.lstrip("#") for t in tags][:30],
            "categoryId": "24",
        },
        "status": {
            "privacyStatus": os.getenv("YT_PRIVACY", "public"),
            "selfDeclaredMadeForKids": False,
        },
    }
    media = MediaFileUpload(
        str(video), chunksize=8 * 1024 * 1024, resumable=True, mimetype="video/mp4"
    )
    req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = None
    while resp is None:
        _, resp = req.next_chunk()
    return {"id": resp["id"], "url": f"https://youtube.com/shorts/{resp['id']}"}


# ---------------- TikTok ----------------

TT = "https://open.tiktokapis.com/v2"


def publish_tiktok(video: Path, caption: str) -> dict:
    token = os.environ["TIKTOK_ACCESS_TOKEN"]
    h = {"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=UTF-8"}
    size = video.stat().st_size
    chunk = min(size, 64 * 1024 * 1024)
    init = requests.post(
        f"{TT}/post/publish/video/init/",
        headers=h,
        json={
            "post_info": {
                "title": caption[:2200],
                "privacy_level": os.getenv("TIKTOK_PRIVACY", "PUBLIC_TO_EVERYONE"),
                "disable_duet": False,
                "disable_comment": False,
                "disable_stitch": False,
            },
            "source_info": {
                "source": "FILE_UPLOAD",
                "video_size": size,
                "chunk_size": chunk,
                "total_chunk_count": -(-size // chunk),
            },
        },
        timeout=60,
    )
    j = init.json()
    if init.status_code != 200 or j.get("error", {}).get("code") not in (None, "ok"):
        raise PublishError(f"tiktok init failed: {j}")
    upload_url = j["data"]["upload_url"]
    publish_id = j["data"]["publish_id"]
    with video.open("rb") as f:
        off = 0
        while off < size:
            data = f.read(chunk)
            end = off + len(data) - 1
            r = requests.put(
                upload_url,
                headers={
                    "Content-Type": "video/mp4",
                    "Content-Length": str(len(data)),
                    "Content-Range": f"bytes {off}-{end}/{size}",
                },
                data=data,
                timeout=600,
            )
            if r.status_code not in (200, 201, 206):
                raise PublishError(f"tiktok chunk upload failed: {r.status_code} {r.text[:200]}")
            off += len(data)
    # poll status
    for _ in range(30):
        s = requests.post(
            f"{TT}/post/publish/status/fetch/",
            headers=h,
            json={"publish_id": publish_id},
            timeout=30,
        ).json()
        st = s.get("data", {}).get("status")
        if st in ("PUBLISH_COMPLETE", "SEND_TO_USER_INBOX"):
            return {"id": publish_id, "status": st}
        if st == "FAILED":
            raise PublishError(f"tiktok publish failed: {s}")
        time.sleep(10)
    return {"id": publish_id, "status": "PROCESSING"}


# ---------------- Instagram Reels ----------------

IG = "https://graph.facebook.com/v21.0"


def publish_instagram(video: Path, caption: str, public_dir: Path) -> dict:
    token = os.environ["IG_ACCESS_TOKEN"]
    user = os.environ["IG_USER_ID"]
    base = os.environ["PUBLIC_BASE_URL"].rstrip("/")
    public_dir.mkdir(parents=True, exist_ok=True)
    served = public_dir / video.name
    if not served.exists():
        shutil.copy(video, served)
    video_url = f"{base}/{video.name}"

    r = requests.post(
        f"{IG}/{user}/media",
        data={
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption[:2200],
            "share_to_feed": "true",
            "access_token": token,
        },
        timeout=60,
    ).json()
    if "id" not in r:
        raise PublishError(f"instagram container failed: {r}")
    cid = r["id"]
    for _ in range(60):
        st = requests.get(
            f"{IG}/{cid}",
            params={"fields": "status_code,status", "access_token": token},
            timeout=30,
        ).json()
        code = st.get("status_code")
        if code == "FINISHED":
            break
        if code == "ERROR":
            raise PublishError(f"instagram processing failed: {st}")
        time.sleep(10)
    pub = requests.post(
        f"{IG}/{user}/media_publish", data={"creation_id": cid, "access_token": token}, timeout=60
    ).json()
    if "id" not in pub:
        raise PublishError(f"instagram publish failed: {pub}")
    return {"id": pub["id"]}


# ---------------- dispatcher ----------------


def publish_all(
    video: Path, hook: str, creator: str, hashtags: list[str], cfg: Config
) -> dict[str, dict]:
    p = cfg.publishing
    tags = list(dict.fromkeys(p.default_hashtags + hashtags))
    title = p.title_template.format(hook=hook, creator=creator)
    caption = p.caption_template.format(hook=hook, creator=creator, hashtags=" ".join(tags))
    results: dict[str, dict] = {}
    targets = [("youtube", p.youtube), ("tiktok", p.tiktok), ("instagram", p.instagram)]
    for name, enabled in targets:
        if not enabled:
            continue
        if p.dry_run:
            log.info("[dry-run] would post %s to %s: %s", video.name, name, title)
            results[name] = {"dry_run": True, "title": title}
            continue
        try:
            if name == "youtube":
                results[name] = publish_youtube(video, title, caption, tags)
            elif name == "tiktok":
                results[name] = publish_tiktok(video, caption)
            else:
                results[name] = publish_instagram(video, caption, cfg.workdir / "public")
            log.info("posted %s to %s: %s", video.name, name, results[name])
        except Exception as e:
            log.error("publish to %s failed: %s", name, e)
            results[name] = {"error": str(e)}
    return results
