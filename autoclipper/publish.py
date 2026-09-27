"""Upload a rendered clip to YouTube Shorts, TikTok and Instagram Reels.

Credentials (all via environment variables):
  YouTube   : YT_CLIENT_SECRETS (path to OAuth client json) + YT_TOKEN (path where the refresh
              token is cached; created by `autoclipper auth youtube`). On a runner with no disk
              (GitHub Actions) put the token file's contents in YT_TOKEN_JSON instead.
  TikTok    : TIKTOK_CLIENT_KEY + TIKTOK_CLIENT_SECRET + TIKTOK_REFRESH_TOKEN (from
              `autoclipper auth tiktok`; access tokens only live 24h so the refresh token is what
              you store) - or a raw short-lived TIKTOK_ACCESS_TOKEN for a one-off.
  Instagram : IG_ACCESS_TOKEN + IG_USER_ID (Graph API, instagram_content_publish) and
              PUBLIC_BASE_URL - Instagram fetches the video from a URL. Set PUBLIC_S3_BUCKET (+
              PUBLIC_S3_ENDPOINT / PUBLIC_S3_KEY / PUBLIC_S3_SECRET) and each video is uploaded
              to that S3-compatible bucket (Cloudflare R2, DigitalOcean Spaces, AWS) before
              posting; otherwise it is copied to `<workdir>/public/`, which you serve yourself.
"""

from __future__ import annotations

import logging
import os
import secrets
import shutil
import time
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import requests

from .config import Config

log = logging.getLogger(__name__)

YT_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",  # `auth check`: channels.list(mine=True)
]


class PublishError(RuntimeError):
    pass


# ---------------- YouTube ----------------


def youtube_auth(interactive: bool = True, manual: bool = False):
    """`manual`: the browser is on another machine - print the consent URL, then take the URL the
    browser lands on afterwards (http://localhost:.../?code=...) pasted back on stdin."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    token_path = Path(os.getenv("YT_TOKEN", "secrets/yt_token.json"))
    if os.getenv("YT_TOKEN_JSON") and not token_path.exists():
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(os.environ["YT_TOKEN_JSON"])
    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), YT_SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    elif not creds or not creds.valid:
        if not interactive:
            raise PublishError("YouTube not authorised: run `autoclipper auth youtube`")
        client = os.getenv("YT_CLIENT_SECRETS", "secrets/yt_client_secret.json")
        flow = InstalledAppFlow.from_client_secrets_file(client, YT_SCOPES)
        if manual:
            os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")  # loopback http redirect
            flow.redirect_uri = "http://localhost:8765/"
            url, _ = flow.authorization_url(access_type="offline", prompt="consent")
            print(f"Open this in a browser and sign in:\n\n{url}\n")
            landed = input(
                "Paste the full URL the browser ends up on (starts http://localhost:8765/): "
            )
            flow.fetch_token(authorization_response=landed.strip())
            creds = flow.credentials
        else:
            creds = flow.run_local_server(port=0, open_browser=not os.getenv("HEADLESS"))
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
TT_SCOPES = "user.info.basic,video.publish"


def tiktok_token() -> str:
    """Access token for this run: refreshed from TIKTOK_REFRESH_TOKEN when the app credentials
    are present (access tokens expire after 24h, refresh tokens after a year), else the raw one."""
    key, sec, ref = (
        os.getenv("TIKTOK_CLIENT_KEY"),
        os.getenv("TIKTOK_CLIENT_SECRET"),
        os.getenv("TIKTOK_REFRESH_TOKEN"),
    )
    if key and sec and ref:
        r = requests.post(
            f"{TT}/oauth/token/",
            data={
                "client_key": key,
                "client_secret": sec,
                "grant_type": "refresh_token",
                "refresh_token": ref,
            },
            timeout=30,
        ).json()
        if "access_token" not in r:
            raise PublishError(f"tiktok token refresh failed: {r}")
        return r["access_token"]
    if os.getenv("TIKTOK_ACCESS_TOKEN"):
        return os.environ["TIKTOK_ACCESS_TOKEN"]
    raise PublishError("TikTok not authorised: run `autoclipper auth tiktok`")


def tiktok_auth(redirect_uri: str) -> dict:
    """One-time browser login (Login Kit). Prints the refresh token to store as
    TIKTOK_REFRESH_TOKEN. TikTok only redirects to the URIs registered on the app, so the
    user pastes the URL they land on back here - works with any registered URI."""
    key, sec = os.environ["TIKTOK_CLIENT_KEY"], os.environ["TIKTOK_CLIENT_SECRET"]
    state = secrets.token_urlsafe(12)
    url = "https://www.tiktok.com/v2/auth/authorize/?" + urlencode(
        {
            "client_key": key,
            "scope": TT_SCOPES,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "state": state,
        }
    )
    print(f"Open this URL and log in to the TikTok account that should post:\n\n{url}\n")
    if not os.getenv("HEADLESS"):
        webbrowser.open(url)
    landed = input("Paste the full URL you were redirected to (or just the code): ").strip()
    q = parse_qs(urlparse(landed).query) if "?" in landed else {"code": [landed]}
    if q.get("state", [state])[0] != state:
        raise PublishError("state mismatch - start over")
    r = requests.post(
        f"{TT}/oauth/token/",
        data={
            "client_key": key,
            "client_secret": sec,
            "code": q["code"][0],
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
        },
        timeout=30,
    ).json()
    if "refresh_token" not in r:
        raise PublishError(f"tiktok code exchange failed: {r}")
    return r


def tiktok_creator_info(token: str) -> dict:
    """Required by TikTok before every post; also tells us which privacy levels the account
    may use (unaudited apps are limited to SELF_ONLY) and the max clip length."""
    h = {"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=UTF-8"}
    r = requests.post(f"{TT}/post/publish/creator_info/query/", headers=h, json={}, timeout=30)
    j = r.json()
    if r.status_code != 200 or j.get("error", {}).get("code") not in (None, "ok"):
        raise PublishError(f"tiktok creator_info failed: {j}")
    return j["data"]


def tiktok_privacy(info: dict) -> str:
    want = os.getenv("TIKTOK_PRIVACY", "PUBLIC_TO_EVERYONE")
    allowed = info.get("privacy_level_options") or [want]
    if want in allowed:
        return want
    log.warning("tiktok: %s not allowed for this account/app, posting as %s", want, allowed[0])
    return allowed[0]


def publish_tiktok(video: Path, caption: str) -> dict:
    token = tiktok_token()
    h = {"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=UTF-8"}
    info = tiktok_creator_info(token)
    size = video.stat().st_size
    chunk = min(size, 64 * 1024 * 1024)
    init = requests.post(
        f"{TT}/post/publish/video/init/",
        headers=h,
        json={
            "post_info": {
                "title": caption[:2200],
                "privacy_level": tiktok_privacy(info),
                "disable_duet": bool(info.get("duet_disabled")),
                "disable_comment": bool(info.get("comment_disabled")),
                "disable_stitch": bool(info.get("stitch_disabled")),
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


def host_public(video: Path, public_dir: Path) -> str:
    """Make the file reachable at a public URL (Instagram downloads it from there).
    With PUBLIC_S3_BUCKET set the file is uploaded to an S3-compatible bucket (R2 / Spaces / S3);
    otherwise it is copied to public_dir for a web server you run."""
    base = os.environ["PUBLIC_BASE_URL"].rstrip("/")
    bucket = os.getenv("PUBLIC_S3_BUCKET")
    if bucket:
        import boto3

        s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("PUBLIC_S3_ENDPOINT") or None,
            aws_access_key_id=os.getenv("PUBLIC_S3_KEY"),
            aws_secret_access_key=os.getenv("PUBLIC_S3_SECRET"),
            region_name=os.getenv("PUBLIC_S3_REGION", "auto"),
        )
        s3.upload_file(str(video), bucket, video.name, ExtraArgs={"ContentType": "video/mp4"})
        log.info("uploaded %s to s3://%s", video.name, bucket)
    else:
        public_dir.mkdir(parents=True, exist_ok=True)
        served = public_dir / video.name
        if not served.exists():
            shutil.copy(video, served)
    return f"{base}/{video.name}"


def publish_instagram(video: Path, caption: str, public_dir: Path) -> dict:
    token = os.environ["IG_ACCESS_TOKEN"]
    user = os.environ["IG_USER_ID"]
    video_url = host_public(video, public_dir)

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


# ---------------- credential check ----------------


def check_credentials(cfg: Config) -> dict[str, str]:
    """Talk to each enabled platform with the configured credentials and report the account
    we would post as (or the error). Nothing is posted."""
    p = cfg.publishing
    out: dict[str, str] = {}
    if p.youtube:
        try:
            from googleapiclient.discovery import build

            yt = build("youtube", "v3", credentials=youtube_auth(interactive=False))
            ch = yt.channels().list(part="snippet", mine=True).execute()
            items = ch.get("items") or [{}]
            out["youtube"] = "ok: " + items[0].get("snippet", {}).get("title", "(no channel)")
        except Exception as e:
            out["youtube"] = f"FAIL: {e}"
    if p.tiktok:
        try:
            info = tiktok_creator_info(tiktok_token())
            out["tiktok"] = (
                f"ok: @{info.get('creator_username', '?')} "
                f"privacy={tiktok_privacy(info)} allowed={info.get('privacy_level_options')} "
                f"max={info.get('max_video_post_duration_sec')}s"
            )
        except Exception as e:
            out["tiktok"] = f"FAIL: {e}"
    if p.instagram:
        try:
            r = requests.get(
                f"{IG}/{os.environ['IG_USER_ID']}",
                params={"fields": "username", "access_token": os.environ["IG_ACCESS_TOKEN"]},
                timeout=30,
            ).json()
            if "username" not in r:
                raise PublishError(str(r))
            base = os.environ["PUBLIC_BASE_URL"]
            how = (
                "s3 bucket " + os.environ["PUBLIC_S3_BUCKET"]
                if os.getenv("PUBLIC_S3_BUCKET")
                else "public dir (you serve it)"
            )
            out["instagram"] = f"ok: @{r['username']} hosting={how} at {base}"
        except Exception as e:
            out["instagram"] = f"FAIL: {e}"
    return out


# ---------------- dispatcher ----------------


def publish_all(
    video: Path,
    hook: str,
    creator: str,
    hashtags: list[str],
    cfg: Config,
    kind: str = "clip",
    only: str = "",
    force: bool = False,
) -> dict[str, dict]:
    """kind: 'clip' (creator/trend clip) or 'top10' (own countdown video) - picks the templates.
    only: restrict to one platform; force: post even when publishing.dry_run is on."""
    p = cfg.publishing
    tags = list(dict.fromkeys(p.default_hashtags + hashtags))
    if kind == "top10":
        title_t, caption_t = p.top10_title_template, p.top10_caption_template
    else:
        title_t, caption_t = p.title_template, p.caption_template
    title = title_t.format(hook=hook, creator=creator)
    caption = caption_t.format(hook=hook, creator=creator, hashtags=" ".join(tags))
    results: dict[str, dict] = {}
    targets = [("youtube", p.youtube), ("tiktok", p.tiktok), ("instagram", p.instagram)]
    for name, enabled in targets:
        if not enabled or (only and name != only):
            continue
        if p.dry_run and not force:
            log.info("[dry-run] would post %s to %s: %s", video.name, name, title)
            results[name] = {"dry_run": True, "title": title, "caption": caption}
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
