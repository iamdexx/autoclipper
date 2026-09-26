# autoclipper

The "clipping side hustle" from the video, fully automated:

```
watch creators' YouTube channels
  -> new stream/VOD appears
  -> download (yt-dlp) + transcribe with word timestamps (faster-whisper)
  -> LLM (Gemini / Claude / OpenAI) picks the moments most likely to go viral
  -> ffmpeg cuts each moment, reframes to 9:16 (blurred background), burns karaoke captions + hook title
  -> auto-posts to YouTube Shorts, TikTok and Instagram Reels
```

No Everygen subscription, no Claude desktop, no manual downloading/posting.

## Quick start

```bash
sudo apt install ffmpeg
python -m venv .venv && . .venv/bin/activate
pip install -e .
export GEMINI_API_KEY=...            # free tier at https://aistudio.google.com/apikey (or ANTHROPIC_API_KEY / OPENAI_API_KEY)

# exactly what the video does by hand, for one stream:
autoclipper clip "https://www.youtube.com/watch?v=VIDEO_ID" --creator "Adin Ross"
#  -> data/videos/VIDEO_ID/clips/01-<hook>.mp4 ...

# the automatic version:
autoclipper run     # one pass over every creator in config.yaml
autoclipper watch   # loop forever (poll_interval_s)
```

Edit `config.yaml` to add creators (handle or channel_id), clip counts/lengths, caption style.
With no LLM key set, a heuristic ranker (speech density, hype words, exclamations) is used.

## Publishing

Everything is `dry_run: true` by default: clips render and the log shows what *would* be posted.
Flip `publishing.dry_run: false` and set the credentials for the platforms you want:

| Platform | Env vars | How to get |
| --- | --- | --- |
| YouTube Shorts | `YT_CLIENT_SECRETS` (OAuth desktop client json), `YT_TOKEN` (cache path) | Google Cloud console -> enable YouTube Data API v3 -> OAuth client (Desktop). Then `autoclipper auth youtube` once. |
| TikTok | `TIKTOK_ACCESS_TOKEN` | developers.tiktok.com app with Content Posting API, scope `video.publish`. Unaudited apps can only post privately (`TIKTOK_PRIVACY=SELF_ONLY`). |
| Instagram Reels | `IG_ACCESS_TOKEN`, `IG_USER_ID`, `PUBLIC_BASE_URL` | Meta app with `instagram_content_publish`, IG Business/Creator account linked to a Page. Instagram pulls the file from a URL: serve `data/public/` (e.g. `python -m http.server` behind your domain, or a bucket) and point `PUBLIC_BASE_URL` at it. |

Posts and processed videos are tracked in `data/state.json` (`autoclipper status`).

## Running it 24/7

**Docker / any VPS**

```bash
docker build -t autoclipper .
docker run -d --restart=always --env-file .env -v $PWD/data:/app/data -v $PWD/config.yaml:/app/config.yaml autoclipper
```

**systemd** — see `deploy/autoclipper.service`.

**GitHub Actions** — `.github/workflows/clip.yml` runs `autoclipper run` every 30 min with the
repo secrets `GEMINI_API_KEY`, `TIKTOK_ACCESS_TOKEN`, etc. and caches `data/state.json` between
runs. Fine for a couple of creators; use a VPS for heavy volume (whisper on a 3h stream takes a
while on CPU — set `whisper_model: tiny` there or run on a GPU box).

## Layout

- `autoclipper/ingest.py` — channel feeds, yt-dlp download, whisper transcript
- `autoclipper/rank.py` — candidate windows, LLM prompt + parsing, heuristic fallback
- `autoclipper/render.py` — ffmpeg reframe + ASS caption generation
- `autoclipper/publish.py` — YouTube / TikTok / Instagram uploaders
- `autoclipper/pipeline.py` — glue; `autoclipper/cli.py` — commands
