# autoclipper

The "clipping side hustle" from the video, fully automated:

```
watch creators' YouTube channels
  -> new stream/VOD appears
  -> download (yt-dlp) + transcribe with word timestamps (faster-whisper)
  -> LLM (Gemini / Claude / OpenAI) picks the moments most likely to go viral
  -> ffmpeg cuts each moment, reframes to 9:16 (blurred background), burns karaoke captions + hook title
  -> a natural TTS narrator speaks a 1-2 sentence hook over the intro, stream audio ducked underneath
  -> auto-posts to YouTube Shorts, TikTok and Instagram Reels
```

No Everygen subscription, no Claude desktop, no manual downloading/posting.

Three content modes run side by side on every pass (`autoclipper run`):

| mode | on when | what it does |
|---|---|---|
| creators | `creators:` non-empty | the flow above for every new upload/VOD of the listed channels |
| trends | `trends.enabled` | scrapes what's blowing up, picks the best source video, clips it like a stream |
| top10 | `top10.enabled` | writes a "Top 10 ..." countdown, pulls footage per entry, narrates the whole thing |

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

## Narration

Each clip opens with an AI narrator (the LLM writes the spoken hook, e.g. *"Imagine getting kicked
out of the company you built from your garage."*). Voices are real neural TTS, not robotic:

| provider (`narration.provider`) | key | default voice |
|---|---|---|
| `azure` | `AZURE_SPEECH_KEY` + `AZURE_SPEECH_REGION` (Speech resource, Free F0 = 500k chars/month) | `en-US-AndrewMultilingualNeural` — also Brian, Davis, Guy, Steffan, Christopher |
| `google` | `GOOGLE_TTS_API_KEY` (Cloud Text-to-Speech, Chirp 3 HD, 1M chars/month free) | `Charon` — also Fenrir, Orus, Puck, Algenib, Iapetus |
| `elevenlabs` | `ELEVENLABS_API_KEY` (most natural, 10k chars/month free) | Liam |
| `gemini` | `GEMINI_API_KEY` (same key as ranking, free tier) | `Puck` — also try Kore, Charon, Fenrir, Aoede, Leda |
| `openai` | `OPENAI_API_KEY` | onyx |

`auto` tries Azure > Google > ElevenLabs > Gemini > OpenAI, moving on to the next one whenever a
call fails (e.g. a monthly quota running out mid-month), so the cron never goes silent. Word
timings for the follow-the-word captions come from ElevenLabs directly; every other provider's
audio is run through Whisper and aligned to the script so the highlight tracks the voice just the
same. Tune `duck_db`, `max_len_s`, or set `provider: off` in `config.yaml` to keep only the
creator's own audio.

## Trends

```
Google Trends daily RSS (per geo)  +  most-viewed YouTube uploads of the last 24h per niche
  -> LLM picks `topics_per_run` topics that fit your niches + writes the YouTube query/angle/hashtags
  -> yt-dlp search, LLM picks the best source video (duration-filtered)
  -> normal clip pipeline (transcribe, rank, cut, caption, narrate, publish)
```

Everything is keyless; a failing source is skipped. Covered topics are remembered in
`state.json` so the same trend isn't clipped twice. `autoclipper trends` prints what's hot and what
would be picked; add `--clip` to actually make the clips.

## Top 10 videos

`autoclipper top10 "craziest NFL plays this week"` (or no topic: picked from trends /
`top10.topics`) makes one vertical ranked video in one of three styles (`top10.style`):

- **`ranking`** (default) - the "Ranking The Most Insane X Moments" meme format, ~20-30s:
  `top10.items` (default 6) real clips back to back with their **own audio**, a colour-per-word
  title banner + small tease line on top, a `1..N` list down the left that fills in with a 1-3
  word label as each clip plays (N first, 1 last), one big word on screen at a time. Liam voices
  a cold-open hook plus one line per clip with the clip audio ducked under him
  (`top10.source_db`); `top10.narrate: false` drops the narrator and captions the clip's own
  speech instead, audio left alone.
- **`countdown`** - "Top 10 ...": narrator (ElevenLabs word timestamps -> synced karaoke
  captions) voices every line and a cold-open intro, source audio ducked (`top10.source_db`),
  big `#10 ... #1` badge (pops in) + entry title on screen; `top10.cuts_per_entry` cuts per
  entry (use 2 here), each with a zoom punch.
- **`listicle`** - "N photos ...": one still per item (Wikimedia Commons, YouTube thumbnail
  fallback) with a slow zoom over a blurred fill, one word on screen at a time, narrator reads
  straight through (`top10.listicle_gap_s` between items), no clip audio.

Common to all:

- LLM writes title, ranked entries (spoken line, list label, footage/image query) - every
  claim goes through the fact-check gate below before anything is fetched or rendered
- footage: yt-dlp grabs a short section of the top YouTube result, ffmpeg reframes to 9:16
  (blurred fill); a whoosh/hit on every rank reveal (`top10.sfx`)
- segments are concatenated into one MP4 in `data/top10/<slug>/`, then one continuous music
  bed runs under the whole video start to finish (level `top10.music_db`). `top10.music:
  elevenlabs` (default) generates an instrumental per video with the ElevenLabs Music API
  (`top10.music_prompt`; your key needs the `music_generation` permission; cached in
  `data/assets/music/`). `top10.music_dir` = folder of your own tracks (random pick, looped/cut,
  faded; also the fallback if generation fails). `synth` = built-in ffmpeg pulse, `off` = none.

## Fact-check gate (`verify:`)

Nothing the narrator says goes out unchecked (`verify.py`):

- **Top 10 entries** - for each entry we pull real evidence (Google News headlines, Wikipedia,
  YouTube titles; no API keys) and the LLM judges the claim *only against that evidence*:
  `supported` / `unsupported` / `unverifiable` + confidence + note. `unsupported` entries are
  dropped (also `unverifiable` when `strict: true`), the script is rewritten to refill them
  (`max_rounds`), and the video is skipped if fewer than `min_entries` survive. Full report with
  sources: `data/top10/<slug>/factcheck.json`.
- **Footage** - a YouTube result is only used if its title actually names the entry (keyword
  overlap), so a random high-view video can't stand in for the narrated event.
- **Creator / trend clips** - the narrator's hook is checked against the clip's own transcript;
  an unfaithful hook is replaced with the checker's faithful rewrite or dropped
  (`data/videos/<id>/factcheck.json`).
- `autoclipper factcheck data/top10/<slug>/script.json` audits an existing script (exit 1 if
  anything fails); `autoclipper factcheck "some claim" --query "search terms"` checks one claim.

## Quality gate (`qa:`)

Every finished video is inspected before it can be published (`autoclipper qa <video>
[--script script.json | --expect "text"]` runs it by hand; the report is written to
`<video>.qa.json` either way):

- file: video + audio streams present, 1080x1920, duration inside the window, A/V lengths match
- audio: average level inside a window (too quiet = narration buried/missing), no clipping,
  no dead air longer than `max_silence_s`
- picture: no black stretches, no frozen frames
- speech: the output is re-transcribed with whisper and at least `min_speech_match` of the
  script's content words must be heard - catches missing/garbled narration and wrong audio

A failing video is kept on disk with its report, recorded in state as `qa_failed`, and never
published. `qa.enabled: false` turns the gate off.

## Publishing

Everything is `dry_run: true` by default: clips render and the log shows what *would* be posted.
Flip `publishing.dry_run: false` and set the credentials for the platforms you want. Only videos
that pass the fact-check and quality gates ever reach this step.

```bash
autoclipper auth check                  # talks to each platform, prints the account it would post as
autoclipper publish out.mp4 --title "..." --platform youtube   # one real post, ignores dry_run
```

| Platform | Env vars | How to get |
| --- | --- | --- |
| YouTube Shorts | `YT_CLIENT_SECRETS` (OAuth desktop client json), `YT_TOKEN` (cache path) or `YT_TOKEN_JSON` (the token file's contents, for GitHub Actions), `YT_PRIVACY` (`public`/`unlisted`/`private`) | Google Cloud console -> enable **YouTube Data API v3** -> OAuth client (Desktop app) -> add your Google account as a test user, or publish the app (tokens of apps left "in testing" expire after 7 days). Then `autoclipper auth youtube` once on your machine and paste `secrets/yt_token.json` into the `YT_TOKEN_JSON` repo secret. Quota: an upload costs 1600 of the 10,000 daily units, i.e. ~6 posts/day until Google raises it. |
| TikTok | `TIKTOK_CLIENT_KEY`, `TIKTOK_CLIENT_SECRET`, `TIKTOK_REFRESH_TOKEN`, `TIKTOK_PRIVACY` | developers.tiktok.com -> app with **Login Kit** + **Content Posting API** (direct post), scopes `user.info.basic,video.publish`, a registered redirect URI. `autoclipper auth tiktok --redirect-uri <that uri>` logs in and prints the refresh token (valid ~1 year; access tokens are minted per run). Until the app passes TikTok's audit the account may only post `SELF_ONLY` (private) - the poster reads the allowed levels from `creator_info` and downgrades with a warning. |
| Instagram Reels | `IG_ACCESS_TOKEN`, `IG_USER_ID`, `PUBLIC_BASE_URL`, optional `PUBLIC_S3_*` | Meta app with `instagram_content_publish` + `instagram_basic`, an IG Business/Creator account linked to a Facebook Page. Use a **System User** token from Business Settings (never expires) - normal user tokens die after 60 days. Instagram pulls the file from a URL: set `PUBLIC_S3_BUCKET/ENDPOINT/KEY/SECRET` to a public S3-compatible bucket (Cloudflare R2 with a custom domain, DigitalOcean Spaces, S3) and each video is uploaded there before posting; without it the clip is copied to `data/public/` for a web server you run (not possible on GitHub Actions). |

Posts and processed videos are tracked in `data/state.json` (`autoclipper status`).

## Running it 24/7

**Docker / any VPS**

```bash
docker build -t autoclipper .
docker run -d --restart=always --env-file .env -v $PWD/data:/app/data -v $PWD/config.yaml:/app/config.yaml autoclipper
```

**systemd** — see `deploy/autoclipper.service`.

**GitHub Actions** — `.github/workflows/clip.yml` runs `autoclipper run` every 30 min with the
repo secrets `GEMINI_API_KEY`, `TIKTOK_REFRESH_TOKEN`, etc. and caches `data/state.json` between
runs. Fine for a couple of creators; use a VPS for heavy volume (whisper on a 3h stream takes a
while on CPU — set `whisper_model: tiny` there or run on a GPU box).

## Layout

- `autoclipper/ingest.py` — channel feeds, yt-dlp download, whisper transcript
- `autoclipper/rank.py` — candidate windows, LLM prompt + parsing, heuristic fallback
- `autoclipper/trends.py` — Google Trends / YouTube hot search, LLM topic + source picking
- `autoclipper/top10.py` — countdown script, footage fetch, segment render + concat
- `autoclipper/render.py` — ffmpeg reframe + ASS caption generation
- `autoclipper/publish.py` — YouTube / TikTok / Instagram uploaders
- `autoclipper/pipeline.py` — glue; `autoclipper/cli.py` — commands
