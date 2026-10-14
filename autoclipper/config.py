from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Creator:
    name: str
    channel_id: str | None = None  # UCxxxx
    handle: str | None = None  # @name (resolved to channel_id on first run)
    min_duration_s: int = 600  # ignore uploads shorter than this (we want streams/long-form)
    max_duration_s: int = 6 * 3600  # skip absurdly long VODs
    hashtags: list[str] = field(default_factory=list)


@dataclass
class ClipRules:
    count: int = 3
    min_len_s: float = 20
    max_len_s: float = 60
    target_len_s: float = 40
    tail_s: float = 1.0  # footage keeps rolling this long after the last line, audio fading out


@dataclass
class CaptionStyle:
    font: str = "DejaVu Sans"
    size: int = 64
    style: str = "highlight"  # highlight: line with the spoken word coloured | word: one big word
    words_per_line: int = 3
    chars_per_line: int = 22  # fits 1080px at size 64 with outline
    hook_chars_per_line: int = 24
    highlight_color: str = "&H0000FFFF"  # ASS BGR: yellow
    base_color: str = "&H00FFFFFF"
    outline: int = 4
    margin_v: int = 420  # px from bottom of 1920px frame


@dataclass
class Look:
    # crop: the 16:9 source is scaled to fill the 9:16 frame and cropped to the speaker's face
    # (centre when no face is found) - the picture fills the phone like every big clip channel.
    # blur: the whole 16:9 picture sits in the middle over a blurred copy of itself.
    reframe: str = "crop"
    watermark: str = ""  # small channel handle in the top corner of every video, "" = none


@dataclass
class Narration:
    provider: str = "auto"  # auto | edge | azure | google | elevenlabs | gemini | openai | off
    edge_voice: str = (
        "en-US-GuyNeural"  # Andrew/Brian ...MultilingualNeural, Christopher/Roger... Neural
    )
    edge_rate: float = 1.05
    azure_voice: str = (
        "en-US-GuyNeural"  # Andrew/Brian ...MultilingualNeural, Davis/Steffan... Neural
    )
    azure_rate: float = 1.05
    azure_style: str = ""  # e.g. excited / cheerful, only on voices that list the style
    google_voice: str = "Charon"  # Chirp 3 HD short name; also Fenrir, Orus, Puck, Algenib ...
    google_rate: float = 1.0  # 0.25-2.0 speaking pace
    gemini_voice: str = "Puck"
    elevenlabs_voice: str = "TX3LPaxmHKxFdv7VOQHJ"  # "Liam"
    openai_voice: str = "onyx"
    duck_db: float = -14.0  # how far the stream audio drops under the narrator
    max_len_s: float = 6.0


@dataclass
class Publishing:
    dry_run: bool = True
    youtube: bool = False
    tiktok: bool = False
    instagram: bool = False
    # Posting budget per platform (real posts only). YouTube's default API quota (10k units/day,
    # 1600 per upload) allows 6 uploads a day; the gap spreads them out instead of dumping a
    # whole run's videos at once. Finished videos wait in workdir/queue and the best-scoring one
    # is posted whenever a slot opens; the rest age out. A run that finds the next slot in the
    # future uploads to YouTube as a scheduled post (private until publishAt), so every slot up
    # to schedule_ahead_h gets filled even when the cron fires a few times a day.
    max_per_day: int = 6
    min_gap_min: int = 150
    schedule_ahead_h: int = 12
    queue_max_age_h: int = 48
    queue_max_items: int = 24
    title_template: str = "{hook} | {creator} #shorts"
    caption_template: str = "{hook}\n\nclip from {creator}'s stream\n{hashtags}"
    top10_title_template: str = "{hook} #shorts"
    top10_caption_template: str = "{hook}\n\n{hashtags}"
    default_hashtags: list[str] = field(default_factory=lambda: ["#shorts"])
    cta: str = ""  # one line added to every description above the hashtags (subscribe nudge)


@dataclass
class Trends:
    enabled: bool = False
    niches: list[str] = field(default_factory=list)  # e.g. ["sports", "gaming", "celebrity drama"]
    geos: list[str] = field(default_factory=lambda: ["US"])  # Google Trends regions
    topics_per_run: int = 2
    min_duration_s: int = 120  # source videos for trend clips can be shorter than streams
    max_duration_s: int = 2 * 3600
    max_source_age_h: float = 24  # never clip a source older than this when its age is known


@dataclass
class Verify:
    """Fact-check gate for narration (see verify.py)."""

    enabled: bool = True
    strict: bool = False  # True: 'unverifiable' entries are dropped too, not just 'unsupported'
    strict_top10: bool = True  # Top-N entries must be positively supported (LLM-invented plays
    # about recent events come back 'unverifiable', never 'unsupported'); hooks stay on `strict`
    min_confidence: float = 0.5  # 'supported' below this is treated as unverifiable
    min_entries: int = 4  # a video needs at least this many verified entries or it's skipped
    max_rounds: int = 2  # script (re)writes to fill dropped entries
    check_hooks: bool = True  # creator/trend clips: narrator hook vs transcript


@dataclass
class QA:
    """Output quality gate run on every finished video before publishing (see qa.py)."""

    enabled: bool = True
    clip_min_s: float = 15.0
    clip_max_s: float = 90.0
    top10_min_s: float = 12.0
    top10_max_s: float = 90.0
    min_mean_db: float = -30.0  # average level; quieter = narration buried / missing
    max_mean_db: float = -8.0
    max_peak_db: float = -0.1  # anything above = clipping
    silence_db: float = -40.0
    max_silence_s: float = 1.5  # dead air longer than this fails
    max_black_s: float = 0.5
    max_freeze_s: float = 3.0
    check_speech: bool = True  # re-transcribe the output and compare with the script
    min_speech_match: float = 0.6  # share of the script's content words that must be heard
    speech_required: bool = True  # fail (not skip) when the output can't be transcribed


@dataclass
class TopTen:
    enabled: bool = False
    per_run: int = 1  # how many Top-10 videos to make per run
    # ranking  = "Ranking The Most ... Moments": ~6 clips back to back with their own audio, a
    #            colour title banner + tease, a 1..N list on the left that fills in per clip,
    #            narrator hook + one line per clip (or, with narrate off, the clip's own speech
    #            captioned one word at a time and its audio left alone)
    # countdown = "Top 10 ...": big rank badge per entry, narrator over ducked footage
    # listicle  = "N photos ...": one still per item with a slow zoom, narrator reads through
    style: str = "ranking"
    narrate: bool = True  # ranking style: full narration (hook + a line per clip)
    items: int = 6
    item_min_s: float = 3.5  # each entry lasts at least this (longer if the narration is)
    listicle_gap_s: float = 0.25  # listicle: pause between items
    outro_s: float = 2.0  # #1's footage plays on (audio back up) this long after the last line
    listicle_outro_s: float = 0.8  # listicle: the last photo holds this long after the last word
    item_max_s: float = 10.0  # how much footage to fetch per entry
    source_max_duration_s: int = 30 * 60  # only clip from sources up to this long
    topics: list[str] = field(default_factory=list)  # fixed ideas; empty = derive from trends
    cuts_per_entry: int = 1  # footage cuts per entry (each with a zoom punch)
    sfx: bool = True  # whoosh/hit on every rank reveal
    # the footage's own audio (crowd, commentary, whatever music the source clip had) under the
    # narration; kept low so it doesn't sound like the soundtrack changes at every rank
    source_db: float = -26.0
    # one continuous bed under the whole video, start to end:
    # elevenlabs = generated per video (Music API), synth = built-in pulse, off = none;
    # music_dir tracks are used whenever set (and as fallback if generation fails)
    music: str = "elevenlabs"
    music_prompt: str = (
        "high-energy instrumental for a viral sports/entertainment countdown video, driving "
        "drums and bass, modern trap-EDM hybrid, 130 bpm, builds tension, no vocals"
    )
    music_dir: str = ""  # your own tracks (mp3/m4a/wav)
    music_db: float = -14.0  # bed level relative to the narration


@dataclass
class Config:
    creators: list[Creator]
    workdir: Path = Path("data")
    whisper_model: str = "base"
    whisper_compute: str = "int8"
    llm_provider: str = "auto"  # auto | anthropic | openai | heuristic
    llm_model: str | None = None
    clips: ClipRules = field(default_factory=ClipRules)
    captions: CaptionStyle = field(default_factory=CaptionStyle)
    look: Look = field(default_factory=Look)
    publishing: Publishing = field(default_factory=Publishing)
    narration: Narration = field(default_factory=Narration)
    trends: Trends = field(default_factory=Trends)
    top10: TopTen = field(default_factory=TopTen)
    verify: Verify = field(default_factory=Verify)
    qa: QA = field(default_factory=QA)
    poll_interval_s: int = 900
    max_videos_per_run: int = 2
    max_video_age_h: float = 72  # creator uploads older than this are never picked up

    @classmethod
    def load(cls, path: str | Path) -> Config:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        creators = [Creator(**c) for c in raw.pop("creators", [])]
        clips = ClipRules(**raw.pop("clips", {}))
        captions = CaptionStyle(**raw.pop("captions", {}))
        look = Look(**raw.pop("look", {}))
        publishing = Publishing(**raw.pop("publishing", {}))
        narration = Narration(**raw.pop("narration", {}))
        trends = Trends(**raw.pop("trends", {}))
        top10 = TopTen(**raw.pop("top10", {}))
        verify = Verify(**raw.pop("verify", {}))
        qa = QA(**raw.pop("qa", {}))
        workdir = Path(os.path.expanduser(raw.pop("workdir", "data")))
        return cls(
            creators=creators,
            workdir=workdir,
            clips=clips,
            captions=captions,
            look=look,
            publishing=publishing,
            narration=narration,
            trends=trends,
            top10=top10,
            verify=verify,
            qa=qa,
            **raw,
        )
