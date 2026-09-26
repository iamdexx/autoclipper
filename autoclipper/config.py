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


@dataclass
class CaptionStyle:
    font: str = "DejaVu Sans"
    size: int = 64
    words_per_line: int = 3
    chars_per_line: int = 22  # fits 1080px at size 64 with outline
    hook_chars_per_line: int = 24
    highlight_color: str = "&H0000FFFF"  # ASS BGR: yellow
    base_color: str = "&H00FFFFFF"
    outline: int = 4
    margin_v: int = 420  # px from bottom of 1920px frame


@dataclass
class Publishing:
    dry_run: bool = True
    youtube: bool = False
    tiktok: bool = False
    instagram: bool = False
    title_template: str = "{hook} | {creator} #shorts"
    caption_template: str = "{hook}\n\nclip from {creator}'s stream\n{hashtags}"
    default_hashtags: list[str] = field(
        default_factory=lambda: ["#clips", "#stream", "#viral", "#fyp"]
    )


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
    publishing: Publishing = field(default_factory=Publishing)
    poll_interval_s: int = 900
    max_videos_per_run: int = 2

    @classmethod
    def load(cls, path: str | Path) -> Config:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        creators = [Creator(**c) for c in raw.pop("creators", [])]
        clips = ClipRules(**raw.pop("clips", {}))
        captions = CaptionStyle(**raw.pop("captions", {}))
        publishing = Publishing(**raw.pop("publishing", {}))
        workdir = Path(os.path.expanduser(raw.pop("workdir", "data")))
        return cls(
            creators=creators,
            workdir=workdir,
            clips=clips,
            captions=captions,
            publishing=publishing,
            **raw,
        )
