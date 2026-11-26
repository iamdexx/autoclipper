"""Filler detector for everything the narrator says. Lines built from hype words carry no
information ("absolutely mind-blowing", "you won't believe", "changes everything"); a line that
contains one is rewritten by nobody - it is simply not spoken."""

from __future__ import annotations

import re

_PHRASES = [
    r"mind[- ]?blowing",
    r"game[- ]?chang(?:er|ing)",
    r"jaw[- ]?dropping",
    r"you (?:won'?t|will not|wont) believe",
    r"you (?:need|have) to see",
    r"wait (?:for|until|till)",
    r"check (?:this )?out",
    r"watch (?:this|until|till)",
    r"stay (?:tuned|till|until)",
    r"chang(?:es?|ed|ing) everything",
    r"redefin(?:es?|ed|ing)",
    r"revolutioni[sz]\w*",
    r"next[- ]level",
    r"like never before",
    r"takes? (?:it|things|this) to (?:a|the) (?:whole )?(?:new|next) level",
    r"visual masterpiece",
    r"masterpiece",
    r"generational leap",
    r"unbelievable",
    r"incredibl[ey]",
    r"insane(?:ly)?",
    r"crazy",
    r"wild",
    r"epic",
    r"stunning",
    r"breathtaking",
    r"hyper[- ]?realistic",
    r"absolute(?:ly)?",
    r"literally",
    r"truly",
    r"completely",
    r"entirely new",
    r"completely fresh",
    r"brand[- ]new",
    r"feels? (?:more )?(?:completely )?alive",
    r"broke the internet",
    r"internet (?:is|was) (?:going|losing)",
    r"fans are (?:going|losing)",
    r"everyone is talking",
    r"this is huge",
    r"huge",
    r"massive",
    r"shocking",
    r"here'?s why (?:it|this) matters",
    r"impactful",
    r"tactical layer",
    r"ultimate(?! editions?\b)",  # "Ultimate Edition" is a product, not hype
]
_RX = re.compile(r"\b(?:" + "|".join(_PHRASES) + r")\b", re.I)


def filler(text: str) -> list[str]:
    """The hype phrases found in `text`, lower-cased, in order (empty = clean)."""
    return [m.group(0).lower() for m in _RX.finditer(text or "")]


def clean(text: str) -> bool:
    return not filler(text)


# what the script writers are told, so they don't waste a draft on words the filter drops
RULES = (
    "Write like a sharp human host, not an ad. Every sentence must carry a concrete fact from "
    "the source (a name, a number, a rule, a place, a quote). Banned: hype adjectives and stock "
    "phrases such as insane, crazy, wild, huge, massive, epic, stunning, mind-blowing, "
    "game-changer, masterpiece, redefines, changes everything, next level, absolutely, "
    "literally, truly, completely, 'you won't believe', 'wait for', 'check out', 'you need to "
    "see'. A line containing any of them is deleted."
)
