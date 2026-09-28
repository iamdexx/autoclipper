"""Fact-check gate for everything the narrator says.

LLM-written lists hallucinate (made-up plays, wrong names, events that never happened). Before a
Top-10 script is voiced we gather real-world evidence for every entry - Google News headlines,
Wikipedia, YouTube titles (all keyless) - and have the LLM judge each claim ONLY against that
evidence. Unsupported entries are dropped and replaced; if too few survive the video is skipped.

For creator / trend clips the narrator's hook is checked against the clip's own transcript so it
never promises something the clip doesn't show.

Grounded search inside Gemini itself is not available on the free tier, hence our own retrieval.
"""

from __future__ import annotations

import html
import json
import logging
import re
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

import requests

from . import rank, trends
from .config import Config

log = logging.getLogger(__name__)

_UA = {"User-Agent": "Mozilla/5.0 (autoclipper fact-check)"}
VERDICTS = ("supported", "unsupported", "unverifiable")


@dataclass
class Evidence:
    kind: str  # news | wiki | youtube
    title: str
    source: str = ""
    date: str = ""
    url: str = ""


@dataclass
class Check:
    claim: str
    verdict: str = "unverifiable"
    confidence: float = 0.0
    note: str = ""
    evidence: list[Evidence] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.verdict == "supported"


# ---------- evidence retrieval ----------


def _strip(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def google_news(query: str, n: int = 6) -> list[Evidence]:
    q = urllib.parse.quote_plus(query)
    r = requests.get(
        f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en",
        headers=_UA,
        timeout=20,
    )
    r.raise_for_status()
    out = []
    for item in ET.fromstring(r.content).iter("item"):
        title = _strip(item.findtext("title"))
        if not title:
            continue
        out.append(
            Evidence(
                "news",
                title,
                source=_strip(item.findtext("source")),
                date=(item.findtext("pubDate") or "")[:16],
                url=item.findtext("link") or "",
            )
        )
        if len(out) >= n:
            break
    return out


def wikipedia(query: str, n: int = 3) -> list[Evidence]:
    r = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": n,
            "utf8": 1,
        },
        headers=_UA,
        timeout=20,
    )
    r.raise_for_status()
    return [
        Evidence(
            "wiki",
            f"{x['title']}: {_strip(x.get('snippet', ''))}",
            source="wikipedia",
            url="https://en.wikipedia.org/wiki/" + urllib.parse.quote(x["title"].replace(" ", "_")),
        )
        for x in r.json().get("query", {}).get("search", [])
    ]


def youtube(query: str, n: int = 5) -> list[Evidence]:
    return [
        Evidence("youtube", c.title, source=c.channel, url=c.url)
        for c in trends.yt_search(query, n)
        if c.views
    ]


def gather(query: str) -> list[Evidence]:
    """All sources in parallel; a failing source contributes nothing."""

    def _run(fn):
        try:
            return fn(query)
        except Exception as e:
            log.debug("%s failed for %r: %s", fn.__name__, query, e)
            return []

    ev: list[Evidence] = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        for res in ex.map(_run, (google_news, wikipedia, youtube)):
            ev += res
    return ev


# ---------- judging ----------

JUDGE_PROMPT = """You are a strict fact-checker for a short-form video channel. For each CLAIM below
decide, using ONLY the EVIDENCE listed under it (headlines, wiki snippets, video titles), whether
the claim describes something that really happened / is true.

verdict rules:
- "supported": evidence clearly refers to this specific event/fact (names, what happened, roughly when)
- "unsupported": evidence contradicts it, or the specific event is absent while evidence covers the
  same people/topic in that period (a real thing would have left a trace)
- "unverifiable": evidence is unrelated or too thin to say either way
Be skeptical of superlatives and very specific numbers. Never use your own memory as a source.
Judge only the factual content: obvious hype / figurative wording ("defied physics", "broke the
internet", "insane") is not a claim and must not by itself make something unsupported.

Return ONLY a JSON array with one object per claim, same order:
[{{"i": <claim index>, "verdict": "supported|unsupported|unverifiable", "confidence": <0-1>,
  "note": "<one sentence citing which evidence, or what is missing>"}}, ...]

{items}
"""


def _fmt(i: int, claim: str, ev: list[Evidence]) -> str:
    lines = [f"CLAIM {i}: {claim}"]
    if not ev:
        lines.append("  EVIDENCE: (none found)")
    for e in ev[:14]:
        meta = " | ".join(x for x in (e.source, e.date) if x)
        lines.append(f"  [{e.kind}] {e.title[:160]}" + (f" ({meta})" if meta else ""))
    return "\n".join(lines)


def check_claims(claims: list[tuple[str, str]], cfg: Config) -> list[Check]:
    """claims = [(claim_text, search_query)] -> one Check per claim (same order)."""
    if not claims:
        return []
    with ThreadPoolExecutor(max_workers=4) as ex:
        evidence = list(ex.map(gather, [q for _, q in claims]))
    checks = [Check(claim=c, evidence=ev) for (c, _), ev in zip(claims, evidence, strict=True)]
    items = "\n\n".join(_fmt(i, c.claim, c.evidence) for i, c in enumerate(checks))
    try:
        raw = trends._llm(JUDGE_PROMPT.format(items=items), cfg)
        for it in rank._parse_json_array(raw):
            i = int(it.get("i", -1))
            if 0 <= i < len(checks):
                v = str(it.get("verdict", "")).lower()
                checks[i].verdict = v if v in VERDICTS else "unverifiable"
                checks[i].confidence = float(it.get("confidence", 0) or 0)
                checks[i].note = str(it.get("note", ""))[:300]
    except Exception as e:
        log.warning("fact-check judge failed (%s); everything stays unverifiable", e)
        for c in checks:
            c.note = f"judge failed: {e}"[:300]
    return checks


def passes(check: Check, cfg: Config, strict: bool | None = None) -> bool:
    v = cfg.verify
    if check.verdict == "unsupported":
        return False
    if check.verdict == "unverifiable":
        return not (v.strict if strict is None else strict)
    return check.confidence >= v.min_confidence


# ---------- hook vs transcript (creator / trend clips) ----------

HOOK_PROMPT = """A narrator will say the HOOK before this clip plays. Check the hook against the clip's
TRANSCRIPT: it must not claim, promise or imply anything the transcript does not actually contain
(no invented quotes, names, outcomes or exaggerations beyond normal hype wording).
Return ONLY JSON: {{"verdict": "supported|unsupported", "confidence": <0-1>,
  "note": "<one sentence>", "fixed": "<if unsupported: a faithful replacement hook, 1-2 sentences,
  max 18 words, same energy, plain spoken words>"}}

HOOK: {hook}
TRANSCRIPT: {transcript}
"""


def check_hook(hook: str, transcript: str, cfg: Config) -> tuple[Check, str]:
    """(check, replacement) - replacement is '' when the hook is fine or no fix was offered."""
    chk = Check(claim=hook)
    if not hook.strip():
        chk.verdict, chk.confidence, chk.note = "supported", 1.0, "nothing to check"
        return chk, ""
    if not transcript.strip():
        chk.note = "no transcript for this window"
        return chk, ""
    try:
        raw = trends._llm(HOOK_PROMPT.format(hook=hook, transcript=transcript[:4000]), cfg)
        j = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        v = str(j.get("verdict", "")).lower()
        chk.verdict = v if v in VERDICTS else "unverifiable"
        chk.confidence = float(j.get("confidence", 0) or 0)
        chk.note = str(j.get("note", ""))[:300]
        return chk, str(j.get("fixed", "")).strip()[:240] if chk.verdict == "unsupported" else ""
    except Exception as e:
        log.warning("hook check failed (%s)", e)
        chk.note = f"check failed: {e}"[:300]
        return chk, ""


def dump(checks: list[Check]) -> list[dict]:
    return [asdict(c) for c in checks]
