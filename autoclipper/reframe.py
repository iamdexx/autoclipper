"""Where to crop a 16:9 frame so the 9:16 Short is filled by the picture instead of a
letterboxed strip. Faces are found on a few sampled frames (OpenCV Haar cascade); the crop is
centred on the dominant face (frontal, else in profile), else on the middle of the frame."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from statistics import median

log = logging.getLogger(__name__)

SAMPLES = 10  # frames inspected per clip
SAMPLE_W = 640
MIN_HITS = 0.3  # share of sampled frames that must show a face to trust it


def _faces(
    source: Path, start: float, dur: float, n: int, profile: bool = False
) -> tuple[list[tuple[float, bool]], int, float] | None:
    """((x centre 0..1, frontal?) of the biggest face in each sampled frame that shows one,
    frames read, the share of the frame's width a full-height 9:16 crop keeps), or None without
    OpenCV. `profile`: a frame with no frontal face is also searched for a face turned to
    either side."""
    try:
        import cv2
    except ImportError:  # pragma: no cover - optional dependency
        return None
    casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    side = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    hits: list[tuple[float, bool]] = []
    read = 0
    window = 1.0
    with tempfile.TemporaryDirectory() as td:
        for i in range(n):
            t = start + dur * (i + 0.5) / n
            f = Path(td) / f"{i}.jpg"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(source)]
                + ["-frames:v", "1", "-vf", f"scale={SAMPLE_W}:-2", str(f)],
                check=False,
            )
            if not f.exists():
                continue
            img = cv2.imread(str(f))
            if img is None:
                continue
            read += 1
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = img.shape[:2]
            window = min(1.0, height * 9 / 16 / width)
            found = [tuple(r) for r in casc.detectMultiScale(gray, 1.1, 5, minSize=(28, 28))]
            frontal = bool(found)
            if not found and profile:
                found = [tuple(r) for r in side.detectMultiScale(gray, 1.1, 5, minSize=(28, 28))]
                # the cascade only knows faces turned one way: the mirrored frame finds the other
                flipped = side.detectMultiScale(cv2.flip(gray, 1), 1.1, 5, minSize=(28, 28))
                found += [(width - x - w, y, w, h) for x, y, w, h in flipped]
            if not found:
                continue
            x, _, w, _ = max(found, key=lambda r: r[2] * r[3])  # the biggest face leads
            hits.append(((x + w / 2) / width, frontal))
    return hits, read, window


def face_share(source: Path, dur: float, samples: int = 40) -> float | None:
    """Share of frames sampled across the whole video that show a face: near zero for a
    text-to-speech voice over screenshots or gameplay, well above it for a person on camera.
    None when it could not be measured."""
    if dur <= 0:
        return None
    found = _faces(source, 0.0, dur, samples)
    if not found or not found[1]:
        return None
    return len(found[0]) / found[1]


def focus_x(source: Path, start: float, dur: float) -> float | None:
    """Horizontal centre (0..1) of the face the clip should stay on, or None when no face is
    seen often enough (wide shots, gameplay)."""
    n = max(1, min(SAMPLES, int(dur)))
    found = _faces(source, start, dur, n, profile=True)
    if found is None:
        log.info("reframe: opencv not installed, no face to crop to")
        return None
    hits, _, window = found

    def near(c: float) -> list[tuple[float, bool]]:
        return [h for h in hits if abs(h[0] - c) <= window / 2]

    # the faces one crop window can hold, most of them, frontal ones first: the median of two
    # speakers on opposite sides would be a window between them showing neither
    best = max((near(c) for c, _ in hits), key=lambda g: (len(g), sum(f for _, f in g)), default=[])
    if len(best) < max(1, round(n * MIN_HITS)):
        log.info("reframe: face in %d/%d frames, no face to crop to", len(best), n)
        return None
    xs = [x for x, _ in best]
    cx = median(xs)
    log.info("reframe: face at x=%.2f (%d/%d frames)", cx, len(xs), n)
    return cx
