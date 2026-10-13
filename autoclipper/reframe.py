"""Where to crop a 16:9 frame so the 9:16 Short is filled by the picture instead of a
letterboxed strip. Faces are found on a few sampled frames (OpenCV Haar cascade); the crop is
centred on the dominant face, else on the middle of the frame."""

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


def focus_x(source: Path, start: float, dur: float) -> float | None:
    """Horizontal centre (0..1) of the face the clip should stay on, or None when no face is
    seen often enough (wide shots, gameplay) - callers then crop the middle."""
    try:
        import cv2
    except ImportError:  # pragma: no cover - optional dependency
        log.info("reframe: opencv not installed, cropping the centre")
        return None
    casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    xs: list[float] = []
    n = max(1, min(SAMPLES, int(dur)))
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
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = casc.detectMultiScale(gray, 1.1, 5, minSize=(28, 28))
            if len(faces) == 0:
                continue
            x, _, w, _ = max(faces, key=lambda r: r[2] * r[3])  # the biggest face leads
            xs.append((x + w / 2) / img.shape[1])
    if len(xs) < max(1, round(n * MIN_HITS)):
        log.info("reframe: face in %d/%d frames, cropping the centre", len(xs), n)
        return None
    cx = median(xs)
    log.info("reframe: face at x=%.2f (%d/%d frames)", cx, len(xs), n)
    return cx
