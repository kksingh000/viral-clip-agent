"""Font resolution, including per-script fallback.

Short-form captions must render Latin *and* Devanagari (the product targets
English, Hindi and Hinglish), so font choice is made per text run rather than
once globally. Resolution order for each script:

1. an explicit path in the caption style,
2. a font shipped in ``assets/fonts``,
3. a known system location (the container installs DejaVu and Noto),
4. Pillow's built-in bitmap font -- legible but not production quality, and
   logged as a warning so the deployment gap is visible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.core.config import BACKEND_ROOT, REPO_ROOT
from app.core.logging import get_logger

logger = get_logger(__name__)

ASSET_FONT_DIRS = (
    REPO_ROOT / "assets" / "fonts",
    BACKEND_ROOT / "assets" / "fonts",
)

_LATIN_CANDIDATES = (
    "Inter-Bold.ttf",
    "Montserrat-Bold.ttf",
    "DejaVuSans-Bold.ttf",
    "NotoSans-Bold.ttf",
    "arialbd.ttf",
    "Arial Bold.ttf",
    "Helvetica-Bold.ttf",
)
_DEVANAGARI_CANDIDATES = (
    "NotoSansDevanagari-Bold.ttf",
    "NotoSansDevanagari-Regular.ttf",
    "Nirmala.ttf",
    "mangal.ttf",
)

_SYSTEM_FONT_DIRS = (
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path("/Library/Fonts"),
    Path("/System/Library/Fonts"),
    Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
)

Script = str  # "latin" | "devanagari"


def detect_script(text: str) -> Script:
    """Devanagari occupies U+0900-U+097F."""
    for char in text:
        if "\u0900" <= char <= "\u097f":
            return "devanagari"
    return "latin"


@dataclass(slots=True)
class ResolvedFont:
    path: Path | None
    script: Script
    is_fallback: bool


@lru_cache(maxsize=8)
def _find_font_file(script: Script) -> ResolvedFont:
    candidates = (
        _DEVANAGARI_CANDIDATES if script == "devanagari" else _LATIN_CANDIDATES
    )
    for directory in ASSET_FONT_DIRS:
        for name in candidates:
            path = directory / name
            if path.exists():
                return ResolvedFont(path=path, script=script, is_fallback=False)
    for directory in _SYSTEM_FONT_DIRS:
        if not directory.exists():
            continue
        for name in candidates:
            direct = directory / name
            if direct.exists():
                return ResolvedFont(path=direct, script=script, is_fallback=False)
            try:
                match = next(directory.rglob(name), None)
            except OSError:  # pragma: no cover - permission-restricted dirs
                match = None
            if match is not None:
                return ResolvedFont(path=match, script=script, is_fallback=False)
    logger.warning(
        "no TrueType font found for script; captions will use a bitmap fallback",
        extra={"script": script},
    )
    return ResolvedFont(path=None, script=script, is_fallback=True)


@lru_cache(maxsize=64)
def load_font(size: int, *, script: Script = "latin", path: str | None = None):
    """Return a Pillow font object for ``size``."""
    from PIL import ImageFont

    if path:
        candidate = Path(path)
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
        logger.warning("configured font not found", extra={"path": path})

    resolved = _find_font_file(script)
    if resolved.path is not None:
        try:
            return ImageFont.truetype(str(resolved.path), size)
        except OSError as exc:  # pragma: no cover - corrupt font file
            logger.warning(
                "font failed to load", extra={"path": str(resolved.path), "error": str(exc)}
            )
    return ImageFont.load_default()


def font_is_fallback(script: Script = "latin") -> bool:
    return _find_font_file(script).is_fallback
