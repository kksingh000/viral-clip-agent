"""Frame-level visual analysis with OpenCV.

Sampling (rather than decoding every frame) keeps this cheap: at 2 fps a
10-minute video costs ~1200 frames, which is enough to place a crop path and
to judge visual quality.
"""

from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path

from app.core.errors import MediaAnalysisError
from app.core.logging import get_logger
from app.providers.base import FaceBox, FrameAnalysis, VisionProvider

logger = get_logger(__name__)

_cascade_lock = threading.Lock()

#: Frames darker than this mean luminance are treated as black.
BLACK_FRAME_THRESHOLD = 12.0

#: Width that frames are downscaled to before Haar detection. Faces smaller
#: than a few percent of the frame are not usable subjects for a vertical
#: crop anyway, so the lost resolution costs nothing.
DETECTION_WIDTH = 640

#: Hard cap on sampled frames, so a pathological input cannot exhaust memory.
MAX_SAMPLES = 20000


@lru_cache(maxsize=4)
def _load_cascade(name: str):
    import cv2

    path = Path(cv2.data.haarcascades) / name
    if not path.exists():  # pragma: no cover - depends on the opencv wheel
        raise MediaAnalysisError(f"OpenCV cascade not found: {path}")
    cascade = cv2.CascadeClassifier(str(path))
    if cascade.empty():  # pragma: no cover
        raise MediaAnalysisError(f"OpenCV cascade failed to load: {path}")
    return cascade


class OpenCVVisionProvider(VisionProvider):
    name = "opencv"

    def __init__(
        self,
        *,
        cascade_name: str = "haarcascade_frontalface_default.xml",
        profile_cascade_name: str = "haarcascade_profileface.xml",
        min_face_fraction: float = 0.04,
    ) -> None:
        self.cascade_name = cascade_name
        self.profile_cascade_name = profile_cascade_name
        self.min_face_fraction = min_face_fraction

    # ------------------------------------------------------------------ faces
    def _detect_faces(self, gray, width: int, height: int) -> list[FaceBox]:
        import cv2

        min_side = max(24, int(min(width, height) * self.min_face_fraction))
        boxes: list[FaceBox] = []
        # Haar detectors are not thread-safe when shared; serialise detection.
        with _cascade_lock:
            frontal = _load_cascade(self.cascade_name).detectMultiScale(
                gray, scaleFactor=1.15, minNeighbors=5, minSize=(min_side, min_side)
            )
            profile = _load_cascade(self.profile_cascade_name).detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=(min_side, min_side)
            )
        for (x, y, w, h) in frontal:
            boxes.append(FaceBox(float(x), float(y), float(w), float(h), 0.9))
        for (x, y, w, h) in profile:
            candidate = FaceBox(float(x), float(y), float(w), float(h), 0.6)
            if not any(_iou(candidate, existing) > 0.35 for existing in boxes):
                boxes.append(candidate)
        # A mirrored pass catches faces looking the other way.
        if not boxes:
            flipped = cv2.flip(gray, 1)
            with _cascade_lock:
                mirrored = _load_cascade(self.profile_cascade_name).detectMultiScale(
                    flipped, scaleFactor=1.2, minNeighbors=5, minSize=(min_side, min_side)
                )
            for (x, y, w, h) in mirrored:
                boxes.append(
                    FaceBox(float(width - x - w), float(y), float(w), float(h), 0.55)
                )
        return sorted(boxes, key=lambda b: b.area, reverse=True)

    @staticmethod
    def _score_mouth_activity(gray, next_gray, faces: list[FaceBox]) -> None:
        """Fill :attr:`FaceBox.mouth_activity` from two adjacent frames.

        The mouth occupies roughly the lower third of a detected face box and
        the middle half horizontally. Averaging the absolute difference over
        that patch separates a talking face from a listening one far more
        reliably than whole-frame motion, which is dominated by camera moves.
        """
        import cv2
        import numpy as np

        height, width = gray.shape[:2]
        for face in faces:
            x0 = int(max(0, face.x + face.width * 0.25))
            x1 = int(min(width, face.x + face.width * 0.75))
            y0 = int(max(0, face.y + face.height * 0.60))
            y1 = int(min(height, face.y + face.height * 0.98))
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            patch_a = gray[y0:y1, x0:x1]
            patch_b = next_gray[y0:y1, x0:x1]
            if patch_a.shape != patch_b.shape or patch_a.size == 0:
                continue
            face.mouth_activity = float(np.mean(cv2.absdiff(patch_a, patch_b)))

    # ----------------------------------------------------------------- frames
    def analyze_frames(
        self,
        video_path: Path,
        *,
        start: float = 0.0,
        end: float | None = None,
        sample_fps: float = 2.0,
        detect_faces: bool = True,
    ) -> list[FrameAnalysis]:
        import cv2
        import numpy as np

        if not Path(video_path).exists():
            raise MediaAnalysisError(f"Video file not found: {video_path}")

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise MediaAnalysisError(f"OpenCV could not open {Path(video_path).name}")

        try:
            native_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
            frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            if native_fps <= 0:
                native_fps = 30.0
            video_end = end if end is not None else (
                frame_count / native_fps if frame_count > 0 else None
            )

            step = 1.0 / max(0.05, sample_fps)
            #: Decode every Nth frame. grab() advances without decoding, which
            #: is roughly an order of magnitude cheaper than seeking per
            #: sample with CAP_PROP_POS_MSEC.
            stride = max(1, int(round(native_fps * step)))

            # One seek to the start, then a purely sequential walk. Frame
            # indices are used rather than CAP_PROP_POS_MSEC, which reports a
            # stale value on the first read after a seek.
            if start > 0.05:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(start * native_fps)))

            results: list[FrameAnalysis] = []
            previous_gray = None
            detection_scale = 1.0

            while True:
                frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                timestamp = frame_index / native_fps
                if video_end is not None and timestamp >= video_end:
                    break

                height, width = frame.shape[:2]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # The *next* decoded frame (one frame later, not one sample
                # later) is what makes mouth movement measurable.
                ok_next, next_frame = capture.read()
                next_gray = (
                    cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
                    if ok_next and next_frame is not None
                    else None
                )

                brightness = float(np.mean(gray))
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                motion = 0.0
                if previous_gray is not None and previous_gray.shape == gray.shape:
                    motion = float(np.mean(cv2.absdiff(previous_gray, gray)))
                previous_gray = gray

                faces: list[FaceBox] = []
                if detect_faces:
                    # Haar cost is quadratic in resolution; detect on a
                    # downscaled copy and map the boxes back.
                    if width > DETECTION_WIDTH:
                        detection_scale = DETECTION_WIDTH / width
                        small = cv2.resize(
                            gray,
                            (DETECTION_WIDTH, int(height * detection_scale)),
                            interpolation=cv2.INTER_AREA,
                        )
                    else:
                        detection_scale = 1.0
                        small = gray
                    detected = self._detect_faces(small, *small.shape[::-1])
                    inverse = 1.0 / detection_scale
                    faces = [
                        FaceBox(
                            b.x * inverse,
                            b.y * inverse,
                            b.width * inverse,
                            b.height * inverse,
                            b.confidence,
                        )
                        for b in detected
                    ]
                if faces and next_gray is not None and next_gray.shape == gray.shape:
                    self._score_mouth_activity(gray, next_gray, faces)

                results.append(
                    FrameAnalysis(
                        timestamp=timestamp,
                        width=width,
                        height=height,
                        faces=faces,
                        brightness=brightness,
                        sharpness=sharpness,
                        motion=motion,
                        is_black=brightness < BLACK_FRAME_THRESHOLD,
                    )
                )
                if len(results) > MAX_SAMPLES:  # safety valve on absurd inputs
                    logger.warning(
                        "frame sampling cap reached", extra={"path": str(video_path)}
                    )
                    break

                # Skip ahead without decoding (two frames were already read).
                for _ in range(max(0, stride - 2)):
                    if not capture.grab():
                        return results
            return results
        finally:
            capture.release()


def _iou(a: FaceBox, b: FaceBox) -> float:
    left = max(a.x, b.x)
    top = max(a.y, b.y)
    right = min(a.x + a.width, b.x + b.width)
    bottom = min(a.y + a.height, b.y + b.height)
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


class MockVisionProvider(VisionProvider):
    """Returns geometry only -- no detection. Used when OpenCV is unavailable."""

    name = "mock"
    is_mock = True

    def analyze_frames(
        self,
        video_path: Path,
        *,
        start: float = 0.0,
        end: float | None = None,
        sample_fps: float = 2.0,
        detect_faces: bool = True,
    ) -> list[FrameAnalysis]:
        from app.video.ffmpeg import probe

        info = probe(video_path)
        stop = end if end is not None else info.duration
        step = 1.0 / max(0.05, sample_fps)
        out: list[FrameAnalysis] = []
        timestamp = max(0.0, start)
        while timestamp < stop:
            out.append(
                FrameAnalysis(
                    timestamp=timestamp,
                    width=info.width or 1920,
                    height=info.height or 1080,
                )
            )
            timestamp += step
        return out
