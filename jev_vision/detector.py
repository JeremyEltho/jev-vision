"""Detector interface: anything that turns an image into per-box class candidates.

Standard object detectors (e.g. YOLO in detection mode) emit a single
top-1 class per box after NMS, which is not enough signal to know when a
box is genuinely ambiguous. ``Detection.candidates`` is a ranked top-k
list instead, so the pipeline can tell "confidently a cat" apart from
"maybe a husky, maybe a malamute, maybe a wolf" and only spend a Jev call
on the latter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

Box = tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass(frozen=True)
class ClassCandidate:
    label: str
    confidence: float


@dataclass(frozen=True)
class Detection:
    box: Box
    candidates: tuple[ClassCandidate, ...]  # sorted descending by confidence

    @property
    def top1(self) -> ClassCandidate:
        return self.candidates[0]


class Detector(Protocol):
    def detect(self, image_path: str, top_k: int = 3) -> Sequence[Detection]:
        """Return detections for ``image_path``, each with up to ``top_k`` candidates."""
        ...


_DEFAULT_FIXTURES: tuple[Detection, ...] = (
    # Clear-cut: detector handles this alone, no Jev call.
    Detection(
        box=(12.0, 40.0, 210.0, 300.0),
        candidates=(
            ClassCandidate("cat", 0.97),
            ClassCandidate("dog", 0.02),
            ClassCandidate("raccoon", 0.01),
        ),
    ),
    # Ambiguous: close top-k spread, worth a Jev disambiguation call.
    Detection(
        box=(240.0, 55.0, 460.0, 310.0),
        candidates=(
            ClassCandidate("husky", 0.42),
            ClassCandidate("malamute", 0.38),
            ClassCandidate("wolf", 0.20),
        ),
    ),
    # Low signal: too weak even to hand to Jev, goes straight to escalation.
    Detection(
        box=(5.0, 200.0, 60.0, 260.0),
        candidates=(
            ClassCandidate("bag", 0.12),
            ClassCandidate("rock", 0.10),
            ClassCandidate("shadow", 0.09),
        ),
    ),
)


class MockDetector:
    """Deterministic stand-in detector for demos and tests.

    Returns a fixed set of detections (or caller-supplied fixtures) so the
    escalation pipeline can be exercised end to end without a real model,
    weights, or an input image.
    """

    def __init__(self, fixtures: Sequence[Detection] | None = None) -> None:
        self._fixtures = tuple(fixtures) if fixtures is not None else _DEFAULT_FIXTURES

    def detect(self, image_path: str, top_k: int = 3) -> Sequence[Detection]:
        del image_path  # unused, kept for interface parity
        return [
            Detection(box=d.box, candidates=d.candidates[:top_k])
            for d in self._fixtures
        ]


class YoloDetector:
    """Real detector backend using Ultralytics YOLO.

    Requires the ``yolo`` extra (``pip install jev-vision[yolo]``).

    Caveat: standard YOLO detection heads emit one class per box after NMS,
    not a probability distribution, so ``top_k`` candidates beyond the
    first are only as good as ``classifier`` can make them. Pass a
    crop-level classifier (anything exposing ``predict_proba(crop) ->
    list[(label, confidence)]``) to get genuine top-k candidates for the
    confidence-gating logic to work with; without one, every detection is
    reported as a single high-certainty candidate and the pipeline will
    never route to Jev.
    """

    def __init__(self, weights: str = "yolov8n.pt", classifier=None) -> None:
        self._weights = weights
        self._classifier = classifier
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO  # lazy import: optional dependency

            self._model = YOLO(self._weights)
        return self._model

    def detect(self, image_path: str, top_k: int = 3) -> Sequence[Detection]:
        model = self._load()
        results = model.predict(image_path, verbose=False)

        detections: list[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
                top1_label = names[int(box.cls[0])]
                top1_conf = float(box.conf[0])

                if self._classifier is not None:
                    crop = result.orig_img[
                        int(xyxy[1]) : int(xyxy[3]), int(xyxy[0]) : int(xyxy[2])
                    ]
                    ranked = sorted(
                        self._classifier.predict_proba(crop),
                        key=lambda pair: pair[1],
                        reverse=True,
                    )[:top_k]
                    candidates = tuple(ClassCandidate(l, c) for l, c in ranked)
                else:
                    candidates = (ClassCandidate(top1_label, top1_conf),)

                detections.append(Detection(box=xyxy, candidates=candidates))
        return detections
