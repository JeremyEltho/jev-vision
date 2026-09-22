"""Confidence-gated escalation: detector handles the easy cases, Jev the ambiguous ones.

Three-way split per detection, driven by ``PipelineConfig``:

  top1 >= high_confidence        -> accept the detector's answer, no Jev call
  ambiguous_floor <= top1 < high -> batch into one Jev fan-out call for disambiguation
  top1 < ambiguous_floor         -> too weak to disambiguate at all, escalate directly

All detections in the second bucket for a given frame are sent to Jev as
one batch of independent Choice questions over shared state (the
"speculative fan-out" pattern: parallel server-side, so batching costs
about the same latency as a single question) rather than one round trip
per detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from jev_vision.config import PipelineConfig
from jev_vision.detector import Detection, Detector
from jev_vision.jev_client import JevClient

Source = Literal["detector", "jev", "escalation"]


@dataclass(frozen=True)
class ResolvedDetection:
    box: tuple[float, float, float, float]
    label: str
    confidence: float
    source: Source
    distribution: dict[str, float] | None = None


def _format_candidates(det: Detection) -> str:
    return ", ".join(f"{c.label} ({c.confidence:.2f})" for c in det.candidates)


class ConfidenceGatedPipeline:
    def __init__(
        self,
        detector: Detector,
        jev_client: JevClient | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.detector = detector
        self.jev_client = jev_client
        self.config = config or PipelineConfig()

    def run(self, image_path: str) -> list[ResolvedDetection]:
        cfg = self.config
        detections = self.detector.detect(image_path, top_k=cfg.top_k)

        resolved: list[ResolvedDetection] = []
        ambiguous: list[tuple[int, Detection]] = []

        for i, det in enumerate(detections):
            top1 = det.top1
            if top1.confidence >= cfg.high_confidence or len(det.candidates) == 1:
                resolved.append(
                    ResolvedDetection(det.box, top1.label, top1.confidence, "detector")
                )
            elif top1.confidence >= cfg.ambiguous_floor:
                ambiguous.append((i, det))
            else:
                resolved.append(
                    ResolvedDetection(det.box, top1.label, top1.confidence, "escalation")
                )

        resolved.extend(self._resolve_ambiguous(ambiguous))
        return resolved

    def _resolve_ambiguous(
        self, ambiguous: list[tuple[int, Detection]]
    ) -> list[ResolvedDetection]:
        if not ambiguous:
            return []

        if self.jev_client is None:
            # No Jev configured: fall back to the detector's best guess.
            return [
                ResolvedDetection(det.box, det.top1.label, det.top1.confidence, "escalation")
                for _, det in ambiguous
            ]

        cfg = self.config
        questions = {
            f"det_{i}": {
                "type": "choice",
                "instructions": (
                    "A vision model detected an object but is unsure of its class. "
                    f"Detector candidates with confidence: {_format_candidates(det)}. "
                    "Given the context, which candidate is correct?"
                ),
                "criteria": {"options": [c.label for c in det.candidates]},
            }
            for i, det in ambiguous
        }

        response = self.jev_client.ask(
            state=cfg.context or "No additional scene context provided.",
            questions=questions,
        )
        answers = response.get("answers", {})

        results: list[ResolvedDetection] = []
        for i, det in ambiguous:
            answer = answers.get(f"det_{i}")
            if answer is None:
                results.append(
                    ResolvedDetection(det.box, det.top1.label, det.top1.confidence, "escalation")
                )
                continue

            jev_confidence = answer.get("confidence", 0.0)
            distribution = answer.get("distribution")
            if jev_confidence >= cfg.jev_confidence_floor:
                results.append(
                    ResolvedDetection(
                        det.box, answer["selected"], jev_confidence, "jev", distribution
                    )
                )
            else:
                results.append(
                    ResolvedDetection(
                        det.box, det.top1.label, det.top1.confidence, "escalation", distribution
                    )
                )
        return results
