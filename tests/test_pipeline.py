from jev_vision.config import PipelineConfig
from jev_vision.detector import ClassCandidate, Detection, MockDetector
from jev_vision.pipeline import ConfidenceGatedPipeline


class FakeJevClient:
    """Stub standing in for JevClient: records calls, returns canned answers."""

    def __init__(self, answers: dict[str, dict]) -> None:
        self.answers = answers
        self.calls: list[dict] = []

    def ask(self, state, questions):
        self.calls.append({"state": state, "questions": questions})
        return {"model": "jev-latest", "answers": self.answers, "usage": {}}


def _fixtures(*confidence_sets: tuple[tuple[str, float], ...]) -> list[Detection]:
    return [
        Detection(
            box=(0.0, 0.0, 1.0, 1.0),
            candidates=tuple(ClassCandidate(label, conf) for label, conf in pairs),
        )
        for pairs in confidence_sets
    ]


def test_high_confidence_detection_skips_jev():
    fixtures = _fixtures((("cat", 0.97), ("dog", 0.02)))
    jev = FakeJevClient(answers={})
    pipeline = ConfidenceGatedPipeline(MockDetector(fixtures), jev_client=jev)

    resolved = pipeline.run("irrelevant.jpg")

    assert len(resolved) == 1
    assert resolved[0].source == "detector"
    assert resolved[0].label == "cat"
    assert jev.calls == []


def test_low_confidence_detection_escalates_without_calling_jev():
    fixtures = _fixtures((("bag", 0.12), ("rock", 0.10), ("shadow", 0.09)))
    jev = FakeJevClient(answers={})
    pipeline = ConfidenceGatedPipeline(MockDetector(fixtures), jev_client=jev)

    resolved = pipeline.run("irrelevant.jpg")

    assert len(resolved) == 1
    assert resolved[0].source == "escalation"
    assert jev.calls == []


def test_ambiguous_detection_is_resolved_by_jev():
    fixtures = _fixtures((("husky", 0.42), ("malamute", 0.38), ("wolf", 0.20)))
    jev = FakeJevClient(
        answers={"det_0": {"type": "choice", "selected": "malamute", "confidence": 0.81}}
    )
    pipeline = ConfidenceGatedPipeline(MockDetector(fixtures), jev_client=jev)

    resolved = pipeline.run("irrelevant.jpg")

    assert len(resolved) == 1
    assert resolved[0].source == "jev"
    assert resolved[0].label == "malamute"
    assert resolved[0].confidence == 0.81
    assert len(jev.calls) == 1
    assert "det_0" in jev.calls[0]["questions"]


def test_low_confidence_jev_answer_still_escalates():
    fixtures = _fixtures((("husky", 0.42), ("malamute", 0.38), ("wolf", 0.20)))
    jev = FakeJevClient(
        answers={"det_0": {"type": "choice", "selected": "malamute", "confidence": 0.31}}
    )
    pipeline = ConfidenceGatedPipeline(MockDetector(fixtures), jev_client=jev)

    resolved = pipeline.run("irrelevant.jpg")

    assert resolved[0].source == "escalation"
    # falls back to the detector's own top-1, not Jev's low-confidence pick
    assert resolved[0].label == "husky"


def test_multiple_ambiguous_detections_batch_into_one_jev_call():
    fixtures = _fixtures(
        (("husky", 0.42), ("malamute", 0.38), ("wolf", 0.20)),
        (("bus", 0.40), ("truck", 0.35), ("van", 0.25)),
    )
    jev = FakeJevClient(
        answers={
            "det_0": {"type": "choice", "selected": "husky", "confidence": 0.7},
            "det_1": {"type": "choice", "selected": "truck", "confidence": 0.9},
        }
    )
    pipeline = ConfidenceGatedPipeline(MockDetector(fixtures), jev_client=jev)

    resolved = pipeline.run("irrelevant.jpg")

    assert len(jev.calls) == 1  # fan-out: one round trip for both detections
    assert {r.label for r in resolved} == {"husky", "truck"}


def test_without_jev_client_ambiguous_detections_escalate():
    fixtures = _fixtures((("husky", 0.42), ("malamute", 0.38), ("wolf", 0.20)))
    pipeline = ConfidenceGatedPipeline(MockDetector(fixtures), jev_client=None)

    resolved = pipeline.run("irrelevant.jpg")

    assert resolved[0].source == "escalation"


def test_config_rejects_inconsistent_thresholds():
    import pytest

    with pytest.raises(ValueError):
        PipelineConfig(high_confidence=0.5, ambiguous_floor=0.9)
