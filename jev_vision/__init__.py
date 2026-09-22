"""jev-vision: confidence-gated escalation from a fast detector to Jev."""

from jev_vision.config import PipelineConfig
from jev_vision.detector import ClassCandidate, Detection, MockDetector, YoloDetector
from jev_vision.jev_client import JevClient, JevError
from jev_vision.pipeline import ConfidenceGatedPipeline, ResolvedDetection

__all__ = [
    "ClassCandidate",
    "ConfidenceGatedPipeline",
    "Detection",
    "JevClient",
    "JevError",
    "MockDetector",
    "PipelineConfig",
    "ResolvedDetection",
    "YoloDetector",
]
