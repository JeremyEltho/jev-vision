#!/usr/bin/env python3
"""Run the confidence-gated pipeline and print where each detection was resolved.

Offline by default (MockDetector, no Jev call) so you can see the routing
logic without any API key or model weights:

    python examples/run_demo.py

Add --live to actually call Jev for the ambiguous fixture (needs
TYPESAFE_API_KEY set, e.g. via .env):

    python examples/run_demo.py --live --context "Camera is mounted in a dog kennel; only breeds, never wolves, are ever present."

Add --real --image path/to.jpg to swap in the YOLO backend instead of
the mock fixtures (needs the `yolo` extra installed).
"""

from __future__ import annotations

import argparse
import sys

from jev_vision.config import PipelineConfig
from jev_vision.detector import MockDetector
from jev_vision.pipeline import ConfidenceGatedPipeline

SOURCE_LABEL = {
    "detector": "detector only",
    "jev": "resolved by Jev",
    "escalation": "escalated to human review",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="demo.jpg", help="image path (ignored by MockDetector)")
    parser.add_argument("--live", action="store_true", help="actually call the Jev API")
    parser.add_argument("--real", action="store_true", help="use YoloDetector instead of MockDetector")
    parser.add_argument("--context", default="", help="scene/application context passed to Jev")
    args = parser.parse_args()

    if args.real:
        from jev_vision.detector import YoloDetector

        detector = YoloDetector()
    else:
        detector = MockDetector()

    jev_client = None
    if args.live:
        from jev_vision.jev_client import JevClient, JevError

        try:
            jev_client = JevClient()
        except JevError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    config = PipelineConfig(context=args.context)
    pipeline = ConfidenceGatedPipeline(detector, jev_client=jev_client, config=config)
    resolved = pipeline.run(args.image)

    print(f"{'label':<12} {'confidence':<12} source")
    print("-" * 40)
    for det in resolved:
        print(f"{det.label:<12} {det.confidence:<12.2f} {SOURCE_LABEL[det.source]}")

    n_jev = sum(1 for d in resolved if d.source == "jev")
    n_escalated = sum(1 for d in resolved if d.source == "escalation")
    print(f"\n{len(resolved)} detections: {n_jev} resolved by Jev, {n_escalated} escalated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
