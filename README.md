# jev-vision

Confidence-gated object recognition: a fast detector handles the easy cases,
[Jev](https://docs.typesafe.ai) (TypeSafe's System One model) disambiguates
the ones it can't call.

## Why

Object detectors are fast at localization but their top-1 label is just an
argmax over a softmax — when two classes are genuinely close (a husky vs. a
malamute, a bus vs. a truck at a bad angle), the detector has no way to use
context it was never trained on: *this camera is mounted in a kennel*,
*this route never sees buses*. Re-running a full vision-language model on
every ambiguous crop is one way to add that context back in, but it's slow
and expensive to do on every frame.

Jev is text-only — it can't see pixels — but it's fast (~150ms) and cheap,
and it's built to take a small set of candidates plus arbitrary context and
return a calibrated probability distribution over them. So the shape that
falls out is:

```
detector: box + ranked candidates
    │
    ├── confident? ──────────────────────► accept, done
    │
    ├── ambiguous? ── candidates + context ──► Jev Choice ──► accept or escalate
    │
    └── no signal? ──────────────────────► escalate directly (skip Jev)
```

Nothing about this makes recognition *faster* per frame with Jev in the
loop — it's an extra network call on the ambiguous slice only. What it buys
you is disambiguation the detector structurally cannot do on its own,
paid for only where it's needed, batched into as few round trips as
possible.

## How it routes

Three thresholds in [`PipelineConfig`](jev_vision/config.py) decide where a
detection goes:

| Detector top-1 confidence | Route | Jev called? |
|---|---|---|
| `>= high_confidence` (default `0.85`) | accepted as-is | no |
| `[ambiguous_floor, high_confidence)` (default `0.15`–`0.85`) | sent to Jev | yes |
| `< ambiguous_floor` (default `0.15`) | escalated to human review | no — too weak to disambiguate |

If Jev answers but its own `confidence` is below `jev_confidence_floor`
(default `0.5`), the detection still escalates rather than trusting a
low-confidence pick.

**Every ambiguous detection in a frame is batched into one Jev call.**
Questions run in parallel server-side (TypeSafe's "speculative fan-out"
pattern), so five ambiguous boxes cost about the same latency as one.

## Project layout

```
jev_vision/
  detector.py     Detector protocol, MockDetector (fixtures, no weights needed),
                   YoloDetector (real backend, optional `yolo` extra)
  jev_client.py    Thin wrapper around TypeSafe's /v1/systemone endpoint
  pipeline.py      ConfidenceGatedPipeline — the routing logic above
  config.py        PipelineConfig — the three thresholds, validated
examples/
  run_demo.py      CLI: prints a routing table for a frame
tests/
  test_pipeline.py Covers all three routes + the low-Jev-confidence fallback
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# offline: MockDetector fixtures, no API key or model weights needed
python examples/run_demo.py
```

```
label        confidence   source
----------------------------------------
cat          0.97         detector only
husky        0.42         escalated to human review
bag          0.12         escalated to human review

3 detections: 0 resolved by Jev, 2 escalated.
```

The husky/malamute/wolf fixture escalates instead of guessing because no
Jev client is configured. Give it one:

```bash
cp .env.example .env   # add your TYPESAFE_API_KEY
export $(cat .env | xargs)
python examples/run_demo.py --live --context \
  "Camera is mounted in a dog kennel; only breeds, never wolves, are ever present."
```

Now the ambiguous case routes through Jev instead of escalating, and the
context (which the vision model never saw) resolves it toward the actual
breed rather than the raw top-1 confidence.

### Using a real detector

```bash
pip install -e ".[yolo]"
python examples/run_demo.py --real --image path/to/photo.jpg
```

`YoloDetector` uses Ultralytics YOLO for localization. Standard YOLO
detection heads emit a single class per box after NMS, not a probability
distribution — so without a crop-level classifier passed as
`YoloDetector(classifier=...)`, every detection reports one high-certainty
candidate and the pipeline never has a reason to call Jev. Plug in a
classifier that exposes `predict_proba(crop) -> list[(label, confidence)]`
to get genuine top-k candidates for the confidence gate to work with.

## Testing

```bash
pytest tests/ -v
```

## Library usage

```python
from jev_vision import ConfidenceGatedPipeline, PipelineConfig, MockDetector, JevClient

pipeline = ConfidenceGatedPipeline(
    detector=MockDetector(),
    jev_client=JevClient(),  # reads TYPESAFE_API_KEY from env
    config=PipelineConfig(
        high_confidence=0.9,
        ambiguous_floor=0.2,
        context="Retail shelf camera; grocery items only.",
    ),
)

for detection in pipeline.run("frame.jpg"):
    print(detection.label, detection.confidence, detection.source)
```

## Design notes

- **Jev is text-only.** It cannot see the image. Everything it's given is
  the detector's candidate labels plus whatever text context the caller
  supplies — it's a semantic judgment layer on top of perception, not a
  replacement for it.
- **The pipeline never invents confidence.** If Jev doesn't return one, or
  it's below `jev_confidence_floor`, the detection escalates rather than
  silently keeping a low-quality guess.
- **`MockDetector` is not a toy left over from testing** — it's the
  intended way to exercise and tune the routing thresholds and Jev
  prompting before wiring up a real, possibly expensive, detector.

## License

MIT — see [LICENSE](LICENSE).
