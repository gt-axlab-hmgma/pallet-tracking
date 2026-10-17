import argparse
from collections.abc import Sequence
from pathlib import Path

from palletvision_infer.backends import OnnxRuntimeBackend
from palletvision_infer.detections import Detector
from palletvision_infer.media.frame_source import parse_source
from palletvision_infer.pipeline import DummyDetector, run


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="pv-infer", description="Run detection on a video or webcam.")
    p.add_argument("--source", required=True, help="camera index (0 = laptop webcam) or video path")
    p.add_argument("--weights", type=Path, help="ONNX model; omit to use the dummy detector")
    p.add_argument("--out", type=Path, help="write the annotated video here")
    p.add_argument("--fps", type=float, default=10.0, help="process at this rate")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--show", action="store_true", help="live window, press q to quit")
    args = p.parse_args(argv)

    detector: Detector = (
        OnnxRuntimeBackend(args.weights, input_size=args.size, conf=args.conf)
        if args.weights
        else DummyDetector()
    )
    n = run(parse_source(args.source), detector, args.out, args.fps, args.show)
    print(f"processed {n} frames")
