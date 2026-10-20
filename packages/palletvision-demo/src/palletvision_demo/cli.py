import argparse
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from palletvision_demo.app import create_app
from palletvision_demo.settings import Settings


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="pv-demo", description="Start the demo API (the UI runs separately)."
    )
    p.add_argument(
        "--source", help="reference frame (image, video or camera index), or the video to track"
    )
    p.add_argument("--camera-id", default="cam01")
    p.add_argument("--device-id", default="demo-laptop")
    p.add_argument("--sink", choices=["jsonl", "postgres", "outbox", "stdout"], default="jsonl")
    p.add_argument("--jsonl", type=Path, default=Path("runs/demo/events.jsonl"))
    p.add_argument("--outbox", type=Path, default=Path("runs/outbox.db"))
    p.add_argument("--db-url", default=None, help="Postgres URL (default: from PV_DB_* / .env)")
    p.add_argument("--hysteresis", type=int, default=1, help="frames needed to confirm a move")
    live = p.add_argument_group("live tracking", "run a video through the detector and tracker")
    live.add_argument(
        "--weights", type=Path, help="ONNX model; with --source video, tracks drive the zones"
    )
    live.add_argument("--fps", type=float, default=10.0, help="frames per second to process")
    live.add_argument("--conf", type=float, default=0.1, help="lowest detection score kept")
    live.add_argument(
        "--lost-seconds", type=float, default=5.0, help="how long a hidden pallet keeps its id"
    )
    live.add_argument("--no-loop", action="store_true", help="stop at the end of the video")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)
    settings = Settings(
        camera_id=args.camera_id,
        device_id=args.device_id,
        source=args.source,
        sink=args.sink,
        jsonl_path=args.jsonl,
        outbox_path=args.outbox,
        db_url=args.db_url,
        hysteresis_frames=args.hysteresis,
        weights=args.weights,
        fps=args.fps,
        conf=args.conf,
        lost_seconds=args.lost_seconds,
        loop=not args.no_loop,
    )
    uvicorn.run(create_app(settings), host=args.host, port=args.port)
