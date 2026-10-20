import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from palletvision_demo.live.runner import (
    LivePlayer,
    camera_label,
    parse_camera,
    redact,
    source_kind,
)
from palletvision_demo.live.state import DemoState, Result, ZoneInputError
from palletvision_demo.live.stream import sse_lines
from palletvision_demo.settings import Settings
from palletvision_infer.backends import OnnxRuntimeBackend
from palletvision_infer.detections import Detector

BOUNDARY = "frame"
VIDEO_SUFFIXES = {".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm"}
NO_MODEL = "No model is loaded. Start pv-demo with --weights path/to/model.onnx."


class ZonesBody(BaseModel):
    zones: list[dict[str, Any]]


class CameraBody(BaseModel):
    device: str


def _result(result: Result) -> dict[str, Any]:
    return {
        "events": result.events,
        "sink_error": result.sink_error,
        "config_error": result.config_error,
    }


def create_app(settings: Settings | None = None, detector: Detector | None = None) -> FastAPI:
    """With ``settings.weights`` a video or camera can be tracked live. ``detector`` replaces
    the ONNX model that would be loaded from them (for tests)."""
    settings = settings or Settings()
    state = DemoState(settings)
    player: LivePlayer | None = None
    if settings.weights is not None:
        detector = detector or OnnxRuntimeBackend(settings.weights, conf=settings.conf)
        player = LivePlayer(state, detector, settings)

    def need_player() -> LivePlayer:
        if player is None:
            raise HTTPException(status_code=409, detail=NO_MODEL)
        return player

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if player and settings.source:
            kind = source_kind(settings.source)
            source: int | str = settings.source
            name = Path(settings.source).name
            if kind == "camera":
                source = parse_camera(settings.source)
                name = camera_label(source)
            try:
                await asyncio.to_thread(player.start, source, kind, name)
            except OSError as exc:
                state.set_source(kind, name, "error", redact(str(exc)))
        yield
        if player:
            await asyncio.to_thread(player.stop)

    app = FastAPI(
        title="palletvision demo",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.demo = state
    app.state.player = player

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "camera_id": settings.camera_id, "device_id": settings.device_id}

    @app.get("/api/frame")
    def frame() -> Response:
        """The frame zones are drawn on. The UI asks with ``?v=`` so a new source is not cached."""
        return Response(
            state.frame_jpeg, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"}
        )

    @app.get("/api/live/stream")
    async def live_stream(request: Request) -> StreamingResponse:
        """The video as motion JPEG, one part per processed frame, for an <img> to play."""

        async def parts() -> AsyncIterator[bytes]:
            seq = 0
            while not await request.is_disconnected():
                seq, jpeg = await asyncio.to_thread(state.wait_frame, seq, 1.0)
                if jpeg is None:
                    continue
                head = f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}"
                yield head.encode() + b"\r\n\r\n" + jpeg + b"\r\n"

        return StreamingResponse(
            parts(),
            media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/state")
    def get_state() -> dict[str, object]:
        return state.snapshot()

    @app.put("/api/source/video")
    async def upload_video(request: Request, name: str) -> dict[str, object]:
        """Save a video sent as the raw request body, then play it."""
        live = need_player()
        safe = Path(name).name  # a bare file name, never a path
        if not safe or safe.startswith(".") or Path(safe).suffix.lower() not in VIDEO_SUFFIXES:
            raise HTTPException(
                status_code=422,
                detail=f"choose a video file ({', '.join(sorted(VIDEO_SUFFIXES))})",
            )
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        dest = settings.upload_dir / safe
        part = dest.with_name(dest.name + ".part")
        size = 0
        with part.open("wb") as out:
            async for chunk in request.stream():
                size += len(chunk)
                await asyncio.to_thread(out.write, chunk)
        if size == 0:
            part.unlink()
            raise HTTPException(status_code=422, detail="that file is empty")
        part.replace(dest)
        try:
            await asyncio.to_thread(live.start, str(dest), "video", safe)
        except OSError as exc:
            raise HTTPException(
                status_code=422, detail=f"cannot play that video: {redact(str(exc))}"
            ) from exc
        return state.snapshot()

    @app.post("/api/source/camera")
    async def connect_camera(body: CameraBody) -> dict[str, object]:
        live = need_player()
        try:
            source = parse_camera(body.device)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        label = camera_label(source)
        try:
            await asyncio.to_thread(live.start, source, "camera", label)
        except OSError as exc:
            # not the library's message: for a stream URL it can carry the password
            raise HTTPException(
                status_code=422, detail=f"cannot open {label}. Is it connected and allowed?"
            ) from exc
        return state.snapshot()

    @app.delete("/api/source")
    async def eject_source() -> dict[str, object]:
        await asyncio.to_thread(need_player().eject)
        return state.snapshot()

    @app.put("/api/zones")
    def put_zones(body: ZonesBody) -> dict[str, object]:
        try:
            result = state.replace_zones(body.zones)
        except ZoneInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {**state.snapshot(), **_result(result)}

    @app.get("/api/events")
    def recent_events(limit: int = 100) -> dict[str, Any]:
        return {"events": state.recent_events(min(limit, 500))}

    @app.get("/api/events/stream")
    async def event_stream(request: Request) -> StreamingResponse:
        return StreamingResponse(
            sse_lines(state.bus, request.is_disconnected),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
