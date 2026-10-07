from fastapi import FastAPI, Response

from palletvision_demo.settings import Settings
from palletvision_demo.state import DemoState


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    state = DemoState(settings)
    app = FastAPI(title="palletvision demo", docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.settings = settings
    app.state.demo = state

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "camera_id": settings.camera_id, "device_id": settings.device_id}

    @app.get("/api/frame")
    def frame() -> Response:
        return Response(
            state.frame_jpeg, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"}
        )

    @app.get("/api/state")
    def get_state() -> dict[str, object]:
        return state.snapshot()

    return app
