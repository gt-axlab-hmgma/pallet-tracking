from fastapi.testclient import TestClient

from palletvision_demo.app import create_app
from palletvision_demo.settings import Settings


def test_health_reports_identity() -> None:
    client = TestClient(create_app(Settings(camera_id="camX", device_id="devY")))
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "camera_id": "camX", "device_id": "devY"}
