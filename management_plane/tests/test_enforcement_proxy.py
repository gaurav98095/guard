"""Tests for the /api/v1/enforce proxy endpoint."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from app.main import app
from app.auth import User, get_current_user
from app.models import ComparisonResult
from app.services.dataplane_client import DataPlaneError

TEST_USER = User(id="tenant_test", email="test@example.com", role="authenticated")


@pytest.fixture(autouse=True)
def override_auth():
    app.dependency_overrides[get_current_user] = lambda: TEST_USER
    yield
    app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def _intent_payload() -> dict:
    return {
        "id": "intent-1",
        "schemaVersion": "v1.3",
        "tenantId": "tenant_test",
        "timestamp": 1699564800.0,
        "actor": {"id": "agent-123", "type": "agent"},
        "action": "read",
        "resource": {"type": "database", "name": "users", "location": "cloud"},
        "data": {"sensitivity": ["internal"], "pii": False, "volume": "single"},
        "risk": {"authn": "required"},
        "layer": "L4",
        "tool_name": "get_weather",
        "tool_method": "query",
        "tool_params": {"location": "SF"},
    }


def test_enforce_intent_proxies_vector(client: TestClient):
    mock_result = ComparisonResult(
        decision=1,
        slice_similarities=[0.9, 0.8, 0.7, 0.6],
        boundaries_evaluated=1,
        timestamp=123.0,
        evidence=[],
    )
    mock_client = MagicMock()
    mock_client.enforce.return_value = mock_result

    with patch("app.endpoints.enforcement.get_data_plane_client", return_value=mock_client), \
         patch("app.endpoints.enforcement.encode_to_128d", return_value=np.ones(128)):
        response = client.post("/api/v1/enforce", json=_intent_payload())

    assert response.status_code == 200
    assert response.json()["decision"] == 1
    mock_client.enforce.assert_called_once()
    _, vector_arg = mock_client.enforce.call_args[0]
    assert isinstance(vector_arg, list)
    assert len(vector_arg) == 128


def test_enforce_intent_handles_dp_error(client: TestClient):
    mock_client = MagicMock()
    mock_client.enforce.side_effect = DataPlaneError("boom")

    with patch("app.endpoints.enforcement.get_data_plane_client", return_value=mock_client), \
         patch("app.endpoints.enforcement.encode_to_128d", return_value=np.ones(128)):
        response = client.post("/api/v1/enforce", json=_intent_payload())

    assert response.status_code == 502
    assert response.json()["detail"].startswith("Data Plane error")
