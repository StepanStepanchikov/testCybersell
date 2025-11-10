import pytest
from httpx import AsyncClient, ASGITransport
from testKsell.main import app


@pytest.mark.asyncio
async def test_classify_success():
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/classify", json={"text": "good"})

    assert response.status_code == 200
    data = response.json()

    assert "labels" in data
    assert isinstance(data["labels"], list)
    assert data["from_cache"] in [True, False]


@pytest.mark.asyncio
async def test_classify_error():
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/classify", json={"wrongField": "nope"})

    assert response.status_code == 422
