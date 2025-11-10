from __future__ import annotations
import hashlib
import os
import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("PROVIDER", "mock")
HF_API_URL = os.getenv(
    "HF_API_URL",
    "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
)
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
EXTERNAL_TIMEOUT = float(os.getenv("EXTERNAL_TIMEOUT", "5.0"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("ml-classifier")

app = FastAPI(title="ML Text Classifier")


class ClassifyRequest(BaseModel):
    text: str


class Label(BaseModel):
    label: str
    score: float


class ClassifyResponse(BaseModel):
    labels: List[Label]
    from_cache: bool
    meta: Dict[str, Any] = {}


class InMemoryCache:
    def __init__(self):
        self._store: Dict[str, Dict] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Dict]:
        async with self._lock:
            rec = self._store.get(key)
            if not rec:
                return None

            if rec["expires_at"] < datetime.utcnow():
                del self._store[key]
                return None

            return rec["value"]

    async def set(self, key: str, value: Dict, ttl: int = CACHE_TTL):
        async with self._lock:
            self._store[key] = {
                "value": value,
                "expires_at": datetime.utcnow() + timedelta(seconds=ttl)
            }


cache = InMemoryCache()


def make_key(text: str) -> str:
    return "classify:" + hashlib.sha256(text.encode()).hexdigest()


async def call_huggingface(text: str) -> List[Dict[str, Any]]:
    headers = {}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"

    logger.info("Calling HF API for text: %s", text[:120])

    async with httpx.AsyncClient(timeout=EXTERNAL_TIMEOUT) as client:
        resp = await client.post(HF_API_URL, json={"inputs": text}, headers=headers)
        resp.raise_for_status()
        return resp.json()


async def call_mock(text: str) -> List[Dict[str, Any]]:
    await asyncio.sleep(0.1)
    t = text.lower()
    if "good" in t or "love" in t:
        return [{"label": "POSITIVE", "score": 0.95}]
    if "bad" in t or "hate" in t:
        return [{"label": "NEGATIVE", "score": 0.9}]
    return [{"label": "NEUTRAL", "score": 0.6}]


async def call_provider(text: str) -> List[Dict[str, Any]]:
    if PROVIDER == "hf":
        return await call_huggingface(text)
    return await call_mock(text)


def normalize(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        if len(raw) > 0 and isinstance(raw[0], list):
            raw = raw[0]
        final = []
        for item in raw:
            if isinstance(item, dict) and "label" in item and "score" in item:
                final.append({
                    "label": str(item["label"]),
                    "score": float(item["score"])
                })
        if final:
            return final
    return []


@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    key = make_key(req.text)

    cached = await cache.get(key)
    if cached:
        return ClassifyResponse(
            labels=[Label(**l) for l in cached["labels"]],
            from_cache=True,
            meta={"provider": cached["provider"]}
        )

    try:
        raw = await call_provider(req.text)
    except httpx.TimeoutException:
        logger.exception("Timeout from external ML API")
        raise HTTPException(502, "External service timeout")
    except httpx.HTTPError:
        logger.exception("HTTP error from external ML API")
        raise HTTPException(502, "External ML service error")
    except Exception:
        logger.exception("Unexpected provider error")
        raise HTTPException(500, "Unexpected ML provider error")

    labels = normalize(raw)
    if not labels:
        logger.warning(
            "Provider returned unexpected format: %s", str(raw)[:300])
        raise HTTPException(502, "Invalid response format from ML provider")

    result = {"labels": labels, "provider": PROVIDER}

    try:
        await cache.set(key, result, ttl=CACHE_TTL)
    except Exception:
        logger.exception("Cache write failed")

    return ClassifyResponse(
        labels=[Label(**l) for l in labels],
        from_cache=False,
        meta={"provider": PROVIDER}
    )


@app.get("/health")
async def health():
    return {"status": "ok", "provider": PROVIDER}
