"""OpenRouter client implementation (video + image generation).

Video: POST /api/v1/videos -> {id, polling_url, status}; poll GET /api/v1/videos/{id}
until completed; download GET /api/v1/videos/{id}/content?index=0 with the same
Authorization header. Image: POST /api/v1/chat/completions with modalities
["image","text"], base64 result in message.images[0].image_url.url.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from dataclasses import replace
from typing import Any, cast

from services.http_client.http_client import HTTPClient, HttpTransportError
from services.openrouter_client.openrouter_client import (
    OpenRouterClientError,
    OpenRouterImageModel,
    OpenRouterVideoModel,
    ProgressCallback,
)
from services.services_utils import JSONValue

logger = logging.getLogger(__name__)

BASE_URL = "https://openrouter.ai/api/v1"

# Transient transport blips on one poll/download must not discard a minutes-long job.
_GET_RETRIES = 3
_POLL_INTERVAL_S = 5.0
_POLL_TIMEOUT_S = 900.0

_IN_PROGRESS_STATUSES = frozenset(
    {"queued", "pending", "processing", "running", "in_progress", "starting", "started"}
)


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _raise_for_response(resp: Any, label: str) -> None:
    if resp.status_code < 400:
        return
    detail = f"{label} failed (HTTP {resp.status_code})"
    provider_message: str | None = None
    try:
        payload = resp.json()
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict) and isinstance(err.get("message"), str):
                provider_message = cast(str, err["message"])
                detail = f"{label} failed: {provider_message}"
            elif isinstance(err, str):
                provider_message = err
                detail = f"{label} failed: {err}"
    except Exception:
        pass
    raise OpenRouterClientError(resp.status_code, detail, provider_message=provider_message)


def _get_with_retries(http: HTTPClient, url: str, api_key: str, label: str) -> Any:
    last_error: Exception | None = None
    for attempt in range(_GET_RETRIES):
        try:
            resp = http.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=120)
            _raise_for_response(resp, label)
            return resp
        except HttpTransportError as e:
            last_error = e
            logger.warning(
                "%s transport error (attempt %d/%d): %s", label, attempt + 1, _GET_RETRIES, e
            )
            time.sleep(2.0 * (attempt + 1))
    raise OpenRouterClientError(502, f"{label} failed after {_GET_RETRIES} attempts: {last_error}")


def _as_str_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(v) for v in value if isinstance(v, (str, int, float)))


def _as_int_tuple(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    out: list[int] = []
    for v in value:
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            continue
    return tuple(sorted(out))


def _min_price_per_second(skus: object) -> float | None:
    """OpenRouter reports per-model pricing_skus; units differ per provider (per second
    or per video token). We surface the smallest positive number as a *relative*
    cheapness hint only — the UI labels it 'price index', never a currency promise."""
    if not isinstance(skus, dict):
        return None
    vals: list[float] = []
    for v in skus.values():
        try:
            f = float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if f > 0:
            vals.append(f)
    return min(vals) if vals else None


class OpenRouterClientImpl:
    """Concrete OpenRouterClient (satisfies the Protocol structurally)."""

    def __init__(self, http: HTTPClient) -> None:
        self._http = http

    # ---- models ----

    def list_video_models(self, *, api_key: str) -> list[OpenRouterVideoModel]:
        resp = _get_with_retries(
            self._http, f"{BASE_URL}/videos/models", api_key, "list video models"
        )
        payload = resp.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise OpenRouterClientError(502, "list video models: unexpected payload shape")
        models: list[OpenRouterVideoModel] = []
        for item in cast(list[JSONValue], payload["data"]):
            if not isinstance(item, dict):
                continue
            mid = item.get("id")
            if not isinstance(mid, str):
                continue
            models.append(
                OpenRouterVideoModel(
                    id=mid,
                    display_name=str(item.get("name") or mid),
                    supported_resolutions=_as_str_tuple(item.get("supported_resolutions")),
                    supported_durations=_as_int_tuple(item.get("supported_durations")),
                    supported_aspect_ratios=_as_str_tuple(item.get("supported_aspect_ratios")),
                    supported_frame_images=_as_str_tuple(item.get("supported_frame_images")),
                    min_price_per_second=_min_price_per_second(item.get("pricing_skus")),
                    supports_audio=bool(item.get("supports_audio", False)),
                )
            )
        return models

    # ---- video ----

    def generate_video(
        self,
        *,
        api_key: str,
        model: str,
        prompt: str,
        duration: int | None,
        resolution: str | None,
        aspect_ratio: str | None,
        generate_audio: bool,
        first_frame_data_url: str | None = None,
        last_frame_data_url: str | None = None,
        seed: int | None = None,
        on_progress: ProgressCallback | None = None,
        raise_if_cancelled: Any | None = None,
    ) -> bytes:
        payload: dict[str, JSONValue] = {
            "model": model,
            "prompt": prompt,
            "generate_audio": generate_audio,
        }
        if duration is not None:
            payload["duration"] = duration
        if resolution is not None:
            payload["resolution"] = resolution
        if aspect_ratio is not None:
            payload["aspect_ratio"] = aspect_ratio
        if seed is not None:
            payload["seed"] = seed
        frame_images: list[dict[str, JSONValue]] = []
        if first_frame_data_url:
            frame_images.append(
                {
                    "type": "image_url",
                    "image_url": {"url": first_frame_data_url},
                    "frame_type": "first_frame",
                }
            )
        if last_frame_data_url:
            frame_images.append(
                {
                    "type": "image_url",
                    "image_url": {"url": last_frame_data_url},
                    "frame_type": "last_frame",
                }
            )
        if frame_images:
            payload["frame_images"] = cast(JSONValue, frame_images)

        logger.info(
            "openrouter submit: model=%s duration=%s resolution=%s frames=%d",
            model, duration, resolution, len(frame_images),
        )
        resp = self._http.post(
            f"{BASE_URL}/videos", headers=_headers(api_key), json_payload=payload, timeout=120
        )
        _raise_for_response(resp, "submit video job")
        job = resp.json()
        if not isinstance(job, dict) or not isinstance(job.get("id"), str):
            raise OpenRouterClientError(502, "submit video job: unexpected payload shape")
        job_id = cast(str, job["id"])

        deadline = time.monotonic() + _POLL_TIMEOUT_S
        status_payload: dict[str, JSONValue] = cast(dict[str, JSONValue], job)
        while True:
            if raise_if_cancelled is not None:
                raise_if_cancelled()
            status = status_payload.get("status")
            if status == "completed":
                break
            if isinstance(status, str) and status not in _IN_PROGRESS_STATUSES:
                err = status_payload.get("error")
                msg = err if isinstance(err, str) else None
                if isinstance(err, dict) and isinstance(err.get("message"), str):
                    msg = cast(str, err["message"])
                raise OpenRouterClientError(
                    502,
                    f"video job ended with status '{status}'{f': {msg}' if msg else ''}",
                    provider_message=msg,
                )
            if time.monotonic() > deadline:
                raise OpenRouterClientError(504, f"video job {job_id} timed out")
            time.sleep(_POLL_INTERVAL_S)
            if on_progress is not None:
                on_progress("inference", 55)
            poll = _get_with_retries(
                self._http, f"{BASE_URL}/videos/{job_id}", api_key, "poll video job"
            )
            fresh = poll.json()
            if isinstance(fresh, dict):
                status_payload = cast(dict[str, JSONValue], fresh)

        if on_progress is not None:
            on_progress("downloading_output", 85)
        content = _get_with_retries(
            self._http, f"{BASE_URL}/videos/{job_id}/content?index=0", api_key, "download video"
        )
        return content.content

    # ---- image ----

    def generate_image(
        self,
        *,
        api_key: str,
        model: str,
        prompt: str,
        aspect_ratio: str | None = None,
        resolution: str | None = None,
        input_data_urls: list[str] | None = None,
    ) -> bytes:
        # Native image endpoint (verified: gen ~7 s, edit ~9 s — faster than
        # chat/completions and it honors aspect_ratio/resolution/input_references).
        payload: dict[str, JSONValue] = {"model": model, "prompt": prompt, "n": 1}
        if aspect_ratio:
            payload["aspect_ratio"] = aspect_ratio
        if resolution:
            payload["resolution"] = resolution
        if input_data_urls:
            payload["input_references"] = [
                {"type": "image_url", "image_url": {"url": u}} for u in input_data_urls
            ]
        resp = self._http.post(
            f"{BASE_URL}/images",
            headers=_headers(api_key),
            json_payload=payload,
            timeout=300,
        )
        _raise_for_response(resp, "generate image")
        data = resp.json()
        try:
            images = data["data"]  # type: ignore[index]
            b64 = images[0]["b64_json"]  # type: ignore[index]
        except (KeyError, IndexError, TypeError) as e:
            raise OpenRouterClientError(502, "generate image: unexpected payload shape") from e
        if not isinstance(b64, str):
            raise OpenRouterClientError(502, "generate image: missing image payload")
        return base64.b64decode(b64)

    # image model catalog: GET /api/v1/images/models (dedicated, richer than /models filter)
    def list_image_models(self, *, api_key: str) -> list[OpenRouterImageModel]:
        resp = _get_with_retries(self._http, f"{BASE_URL}/images/models", api_key, "list image models")
        payload = resp.json()
        out: list[OpenRouterImageModel] = []
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            return out
        for item in cast(list[JSONValue], data):
            if not isinstance(item, dict):
                continue
            mid = item.get("id")
            if not isinstance(mid, str):
                continue
            params = item.get("supported_parameters")
            ratios: tuple[str, ...] = ()
            has_resolution = False
            max_refs = 0
            # Some models (e.g. meta/muse-image) ship an empty supported_parameters
            # block but accept reference images anyway — the reliable signal is
            # architecture.input_modalities containing "image".
            arch = item.get("architecture")
            if isinstance(arch, dict):
                mods = arch.get("input_modalities")
                if isinstance(mods, list) and "image" in mods:
                    max_refs = max(max_refs, 1)
            if isinstance(params, dict):
                ar = params.get("aspect_ratio")
                if isinstance(ar, dict) and isinstance(ar.get("values"), list):
                    ratios = tuple(str(v) for v in cast(list[JSONValue], ar["values"]))
                has_resolution = "resolution" in params
                refs = params.get("input_references")
                if isinstance(refs, dict):
                    try:
                        max_refs = int(refs.get("max", 0))  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        max_refs = 0
            price: float | None = None
            pricing = item.get("pricing")
            if isinstance(pricing, dict):
                img = pricing.get("image")
                try:
                    f = float(img)  # type: ignore[arg-type]
                    if f > 0:
                        price = f
                except (TypeError, ValueError):
                    pass
            out.append(
                OpenRouterImageModel(
                    id=mid,
                    display_name=str(item.get("name") or mid),
                    price_per_image=price,
                    supported_aspect_ratios=ratios,
                    supports_resolution=has_resolution,
                    max_input_references=max_refs,
                )
            )
        _enrich_image_prices(self._http, api_key, out)
        return out


def _endpoint_pricing(http: HTTPClient, api_key: str, model_id: str) -> list[dict[str, JSONValue]]:
    resp = http.get(
        f"{BASE_URL}/images/models/{model_id}/endpoints",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=15,
    )
    payload = resp.json()
    eps = payload.get("endpoints") if isinstance(payload, dict) else None
    if not isinstance(eps, list):
        return []
    entries: list[dict[str, JSONValue]] = [
        e for e in cast(list[JSONValue], eps) if isinstance(e, dict)
    ]

    # Cheapest provider's pricing table (OpenRouter routes to the cheapest anyway).
    def _cheapest(entry: dict[str, JSONValue]) -> float:
        pr = entry.get("pricing")
        if not isinstance(pr, list):
            return float("inf")
        costs: list[float] = []
        for p in cast(list[JSONValue], pr):
            if isinstance(p, dict):
                cost = p.get("cost_usd")
                if isinstance(cost, (int, float)):
                    costs.append(float(cost))
        return min(costs) if costs else float("inf")

    best = min(entries, key=_cheapest, default=None)
    if best is None:
        return []
    pr = best.get("pricing")
    if not isinstance(pr, list):
        return []
    return [p for p in cast(list[JSONValue], pr) if isinstance(p, dict)]


def _enrich_image_prices(http: HTTPClient, api_key: str, models: list[OpenRouterImageModel]) -> None:
    """Fill price_per_image from GET /images/models/{id}/endpoints.

    The image catalog itself carries no pricing; the per-model endpoints
    endpoint does. unit=image entries are exact per-image costs; unit=token
    models (per-token billing) get a price_note instead so the picker never
    shows a misleading flat price. Mutates in place; a failed lookup just
    leaves the price unset.
    """
    for i, m in enumerate(models):
        if m.price_per_image is not None:
            continue
        try:
            pricing = _endpoint_pricing(http, api_key, m.id)
        except (HttpTransportError, OpenRouterClientError, ValueError) as e:
            logger.debug("openrouter image price lookup failed for %s: %s", m.id, e)
            continue
        per_image: float | None = None
        per_mtok: float | None = None
        for p in pricing:
            try:
                cost = float(p["cost_usd"])  # type: ignore[arg-type]
            except (KeyError, TypeError, ValueError):
                continue
            unit = p.get("unit")
            billable = str(p.get("billable") or "")
            if unit == "image" and billable == "output_image" and cost > 0:
                per_image = cost if per_image is None else min(per_image, cost)
            elif unit == "token" and billable == "output_image" and cost > 0:
                per_mtok = cost * 1_000_000 if per_mtok is None else min(per_mtok, cost * 1_000_000)
        if per_image is not None:
            models[i] = replace(m, price_per_image=per_image)
        elif per_mtok is not None:
            models[i] = replace(m, price_note=f"~${per_mtok:,.2f}/M out-tokens")
