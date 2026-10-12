"""Translate the OpenRouter video-model catalog into Desktop model specs.

Each OpenRouter model becomes one Desktop spec item whose `pipeline` id is
"or:<openrouter-id>". The Desktop option pickers (model/resolution/duration/
aspect ratio) are driven entirely by these specs, so the catalog drives the UI.
"""

from __future__ import annotations

import logging
import time
from typing import cast

from api_types import (
    LTXOfferingCapabilitiesSpec,
    LTXVideoGenAspectRatio,
    LTXVideoGenDuration,
    LTXVideoGenFps,
    LTXVideoGenResolution,
    LTXVideoGenerationModelSpecItem,
    LTXVideoGenerationResolutionSpec,
    LTXVideoGenerationSpec,
    OPENROUTER_PIPELINE_PREFIX,
)
from services.openrouter_client.openrouter_client import (
    OpenRouterClient,
    OpenRouterClientError,
    OpenRouterVideoModel,
)

logger = logging.getLogger(__name__)

# Desktop fps is a request field but OpenRouter ignores it; advertise 24 only.
_OR_FPS: LTXVideoGenFps = 24

# Values the Desktop pickers/Literals can express; anything else in a catalog row is
# filtered out rather than shown broken.
_ALLOWED_DURATIONS: tuple[LTXVideoGenDuration, ...] = (2, 3, 4, 5, 6, 8, 10, 12, 14, 15, 16, 18, 20, 24, 25, 30)
_ALLOWED_RESOLUTIONS: tuple[LTXVideoGenResolution, ...] = (
    "270p", "360p", "480p", "540p", "720p", "768p", "1080p", "1440p", "2K", "2160p", "4K"
)
_ALLOWED_ASPECTS: tuple[LTXVideoGenAspectRatio, ...] = (
    "21:9", "16:9", "3:2", "4:3", "1:1", "4:5", "9:16", "3:4", "9:21"
)


def openrouter_pipeline_id(or_model_id: str) -> str:
    return f"{OPENROUTER_PIPELINE_PREFIX}{or_model_id}"


def or_model_id_from_pipeline(pipeline: str) -> str | None:
    if pipeline.startswith(OPENROUTER_PIPELINE_PREFIX):
        return pipeline[len(OPENROUTER_PIPELINE_PREFIX):]
    return None


def is_openrouter_pipeline(pipeline: str) -> bool:
    return pipeline.startswith(OPENROUTER_PIPELINE_PREFIX)


def spec_from_openrouter_model(model: OpenRouterVideoModel) -> LTXVideoGenerationModelSpecItem | None:
    """One catalog row -> one spec item, or None when nothing usable survives filtering."""
    resolutions = cast(
        tuple[LTXVideoGenResolution, ...],
        tuple(r for r in _ALLOWED_RESOLUTIONS if r in model.supported_resolutions),
    )
    durations = cast(
        tuple[LTXVideoGenDuration, ...],
        tuple(d for d in _ALLOWED_DURATIONS if d in model.supported_durations),
    )
    aspects = cast(
        tuple[LTXVideoGenAspectRatio, ...],
        tuple(a for a in _ALLOWED_ASPECTS if a in model.supported_aspect_ratios),
    )
    if not resolutions or not durations:
        logger.info("openrouter spec skipped (no usable res/duration): %s", model.id)
        return None
    if not aspects:
        aspects = ("16:9",)
    resolution_map: dict[LTXVideoGenResolution, LTXVideoGenerationResolutionSpec] = {
        r: LTXVideoGenerationResolutionSpec(
            fps_to_durations={_OR_FPS: list(durations)},
            aspect_ratios=list(aspects),
        )
        for r in resolutions
    }
    has_first = "first_frame" in model.supported_frame_images
    has_last = "last_frame" in model.supported_frame_images
    price_hint = model.min_price_per_second
    display = model.display_name
    # Frame capability so the picker shows what each model takes before
    # the user hits generate ("Start+End" vs "Start or End" vs "Start").
    if has_first and has_last:
        frames = "Start+End" if model.supports_frame_combo else "Start or End"
    elif has_first:
        frames = "Start"
    else:
        frames = ""
    bits = [
        b for b in (
            frames,
            f"~${price_hint:,.4f}/s" if price_hint is not None else "",
        ) if b
    ]
    if bits:
        display = f"{display} ({' · '.join(bits)})"
    return LTXVideoGenerationModelSpecItem(
        pipeline=openrouter_pipeline_id(model.id),
        spec=LTXVideoGenerationSpec(
            display_name=display,
            supported_resolutions_durations=resolution_map,
            capabilities=LTXOfferingCapabilitiesSpec(
                t2v=True,
                i2v=has_first,
                a2v=False,
                ic_lora=False,
                retake=False,
                extend=False,
                multi_keyframe=False,
                multi_keyframe_max_count=0,
                user_loras=False,
                camera_motion=False,
                auto_duration=False,
                last_frame=has_last,
            ),
        ),
    )


def specs_from_openrouter_catalog(models: list[OpenRouterVideoModel]) -> list[LTXVideoGenerationModelSpecItem]:
    # Cheapest first (price index); models without a price go last.
    ordered = sorted(models, key=lambda m: (m.min_price_per_second is None,
                                            m.min_price_per_second if m.min_price_per_second is not None else 0.0,
                                            m.id))
    items: list[LTXVideoGenerationModelSpecItem] = []
    for m in ordered:
        item = spec_from_openrouter_model(m)
        if item is not None:
            items.append(item)
    return items


class OpenRouterCatalogCache:
    """TTL cache so opening the picker doesn't hit OpenRouter on every request."""

    def __init__(self, ttl_seconds: float = 300.0) -> None:
        self._ttl = ttl_seconds
        self._fetched_at = 0.0
        self._key: str = ""
        self._items: list[LTXVideoGenerationModelSpecItem] = []
        self._raw: list[OpenRouterVideoModel] = []

    def get(
        self, client: OpenRouterClient, api_key: str
    ) -> tuple[list[LTXVideoGenerationModelSpecItem], list[OpenRouterVideoModel]]:
        now = time.monotonic()
        if self._key == api_key and self._raw and (now - self._fetched_at) < self._ttl:
            return self._items, self._raw
        try:
            raw = client.list_video_models(api_key=api_key)
        except OpenRouterClientError as e:
            logger.warning("openrouter catalog fetch failed: %s", e.detail)
            # Serve stale rather than an empty picker; empty only if we never fetched.
            return self._items, self._raw
        self._raw = raw
        self._items = specs_from_openrouter_catalog(raw)
        self._fetched_at = now
        self._key = api_key
        return self._items, raw

    def find(self, or_model_id: str) -> OpenRouterVideoModel | None:
        for m in self._raw:
            if m.id == or_model_id:
                return m
        return None

    def invalidate(self) -> None:
        self._fetched_at = 0.0
        self._key = ""
