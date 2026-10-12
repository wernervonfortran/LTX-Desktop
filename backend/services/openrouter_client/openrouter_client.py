"""OpenRouter video/image generation client service protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


class OpenRouterClientError(RuntimeError):
    def __init__(
        self,
        status_code: int,
        detail: str,
        *,
        provider_message: str | None = None,
    ) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.provider_message = provider_message


@dataclass(frozen=True, slots=True)
class OpenRouterVideoModel:
    """One entry of GET /api/v1/videos/models, normalized for the Desktop UI."""

    id: str
    display_name: str
    supported_resolutions: tuple[str, ...]
    supported_durations: tuple[int, ...]
    supported_aspect_ratios: tuple[str, ...]
    supported_frame_images: tuple[str, ...]
    min_price_per_second: float | None
    supports_audio: bool
    # True when the model accepts first_frame AND last_frame in one job.
    # The catalog's supported_frame_images only lists individually accepted
    # slots; MiniMax H3/H3 Max list both but reject the pair at submit time
    # ("single keyframe image") — confirmed by live probe 2026-10-13.
    supports_frame_combo: bool = False


@dataclass(frozen=True, slots=True)
class OpenRouterImageModel:
    id: str
    display_name: str
    price_per_image: float | None
    supported_aspect_ratios: tuple[str, ...] = ()
    supports_resolution: bool = False
    max_input_references: int = 0
    price_note: str | None = None


ProgressCallback = Callable[[str, int], None]


class OpenRouterClient(Protocol):
    def list_video_models(self, *, api_key: str) -> list[OpenRouterVideoModel]: ...

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
        raise_if_cancelled: Callable[[], None] | None = None,
    ) -> bytes: ...

    def generate_image(
        self,
        *,
        api_key: str,
        model: str,
        prompt: str,
        aspect_ratio: str | None = None,
        resolution: str | None = None,
        input_data_urls: list[str] | None = None,
    ) -> bytes: ...

    def list_image_models(self, *, api_key: str) -> list[OpenRouterImageModel]: ...
