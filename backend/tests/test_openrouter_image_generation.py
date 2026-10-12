"""OpenRouter fork: hosted image generation routed through the OpenRouter client.

Covers: GET /api/generate-image/models-specs (key-gated catalog), POST routing for
"or:<model>" selections (gen + edit), validation gates (missing key, unknown model,
edit on a model without input references), and aspect/resolution mapping.
"""

from __future__ import annotations

from pathlib import Path

from services.openrouter_client.openrouter_client import (
    OpenRouterClientError,
    OpenRouterImageModel,
)
from tests.http_error_assertions import assert_http_error

_EDITABLE = OpenRouterImageModel(
    id="test/editable",
    display_name="Editable Test",
    price_per_image=0.03,
    supported_aspect_ratios=("1:1", "16:9", "9:16"),
    supports_resolution=True,
    max_input_references=3,
)

_GEN_ONLY = OpenRouterImageModel(
    id="test/gen-only",
    display_name="Gen Only Test",
    price_per_image=None,
    supported_aspect_ratios=(),
    supports_resolution=False,
    max_input_references=0,
)

_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200


def _use_openrouter_images(test_state, *, key: str = "or-key") -> None:
    test_state.state.app_settings.openrouter_api_key = key
    fake = test_state.openrouter_client
    fake.image_models = [_EDITABLE, _GEN_ONLY]
    fake.image_result = _PNG_BYTES


def _img_request(**overrides) -> dict:
    body = {"prompt": "a red cube", "width": 1024, "height": 576, "model": "or:test/editable"}
    body.update(overrides)
    return body


class TestImageModelSpecs:
    def test_no_key_returns_empty(self, client, fake_services):
        r = client.get("/api/generate-image/models-specs")
        assert r.status_code == 200
        assert r.json()["openrouter_models"] == []
        assert fake_services.openrouter_client.list_image_models_calls == []

    def test_key_returns_catalog_with_flags(self, client, test_state, fake_services):
        _use_openrouter_images(test_state)
        r = client.get("/api/generate-image/models-specs")
        assert r.status_code == 200
        items = r.json()["openrouter_models"]
        assert [i["model"] for i in items] == ["or:test/editable", "or:test/gen-only"]
        assert items[0]["supports_edit"] is True
        assert items[0]["price_per_image"] == 0.03
        assert items[0]["supported_aspect_ratios"] == ["1:1", "16:9", "9:16"]
        assert items[1]["supports_edit"] is False

    def test_catalog_fetch_failure_serves_empty_not_crash(self, client, test_state):
        _use_openrouter_images(test_state)
        test_state.openrouter_client.raise_on_list_image_models = OpenRouterClientError(
            502, "boom"
        )
        r = client.get("/api/generate-image/models-specs")
        assert r.status_code == 200
        assert r.json()["openrouter_models"] == []


class TestOpenRouterImageGenerate:
    def test_routes_to_openrouter_client(self, client, test_state, fake_services):
        _use_openrouter_images(test_state)
        r = client.post("/api/generate-image", json=_img_request())
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "complete"
        assert Path(data["image_paths"][0]).exists()
        assert Path(data["image_paths"][0]).read_bytes() == _PNG_BYTES

        calls = fake_services.openrouter_client.generate_image_calls
        assert len(calls) == 1
        assert calls[0]["model"] == "test/editable"
        # 1024x576 -> closest allowed ratio 16:9, long side < 1800 -> 1K
        assert calls[0]["aspect_ratio"] == "16:9"
        assert calls[0]["resolution"] == "1K"
        assert calls[0]["input_data_urls"] is None

    def test_gen_only_model_omits_aspect_and_resolution(self, client, test_state, fake_services):
        _use_openrouter_images(test_state)
        r = client.post(
            "/api/generate-image", json=_img_request(model="or:test/gen-only")
        )
        assert r.status_code == 200
        calls = fake_services.openrouter_client.generate_image_calls
        assert calls[-1]["aspect_ratio"] is None
        assert calls[-1]["resolution"] is None

    def test_edit_passes_input_data_url(self, client, test_state, fake_services, tmp_path):
        from PIL import Image

        _use_openrouter_images(test_state)
        src = tmp_path / "src.png"
        Image.new("RGB", (64, 64), "red").save(src, format="PNG")

        r = client.post(
            "/api/generate-image",
            json=_img_request(imagePath=str(src), strength=0.5),
        )
        assert r.status_code == 200
        calls = fake_services.openrouter_client.generate_image_calls
        urls = calls[-1]["input_data_urls"]
        assert urls and urls[0].startswith("data:image/")

    def test_edit_on_model_without_references_rejected(self, client, test_state, tmp_path):
        from PIL import Image

        _use_openrouter_images(test_state)
        src = tmp_path / "src.png"
        Image.new("RGB", (64, 64), "red").save(src, format="PNG")

        r = client.post(
            "/api/generate-image",
            json=_img_request(model="or:test/gen-only", imagePath=str(src)),
        )
        assert_http_error(
            r,
            status_code=422,
            message="Model test/gen-only does not support image editing (no input references)",
            code="INVALID_IMAGE_GENERATION_SPEC",
        )

    def test_missing_key_rejected(self, client, test_state):
        test_state.openrouter_client.image_models = [_EDITABLE]
        r = client.post("/api/generate-image", json=_img_request())
        assert_http_error(
            r,
            status_code=500,
            code="OPENROUTER_API_KEY_NOT_CONFIGURED",
            message="OPENROUTER_API_KEY_NOT_CONFIGURED",
        )

    def test_unknown_model_rejected(self, client, test_state):
        _use_openrouter_images(test_state)
        r = client.post("/api/generate-image", json=_img_request(model="or:test/nope"))
        assert_http_error(
            r,
            status_code=422,
            message="Unknown OpenRouter image model: test/nope",
            code="INVALID_IMAGE_GENERATION_SPEC",
        )

    def test_provider_error_maps_to_http(self, client, test_state):
        _use_openrouter_images(test_state)
        test_state.openrouter_client.raise_on_generate_image = OpenRouterClientError(
            429, "rate limited"
        )
        r = client.post("/api/generate-image", json=_img_request())
        assert r.status_code == 429

    def test_local_path_untouched_without_model_field(self, client, create_fake_model_files):
        create_fake_model_files(include_zit=True)
        r = client.post(
            "/api/generate-image", json={"prompt": "local still works", "width": 64, "height": 64}
        )
        assert r.status_code == 200
        assert r.json()["status"] == "complete"
