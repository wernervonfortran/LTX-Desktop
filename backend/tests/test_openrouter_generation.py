"""OpenRouter fork: cloud video generation routed through the OpenRouter client.

Covers: settings key probe, catalog merge into model specs, /api/generate routing
for "or:<model>" pipelines, and the validation gates (missing key, unknown model,
unsupported duration/resolution/frame images).
"""

from __future__ import annotations

import pytest

from services.openrouter_client.openrouter_client import (
    OpenRouterClientError,
    OpenRouterVideoModel,
)
from tests.fakes import FakeResponse
from tests.http_error_assertions import assert_http_error

# A plausible fake catalog entry: Seedance-style mini model.
_MINI = OpenRouterVideoModel(
    id="test/mini",
    display_name="Mini Test",
    supported_resolutions=("480p", "720p", "1080p"),
    supported_durations=(4, 5, 8),
    supported_aspect_ratios=("16:9", "9:16", "1:1"),
    supported_frame_images=("first_frame", "last_frame"),
    min_price_per_second=0.03,
    supports_audio=True,
    supports_frame_combo=True,
)

# Lists both frame slots but rejects the pair (MiniMax H3 behaviour).
_SINGLE_KEY = OpenRouterVideoModel(
    id="test/single-key",
    display_name="Single Key",
    supported_resolutions=("720p",),
    supported_durations=(5,),
    supported_aspect_ratios=("16:9",),
    supported_frame_images=("first_frame", "last_frame"),
    min_price_per_second=0.05,
    supports_audio=False,
    supports_frame_combo=False,
)

_NO_FRAMES = OpenRouterVideoModel(
    id="test/no-frames",
    display_name="No Frames",
    supported_resolutions=("720p",),
    supported_durations=(5,),
    supported_aspect_ratios=("16:9",),
    supported_frame_images=(),
    min_price_per_second=0.02,
    supports_audio=False,
)

# Seedance-2.x behaviour (live probe 2026-10-13): accepts audio_url + image
# reference; audio alone is rejected by the provider.
_AUDIO = OpenRouterVideoModel(
    id="test/audio-capable",
    display_name="Audio Capable",
    supported_resolutions=("480p", "720p"),
    supported_durations=(4, 5),
    supported_aspect_ratios=("16:9",),
    supported_frame_images=("first_frame",),
    min_price_per_second=0.035,
    supports_audio=False,
    accepts_audio_reference=True,
)

def test_min_price_per_second_unit_handling():
    from services.openrouter_client.openrouter_client_impl import _min_price_per_second

    # USD per second via duration SKUs
    assert _min_price_per_second({"duration_seconds": "0.08", "duration_seconds_480p": "0.05"}) == 0.05
    # cents per second must not be read as dollars
    assert _min_price_per_second({"cents_per_second_output": "17", "cents_per_second_output_1080p": "29"}) == 0.17
    # token-only SKUs are not a per-second rate: no price, not a wrong one
    assert _min_price_per_second({"video_tokens": "0.0000035"}) is None
    # mixed: only the per-second SKUs count
    assert _min_price_per_second({"video_tokens": "0.0000035", "duration_seconds": "0.1"}) == 0.1
    assert _min_price_per_second(None) is None
    assert _min_price_per_second({}) is None


_MP4_BYTES = b"\x00\x00\x00\x1cftypisom" + b"\x00" * 500


def _use_openrouter(test_state, *, key: str = "or-key") -> None:
    test_state.state.app_settings.openrouter_api_key = key
    test_state.state.app_settings.user_prefers_openrouter_video_generations = True
    fake = test_state.openrouter_client
    fake.video_models = [_MINI, _NO_FRAMES, _SINGLE_KEY, _AUDIO]
    fake.video_result = _MP4_BYTES


def _write_wav(path) -> None:
    """Minimal valid 8-bit mono WAV so validate_audio_file's sniffing passes."""
    import wave

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)
        w.setframerate(8000)
        w.writeframes(bytes(160))


def _or_request(**overrides) -> dict:
    body = {
        "prompt": "a robot dancing",
        "resolution": "720p",
        "model": "or:test/mini",
        "duration": 5,
        "fps": 24,
        "aspectRatio": "16:9",
    }
    body.update(overrides)
    return body


class TestOpenRouterKeyProbe:
    def test_openrouter_key_is_probed_and_saved(self, client, test_state):
        test_state.http.queue("get", FakeResponse(status_code=200, json_payload={"label": "k"}))
        r = client.post("/api/settings", json={"openrouterApiKey": "  or-key  "})
        assert r.status_code == 200
        assert test_state.state.app_settings.openrouter_api_key == "or-key"
        call = test_state.http.calls[0]
        assert call.url == "https://openrouter.ai/api/v1/key"
        assert call.headers is not None
        assert call.headers["Authorization"] == "Bearer or-key"

    def test_rejected_openrouter_key_is_not_saved(self, client, test_state):
        test_state.state.app_settings.openrouter_api_key = "real-or-key"
        test_state.http.queue("get", FakeResponse(status_code=401, text="Invalid API key"))
        r = client.post("/api/settings", json={"openrouterApiKey": "random-junk"})
        assert_http_error(
            r,
            status_code=400,
            code="OPENROUTER_INVALID_API_KEY",
            message="This key isn’t valid.",
        )
        assert test_state.state.app_settings.openrouter_api_key == "real-or-key"


class TestOpenRouterModelSpecs:
    def test_catalog_merged_when_key_set(self, client, test_state):
        _use_openrouter(test_state)
        r = client.get("/api/generate/models-specs")
        assert r.status_code == 200
        payload = r.json()
        api_pipelines = [item["pipeline"] for item in payload["api_models"]]
        assert "or:test/mini" in api_pipelines
        assert "or:test/no-frames" in api_pipelines
        # last_frame capability flag reaches the frontend spec for frame-capable models
        mini = next(i for i in payload["api_models"] if i["pipeline"] == "or:test/mini")
        assert mini["spec"]["capabilities"]["last_frame"] is True
        no_frames = next(i for i in payload["api_models"] if i["pipeline"] == "or:test/no-frames")
        assert no_frames["spec"]["capabilities"]["last_frame"] is False

    def test_audio_capable_model_gets_synthesized_540p_a2v_cell(self, client, test_state):
        _use_openrouter(test_state)
        r = client.get("/api/generate/models-specs")
        assert r.status_code == 200
        audio = next(
            i for i in r.json()["api_models"] if i["pipeline"] == "or:test/audio-capable"
        )
        assert audio["spec"]["capabilities"]["a2v"] is True
        a2v_map = audio["spec"]["a2v_supported_resolutions_durations"]
        # catalog has 480p/720p only; the A2V screen's fixed 540p/24 cell is synthesized
        assert "540p" in a2v_map
        assert a2v_map["540p"]["fps_to_durations"]["24"] == [4, 5]
        mini = next(i for i in r.json()["api_models"] if i["pipeline"] == "or:test/mini")
        assert mini["spec"]["a2v_supported_resolutions_durations"] == {}

    def test_no_openrouter_rows_without_key(self, client, test_state):
        test_state.state.app_settings.openrouter_api_key = ""
        r = client.get("/api/generate/models-specs")
        assert r.status_code == 200
        api_pipelines = [item["pipeline"] for item in r.json()["api_models"]]
        assert not any(p.startswith("or:") for p in api_pipelines)


class TestOpenRouterGenerate:
    def test_or_pipeline_routes_to_openrouter_client(self, client, test_state):
        _use_openrouter(test_state)
        r = client.post("/api/generate", json=_or_request())
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "complete"
        calls = test_state.openrouter_client.generate_video_calls
        assert len(calls) == 1
        call = calls[0]
        assert call["model"] == "test/mini"
        assert call["api_key"] == "or-key"
        assert call["duration"] == 5
        assert call["resolution"] == "720p"
        assert call["aspect_ratio"] == "16:9"
        assert call["first_frame_data_url"] is None
        assert call["last_frame_data_url"] is None
        # The written video carries the client's bytes.
        from pathlib import Path
        assert Path(body["video_path"]).read_bytes() == _MP4_BYTES

    def test_missing_key_rejected(self, client, test_state):
        _use_openrouter(test_state, key="")
        r = client.post("/api/generate", json=_or_request())
        assert_http_error(
            r,
            status_code=422,
            code="OPENROUTER_API_KEY_MISSING",
            message="OpenRouter API key is not configured. Add it under Settings -> API Keys.",
        )
        assert test_state.openrouter_client.generate_video_calls == []

    def test_unknown_model_rejected(self, client, test_state):
        _use_openrouter(test_state)
        r = client.post("/api/generate", json=_or_request(model="or:test/ghost"))
        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message="Unknown OpenRouter model 'test/ghost'. Refresh the model list and retry.",
        )
        assert test_state.openrouter_client.generate_video_calls == []

    def test_unsupported_duration_rejected(self, client, test_state):
        _use_openrouter(test_state)
        # 10 is valid in the global duration Literal but absent from the fake
        # catalog's supported_durations (4, 5, 8) — so the catalog gate must fire.
        r = client.post("/api/generate", json=_or_request(duration=10))
        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message="Duration 10s not supported by 'Mini Test' (supported: 4, 5, 8)",
        )

    def test_last_frame_without_first_rejected(self, client, test_state):
        _use_openrouter(test_state)
        r = client.post(
            "/api/generate",
            json=_or_request(lastImagePath="/tmp/does-not-matter.png"),
        )
        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message="Last frame requires a first-frame image",
        )

    def test_first_last_frames_passed_as_data_urls(self, client, test_state, tmp_path):
        from PIL import Image

        _use_openrouter(test_state)
        first = tmp_path / "first.png"
        last = tmp_path / "last.png"
        Image.new("RGB", (64, 36), "red").save(first)
        Image.new("RGB", (64, 36), "blue").save(last)
        r = client.post(
            "/api/generate",
            json=_or_request(imagePath=str(first), lastImagePath=str(last)),
        )
        assert r.status_code == 200
        call = test_state.openrouter_client.generate_video_calls[-1]
        assert call["first_frame_data_url"].startswith("data:image/")
        assert call["last_frame_data_url"].startswith("data:image/")

    def test_frame_images_are_normalized_to_jpeg(self, client, test_state, tmp_path):
        from PIL import Image

        _use_openrouter(test_state)
        webp = tmp_path / "frame.webp"
        rgba = tmp_path / "alpha.png"
        Image.new("RGB", (64, 36), "red").save(webp, format="WEBP")
        Image.new("RGBA", (64, 36), (0, 255, 0, 128)).save(rgba)
        r = client.post(
            "/api/generate",
            json=_or_request(imagePath=str(webp), lastImagePath=str(rgba)),
        )
        assert r.status_code == 200
        call = test_state.openrouter_client.generate_video_calls[-1]
        # every provider accepts JPEG; alpha is flattened, not rejected
        assert call["first_frame_data_url"].startswith("data:image/jpeg;base64,")
        assert call["last_frame_data_url"].startswith("data:image/jpeg;base64,")
        import base64
        from io import BytesIO

        img = Image.open(BytesIO(base64.b64decode(call["last_frame_data_url"].split(",", 1)[1])))
        assert img.format == "JPEG" and img.mode == "RGB"

    def test_model_without_frame_support_rejects_image(self, client, test_state, tmp_path):
        from PIL import Image

        _use_openrouter(test_state)
        first = tmp_path / "first.png"
        Image.new("RGB", (64, 36), "red").save(first)
        r = client.post(
            "/api/generate",
            json=_or_request(model="or:test/no-frames", imagePath=str(first)),
        )
        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message="'No Frames' does not support image-to-video.",
        )

    def test_single_keyframe_model_rejects_both_frames(self, client, test_state, tmp_path):
        from PIL import Image

        _use_openrouter(test_state)
        first = tmp_path / "first.png"
        last = tmp_path / "last.png"
        Image.new("RGB", (64, 36), "red").save(first)
        Image.new("RGB", (64, 36), "blue").save(last)
        r = client.post(
            "/api/generate",
            json=_or_request(model="or:test/single-key", imagePath=str(first), lastImagePath=str(last)),
        )
        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message=(
                "'Single Key' accepts only ONE keyframe per job (start OR end frame, "
                "not both). Remove one image, or pick a model that supports start+end: "
                "Mini Test."
            ),
        )

    def test_a2v_rejected_on_audio_incapable_model(self, client, test_state, tmp_path):
        from PIL import Image

        _use_openrouter(test_state)
        first = tmp_path / "first.png"
        audio = tmp_path / "tone.wav"
        Image.new("RGB", (64, 36), "red").save(first)
        _write_wav(audio)
        r = client.post(
            "/api/generate",
            json=_or_request(imagePath=str(first), audioPath=str(audio)),
        )
        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message=(
                "Audio-to-video is not supported by 'Mini Test'. "
                "Audio-capable models: Audio Capable."
            ),
        )

    def test_a2v_requires_reference_image(self, client, test_state, tmp_path):
        _use_openrouter(test_state)
        audio = tmp_path / "tone.wav"
        _write_wav(audio)
        r = client.post(
            "/api/generate",
            json=_or_request(model="or:test/audio-capable", audioPath=str(audio)),
        )
        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message=(
                "OpenRouter audio-to-video needs a reference image alongside the audio "
                "(the provider rejects audio-only requests)."
            ),
        )

    def test_a2v_uploads_audio_and_passes_public_url(self, client, test_state, tmp_path, monkeypatch):
        from PIL import Image

        import handlers.video_generation_handler as vgh

        _use_openrouter(test_state)
        first = tmp_path / "first.png"
        audio = tmp_path / "tone.wav"
        Image.new("RGB", (64, 36), "red").save(first)
        _write_wav(audio)

        uploads: list[str] = []

        def fake_upload(path):
            uploads.append(str(path))
            return "https://files.example.org/tone.wav"

        monkeypatch.setattr(vgh, "upload_public_file", fake_upload)
        r = client.post(
            "/api/generate",
            json=_or_request(model="or:test/audio-capable", imagePath=str(first), audioPath=str(audio)),
        )
        assert r.status_code == 200
        assert uploads == [str(audio)]
        call = test_state.openrouter_client.generate_video_calls[-1]
        assert call["audio_public_url"] == "https://files.example.org/tone.wav"
        # the image still rides as a JPEG-normalized data URL
        assert call["first_frame_data_url"].startswith("data:image/jpeg;base64,")

    def test_a2v_540p_cell_maps_and_duration_snaps_to_audio(self, client, test_state, tmp_path, monkeypatch):
        from PIL import Image

        import handlers.video_generation_handler as vgh

        _use_openrouter(test_state)
        first = tmp_path / "first.png"
        audio = tmp_path / "tone.wav"
        Image.new("RGB", (64, 36), "red").save(first)
        _write_wav(audio)
        monkeypatch.setattr(
            vgh, "upload_public_file", lambda path: "https://files.example.org/tone.wav"
        )
        r = client.post(
            "/api/generate",
            json=_or_request(
                model="or:test/audio-capable",
                resolution="540p",
                duration=None,
                imagePath=str(first),
                audioPath=str(audio),
            ),
        )
        assert r.status_code == 200
        call = test_state.openrouter_client.generate_video_calls[-1]
        # 540p cell -> nearest real catalog resolution; duration follows the audio
        assert call["resolution"] == "720p"
        assert call["duration"] == 4

    def test_client_error_maps_to_http_error(self, client, test_state):
        _use_openrouter(test_state)
        test_state.openrouter_client.raise_on_generate_video = OpenRouterClientError(
            429, "Rate limited", provider_message="slow down"
        )
        r = client.post("/api/generate", json=_or_request())
        assert r.status_code == 429
        assert r.json()["code"] == "OPENROUTER_GENERATION_FAILED"

    def test_catalog_fetch_failure_serves_empty_not_crash(self, client, test_state):
        _use_openrouter(test_state)
        test_state.openrouter_client.raise_on_list_video_models = OpenRouterClientError(
            500, "catalog down"
        )
        r = client.post("/api/generate", json=_or_request())
        # Empty catalog -> model unknown -> 422, not a 500.
        assert_http_error(
            r,
            status_code=422,
            code="INVALID_VIDEO_GENERATION_SPEC",
            message="Unknown OpenRouter model 'test/mini'. Refresh the model list and retry.",
        )


class TestOpenRouterPipelineType:
    def test_or_pipeline_id_validates_in_request_model(self):
        # Widened LTXVideoGenPipeline accepts the "or:" format.
        from api_types import GenerateVideoRequest

        req = GenerateVideoRequest.model_validate(_or_request())
        assert isinstance(req.model, str)
        assert req.model.startswith("or:")
