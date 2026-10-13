"""Upload a local file to a public host so video providers can fetch it.

OpenRouter's video providers reject data: URLs for audio references
("Only HTTPS URLs are allowed") and cannot fetch OpenRouter's own Files API
URLs ("resource download failed" — live probe 2026-10-13). A public HTTPS
URL is the only accepted form. catbox.moe is used because it needs no
account and no API key.

Privacy note: uploaded files become public and stay up. Only non-sensitive
media (generated audio/video) should ever go through this path.
"""
from __future__ import annotations

import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_CATBOX_ENDPOINT = "https://catbox.moe/user/api.php"
_MAX_UPLOAD_BYTES = 64 * 1024 * 1024


class PublicUploadError(RuntimeError):
    """Raised when the public host refuses or the response is not a URL."""


def upload_public_file(path: Path) -> str:
    """Upload `path` and return its public HTTPS URL."""
    data = path.read_bytes()
    if len(data) > _MAX_UPLOAD_BYTES:
        raise PublicUploadError(
            f"File too large for public upload: {len(data)} bytes (max {_MAX_UPLOAD_BYTES})"
        )
    resp = requests.post(
        _CATBOX_ENDPOINT,
        data={"reqtype": "fileupload"},
        files={"fileToUpload": (path.name, data)},
        timeout=120,
    )
    url = resp.text.strip()
    if resp.status_code != 200 or not url.startswith("https://"):
        raise PublicUploadError(
            f"Public upload failed (HTTP {resp.status_code}): {url[:200]}"
        )
    logger.info("public upload: %s -> %s", path.name, url)
    return url
