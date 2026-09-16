from __future__ import annotations

import ipaddress
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from app.config import Settings
from app.errors import ValidationError


def _host_allowed(host: str, allowed: list[str]) -> bool:
    host = host.lower().strip(".")
    for allowed_host in allowed:
        allowed_host = allowed_host.lower().strip(".")
        if host == allowed_host or host.endswith(f".{allowed_host}"):
            return True
    return False


def _reject_ip_literals(host: str) -> None:
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
        raise ValidationError("Private and local network URLs are not allowed.")


def validate_public_video_url(raw_url: str, settings: Settings) -> tuple[str, str]:
    parsed = urlparse(raw_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValidationError("Only http and https video URLs are supported.")
    if not parsed.hostname:
        raise ValidationError("The URL is missing a host.")
    _reject_ip_literals(parsed.hostname)
    if not _host_allowed(parsed.hostname, settings.allowed_url_hosts):
        allowed = ", ".join(settings.allowed_url_hosts)
        raise ValidationError(f"Unsupported video host. Allowed hosts: {allowed}.")

    host = parsed.hostname.lower()
    if "youtube.com" in host or host == "youtu.be":
        return raw_url, "youtube"
    if host == "drive.google.com" or host.endswith(".drive.google.com"):
        return normalize_google_drive_url(raw_url), "google_drive"
    raise ValidationError("Unsupported video URL.")


def normalize_google_drive_url(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    parts = [part for part in parsed.path.split("/") if part]
    file_id = None
    if "file" in parts and "d" in parts:
        d_index = parts.index("d")
        if d_index + 1 < len(parts):
            file_id = parts[d_index + 1]
    if not file_id:
        query = parse_qs(parsed.query)
        file_id = (query.get("id") or [None])[0]
    if not file_id:
        raise ValidationError("Google Drive links must contain a public file id.")
    query = urlencode({"id": file_id, "export": "download"})
    return urlunparse(("https", "drive.google.com", "/uc", "", query, ""))
