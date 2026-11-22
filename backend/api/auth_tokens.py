"""Minimal HS256 JWT + scrypt password hashing (stdlib-friendly)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any, Dict, Optional


def jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET", "").strip()
    if secret:
        return secret
    # Dev fallback — production multi mode should set JWT_SECRET.
    return "epochflow-dev-jwt-secret-change-me"


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def create_access_token(*, user_id: str, email: str, ttl_seconds: int = 60 * 60 * 24 * 7) -> str:
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": user_id,
        "email": email,
        "iat": now,
        "exp": now + int(ttl_seconds),
    }
    head = _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
    body = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{head}.{body}".encode()
    sig = hmac.new(jwt_secret().encode(), signing_input, hashlib.sha256).digest()
    return f"{head}.{body}.{_b64url_encode(sig)}"


def decode_access_token(token: str) -> Dict[str, Any]:
    try:
        head, body, sig = token.split(".")
    except ValueError as ex:
        raise ValueError("invalid token format") from ex
    signing_input = f"{head}.{body}".encode()
    expected = hmac.new(jwt_secret().encode(), signing_input, hashlib.sha256).digest()
    if not hmac.compare_digest(_b64url_encode(expected), sig):
        raise ValueError("invalid token signature")
    payload = json.loads(_b64url_decode(body))
    if int(payload.get("exp") or 0) < int(time.time()):
        raise ValueError("token expired")
    if not payload.get("sub"):
        raise ValueError("token missing subject")
    return payload


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=2**14,
        r=8,
        p=1,
        dklen=32,
    )
    return f"scrypt${_b64url_encode(salt)}${_b64url_encode(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algo, salt_b64, digest_b64 = encoded.split("$", 2)
    except ValueError:
        return False
    if algo != "scrypt":
        return False
    salt = _b64url_decode(salt_b64)
    expected = _b64url_decode(digest_b64)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=2**14,
        r=8,
        p=1,
        dklen=len(expected) or 32,
    )
    return hmac.compare_digest(digest, expected)


def bearer_token_from_header(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None
