"""Request-scoped user identity for tools (KB) that cannot read FastAPI Depends."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass(frozen=True)
class AuthUser:
    id: str
    email: str


_current_user: ContextVar[Optional[AuthUser]] = ContextVar("epochflow_current_user", default=None)


def get_request_user() -> Optional[AuthUser]:
    return _current_user.get()


def require_request_user() -> AuthUser:
    user = _current_user.get()
    if user is None:
        raise RuntimeError("当前请求未绑定登录用户")
    return user


@contextmanager
def bind_request_user(user: Optional[AuthUser]) -> Iterator[None]:
    token = _current_user.set(user)
    try:
        yield
    finally:
        _current_user.reset(token)
