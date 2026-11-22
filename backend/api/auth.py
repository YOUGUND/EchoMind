"""JWT register / login and FastAPI dependency."""

from __future__ import annotations

import os
import re
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from api.auth_tokens import (
    bearer_token_from_header,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from api.user_context import AuthUser
from storage.db import get_session_factory
from storage.models import User

router = APIRouter(prefix="/auth", tags=["Auth"])

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class RegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=8, max_length=128)


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str


def multi_user_enabled() -> bool:
    # Product path is multi-only (JWT + PostgreSQL + Milvus).
    return True


def _normalize_email(email: str) -> str:
    value = (email or "").strip().lower()
    if not _EMAIL_RE.match(value):
        raise HTTPException(400, "邮箱格式不正确")
    return value


async def get_current_user(
    authorization: Optional[str] = Header(default=None),
) -> AuthUser:
    token = bearer_token_from_header(authorization)
    if not token:
        raise HTTPException(401, "请先登录（缺少 Bearer Token）")
    try:
        payload = decode_access_token(token)
    except ValueError as ex:
        raise HTTPException(401, f"登录已失效：{ex}") from ex
    user_id = str(payload["sub"])
    email = str(payload.get("email") or "")
    factory = get_session_factory()
    async with factory() as session:
        user = await session.get(User, user_id)
        if user is None:
            raise HTTPException(401, "用户不存在或已被删除")
        return AuthUser(id=user.id, email=user.email or email)


async def get_optional_user(
    authorization: Optional[str] = Header(default=None),
) -> Optional[AuthUser]:
    if not multi_user_enabled():
        return None
    token = bearer_token_from_header(authorization)
    if not token:
        return None
    try:
        return await get_current_user(authorization=authorization)
    except HTTPException:
        return None


async def require_user_if_multi(
    authorization: Optional[str] = Header(default=None),
) -> Optional[AuthUser]:
    """Chat/KB dependency: always require JWT user."""
    return await get_current_user(authorization=authorization)


@router.post("/register", response_model=AuthResponse)
async def register(body: RegisterRequest):
    email = _normalize_email(body.email)
    factory = get_session_factory()
    async with factory() as session:
        existing = await session.scalar(select(User).where(User.email == email))
        if existing:
            raise HTTPException(409, "该邮箱已注册")
        user = User(email=email, password_hash=hash_password(body.password))
        session.add(user)
        await session.commit()
        await session.refresh(user)
        token = create_access_token(user_id=user.id, email=user.email)
        return AuthResponse(access_token=token, user_id=user.id, email=user.email)


@router.post("/login", response_model=AuthResponse)
async def login(body: LoginRequest):
    email = _normalize_email(body.email)
    factory = get_session_factory()
    async with factory() as session:
        user = await session.scalar(select(User).where(User.email == email))
        if user is None or not verify_password(body.password, user.password_hash):
            raise HTTPException(401, "邮箱或密码错误")
        token = create_access_token(user_id=user.id, email=user.email)
        return AuthResponse(access_token=token, user_id=user.id, email=user.email)


@router.get("/me")
async def me(user: AuthUser = Depends(get_current_user)):
    return {"user_id": user.id, "email": user.email}
