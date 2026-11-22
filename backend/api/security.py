"""Access, body-size, concurrency and rejection-audit boundaries for a small free service."""
import logging
import os
import secrets
import time
import uuid
from collections import Counter, deque
from typing import Any, Deque, Dict

from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)


def model_configured():
    return os.getenv('ANTHROPIC_API_KEY', '').strip() not in {'', 'your_api_key', 'changeme'}


def public_demo_enabled():
    return os.getenv('PUBLIC_DEMO', '').strip().lower() in {'1', 'true', 'yes', 'on'}


def multi_user_enabled():
    # Product path is multi-only (JWT required for chat/KB).
    return True


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, '').strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


class SecurityAudit:
    """In-memory rejection / admission audit for /monitor and interviews."""

    def __init__(self, maxlen: int = 200):
        self.events: Deque[Dict[str, Any]] = deque(maxlen=maxlen)
        self.counts: Counter = Counter()

    def record(self, *, code: str, status: int, path: str, request_id: str, detail: str) -> None:
        self.counts[code] += 1
        self.events.appendleft(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "code": code,
                "status": status,
                "path": path,
                "request_id": request_id,
                "detail": detail[:200],
            }
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "rejection_counts": dict(self.counts),
            "recent_rejections": list(self.events)[:20],
            "model_rpm_limit": _env_int("MODEL_RPM_LIMIT", 12),
            "model_concurrency_limit": _env_int("MODEL_CONCURRENCY_LIMIT", 2),
            "public_demo": public_demo_enabled(),
            "multi_user": multi_user_enabled(),
            "access_token_required": (
                bool(os.getenv("APP_ACCESS_TOKEN", "").strip())
                and not public_demo_enabled()
                and not multi_user_enabled()
            ),
        }


security_audit = SecurityAudit()


class RequestBoundary:
    """Bound request bodies before multipart parsing, including chunked uploads."""
    MAX_BODY = 1200 * 1024

    def __init__(self, app: ASGIApp):
        self.app = app
        self.model_calls = deque()
        self.active_models = 0
        self.rpm_limit = _env_int("MODEL_RPM_LIMIT", 12)
        self.concurrency_limit = _env_int("MODEL_CONCURRENCY_LIMIT", 2)

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        request_id = uuid.uuid4().hex
        path, method = scope['path'], scope['method']
        headers = dict(scope.get('headers', []))

        async def reject(status: int, message: str, code: str):
            security_audit.record(
                code=code, status=status, path=path, request_id=request_id, detail=message
            )
            await JSONResponse(
                {'detail': message, 'request_id': request_id, 'code': code},
                status_code=status,
                headers={'X-Request-ID': request_id},
            )(scope, receive, send)

        if method != 'OPTIONS' and path != '/health' and not public_demo_enabled() and not multi_user_enabled():
            token = os.getenv('APP_ACCESS_TOKEN', '').strip()
            if os.getenv('APP_ENV') == 'production' and not token:
                return await reject(503, '请在 Render 设置 APP_ACCESS_TOKEN 后访问工作台', 'missing_access_token')
            supplied = headers.get(b'x-access-token', b'')
            if token and not secrets.compare_digest(supplied, token.encode()):
                return await reject(401, '访问码无效或未填写，请前往工作台设置填写访问码', 'unauthorized')

        # Multi-user mode: protected routes require Bearer JWT (validated later by Depends).
        if multi_user_enabled() and method != 'OPTIONS':
            public_prefixes = (
                '/health', '/auth/register', '/auth/login', '/docs', '/openapi.json', '/redoc', '/metrics',
            )
            needs_jwt = not any(path == p or path.startswith(p + '/') for p in public_prefixes)
            # Keep model catalog/browse readable; lock chat/knowledge/user data.
            jwt_paths = (
                '/chat', '/knowledge', '/search', '/trace', '/auth/me', '/monitor',
            )
            needs_jwt = needs_jwt and any(path == p or path.startswith(p + '/') for p in jwt_paths)
            if needs_jwt:
                auth = headers.get(b'authorization', b'').decode('latin-1', errors='ignore')
                if not auth.lower().startswith('bearer '):
                    return await reject(401, '请先登录（需要 Authorization: Bearer <token>）', 'jwt_required')

        is_model = method == 'POST' and path in {
            '/chat', '/chat/resume', '/chat/fork', '/eval/run', '/models/run', '/models/video/status',
        }
        if is_model:
            if not model_configured():
                return await reject(503, '模型未配置，请在后端设置 ANTHROPIC_API_KEY；未调用模型', 'model_not_configured')
            now = time.monotonic()
            while self.model_calls and now - self.model_calls[0] > 60:
                self.model_calls.popleft()
            if len(self.model_calls) >= self.rpm_limit or self.active_models >= self.concurrency_limit:
                return await reject(
                    429,
                    f'当前实验较多，请稍后再试（最多同时运行 {self.concurrency_limit} 项、每分钟 {self.rpm_limit} 项）',
                    'rate_limited',
                )

        chunks, size = [], 0
        if method in {'POST', 'PUT', 'PATCH'}:
            while True:
                event = await receive()
                if event['type'] == 'http.disconnect':
                    return
                size += len(event.get('body', b''))
                if size > self.MAX_BODY:
                    return await reject(413, '请求过大，请上传不超过 1MB 的文件或减少文本长度', 'payload_too_large')
                chunks.append(event.get('body', b''))
                if not event.get('more_body'):
                    break

        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if not delivered and method in {'POST', 'PUT', 'PATCH'}:
                delivered = True
                return {'type': 'http.request', 'body': b''.join(chunks), 'more_body': False}
            return await receive()

        async def response_send(event):
            if event['type'] == 'http.response.start':
                event.setdefault('headers', []).extend([
                    (b'x-request-id', request_id.encode()),
                    (b'x-content-type-options', b'nosniff'),
                    (b'cache-control', b'no-store'),
                ])
            await send(event)

        if is_model:
            # Body collection awaits network input; recheck admission atomically afterwards.
            if len(self.model_calls) >= self.rpm_limit or self.active_models >= self.concurrency_limit:
                return await reject(
                    429,
                    f'当前实验较多，请稍后再试（最多同时运行 {self.concurrency_limit} 项、每分钟 {self.rpm_limit} 项）',
                    'rate_limited',
                )
            self.active_models += 1
            self.model_calls.append(time.monotonic())
        try:
            await self.app(scope, bounded_receive, response_send)
        finally:
            if is_model:
                self.active_models -= 1
