"""Reject foreign browser origins and DNS-rebinding hosts before parsing an upload.

CORS controls who can read a response; it does not prevent a simple cross-origin
POST from starting work. This service is for a single local machine only.
"""
from urllib.parse import urlsplit

from starlette.responses import JSONResponse

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class LocalBrowserGuard:
    def __init__(self, app, origins):
        self.app = app
        self.origins = frozenset(origins)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        host = headers.get(b"host", b"").decode("latin-1")
        try:
            parsed = urlsplit("http://" + host)
            safe_host = (
                parsed.hostname in LOOPBACK_HOSTS
                and parsed.username is None
                and parsed.password is None
                and parsed.path == ""
                and not parsed.query
                and not parsed.fragment
            )
            _ = parsed.port  # malformed port numbers are invalid hosts too
        except ValueError:
            safe_host = False
        if not safe_host:
            await JSONResponse({"detail": "This service accepts loopback hosts only."}, 400)(
                scope, receive, send
            )
            return
        origin = headers.get(b"origin")
        if origin is not None and origin.decode("latin-1") not in self.origins:
            await JSONResponse({"detail": "This browser origin is not allowed."}, 403)(
                scope, receive, send
            )
            return
        await self.app(scope, receive, send)
