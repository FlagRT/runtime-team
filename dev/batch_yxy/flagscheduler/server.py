"""Small loopback-only HTTP transport for a node runtime.

This is a local control/inference API, not cross-node routing or an Internet
gateway. Payload sizes, headers, connection lifetime and admission are bounded.
"""
from __future__ import annotations

import asyncio
import json
from urllib.parse import unquote, urlsplit

from .contracts import Request

MAX_BODY = 1024 * 1024


async def start_server(runtime, host="127.0.0.1", port=8765):
    if host not in ("127.0.0.1", "::1"):
        raise ValueError("node API binds only to loopback")

    async def handle(reader, writer):
        status = 200
        try:
            header = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 10)
            lines = header.decode("ascii").split("\r\n")
            method, path, protocol = lines[0].split(" ")
            if protocol != "HTTP/1.1":
                raise ValueError("HTTP/1.1 required")
            headers = {}
            for line in lines[1:]:
                if not line:
                    continue
                key, value = line.split(":", 1)
                key = key.lower().strip()
                if key in headers:
                    raise ValueError("duplicate headers are unsupported")
                headers[key] = value.strip()
            if "transfer-encoding" in headers:
                raise ValueError("chunked transfer encoding is unsupported")
            length = int(headers.get("content-length", "0"))
            if not 0 <= length <= MAX_BODY:
                raise ValueError("request body exceeds 1 MiB limit")
            body = await asyncio.wait_for(reader.readexactly(length), 10)
            path = urlsplit(path).path
            if method == "GET" and path == "/healthz":
                value = {"schema_version": 1, "state": runtime.snapshot()["state"]}
            elif method == "GET" and path == "/v1/status":
                value = runtime.snapshot()
            elif method == "GET" and path.startswith("/v1/results/"):
                value = runtime.result(unquote(path[len("/v1/results/"):]))
                if value is None:
                    status, value = 404, {"error": "unknown or evicted request ID"}
            elif method == "POST" and path == "/v1/infer":
                request = Request.from_dict(json.loads(body))
                value = await asyncio.shield(runtime.submit(request))
                if value["status"] == "failed":
                    status = 502
            else:
                status, value = 404, {"error": "unknown route"}
        except OverflowError as exc:
            status, value = 429, {"error": str(exc)}
        except (ValueError, KeyError, TypeError, asyncio.IncompleteReadError, asyncio.LimitOverrunError, asyncio.TimeoutError) as exc:
            status, value = 400, {"error": str(exc)}
        except Exception as exc:
            status, value = 500, {"error": type(exc).__name__, "message": str(exc)}
        try:
            raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            writer.write(f"HTTP/1.1 {status} Response\r\nContent-Type: application/json\r\nContent-Length: {len(raw)}\r\nConnection: close\r\n\r\n".encode() + raw)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    return await asyncio.start_server(handle, host, port, limit=16384)
