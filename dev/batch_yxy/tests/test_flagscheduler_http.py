import asyncio
import json
from pathlib import Path

from flagscheduler.config import load_config
from flagscheduler.runtime import NodeRuntime
from flagscheduler.server import start_server


def test_loopback_status_inference_lookup_and_rejection():
    async def scenario():
        node = NodeRuntime(load_config(str(Path(__file__).resolve().parents[1] / "examples/flagscheduler.mock.json")))
        await node.start()
        server = await start_server(node, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        async def request(method, path, body=b""):
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(f"{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body)
            await writer.drain()
            raw = await reader.read()
            writer.close()
            await writer.wait_closed()
            head, value = raw.split(b"\r\n\r\n", 1)
            return int(head.split()[1]), json.loads(value)
        try:
            status, state = await request("GET", "/v1/status")
            assert status == 200 and state["state"] == "accepting"
            body = json.dumps({"request_id": "http-1", "binding": "qwen-embed", "payload": "hello"}).encode()
            status, result = await request("POST", "/v1/infer", body)
            assert status == 200 and result["output"]["mock"]
            status, found = await request("GET", "/v1/results/http-1")
            assert found == result
            assert (await request("POST", "/v1/infer", body))[0] == 400
            assert (await request("GET", "/v1/results/unknown"))[0] == 404
            assert (await request("POST", "/v1/infer", b"{bad json"))[0] == 400
        finally:
            server.close()
            await server.wait_closed()
            await node.stop()
    asyncio.run(asyncio.wait_for(scenario(), 5))
