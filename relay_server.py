#!/usr/bin/env python3
"""
Production relay server for FaceSwap Pro.
Runs on a cheap VPS with a fixed domain.
Auto-spins RunPod GPU pods on demand, proxies WebSocket traffic.
"""
import os, json, time, asyncio, subprocess, logging, base64
from pathlib import Path

from aiohttp import web, ClientSession, WSMsgType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("relay")

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
PORT = int(os.environ.get("PORT", 80))

# Template ID for our facefusion setup
TEMPLATE_ID = os.environ.get("TEMPLATE_ID", "")

class PodManager:
    """Manages RunPod GPU pod lifecycle."""

    def __init__(self, api_key):
        self.api_key = api_key
        self.pod_id = None
        self.pod_url = None
        self.last_spawn = 0
        self.idle_start = None
        self.IDLE_TIMEOUT = 600  # 10 min idle = shutdown

    async def ensure_pod(self):
        """Get or create a running pod. Returns the proxy URL."""
        # Check existing pods first
        pods = await self._list_pods()
        for p in pods:
            if p.get("name", "").startswith("faceswap"):
                pid = p["id"]
                status = p.get("desiredStatus", "")
                if status == "RUNNING":
                    self.pod_id = pid
                    self.pod_url = f"wss://{pid}-8888.proxy.runpod.net/ws"
                    self.idle_start = None
                    logger.info(f"Found running pod: {pid}")
                    return self.pod_url
                elif status == "STOPPED":
                    logger.info(f"Starting stopped pod: {pid}")
                    await self._start_pod(pid)
                    return await self._wait_for_pod(pid)

        # No existing pod — create new one
        logger.info("Creating new GPU pod...")
        pod = await self._create_pod()
        self.pod_id = pod["id"]
        return await self._wait_for_pod(self.pod_id)

    async def _list_pods(self):
        async with ClientSession() as sess:
            async with sess.get(
                "https://api.runpod.io/v2/pods",
                headers={"Authorization": f"Bearer {self.api_key}"}
            ) as resp:
                data = await resp.json()
                return data.get("pods", [])

    async def _create_pod(self):
        async with ClientSession() as sess:
            payload = {
                "name": "faceswap-pro",
                "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
                "containerDiskSizeGb": 30,
                "ports": "8888/http",
                "gpuTypeIds": ["NVIDIA L4"],
                "gpuCount": 1,
                "startupScript": Path("/workspace/startup.sh").read_text() if Path("/workspace/startup.sh").exists() else "echo starting",
            }
            async with sess.post(
                "https://api.runpod.io/v2/pods",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload
            ) as resp:
                return await resp.json()

    async def _start_pod(self, pod_id):
        async with ClientSession() as sess:
            await sess.post(
                f"https://api.runpod.io/v2/pods/{pod_id}/start",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )

    async def _wait_for_pod(self, pod_id, timeout=240):
        """Poll until pod is RUNNING and has a proxy URL."""
        start = time.time()
        while time.time() - start < timeout:
            await asyncio.sleep(5)
            async with ClientSession() as sess:
                async with sess.get(
                    f"https://api.runpod.io/v2/pods/{pod_id}",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                ) as resp:
                    data = await resp.json()
                    pod = data if "id" in data else data.get("pod", {})
                    if pod.get("desiredStatus") == "RUNNING":
                        runtime = pod.get("runtime", {})
                        if runtime and runtime.get("ports"):
                            self.pod_url = f"wss://{pod_id}-8888.proxy.runpod.net/ws"
                            self.pod_id = pod_id
                            logger.info(f"Pod ready: {self.pod_url}")
                            return self.pod_url
        raise TimeoutError("Pod failed to start")

    async def shutdown_idle(self):
        """Stop the pod if idle for too long."""
        if self.pod_id and self.idle_start:
            if time.time() - self.idle_start > self.IDLE_TIMEOUT:
                logger.info(f"Pod {self.pod_id} idle too long, shutting down...")
                async with ClientSession() as sess:
                    await sess.post(
                        f"https://api.runpod.io/v2/pods/{self.pod_id}/stop",
                        headers={"Authorization": f"Bearer {self.api_key}"}
                    )
                self.pod_id = None
                self.pod_url = None
                self.idle_start = None

    def mark_active(self):
        self.idle_start = None

    def mark_idle(self):
        if self.idle_start is None:
            self.idle_start = time.time()


# --- Web Server ---

pod_manager = PodManager(RUNPOD_API_KEY)
active_connections = 0

async def index_handler(request):
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        html = html_path.read_text()
        # Replace WS URL with relay endpoint
        host = request.headers.get("Host", "localhost")
        scheme = "wss" if request.headers.get("X-Forwarded-Proto", "http") == "https" else "ws"
        ws_url = f"{scheme}://{host}/ws"
        html = html.replace('value="wss://', f'value="{ws_url}')
        return web.Response(text=html, content_type="text/html")
    return web.Response(text="<h1>FaceSwap Pro</h1><p>Connect via WebSocket at /ws</p>", content_type="text/html")

async def ws_handler(request):
    global active_connections
    active_connections += 1
    pod_manager.mark_active()

    try:
        # Get or create GPU pod
        await ws.send_json({"status": "provisioning", "message": "Starting GPU pod..."})
        target_url = await pod_manager.ensure_pod()

        # Proxy the connection
        async with ClientSession() as sess:
            async with sess.ws_connect(target_url) as remote_ws:
                local_ws = web.WebSocketResponse()
                await local_ws.prepare(request)

                async def forward_local():
                    async for msg in local_ws:
                        if msg.type == WSMsgType.TEXT:
                            await remote_ws.send_str(msg.data)
                        elif msg.type == WSMsgType.BINARY:
                            await remote_ws.send_bytes(msg.data)
                        elif msg.type == WSMsgType.CLOSED:
                            break

                async def forward_remote():
                    async for msg in remote_ws:
                        if msg.type == WSMsgType.TEXT:
                            await local_ws.send_str(msg.data)
                        elif msg.type == WSMsgType.BINARY:
                            await local_ws.send_bytes(msg.data)
                        elif msg.type == WSMsgType.CLOSED:
                            break

                await asyncio.gather(forward_local(), forward_remote())
    except Exception as e:
        logger.error(f"Relay error: {e}")
        raise
    finally:
        active_connections -= 1
        if active_connections == 0:
            pod_manager.mark_idle()

    return web.Response()

async def health_handler(request):
    return web.json_response({
        "status": "ok",
        "pod_id": pod_manager.pod_id,
        "pod_url": pod_manager.pod_url,
        "active_connections": active_connections,
        "idle_seconds": int(time.time() - pod_manager.idle_start) if pod_manager.idle_start else 0,
    })

async def idle_checker():
    """Background task to shut down idle pods."""
    while True:
        await asyncio.sleep(30)
        await pod_manager.shutdown_idle()

async def main():
    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_get("/health", health_handler)
    app.router.add_get("/ws", ws_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()

    asyncio.create_task(idle_checker())

    logger.info(f"Relay server on http://0.0.0.0:{PORT}")
    await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
