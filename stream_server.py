"""
Real-time face swap using insightface + ONNX on CUDA.
No FaceFusion overhead. Single port (HTTP + WS at /ws).
"""
import os, sys, json, asyncio, time, struct, logging, base64, re
from pathlib import Path

import numpy as np
import cv2
import onnxruntime
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

from aiohttp import web

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stream_server")

PORT = int(os.environ.get("STREAM_PORT", 8888))
MAX_USERS = int(os.environ.get("MAX_USERS", 2))
CALL_DURATION = int(os.environ.get("CALL_DURATION", 65))

face_app = None
swapper = None

def init_models():
    global face_app, swapper
    logger.info("Initializing insightface with CUDA...")

    available = onnxruntime.get_available_providers()
    logger.info(f"Available providers: {available}")

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    if 'CUDAExecutionProvider' not in available:
        providers = ['CPUExecutionProvider']
        logger.warning("CUDA not available, using CPU")

    # Face analysis uses onnx sessions internally
    face_app = FaceAnalysis(name="buffalo_l", providers=providers)
    face_app.prepare(ctx_id=0, det_size=(320, 320))

    # Inswapper model
    swapper = get_model("inswapper_128.onnx", providers=providers)

    logger.info(f"Models ready (providers: {providers})")


def extract_source_face(image_bytes: bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode source image")
    faces = face_app.get(img)
    if not faces:
        raise ValueError("No face detected in source image")
    return faces[0]


class SessionWS:
    def __init__(self, ws):
        self.ws = ws

    async def recv(self):
        msg = await self.ws.receive()
        if msg.type == web.WSMsgType.TEXT:
            return msg.data
        elif msg.type == web.WSMsgType.BINARY:
            return msg.data
        raise ConnectionResetError("WebSocket closed")

    async def send(self, data):
        if isinstance(data, str):
            await self.ws.send_str(data)
        elif isinstance(data, bytes):
            await self.ws.send_bytes(data)
        else:
            await self.ws.send_json(data)

    async def close(self):
        await self.ws.close()


class UserSession:
    def __init__(self, ws):
        self.ws = ws
        self.source_face = None
        self.start_time = None
        self.call_active = False
        self.frame_count = 0

    async def handle(self):
        try:
            msg = await asyncio.wait_for(self.ws.recv(), timeout=30)
            setup = json.loads(msg)
            face_b64 = setup.get("source_face")

            if not face_b64:
                await self.ws.send(json.dumps({"error": "source_face required"}))
                return

            self.source_face = extract_source_face(base64.b64decode(face_b64))
            await self.ws.send(json.dumps({"status": "face_loaded"}))
            await self.ws.send(json.dumps({"status": "voice_skipped"}))

            self.start_time = time.time()
            self.call_active = True
            await self.ws.send(json.dumps({"status": "ready", "duration": CALL_DURATION}))
            await self._stream_loop()

        except (asyncio.TimeoutError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as e:
            logger.error(f"Session error: {e}", exc_info=True)

    async def _stream_loop(self):
        try:
            while self.call_active:
                if time.time() - self.start_time > CALL_DURATION:
                    break
                try:
                    msg = await asyncio.wait_for(self.ws.recv(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                except (ConnectionResetError, ConnectionAbortedError):
                    break

                if isinstance(msg, str):
                    try:
                        cmd = json.loads(msg)
                        if cmd.get("action") == "end_call":
                            break
                    except: pass
                    continue

                if msg[0] == 0:
                    await self._proc_frame(msg)
                elif msg[0] == 1:
                    pass
        finally:
            self.call_active = False
            elapsed = time.time() - self.start_time if self.start_time else 0
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            logger.info(f"Session ended. Frames: {self.frame_count}, FPS: {fps:.1f}")

    async def _proc_frame(self, msg):
        self.frame_count += 1
        ts = struct.unpack("!I", msg[1:5])[0]
        arr = np.frombuffer(msg[5:], np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        t0 = time.time()
        faces = face_app.get(frame)
        if faces:
            frame = swapper.get(frame, faces[0], self.source_face, paste_back=True)
        elapsed = time.time() - t0

        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        try:
            await self.ws.send(struct.pack("!B I", 0, ts) + jpg.tobytes())
        except:
            self.call_active = False

        if self.frame_count % 30 == 0:
            logger.info(f"Frame {self.frame_count}: {elapsed*1000:.0f}ms (fps={1/elapsed:.1f})")


sessions = set()

async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    if len(sessions) >= MAX_USERS:
        await ws.send_json({"error": "server_full"})
        return ws

    wrapped = SessionWS(ws)
    s = UserSession(wrapped)
    sessions.add(s)
    try:
        await s.handle()
    finally:
        sessions.discard(s)
    return ws


async def index_handler(request):
    html_path = Path(__file__).parent / "index.html"
    if not html_path.exists():
        return web.Response(text="index.html not found", status=404)

    html = html_path.read_text()
    match = re.search(r'https://(.+?)-\d+\.proxy\.runpod\.net', str(request.url))
    if match:
        pod_id = match.group(1)
        ws_url = f"wss://{pod_id}-{PORT}.proxy.runpod.net/ws"
        html = re.sub(r'value="wss://[^"]*"', f'value="{ws_url}"', html)
    return web.Response(text=html, content_type="text/html")


async def main():
    init_models()

    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_get("/index.html", index_handler)
    app.router.add_get("/ws", ws_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()

    logger.info(f"Server on http://0.0.0.0:{PORT}")
    await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
