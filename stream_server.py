"""
Real-time face swap + voice changer streaming server.
Uses FaceFusion 3.1.0 internal API, RVC for voice conversion.
Single port: HTTP (index.html) + WebSocket (/ws)
"""
import os, sys, json, asyncio, time, struct, logging, base64, re
from pathlib import Path

import numpy as np
import cv2

from aiohttp import web

# FaceFusion 3.1.0 imports
from facefusion import state_manager
from facefusion.face_analyser import get_many_faces, get_one_face
from facefusion.processors.modules.face_swapper import swap_face
from facefusion.face_store import get_static_faces, set_static_faces
from facefusion.typing import VisionFrame, Face

# Initialize FaceFusion state with required defaults
import facefusion.choices as ff_choices
def init_facefusion_state():
    """Initialize FaceFusion state manager with default values."""
    defaults = {
        'face_detector_model': 'many',
        'face_detector_size': '640x640',
        'face_detector_angles': [0, 90, 180, 270],
        'face_detector_score': 0.5,
        'face_landmarker_model': 'many',
        'face_landmarker_score': 0.5,
        'face_selector_mode': 'many',
        'face_selector_order': 'left-right',
        'face_selector_gender': None,
        'face_selector_race': None,
        'face_selector_age': None,
        'reference_face_distance': 0.6,
        'reference_face_position': 0,
        'reference_frame_number': 0,
        'face_mask_types': ['box'],
        'face_mask_blur': 0.3,
        'face_mask_padding': (0, 0, 0, 0),
        'face_mask_regions': None,
        'face_occluder_model': 'xseg_1',
        'face_parser_model': 'bisenet_resnet_18',
        'face_swapper_model': 'inswapper_128',
        'face_swapper_pixel_boost': '0x0',
        'face_swapper_pixel_boost_type': 'cpu',
        'face_swapper_pixel_boost_scale': 1,
        'execution_providers': ['cuda', 'cpu'],
        'execution_device_id': 0,
        'execution_thread_count': 4,
        'execution_queue_count': 1,
        'video_memory_strategy': 'moderate',
        'system_memory_limit': 0,
        'log_level': 'info',
        'download_providers': ['github', 'huggingface'],
        'download_scope': 'full',
        'temp_path': '/tmp',
        'temp_frame_format': 'jpg',
        'output_path': '/workspace/output',
        'output_image_quality': 80,
        'output_audio_encoder': 'aac',
        'output_video_encoder': 'libx264',
        'output_video_preset': 'medium',
        'output_video_quality': 80,
        'trim_frame_start': None,
        'trim_frame_end': None,
        'keep_temp': False,
        'skip_audio': False,
        'command': 'run',
        'source_paths': None,
        'target_path': None,
        'output_path': '/workspace/output',
    }
    for key, value in defaults.items():
        state_manager.set_item(key, value)
    
    # Patch get_item to return safe defaults for unset keys
    # FaceFusion crashes when None propagates through string split/int operations
    _orig_get_item = state_manager.get_item
    _SAFE_DEFAULTS = {
        'face_swapper_pixel_boost': '0x0',
        'face_swapper_pixel_boost_type': 'cpu',
        'face_swapper_pixel_boost_scale': 1,
        'face_enhancer_model': 'none',
        'face_enhancer_blend': 80,
        'face_enhancer_pixel_boost': '0x0',
        'face_enhancer_pixel_boost_type': 'cpu',
        'face_enhancer_pixel_boost_scale': 1,
        'face_detector_model': 'many',
        'face_detector_size': '640x640',
        'face_detector_score': 0.5,
        'face_landmarker_score': 0.5,
        'face_selector_mode': 'many',
        'face_selector_order': 'left-right',
        'reference_face_distance': 0.6,
        'face_mask_blur': 0.3,
        'face_mask_padding': (0, 0, 0, 0),
        'execution_thread_count': 4,
        'execution_queue_count': 1,
        'video_memory_strategy': 'moderate',
        'system_memory_limit': 0,
    }
    def safe_get_item(key):
        val = _orig_get_item(key)
        if val is None and key in _SAFE_DEFAULTS:
            return _SAFE_DEFAULTS[key]
        return val
    state_manager.get_item = safe_get_item
    
    # Verify state was stored
    check = state_manager.get_item('download_providers')
    logger.info(f"State init complete. download_providers = {check}")

try:
    from rvc_python import RVC
    RVC_AVAILABLE = True
except ImportError:
    RVC_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stream_server")

PORT = int(os.environ.get("STREAM_PORT", 8888))
MAX_USERS = int(os.environ.get("MAX_USERS", 2))
CALL_DURATION = int(os.environ.get("CALL_DURATION", 65))
GPU_STATUS = "Checking GPU..."

if not RVC_AVAILABLE:
    class RVC:
        def train(self, audio_path): pass
        def infer(self, audio_array): return audio_array


def get_gpu_info():
    """Detect GPU and return status string."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(', ')
            gpu_name = parts[0] if len(parts) > 0 else 'unknown'
            mem_total = parts[1] if len(parts) > 1 else '?'
            mem_used = parts[2] if len(parts) > 2 else '?'
            gpu_util = parts[3] if len(parts) > 3 else '?'
            return f"{gpu_name} | {mem_used}/{mem_total} MiB | GPU {gpu_util}%"
    except Exception as e:
        logger.warning(f"GPU detection failed: {e}")
    return "No GPU detected (CPU mode)"


def init_models():
    global GPU_STATUS
    logger.info("Initializing FaceFusion state...")
    init_facefusion_state()
    
    # Detect GPU
    GPU_STATUS = get_gpu_info()
    logger.info(f"GPU: {GPU_STATUS}")
    
    logger.info("Warming up FaceFusion models...")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    faces = get_many_faces([dummy])
    logger.info(f"FaceFusion models ready ({len(faces)} faces in dummy frame)")


def extract_source_face(image_bytes: bytes) -> Face:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode source image")
    faces = get_many_faces([img])
    if not faces:
        raise ValueError("No face detected in source image")
    return faces[0]


def swap_frame(frame: np.ndarray, source_face: Face) -> np.ndarray:
    many_faces = get_many_faces([frame])
    if not many_faces:
        return frame
    for target_face in many_faces:
        frame = swap_face(source_face, target_face, frame)
    return frame


class SessionWS:
    """Wraps aiohttp WebSocketResponse for our handler."""
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
        self.rvc_model = None
        self.start_time = None
        self.call_active = False
        self.frame_count = 0

    async def handle(self):
        try:
            msg = await asyncio.wait_for(self.ws.recv(), timeout=30)
            setup = json.loads(msg)

            face_b64 = setup.get("source_face")
            audio_b64 = setup.get("reference_audio")
            voice_on = setup.get("enable_voice", True)

            if not face_b64:
                await self.ws.send(json.dumps({"error": "source_face required"}))
                return

            self.source_face = extract_source_face(base64.b64decode(face_b64))
            await self.ws.send(json.dumps({"status": "face_loaded"}))
            await self.ws.send(json.dumps({"status": "gpu_info", "gpu": GPU_STATUS}))

            if voice_on and audio_b64:
                await self.ws.send(json.dumps({"status": "training_voice"}))
                try:
                    audio_bytes = base64.b64decode(audio_b64)
                    audio_path = f"/tmp/ref_{id(self)}.wav"
                    with open(audio_path, "wb") as f:
                        f.write(audio_bytes)
                    self.rvc_model = RVC(model_path=None)
                    self.rvc_model.train(audio_path)
                    await self.ws.send(json.dumps({"status": "voice_ready"}))
                except Exception as e:
                    logger.error(f"RVC error: {e}")
                    await self.ws.send(json.dumps({"status": "voice_failed", "error": str(e)}))
                    self.rvc_model = None
            else:
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
                    await self._proc_audio(msg)
        finally:
            self.call_active = False
            logger.info(f"Session ended. Frames: {self.frame_count}")

    async def _proc_frame(self, msg):
        self.frame_count += 1
        ts = struct.unpack("!I", msg[1:5])[0]
        arr = np.frombuffer(msg[5:], np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
        frame = swap_frame(frame, self.source_face)
        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        try:
            await self.ws.send(struct.pack("!B I", 0, ts) + jpg.tobytes())
        except:
            self.call_active = False

    async def _proc_audio(self, msg):
        ts = struct.unpack("!I", msg[1:5])[0]
        payload = msg[5:]
        if self.rvc_model:
            try:
                arr = np.frombuffer(payload, dtype=np.float32)
                out = self.rvc_model.infer(arr)
                payload = out.astype(np.float32).tobytes()
            except Exception as e:
                logger.error(f"RVC infer error: {e}")
        try:
            await self.ws.send(struct.pack("!B I", 1, ts) + payload)
        except:
            self.call_active = False


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

    # Auto-fill WS URL from the page URL
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

    logger.info(f"Server on http://0.0.0.0:{PORT} (index.html at /, WS at /ws)")
    await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
