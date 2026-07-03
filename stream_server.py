"""
Real-time face swap + voice changer streaming server.
Uses FaceFusion's internal API, RVC for voice conversion.
"""
import os, sys, json, asyncio, time, struct, logging, base64
from pathlib import Path
import websockets
import numpy as np
import cv2
import torch

from facefusion.face_analyser import get_face_analyser, get_many_faces
from facefusion.processors.frame.modules.face_swapper import get_face_swapper, swap_face

try:
    from rvc_python import RVC
    RVC_AVAILABLE = True
except ImportError:
    RVC_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stream_server")

PORT = int(os.environ.get("STREAM_PORT", 8765))
MAX_USERS = int(os.environ.get("MAX_USERS", 1))
CALL_DURATION = int(os.environ.get("CALL_DURATION", 65))

if not RVC_AVAILABLE:
    class RVC:
        def train(self, audio_path): pass
        def infer(self, audio_array): return audio_array

swapper = None

def init_facefusion():
    global swapper
    logger.info("Loading FaceFusion...")
    get_face_analyser()
    swapper = get_face_swapper()
    logger.info("FaceFusion ready")

def extract_source_face(image_bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    faces = get_many_faces(img)
    if not faces:
        raise ValueError("No face detected in source image")
    return faces[0]

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

            face_bytes = base64.b64decode(face_b64)
            self.source_face = extract_source_face(face_bytes)
            await self.ws.send(json.dumps({"status": "face_loaded"}))

            if voice_on and audio_b64:
                await self.ws.send(json.dumps({"status": "training_voice"}))
                audio_bytes = base64.b64decode(audio_b64)
                audio_path = f"/tmp/ref_{id(self)}.wav"
                with open(audio_path, "wb") as f:
                    f.write(audio_bytes)
                self.rvc_model = RVC(model_path=None)
                self.rvc_model.train(audio_path)
                await self.ws.send(json.dumps({"status": "voice_ready"}))

            self.start_time = time.time()
            self.call_active = True
            await self.ws.send(json.dumps({"status": "ready", "duration": CALL_DURATION}))
            await self._stream_loop()
        except Exception as e:
            logger.error(f"Session error: {e}")

    async def _stream_loop(self):
        while self.call_active:
            if time.time() - self.start_time > CALL_DURATION:
                break
            try:
                msg = await asyncio.wait_for(self.ws.recv(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
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
        self.call_active = False

    async def _proc_frame(self, msg):
        self.frame_count += 1
        ts = struct.unpack("!I", msg[1:5])[0]
        arr = np.frombuffer(msg[5:], np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
        faces = get_many_faces(frame)
        if faces:
            for f in faces:
                frame = swap_face(self.source_face, f, frame)
        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        try:
            await self.ws.send(struct.pack("!B I", 0, ts) + jpg.tobytes())
        except:
            self.call_active = False

    async def _proc_audio(self, msg):
        ts = struct.unpack("!I", msg[1:5])[0]
        payload = msg[5:]
        if self.rvc_model:
            arr = np.frombuffer(payload, dtype=np.float32)
            out = self.rvc_model.infer(arr).astype(np.float32).tobytes()
        else:
            out = payload
        try:
            await self.ws.send(struct.pack("!B I", 1, ts) + out)
        except:
            self.call_active = False

sessions = set()

async def handler(ws):
    if len(sessions) >= MAX_USERS:
        await ws.send(json.dumps({"error": "server_full"}))
        await ws.close()
        return
    s = UserSession(ws)
    sessions.add(s)
    try:
        await s.handle()
    finally:
        sessions.discard(s)

async def main():
    init_facefusion()
    logger.info(f"Stream server on ws://0.0.0.0:{PORT}")
    async with websockets.serve(handler, "0.0.0.0", PORT):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
