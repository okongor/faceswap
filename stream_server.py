"""
Real-time face swap + voice changer streaming server.
Runs on RunPod GPU pod as a persistent warm worker.

Connects via WebSocket:
  1. Client sends source_face + reference_audio
  2. Server loads models, trains RVC voice
  3. Client starts streaming video frames + audio chunks
  4. Server returns swapped frames + converted audio in real-time
  5. After 60s, connection drops
"""
import os, sys, io, json, asyncio, time, struct, logging
import base64
from pathlib import Path

import asyncio
import websockets
import numpy as np
import cv2
import torch

# --- Face swap ---
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# --- RVC ---
# Using rvc_python for real-time voice conversion
# Falls back to a simpler approach if not available
try:
    from rvc_python import RVC
    RVC_AVAILABLE = True
except ImportError:
    RVC_AVAILABLE = False
    logging.warning("rvc_python not available, voice conversion disabled")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stream_server")

PORT = int(os.environ.get("STREAM_PORT", 8765))
MAX_USERS = int(os.environ.get("MAX_USERS", 1))
CALL_DURATION = int(os.environ.get("CALL_DURATION", 65))  # 65s grace, kill at 60

# ──────────────────────────────────────────
# Global models (shared across users)
# ──────────────────────────────────────────

face_swapper = None
face_analyzer = None
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")


def load_face_models():
    global face_analyzer, face_swapper
    logger.info("Loading face analysis model...")
    face_analyzer = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

    logger.info("Loading face swapper model...")
    model_path = insightface.model_zoo.get_model("inswapper_128.onnx", download=True, download_zip=True)
    face_swapper = get_model(model_path)
    face_swapper.prepare(ctx_id=0)

    logger.info("Face models loaded.")


# ──────────────────────────────────────────
# Per-user session
# ──────────────────────────────────────────

class UserSession:
    def __init__(self, websocket):
        self.ws = websocket
        self.source_face = None  # cropped source face embedding
        self.source_face_img = None
        self.rvc_model = None
        self.start_time = None
        self.call_active = False
        self.fps = 0
        self.frame_count = 0

    async def handle(self):
        """Main handler for one user session."""
        try:
            # Phase 1: Receive setup data
            logger.info("Waiting for setup data...")
            msg = await asyncio.wait_for(self.ws.recv(), timeout=30)
            setup = json.loads(msg)

            source_face_b64 = setup.get("source_face")
            reference_audio_b64 = setup.get("reference_audio")
            enable_voice = setup.get("enable_voice", True)

            if not source_face_b64:
                await self.ws.send(json.dumps({"error": "source_face required"}))
                return

            # Decode source face
            face_bytes = base64.b64decode(source_face_b64)
            face_arr = np.frombuffer(face_bytes, np.uint8)
            self.source_face_img = cv2.imdecode(face_arr, cv2.IMREAD_COLOR)

            # Extract source face embedding
            self.source_face = self._extract_face(self.source_face_img)
            if self.source_face is None:
                await self.ws.send(json.dumps({"error": "No face detected in source image"}))
                return

            await self.ws.send(json.dumps({"status": "face_loaded"}))

            # Load RVC model from reference audio (async)
            if enable_voice and RVC_AVAILABLE and reference_audio_b64:
                await self.ws.send(json.dumps({"status": "training_voice"}))
                try:
                    self.rvc_model = await self._load_rvc_model(reference_audio_b64)
                    await self.ws.send(json.dumps({"status": "voice_ready"}))
                except Exception as e:
                    logger.error(f"RVC load failed: {e}")
                    await self.ws.send(json.dumps({"status": "voice_failed", "error": str(e)}))
                    self.rvc_model = None
            else:
                await self.ws.send(json.dumps({"status": "voice_skipped"}))

            # Signal ready for stream
            self.start_time = time.time()
            self.call_active = True
            await self.ws.send(json.dumps({"status": "ready", "duration": CALL_DURATION}))

            # Phase 2: Stream processing loop
            await self._process_stream()

        except asyncio.TimeoutError:
            logger.warning("Setup timeout")
            try:
                await self.ws.send(json.dumps({"error": "setup_timeout"}))
            except: pass
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected during setup")
        except Exception as e:
            logger.error(f"Session error: {e}", exc_info=True)
            try:
                await self.ws.send(json.dumps({"error": str(e)}))
            except: pass

    def _extract_face(self, img):
        """Detect and extract the first face from an image."""
        faces = face_analyzer.get(img)
        if not faces:
            return None
        return faces[0]

    async def _load_rvc_model(self, audio_b64):
        """Build RVC model from reference audio sample."""
        audio_bytes = base64.b64decode(audio_b64)

        # Write to temp file
        audio_path = f"/tmp/ref_audio_{id(self)}.wav"
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        rvc = RVC(model_path=None)  # Will train on the fly
        rvc.train(audio_path)
        return rvc

    async def _process_stream(self):
        """Process incoming video frames and audio chunks."""
        try:
            frame_skip = 0  # Process every frame
            audio_buffer = b""

            while self.call_active:
                elapsed = time.time() - self.start_time
                if elapsed > CALL_DURATION:
                    logger.info("Call duration reached")
                    break

                try:
                    msg = await asyncio.wait_for(self.ws.recv(), timeout=0.5)
                except asyncio.TimeoutError:
                    # Check if still connected
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logger.info("Client disconnected during stream")
                    break

                if isinstance(msg, str):
                    try:
                        cmd = json.loads(msg)
                        if cmd.get("action") == "end_call":
                            break
                        elif cmd.get("action") == "ping":
                            await self.ws.send(json.dumps({"action": "pong", "t": cmd.get("t", 0)}))
                    except json.JSONDecodeError:
                        pass
                    continue

                # Binary messages: interleaved video and audio
                # Format: [1 byte type] [4 byte timestamp] [payload]
                msg_type = msg[0]
                ts = struct.unpack("!I", msg[1:5])[0]
                payload = msg[5:]

                if msg_type == 0:  # Video frame
                    await self._process_frame(payload, ts)
                elif msg_type == 1:  # Audio chunk
                    await self._process_audio(payload, ts)

        except Exception as e:
            logger.error(f"Stream error: {e}")

        finally:
            self.call_active = False
            logger.info(f"Session ended. Processed {self.frame_count} frames")

    async def _process_frame(self, jpeg_bytes, timestamp_ms):
        """Swap face in a single frame."""
        self.frame_count += 1

        # Decode JPEG
        frame_arr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        # Detect target faces
        target_faces = face_analyzer.get(frame)
        if not target_faces:
            # No face found, send frame back unchanged
            await self._send_frame(jpeg_bytes, timestamp_ms)
            return

        # Swap each detected face with source
        for target_face in target_faces:
            frame = face_swapper.get(frame, target_face, self.source_face, paste_back=True)

        # Encode back to JPEG
        _, out_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        await self._send_frame(out_buf.tobytes(), timestamp_ms)

    async def _send_frame(self, jpeg_bytes, timestamp_ms):
        """Send processed frame back to client."""
        header = struct.pack("!B I", 0, timestamp_ms)
        try:
            await self.ws.send(header + jpeg_bytes)
        except websockets.exceptions.ConnectionClosed:
            self.call_active = False

    async def _process_audio(self, pcm_bytes, timestamp_ms):
        """Convert voice using RVC."""
        if self.rvc_model is None:
            # Echo back original audio if no RVC
            await self._send_audio(pcm_bytes, timestamp_ms)
            return

        try:
            # Convert audio chunk through RVC
            pcm_array = np.frombuffer(pcm_bytes, dtype=np.float32)
            converted = self.rvc_model.infer(pcm_array)
            out_bytes = converted.astype(np.float32).tobytes()
        except Exception as e:
            logger.error(f"RVC inference error: {e}")
            out_bytes = pcm_bytes

        await self._send_audio(out_bytes, timestamp_ms)

    async def _send_audio(self, pcm_bytes, timestamp_ms):
        """Send processed audio back to client."""
        header = struct.pack("!B I", 1, timestamp_ms)
        try:
            await self.ws.send(header + pcm_bytes)
        except websockets.exceptions.ConnectionClosed:
            self.call_active = False


# ──────────────────────────────────────────
# WebSocket Server
# ──────────────────────────────────────────

async def handler(websocket):
    """New WebSocket connection from a user."""
    # Check concurrent user limit
    if len(sessions) >= MAX_USERS:
        await websocket.send(json.dumps({"error": "server_full", "message": "All slots busy"}))
        await websocket.close()
        return

    session = UserSession(websocket)
    sessions.add(session)
    try:
        await session.handle()
    finally:
        sessions.discard(session)


sessions = set()


async def main():
    logger.info(f"Starting stream server on port {PORT}, max users: {MAX_USERS}")

    # Pre-load face models
    load_face_models()

    async with websockets.serve(handler, "0.0.0.0", PORT, ping_interval=20, ping_timeout=10):
        logger.info(f"Stream server ready on ws://0.0.0.0:{PORT}")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
