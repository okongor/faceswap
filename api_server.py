"""
API Server - Manages queue, quotas, and GPU pod orchestration.
"""
import os, json, uuid, time, logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("api_server")

app = FastAPI(title="FaceSwap + Voice Clone API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", 5))
MAX_CALLS_PER_DAY = int(os.environ.get("MAX_CALLS_PER_DAY", 2))
CALL_DURATION = int(os.environ.get("CALL_DURATION", 60))
WARM_POD_URL = os.environ.get("WARM_POD_URL", "")

@dataclass
class CallSession:
    call_id: str
    user_id: str
    status: str = "queued"
    queue_position: int = 0
    pod_url: str = ""
    created_at: float = 0.0
    started_at: Optional[float] = None

quotas: dict = {}
sessions: dict = {}
queue: list = []
active_count = 0

class StartCallRequest(BaseModel):
    user_id: str
    source_face_url: Optional[str] = None
    reference_audio_url: Optional[str] = None
    enable_voice: bool = True

def today():
    return datetime.utcnow().strftime("%Y-%m-%d")

@app.post("/api/start-call")
async def start_call(req: StartCallRequest):
    global active_count
    call_id = str(uuid.uuid4())[:8]
    today_str = today()
    user_key = f"{req.user_id}:{today_str}"
    calls_today = quotas.get(user_key, 0)
    if calls_today >= MAX_CALLS_PER_DAY:
        raise HTTPException(status_code=429, detail="Daily limit reached")
    if active_count >= MAX_CONCURRENT:
        queue.append(call_id)
        sessions[call_id] = CallSession(call_id=call_id, user_id=req.user_id, status="queued", queue_position=len(queue), created_at=time.time())
        return {"call_id": call_id, "status": "queued", "queue_position": len(queue)}
    active_count += 1
    quotas[user_key] = calls_today + 1
    sessions[call_id] = CallSession(call_id=call_id, user_id=req.user_id, status="connecting", pod_url=WARM_POD_URL, created_at=time.time())
    return {"call_id": call_id, "status": "connecting", "pod_url": WARM_POD_URL, "duration": CALL_DURATION}

@app.get("/api/status/{call_id}")
async def get_status(call_id: str):
    s = sessions.get(call_id)
    if not s:
        raise HTTPException(404, "Call not found")
    return {"call_id": call_id, "status": s.status, "queue_position": s.queue_position if s.status == "queued" else None, "pod_url": s.pod_url or None}

@app.post("/api/end-call/{call_id}")
async def end_call(call_id: str):
    global active_count
    s = sessions.get(call_id)
    if not s:
        raise HTTPException(404, "Call not found")
    s.status = "completed"
    active_count = max(0, active_count - 1)
    if queue:
        next_id = queue.pop(0)
        ns = sessions[next_id]
        ns.status = "connecting"
        ns.pod_url = WARM_POD_URL
        active_count += 1
    return {"status": "completed"}

@app.get("/api/health")
async def health():
    return {"active": active_count, "queue_length": len(queue), "max_concurrent": MAX_CONCURRENT}
