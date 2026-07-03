"""
API Server - Manages queue, quotas, and GPU pod orchestration.

Endpoints:
  POST /api/start-call - Submit a call request
  GET  /api/status/:id  - Check call status
  POST /api/end-call/:id - End a call early

Runs on a cheap VPS or RunPod CPU instance.
"""
import os, json, uuid, time, threading, logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("api_server")

app = FastAPI(title="FaceSwap + Voice Clone API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ──────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────

MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", 5))
MAX_CALLS_PER_DAY = int(os.environ.get("MAX_CALLS_PER_DAY", 2))
CALL_DURATION = int(os.environ.get("CALL_DURATION", 60))
WARM_POD_URL = os.environ.get("WARM_POD_URL", "")  # ws://<pod-ip>:8765

# ──────────────────────────────────────────
# In-memory state (use Redis in production)
# ──────────────────────────────────────────

@dataclass
class UserQuota:
    user_id: str
    calls_today: int = 0
    last_call_date: str = ""  # YYYY-MM-DD

@dataclass
class CallSession:
    call_id: str
    user_id: str
    status: str = "queued"  # queued | connecting | active | completed | failed
    queue_position: int = 0
    pod_url: str = ""
    created_at: float = 0.0
    started_at: Optional[float] = None
    expires_at: Optional[float] = None

quotas: dict[str, UserQuota] = {}
sessions: dict[str, CallSession] = {}
queue: list[str] = []  # call_id order
active_count = 0
lock = threading.Lock()

# ──────────────────────────────────────────
# Models
# ──────────────────────────────────────────

class StartCallRequest(BaseModel):
    user_id: str
    source_face_url: Optional[str] = None
    reference_audio_url: Optional[str] = None
    enable_voice: bool = True

class StartCallResponse(BaseModel):
    call_id: str
    status: str
    queue_position: Optional[int] = None
    pod_url: Optional[str] = None
    duration: int = CALL_DURATION

class StatusResponse(BaseModel):
    call_id: str
    status: str
    queue_position: Optional[int] = None
    pod_url: Optional[str] = None
    remaining_seconds: Optional[int] = None

# ──────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────

def today_str():
    return datetime.utcnow().strftime("%Y-%m-%d")

def get_or_create_quota(user_id: str) -> UserQuota:
    today = today_str()
    if user_id not in quotas or quotas[user_id].last_call_date != today:
        quotas[user_id] = UserQuota(user_id=user_id, calls_today=0, last_call_date=today)
    return quotas[user_id]

def check_quota(user_id: str) -> bool:
    q = get_or_create_quota(user_id)
    return q.calls_today < MAX_CALLS_PER_DAY

def increment_quota(user_id: str):
    q = get_or_create_quota(user_id)
    q.calls_today += 1

# ──────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────

@app.post("/api/start-call", response_model=StartCallResponse)
async def start_call(req: StartCallRequest):
    call_id = str(uuid.uuid4())[:8]
    with lock:
        global active_count

        # Check quota
        if not check_quota(req.user_id):
            raise HTTPException(status_code=429, detail="Daily call limit reached")

        # Check concurrent limit
        if active_count >= MAX_CONCURRENT:
            # Add to queue
            queue.append(call_id)
            sessions[call_id] = CallSession(
                call_id=call_id,
                user_id=req.user_id,
                status="queued",
                queue_position=len(queue),
                created_at=time.time(),
            )
            logger.info(f"User {req.user_id} queued (position {len(queue)})")
            return StartCallResponse(
                call_id=call_id,
                status="queued",
                queue_position=len(queue),
            )

        # Active slot available
        active_count += 1
        increment_quota(req.user_id)

        session = CallSession(
            call_id=call_id,
            user_id=req.user_id,
            status="connecting",
            pod_url=WARM_POD_URL,
            created_at=time.time(),
        )
        sessions[call_id] = session
        logger.info(f"User {req.user_id} starting call {call_id}")

    return StartCallResponse(
        call_id=call_id,
        status="connecting",
        pod_url=WARM_POD_URL,
        duration=CALL_DURATION,
    )


@app.get("/api/status/{call_id}", response_model=StatusResponse)
async def get_status(call_id: str):
    session = sessions.get(call_id)
    if not session:
        raise HTTPException(status_code=404, detail="Call not found")

    remaining = None
    if session.started_at:
        elapsed = time.time() - session.started_at
        remaining = max(0, CALL_DURATION - int(elapsed))

    return StatusResponse(
        call_id=call_id,
        status=session.status,
        queue_position=session.queue_position if session.status == "queued" else None,
        pod_url=session.pod_url or None,
        remaining_seconds=remaining,
    )


@app.post("/api/end-call/{call_id}")
async def end_call(call_id: str):
    with lock:
        global active_count
        session = sessions.get(call_id)
        if not session:
            raise HTTPException(status_code=404, detail="Call not found")

        session.status = "completed"
        active_count = max(0, active_count - 1)

        # Dequeue next waiting user
        if queue:
            next_id = queue.pop(0)
            next_session = sessions[next_id]
            next_session.status = "connecting"
            next_session.pod_url = WARM_POD_URL
            next_session.started_at = time.time()
            active_count += 1
            logger.info(f"Dequeued call {next_id}")

    return {"status": "completed"}


@app.get("/api/health")
async def health():
    with lock:
        return {
            "active": active_count,
            "queue_length": len(queue),
            "max_concurrent": MAX_CONCURRENT,
            "warm_pod": bool(WARM_POD_URL),
        }
