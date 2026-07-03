# Real-time FaceSwap + Voice Clone on RunPod

Live face swap and voice cloning for 60-second video calls.

## Architecture

- **GPU Pod (warm):** Runs the streaming server (face swap + RVC voice conversion)
- **API Server:** Manages queue, quotas, user authentication
- **Client:** Web app that captures webcam + mic, streams to server

## Files

| File | Purpose |
|------|---------|
| `stream_server.py` | Real-time WebSocket server (face swap + voice) |
| `api_server.py` | Queue + quota management API |
| `test_client.html` | Browser test client |
| `Dockerfile` | GPU pod container (CUDA 12.4 + insightface + RVC) |
| `requirements.txt` | Python dependencies |

## Deploy

### 1. GPU Pod (Warm)

On RunPod:
- Deploy as a **GPU Pod** (not serverless)
- Use the Dockerfile in this repo
- Expose port **8765** (WebSocket)
- Set env: `MAX_USERS=1`, `CALL_DURATION=65`

### 2. API Server

Run on a cheap CPU instance:
```bash
pip install fastapi uvicorn
uvicorn api_server:app --host 0.0.0.0 --port 8000
```
Set env: `WARM_POD_URL=ws://<gpu-pod-ip>:8765`

### 3. Test Client

Open `test_client.html` in a browser, enter the WebSocket URL, upload a face image and reference audio, click Start.

## Environment Variables

### GPU Pod
- `STREAM_PORT` - WebSocket port (default: 8765)
- `MAX_USERS` - Concurrent users per GPU (default: 1)
- `CALL_DURATION` - Max call seconds (default: 65)

### API Server
- `MAX_CONCURRENT` - Max simultaneous calls (default: 5)
- `MAX_CALLS_PER_DAY` - Per user daily limit (default: 2)
- `CALL_DURATION` - Call length in seconds (default: 60)
- `WARM_POD_URL` - WebSocket URL of the GPU pod
